//! Extension state that takes async work to create.
//!
//! Run with: `cargo run --example async_extensions`
//!
//! `OnceTypeMap::get_or_init` blocks the thread while another caller runs the
//! initializer. That is fine for quick setup, but in async code it would
//! stall the executor. The fix is to store an async cell as the value:
//!
//! 1. The map's `get_or_init` only creates the empty cell, which is instant.
//! 2. The cell's async `get_or_init` does the slow work. Tasks that arrive
//!    while it runs wait for it without blocking the thread.
//!
//! The runtime below is single-threaded on purpose. If anything blocked the
//! thread, the heartbeat would stop ticking while the token is fetched.

/// The library crate, the same as in the `extensions` example.
mod http_client {
    use once_type_map::OnceTypeMap;

    pub struct Client {
        base_url: String,
        extensions: OnceTypeMap,
    }

    impl Client {
        pub fn new(base_url: &str) -> Self {
            Client {
                base_url: base_url.to_string(),
                extensions: OnceTypeMap::new(),
            }
        }

        pub fn base_url(&self) -> &str {
            &self.base_url
        }

        pub fn extensions(&self) -> &OnceTypeMap {
            &self.extensions
        }
    }
}

/// An extension crate whose state comes from a slow login call.
mod auth {
    use crate::http_client::Client;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::OnceCell;

    // Private, so the map key `OnceCell<Token>` is private too: no other
    // crate can reach this extension's state.
    struct Token(String);

    static LOGINS: AtomicUsize = AtomicUsize::new(0);

    // `async fn` keeps this example short. In a library's public trait,
    // promise a `Send` future instead, so callers can use it with
    // `tokio::spawn` (rustc's `async_fn_in_trait` lint asks for this):
    //
    //     fn token(&self) -> impl Future<Output = &str> + Send;
    //
    // The `async fn` in the impl below works with either signature.
    pub trait ClientAuthExt {
        /// Returns the token, logging in on first use.
        async fn token(&self) -> &str;
    }

    impl ClientAuthExt for Client {
        async fn token(&self) -> &str {
            let cell = self.extensions().get_or_init(OnceCell::<Token>::new);
            let token = cell.get_or_init(|| login(self.base_url())).await;
            &token.0
        }
    }

    /// Stands in for a slow call to an auth server.
    async fn login(base_url: &str) -> Token {
        LOGINS.fetch_add(1, Ordering::Relaxed);
        println!("auth: logging in at {base_url}/login ...");
        tokio::time::sleep(Duration::from_millis(350)).await;
        println!("auth: got a token");
        Token("s3cret".to_string())
    }

    pub fn login_count() -> usize {
        LOGINS.load(Ordering::Relaxed)
    }
}

// The application.
use auth::ClientAuthExt;
use http_client::Client;
use std::time::{Duration, Instant};

async fn send(client: &Client, path: &str) {
    let token = client.token().await;
    println!(
        "GET {}{path} (token: {} chars)",
        client.base_url(),
        token.len()
    );
}

/// Stands in for other work the application does in the meantime.
async fn heartbeat() {
    let start = Instant::now();
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        println!("heartbeat: {} ms", start.elapsed().as_millis());
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let client = Client::new("https://api.example.com");

    // Three requests start at once. The first one logs in; the other two wait
    // for that login instead of starting their own. The heartbeat keeps
    // ticking the whole time, so nothing blocks the thread.
    let requests = async {
        tokio::join!(
            send(&client, "/users"),
            send(&client, "/orders"),
            send(&client, "/status"),
        )
    };
    tokio::select! {
        _ = requests => {}
        _ = heartbeat() => {}
    }

    // The token is stored now, so this request doesn't wait.
    send(&client, "/users/42").await;

    assert_eq!(auth::login_count(), 1);
}
