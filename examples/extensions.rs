//! Extending a struct from code that its author never saw.
//!
//! Run with: `cargo run --example extensions`
//!
//! The three modules below stand in for three separate crates:
//!
//! - `http_client`: a library that defines `Client`. It embeds a
//!   `OnceTypeMap` and knows nothing about what will be stored in it.
//! - `auth` and `stats`: two independent extension crates. Each one adds
//!   methods to `Client` through an extension trait and keeps its state in
//!   the client's map.
//! - `main`: the application. It imports the extension traits and calls
//!   their methods as if `Client` had always had them.

/// The library crate. It owns the struct and offers a place for extensions.
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

        /// Storage for state that other crates attach to this client.
        pub fn extensions(&self) -> &OnceTypeMap {
            &self.extensions
        }
    }
}

/// An extension crate that adds an auth token, set once by the application.
mod auth {
    use crate::http_client::Client;

    // Private newtype: it is the key in the map, so no other crate can read
    // or overwrite this extension's state, and it can't collide with some
    // other crate that also stores a `String`.
    struct Token(String);

    pub trait ClientAuthExt {
        /// Sets the token. Returns `false` if a token was already set.
        fn set_token(&self, token: &str) -> bool;
        fn token(&self) -> Option<&str>;
    }

    impl ClientAuthExt for Client {
        fn set_token(&self, token: &str) -> bool {
            self.extensions().insert(Token(token.to_string())).is_ok()
        }

        fn token(&self) -> Option<&str> {
            // A plain `&str` borrowed from the client: no lock guard.
            self.extensions().get::<Token>().map(|t| t.0.as_str())
        }
    }
}

/// Another extension crate, independent of `auth`. It creates its state
/// lazily the first time it is used.
mod stats {
    use crate::http_client::Client;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Default)]
    pub struct Stats {
        requests: AtomicU64,
    }

    impl Stats {
        pub fn record_request(&self) {
            self.requests.fetch_add(1, Ordering::Relaxed);
        }

        pub fn requests(&self) -> u64 {
            self.requests.load(Ordering::Relaxed)
        }
    }

    pub trait ClientStatsExt {
        fn stats(&self) -> &Stats;
    }

    impl ClientStatsExt for Client {
        fn stats(&self) -> &Stats {
            self.extensions().get_or_init(Stats::default)
        }
    }
}

// The application. Importing the traits is all it takes to use the
// extensions.
use auth::ClientAuthExt;
use http_client::Client;
use stats::ClientStatsExt;

fn send(client: &Client, path: &str) {
    client.stats().record_request();
    let signed_in = if client.token().is_some() {
        "with token"
    } else {
        "anonymous"
    };
    println!("GET {}{path} ({signed_in})", client.base_url());
}

fn main() {
    let client = Client::new("https://api.example.com");

    send(&client, "/status");

    assert!(client.set_token("s3cret"));
    assert!(!client.set_token("other"), "the token is set only once");

    // Extensions work through `&Client`, so threads can share the client.
    std::thread::scope(|s| {
        s.spawn(|| send(&client, "/users"));
        s.spawn(|| send(&client, "/orders"));
    });

    println!("{} requests sent", client.stats().requests());
    assert_eq!(client.stats().requests(), 3);
    assert_eq!(client.token(), Some("s3cret"));
}
