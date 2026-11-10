//! Extension state whose setup can fail.
//!
//! Run with: `cargo run --example fallible_extensions`
//!
//! `OnceTypeMap::get_or_init` takes an initializer that can't fail, and std's
//! `OnceLock::get_or_try_init` isn't stable yet. What to store depends on
//! what a failure means:
//!
//! - The failure is final (`config` below). Store the `Result` itself. Every
//!   caller sees the same outcome, and setup runs once. This needs nothing
//!   beyond the map.
//! - The failure is temporary (`db` below). Store a cell that supports a
//!   fallible initializer, here `once_cell::sync::OnceCell`. A failed attempt
//!   stores nothing, so the next call tries again. As with `get_or_init`,
//!   only one attempt runs at a time and other callers wait for it.

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

/// An extension whose setup fails for good: a malformed setting won't fix
/// itself, so retrying would only repeat the same error.
mod config {
    use crate::http_client::Client;
    use std::fmt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Config {
        pub timeout_ms: u64,
    }

    #[derive(Debug)]
    pub struct ConfigError(String);

    impl fmt::Display for ConfigError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "bad config: {}", self.0)
        }
    }

    // Private wrapper, so the map key is private to this module.
    struct Loaded(Result<Config, ConfigError>);

    static LOADS: AtomicUsize = AtomicUsize::new(0);

    pub trait ClientConfigExt {
        fn config(&self) -> Result<&Config, &ConfigError>;
    }

    impl ClientConfigExt for Client {
        fn config(&self) -> Result<&Config, &ConfigError> {
            let loaded = self.extensions().get_or_init(|| Loaded(load()));
            loaded.0.as_ref()
        }
    }

    /// Stands in for reading and parsing a settings file.
    fn load() -> Result<Config, ConfigError> {
        LOADS.fetch_add(1, Ordering::Relaxed);
        let raw = "fast";
        let timeout_ms = raw
            .parse()
            .map_err(|_| ConfigError(format!("timeout_ms = {raw:?} is not a number")))?;
        Ok(Config { timeout_ms })
    }

    pub fn load_count() -> usize {
        LOADS.load(Ordering::Relaxed)
    }
}

/// An extension whose setup can fail for a while: the database may not be
/// up yet, so a later call should try again.
mod db {
    use crate::http_client::Client;
    use once_cell::sync::OnceCell;
    use std::fmt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Connection {
        id: usize,
    }

    impl Connection {
        pub fn id(&self) -> usize {
            self.id
        }
    }

    #[derive(Debug)]
    pub struct ConnectError;

    impl fmt::Display for ConnectError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("database not ready")
        }
    }

    // Private wrapper, so the map key is private to this module.
    struct DbCell(OnceCell<Connection>);

    static ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

    pub trait ClientDbExt {
        fn db(&self) -> Result<&Connection, ConnectError>;
    }

    impl ClientDbExt for Client {
        fn db(&self) -> Result<&Connection, ConnectError> {
            let cell = self.extensions().get_or_init(|| DbCell(OnceCell::new()));
            cell.0.get_or_try_init(|| connect(self.base_url()))
        }
    }

    /// Stands in for opening a connection. The first attempt fails.
    fn connect(base_url: &str) -> Result<Connection, ConnectError> {
        let attempt = ATTEMPTS.fetch_add(1, Ordering::Relaxed) + 1;
        println!("db: connecting to {base_url}/db, attempt {attempt}");
        if attempt == 1 {
            Err(ConnectError)
        } else {
            Ok(Connection { id: attempt })
        }
    }

    pub fn attempt_count() -> usize {
        ATTEMPTS.load(Ordering::Relaxed)
    }
}

// The application.
use config::ClientConfigExt;
use db::ClientDbExt;
use http_client::Client;

fn main() {
    let client = Client::new("https://api.example.com");

    // A final failure: both calls get the same stored error.
    for _ in 0..2 {
        match client.config() {
            Ok(config) => println!("config: timeout {} ms", config.timeout_ms),
            Err(error) => println!("config: {error}"),
        }
    }
    assert_eq!(config::load_count(), 1);

    // A temporary failure: the first call fails, the second connects, and
    // the third reuses that connection.
    for _ in 0..3 {
        match client.db() {
            Ok(conn) => println!("db: using connection #{}", conn.id()),
            Err(error) => println!("db: {error}, will retry"),
        }
    }
    assert_eq!(db::attempt_count(), 2);
}
