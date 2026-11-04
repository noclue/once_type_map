# once_type_map

A thread-safe, insert-once type map that hands out plain `&T` references, not
lock guards.

You can think of it as a `OnceLock` per type, for types you don't know when you
write the owning struct. Library authors can embed one `OnceTypeMap` in a
long-lived object (a client, a session, an app context). Downstream crates can
then attach their own state to that object, such as caches, adapters, or
implementers of extra traits, without the original struct knowing about them.

```rust
use once_type_map::OnceTypeMap;

struct Client {
    extensions: OnceTypeMap,
}

impl Client {
    pub fn extensions(&self) -> &OnceTypeMap {
        &self.extensions
    }
}

// Somewhere in a downstream crate:
struct MyCache { /* ... */ }

fn cache(client: &Client) -> &MyCache {
    client.extensions().get_or_init(|| MyCache { /* expensive setup */ })
}
```

## Why references instead of guards?

Typical type maps (`http::Extensions`, `anymap`, `RwLock<HashMap<TypeId, Box<dyn Any>>>`)
either need `&mut self` to insert or return a guard that borrows the lock. Both
make it hard for a wrapper type to expose `fn foo(&self) -> &Foo`.

`OnceTypeMap` returns a `&T` that lives as long as the borrow of the map. This
is sound because of three rules:

1. Each type's value lives in its own heap-allocated `OnceLock<T>` slot. When
   the internal hash map grows, it moves the pointer to the slot, not the slot.
2. Through `&self`, a slot can only be created and filled once.
3. Slots are only mutated, removed or dropped through `&mut self` or by
   dropping the map. The borrow checker allows that only when no `&T` handed
   out by the map is still alive.

The same reasoning is why `OnceLock` itself can offer `get_mut` and `take`.

## API

| Method | Receiver | Behavior |
| --- | --- | --- |
| `get::<T>()` | `&self` | `Some(&T)` if set |
| `get_or_init(f)` | `&self` | Returns the value, running `f` exactly once per type if unset |
| `insert(value)` | `&self` | `Ok(&T)` if it was unset, otherwise gives the value back in `Err` |
| `contains::<T>()` | `&self` | Whether a value is set |
| `len()` / `is_empty()` | `&self` | Number of types whose value is set |
| `get_mut::<T>()` | `&mut self` | `Some(&mut T)` if set |
| `remove::<T>()` | `&mut self` | Takes the value out; the type can be set again |
| `clear()` | `&mut self` | Removes all values |

## Initialization semantics

- `get_or_init` runs the initializer outside the map's lock. A slow
  initializer for one type doesn't block other types, and an initializer may
  use the map for other types.
- For a given type, the initializer runs exactly once, even when many threads
  race. The other threads block until the value is ready.
- If the initializer panics, the value stays unset and the next call retries.
- Calling `get_or_init::<T>` from inside the initializer for the same `T` is a
  reentrant `OnceLock` initialization. The standard library leaves the outcome
  unspecified, and today it deadlocks.

## Caveats

- One value per type. Use newtypes to store several values of the same
  underlying type.
- Values must be `Send + Sync + 'static`.
- Initializers are synchronous and can't fail. If you need fallible or async
  initialization, store an `OnceLock`-like cell of your own (for example
  `tokio::sync::OnceCell`) as the value.
- A lookup takes a shared read lock. Because the returned reference isn't tied
  to a guard, hot paths can look a value up once and keep the `&T`.

## Checking the unsafe code

The crate has one `unsafe` block, which extends a slot reference from the lock
guard's lifetime to the borrow of the map. The test suite includes a test that
keeps a reference alive while the map grows. Run it under
[Miri](https://github.com/rust-lang/miri) with both aliasing models:

```sh
cargo +nightly miri test
MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test
```

## License

Apache-2.0
