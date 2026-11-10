//! A thread-safe, insert-once type map that hands out plain references.
//!
//! [`OnceTypeMap`] stores at most one value per type. Think of it as a
//! [`OnceLock`] per type, for types that are not known when the owning struct
//! is written. Values are set through `&self` and returned as plain `&T`
//! references tied to the borrow of the map — the caller never holds a lock
//! guard.
//!
//! This lets an object embed one `OnceTypeMap` and let other code attach
//! state to it (caches, adapters, implementers of extra traits) while the
//! object keeps exposing ordinary `fn foo(&self) -> &Foo` accessors.
//!
//! ```rust
//! use once_type_map::OnceTypeMap;
//!
//! struct Client {
//!     extensions: OnceTypeMap,
//! }
//!
//! impl Client {
//!     fn extensions(&self) -> &OnceTypeMap {
//!         &self.extensions
//!     }
//! }
//!
//! struct Metrics {
//!     name: &'static str,
//! }
//!
//! let client = Client { extensions: OnceTypeMap::new() };
//! let metrics: &Metrics = client.extensions().get_or_init(|| Metrics { name: "requests" });
//! assert_eq!(metrics.name, "requests");
//! ```
//!
//! # Why the references outlive the internal lock
//!
//! Each type gets its own heap-allocated [`OnceLock<T>`] slot. Through `&self`
//! a slot can only be created and filled once; it is never moved, replaced or
//! dropped. The internal hash map may reallocate when other types are added,
//! but that moves only the pointer to a slot, not the slot itself. A `&T`
//! therefore stays valid for as long as the map is borrowed.
//!
//! Mutating or removing values requires `&mut self` ([`get_mut`], [`remove`],
//! [`clear`]). The borrow checker allows that only when no `&T` handed out by
//! the map is alive — the same reasoning that lets [`OnceLock::get_mut`] and
//! [`OnceLock::take`] exist.
//!
//! # What a live reference allows
//!
//! While a reference obtained from the map is alive, the map is *shared*, not
//! frozen. Anything that takes `&self` still works, including adding new
//! types, from this thread or others:
//!
//! ```rust
//! use once_type_map::OnceTypeMap;
//!
//! let map = OnceTypeMap::new();
//! let name: &String = map.get_or_init(|| "first".to_string());
//!
//! map.get_or_init(|| 42i32);
//! map.insert(2.5f64).unwrap();
//! std::thread::scope(|s| {
//!     s.spawn(|| map.get_or_init(|| 7u8));
//! });
//!
//! assert_eq!(map.len(), 4);
//! assert_eq!(name, "first");
//! ```
//!
//! Anything that takes `&mut self` — [`get_mut`], [`remove`], [`clear`] — is
//! rejected by the borrow checker while the reference is alive:
//!
//! ```compile_fail,E0502
//! use once_type_map::OnceTypeMap;
//!
//! let mut map = OnceTypeMap::new();
//! let name: &String = map.get_or_init(|| "first".to_string());
//!
//! map.clear(); // error: `map` is already borrowed as immutable
//!
//! println!("{name}");
//! ```
//!
//! So is moving or dropping the map, which would free the value:
//!
//! ```compile_fail,E0505
//! use once_type_map::OnceTypeMap;
//!
//! let map = OnceTypeMap::new();
//! let name: &String = map.get_or_init(|| "first".to_string());
//!
//! drop(map); // error: cannot move out of `map` because it is borrowed
//!
//! println!("{name}");
//! ```
//!
//! Once the reference is no longer used, the borrow ends and mutation is
//! allowed again:
//!
//! ```rust
//! use once_type_map::OnceTypeMap;
//!
//! let mut map = OnceTypeMap::new();
//! let count: &i32 = map.get_or_init(|| 1);
//! assert_eq!(*count, 1);
//!
//! *map.get_mut::<i32>().unwrap() += 1;
//! assert_eq!(map.remove::<i32>(), Some(2));
//! ```
//!
//! # Initialization
//!
//! [`get_or_init`] runs the initializer outside the map's lock, so a slow
//! initializer for one type does not block other types, and an initializer may
//! itself use the map for *other* types.
//!
//! When many threads race to initialize the same type, only one initializer
//! runs; the others block until the value is ready, then return it. Once an
//! initializer returns, no other initializer for that type runs until the
//! value is removed. If the initializer panics, the slot stays empty and the
//! next call runs its own initializer. So an initializer can run more than
//! once over the life of the map: after a panic, and after [`remove`] or
//! [`clear`].
//!
//! ## Deadlocks
//!
//! Calling [`get_or_init::<T>`](OnceTypeMap::get_or_init) or
//! [`insert::<T>`](OnceTypeMap::insert) from inside the initializer for the
//! same `T` is a reentrant `OnceLock` initialization. The standard library
//! leaves the outcome unspecified, and today it deadlocks.
//!
//! Initializers that depend on each other in a cycle deadlock too. If the
//! initializer for `A` uses `B` and the initializer for `B` uses `A`, then:
//!
//! - on one thread, initializing `A` initializes `B`, which re-enters `A`'s
//!   initialization (the reentrant case above);
//! - on two threads, one starting with `A` and the other with `B`, each thread
//!   blocks forever waiting for the slot the other is initializing.
//!
//! Neither case panics or reports an error. Keep the dependencies between
//! initializers acyclic.
//!
//! ## Empty slots
//!
//! The map creates a type's slot the first time [`get_or_init`] or
//! [`insert`] is called for it, before the value exists. If the initializer
//! panics, the empty slot stays in the map. Nothing called through `&self`
//! removes it, though a later successful initialization reuses it. This costs
//! at most one small allocation per distinct type, but [`len`] and
//! [`is_empty`] scan empty slots along with full ones. [`remove::<T>`] and
//! [`clear`] free empty slots too.
//!
//! # Limitations
//!
//! - One value per type. Use newtypes to store several values of the same
//!   underlying type.
//! - Initializers are synchronous and can't fail. See below for how to handle
//!   setup that can fail or is async.
//!
//! # Fallible and async initialization
//!
//! std's `OnceLock::get_or_try_init` is not stable yet, so the map has no
//! `get_or_try_init`. What to store instead depends on what a failure means:
//!
//! - **The failure is final.** Store the `Result` itself. Every caller sees
//!   the same outcome, and setup runs once:
//!
//!   ```rust
//!   # use once_type_map::OnceTypeMap;
//!   # let map = OnceTypeMap::new();
//!   struct Port(Result<u16, std::num::ParseIntError>);
//!
//!   let port = &map.get_or_init(|| Port("80a".parse())).0;
//!   assert!(port.is_err());
//!   ```
//!
//! - **The failure is temporary.** Store a cell that supports a fallible
//!   initializer, such as `once_cell::sync::OnceCell`, and call its
//!   `get_or_try_init`. A failed attempt stores nothing, so the next call
//!   tries again, and only one attempt runs at a time.
//!
//! - **Setup is async.** Store an async cell, such as
//!   `tokio::sync::OnceCell`. The map's `get_or_init` only creates the empty
//!   cell, and the cell's async `get_or_init` (or `get_or_try_init`) does the
//!   work without blocking the executor. Don't run slow or blocking setup in
//!   the map's own `get_or_init` from async code: callers that wait on it
//!   block their thread.
//!
//! The `fallible_extensions` and `async_extensions` examples in the
//! repository show these patterns. Once std stabilizes
//! `OnceLock::get_or_try_init`, the map will gain a matching method.
//!
//! # Cost
//!
//! A lookup takes a shared read lock and does one hash map probe keyed by
//! [`TypeId`]. Since the returned reference is not tied to a guard, callers on
//! hot paths can look a value up once and keep the `&T`.
//!
//! [`get_or_init`]: OnceTypeMap::get_or_init
//! [`insert`]: OnceTypeMap::insert
//! [`len`]: OnceTypeMap::len
//! [`is_empty`]: OnceTypeMap::is_empty
//! [`get_mut`]: OnceTypeMap::get_mut
//! [`remove`]: OnceTypeMap::remove
//! [`remove::<T>`]: OnceTypeMap::remove
//! [`clear`]: OnceTypeMap::clear

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt,
    hash::{BuildHasherDefault, Hasher},
    sync::{OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

type Slots = HashMap<TypeId, Box<dyn Slot>, BuildHasherDefault<TypeIdHasher>>;

/// A thread-safe map holding at most one value per type, set once through
/// `&self` and read through plain references.
///
/// See the [crate documentation](crate) for the guarantees that make the
/// guard-free references sound.
///
/// # Examples
///
/// ```rust
/// use once_type_map::OnceTypeMap;
///
/// let map = OnceTypeMap::new();
///
/// assert_eq!(*map.get_or_init(|| 42i32), 42);
/// // The value is already set, so this initializer does not run.
/// assert_eq!(*map.get_or_init(|| 99i32), 42);
///
/// assert_eq!(map.insert("hello".to_string()), Ok(&"hello".to_string()));
/// assert_eq!(map.get::<String>().map(String::as_str), Some("hello"));
/// assert_eq!(map.get::<u8>(), None);
/// ```
#[derive(Default)]
pub struct OnceTypeMap {
    slots: RwLock<Slots>,
}

impl OnceTypeMap {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the value of type `T`, or `None` if it has not been set.
    ///
    /// Returns `None` while another thread is still running the initializer
    /// for `T`; it does not wait for it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use once_type_map::OnceTypeMap;
    ///
    /// let map = OnceTypeMap::new();
    /// assert_eq!(map.get::<i32>(), None);
    /// map.get_or_init(|| 5i32);
    /// assert_eq!(map.get::<i32>(), Some(&5));
    /// ```
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.slot::<T>()?.get()
    }

    /// Returns the value of type `T`, initializing it with `init` if it has
    /// not been set.
    ///
    /// When called concurrently from many threads, only one initializer runs;
    /// the other callers block until the value is ready. Once an initializer
    /// returns, no other initializer for `T` runs until the value is removed
    /// with [`remove`](Self::remove) or [`clear`](Self::clear). If `init`
    /// panics, the panic propagates, the value stays unset and a later call
    /// runs its own initializer.
    ///
    /// Calling `get_or_init::<T>` or [`insert::<T>`](Self::insert) from
    /// inside `init` deadlocks, and so do initializers that depend on each
    /// other in a cycle. See [Deadlocks](crate#deadlocks).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use once_type_map::OnceTypeMap;
    ///
    /// let map = OnceTypeMap::new();
    /// let first: &i32 = map.get_or_init(|| 42);
    /// let second: &i32 = map.get_or_init(|| unreachable!());
    /// assert!(std::ptr::eq(first, second));
    /// ```
    pub fn get_or_init<T: Send + Sync + 'static>(&self, init: impl FnOnce() -> T) -> &T {
        self.slot_or_create::<T>().get_or_init(init)
    }

    /// Sets the value of type `T` if it has not been set.
    ///
    /// Returns a reference to the stored value, or gives `value` back in
    /// `Err` if a value of type `T` is already present. If another thread is
    /// initializing `T`, this waits for it to finish.
    ///
    /// Calling `insert::<T>` from inside an initializer for the same `T`
    /// deadlocks instead of returning `Err`. See [Deadlocks](crate#deadlocks).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use once_type_map::OnceTypeMap;
    ///
    /// let map = OnceTypeMap::new();
    /// assert_eq!(map.insert(1u32), Ok(&1));
    /// assert_eq!(map.insert(2u32), Err(2));
    /// assert_eq!(map.get::<u32>(), Some(&1));
    /// ```
    pub fn insert<T: Send + Sync + 'static>(&self, value: T) -> Result<&T, T> {
        let slot = self.slot_or_create::<T>();
        slot.set(value)?;
        // Only `&mut self` can empty the slot, so the value set above is
        // still there.
        Ok(slot.get().expect("value was just set"))
    }

    /// Returns `true` if a value of type `T` has been set.
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.get::<T>().is_some()
    }

    /// Returns a mutable reference to the value of type `T`, if set.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use once_type_map::OnceTypeMap;
    ///
    /// let mut map = OnceTypeMap::new();
    /// map.get_or_init(|| vec![1, 2]);
    /// map.get_mut::<Vec<i32>>().unwrap().push(3);
    /// assert_eq!(map.get::<Vec<i32>>(), Some(&vec![1, 2, 3]));
    /// ```
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        let slot = self.slots_mut().get_mut(&TypeId::of::<T>())?;
        downcast_slot_mut::<T>(&mut **slot).get_mut()
    }

    /// Removes the value of type `T` and returns it, if set.
    ///
    /// After removal, the type can be set again. This also frees the empty
    /// slot left behind by an initializer that panicked; in that case it
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use once_type_map::OnceTypeMap;
    ///
    /// let mut map = OnceTypeMap::new();
    /// map.get_or_init(|| 7u8);
    /// assert_eq!(map.remove::<u8>(), Some(7));
    /// assert_eq!(map.get::<u8>(), None);
    /// assert_eq!(*map.get_or_init(|| 8u8), 8);
    /// ```
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        let slot = self.slots_mut().remove(&TypeId::of::<T>())?;
        downcast_slot_box::<T>(slot).into_inner()
    }

    /// Removes all values, and frees all slots, including empty ones.
    pub fn clear(&mut self) {
        self.slots_mut().clear();
    }

    /// Returns the number of types whose value is set.
    ///
    /// Types whose initializer is still running, or panicked, are not
    /// counted. This walks all slots, so it is `O(n)` in the number of types
    /// ever used, including types whose slot was left empty by a panicking
    /// initializer (see [Empty slots](crate#empty-slots)).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use once_type_map::OnceTypeMap;
    ///
    /// let map = OnceTypeMap::new();
    /// assert_eq!(map.len(), 0);
    /// map.get_or_init(|| 5i32);
    /// assert_eq!(map.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.read().values().filter(|slot| slot.is_set()).count()
    }

    /// Returns `true` if no value is set.
    ///
    /// Like [`len`](Self::len), this may scan every slot, including empty
    /// ones.
    pub fn is_empty(&self) -> bool {
        !self.read().values().any(|slot| slot.is_set())
    }

    fn slot<T: Send + Sync + 'static>(&self) -> Option<&OnceLock<T>> {
        let slots = self.read();
        let slot = downcast_slot::<T>(&**slots.get(&TypeId::of::<T>())?);
        // SAFETY: `slot` is owned by `self.slots`.
        Some(unsafe { self.detach(slot) })
    }

    fn slot_or_create<T: Send + Sync + 'static>(&self) -> &OnceLock<T> {
        if let Some(slot) = self.slot::<T>() {
            return slot;
        }
        self.create_slot::<T>()
    }

    /// The write-locked half of [`slot_or_create`](Self::slot_or_create).
    /// Another thread may have created the slot after our read-locked
    /// lookup missed, so the entry may already be occupied, and that slot's
    /// value may be borrowed. `or_insert_with` then hands out a `&mut` to the
    /// `Box` pointer stored in the map, but not to the `OnceLock` it points
    /// to, so outstanding `&T`s stay valid. The
    /// `create_slot_on_occupied_entry_keeps_references_valid` test checks
    /// this under Miri.
    fn create_slot<T: Send + Sync + 'static>(&self) -> &OnceLock<T> {
        let mut slots = self.write();
        let slot = slots
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(OnceLock::<T>::new()));
        let slot = downcast_slot::<T>(&**slot);
        // SAFETY: `slot` is owned by `self.slots`.
        unsafe { self.detach(slot) }
    }

    /// Extends the lifetime of a slot reference from the lock guard it was
    /// obtained through to the borrow of `self`.
    ///
    /// # Safety
    ///
    /// `slot` must point into a slot owned by `self.slots`. This is sound
    /// because slots are boxed, so rehashing never moves them, and slots are
    /// only removed or replaced through `&mut self`, which cannot coexist with
    /// the `&'a self` borrow the result is tied to.
    unsafe fn detach<'a, T>(&'a self, slot: &OnceLock<T>) -> &'a OnceLock<T> {
        unsafe { &*(slot as *const OnceLock<T>) }
    }

    // No user code runs while the lock is held, and the hash map stays
    // consistent if one of its operations unwinds, so poisoning is ignored.
    fn read(&self) -> RwLockReadGuard<'_, Slots> {
        self.slots.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Slots> {
        self.slots.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn slots_mut(&mut self) -> &mut Slots {
        self.slots.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

impl fmt::Debug for OnceTypeMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnceTypeMap")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

// Every slot is inserted under the `TypeId` of its own `T`, so these
// downcasts cannot fail. If that invariant is ever broken, all access paths
// panic the same way rather than some of them reporting a missing value (and
// `remove` silently dropping it).
const SLOT_TYPE_MISMATCH: &str = "slot type always matches its TypeId key";

// Each helper upcasts `dyn Slot` to `dyn Any` (trait upcasting, Rust 1.86)
// and then downcasts to the concrete slot type.

fn downcast_slot<T: Send + Sync + 'static>(slot: &dyn Slot) -> &OnceLock<T> {
    let slot: &dyn Any = slot;
    slot.downcast_ref().expect(SLOT_TYPE_MISMATCH)
}

fn downcast_slot_mut<T: Send + Sync + 'static>(slot: &mut dyn Slot) -> &mut OnceLock<T> {
    let slot: &mut dyn Any = slot;
    slot.downcast_mut().expect(SLOT_TYPE_MISMATCH)
}

fn downcast_slot_box<T: Send + Sync + 'static>(slot: Box<dyn Slot>) -> Box<OnceLock<T>> {
    let slot: Box<dyn Any> = slot;
    slot.downcast().expect(SLOT_TYPE_MISMATCH)
}

/// Type-erased `OnceLock<T>`.
trait Slot: Any + Send + Sync {
    fn is_set(&self) -> bool;
}

impl<T: Send + Sync + 'static> Slot for OnceLock<T> {
    fn is_set(&self) -> bool {
        self.get().is_some()
    }
}

/// `TypeId` is already a hash, so it is passed through instead of being
/// hashed again with SipHash.
///
/// `TypeId` stores 128 bits, but its `Hash` impl feeds only the low 64 bits
/// to `write_u64`, so `write_u128` is not needed. `write` is a fallback in
/// case that changes; the `type_id_hashes_through_write_u64` test catches it.
#[derive(Default)]
struct TypeIdHasher(u64);

impl Hasher for TypeIdHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 ^= n;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OnceTypeMap>();
    }

    #[test]
    fn get_or_init_then_get() {
        let map = OnceTypeMap::new();
        assert_eq!(map.get::<i32>(), None);
        assert_eq!(*map.get_or_init(|| 42i32), 42);
        assert_eq!(map.get::<i32>(), Some(&42));
        assert!(map.contains::<i32>());
        assert!(!map.contains::<u32>());
    }

    #[test]
    fn multiple_types_are_independent() {
        let map = OnceTypeMap::new();
        map.get_or_init(|| 42i32);
        map.get_or_init(|| "hello".to_string());
        map.get_or_init(|| 2.5f64);

        assert_eq!(map.len(), 3);
        assert_eq!(map.get::<i32>(), Some(&42));
        assert_eq!(map.get::<String>().map(String::as_str), Some("hello"));
        assert_eq!(map.get::<f64>(), Some(&2.5));
    }

    #[test]
    fn initializer_does_not_run_when_set() {
        let map = OnceTypeMap::new();
        let calls = AtomicUsize::new(0);
        for _ in 0..3 {
            let value = map.get_or_init(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                42i32
            });
            assert_eq!(*value, 42);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn references_point_to_the_same_value() {
        let map = OnceTypeMap::new();
        let a = map.get_or_init(|| 42i32);
        let b = map.get::<i32>().unwrap();
        let c = map.get_or_init::<i32>(|| unreachable!());
        assert!(std::ptr::eq(a, b));
        assert!(std::ptr::eq(b, c));
    }

    #[test]
    fn insert_sets_once_and_returns_rejected_value() {
        let map = OnceTypeMap::new();
        assert_eq!(map.insert(1u32), Ok(&1));
        assert_eq!(map.insert(2u32), Err(2));
        assert_eq!(*map.get_or_init(|| 3u32), 1);
    }

    #[test]
    fn panicking_initializer_leaves_value_unset() {
        let map = OnceTypeMap::new();
        let result = catch_unwind(AssertUnwindSafe(|| {
            map.get_or_init::<i32>(|| panic!("init failed"));
        }));
        assert!(result.is_err());
        assert_eq!(map.get::<i32>(), None);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        assert_eq!(*map.get_or_init(|| 42i32), 42);
    }

    #[test]
    fn insert_after_panicking_initializer() {
        let map = OnceTypeMap::new();
        let _ = catch_unwind(AssertUnwindSafe(|| {
            map.get_or_init::<u8>(|| panic!("init failed"));
        }));
        assert_eq!(map.insert(5u8), Ok(&5));
    }

    #[test]
    fn remove_and_clear_free_empty_slots() {
        let mut map = OnceTypeMap::new();
        for _ in 0..2 {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                map.get_or_init::<i32>(|| panic!("init failed"));
            }));
        }
        let _ = catch_unwind(AssertUnwindSafe(|| {
            map.get_or_init::<u8>(|| panic!("init failed"));
        }));
        // Retrying reuses the slot instead of adding another one.
        assert_eq!(map.slots_mut().len(), 2);
        assert!(map.is_empty());

        assert_eq!(map.remove::<i32>(), None);
        assert_eq!(map.slots_mut().len(), 1);

        map.clear();
        assert_eq!(map.slots_mut().len(), 0);
    }

    #[test]
    fn initializer_can_use_other_types() {
        let map = OnceTypeMap::new();
        let outer = map.get_or_init(|| {
            let inner: &String = map.get_or_init(|| "inner".to_string());
            inner.len()
        });
        assert_eq!(*outer, 5);
        assert_eq!(map.get::<String>().map(String::as_str), Some("inner"));
    }

    struct Marker<const N: usize>;

    macro_rules! fill {
        ($map:expr, $($n:literal)*) => { $( $map.get_or_init(|| Marker::<$n>); )* };
    }

    #[test]
    fn references_survive_map_growth() {
        let map = OnceTypeMap::new();
        let pinned: &String = map.get_or_init(|| "pinned".to_string());
        let address = pinned as *const String;

        fill!(map, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
                   20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39);

        assert_eq!(map.len(), 41);
        assert_eq!(pinned, "pinned");
        assert!(std::ptr::eq(map.get::<String>().unwrap(), address));
    }

    /// The race in `slot_or_create`, made deterministic: the read-locked
    /// lookup misses, another caller then creates and fills the slot and
    /// keeps a `&T`, and only then does this caller take the write lock and
    /// find the entry occupied.
    #[test]
    fn create_slot_on_occupied_entry_keeps_references_valid() {
        let map = OnceTypeMap::new();
        let held: &String = map.get_or_init(|| "held".to_string());

        let again = map.create_slot::<String>();
        assert_eq!(held, "held");
        assert!(std::ptr::eq(again.get().unwrap(), held));

        map.create_slot::<String>().get_or_init(|| unreachable!());
        assert_eq!(held, "held");
    }

    #[test]
    fn zero_sized_values() {
        let mut map = OnceTypeMap::new();
        fill!(map, 0 1);
        assert!(map.get::<Marker<0>>().is_some());
        assert!(map.get_mut::<Marker<1>>().is_some());
        assert!(map.remove::<Marker<0>>().is_some());
        assert!(map.get::<Marker<0>>().is_none());
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn type_id_hashes_through_write_u64() {
        use std::hash::Hash;

        #[derive(Default)]
        struct OnlyWriteU64;
        impl Hasher for OnlyWriteU64 {
            fn write(&mut self, _: &[u8]) {
                panic!("TypeId no longer hashes through write_u64");
            }
            fn write_u64(&mut self, _: u64) {}
            fn finish(&self) -> u64 {
                0
            }
        }

        TypeId::of::<String>().hash(&mut OnlyWriteU64);
    }

    #[test]
    fn get_mut_remove_and_clear() {
        let mut map = OnceTypeMap::new();
        map.get_or_init(|| vec![1, 2]);
        map.get_or_init(|| 7u8);

        map.get_mut::<Vec<i32>>().unwrap().push(3);
        assert_eq!(map.get::<Vec<i32>>(), Some(&vec![1, 2, 3]));
        assert_eq!(map.get_mut::<u16>(), None);

        assert_eq!(map.remove::<Vec<i32>>(), Some(vec![1, 2, 3]));
        assert_eq!(map.remove::<Vec<i32>>(), None);
        assert_eq!(map.len(), 1);

        map.clear();
        assert!(map.is_empty());
        assert_eq!(*map.get_or_init(|| 9u8), 9);
    }

    #[test]
    fn values_are_dropped() {
        let tracker = Arc::new(());
        let mut map = OnceTypeMap::new();
        map.get_or_init(|| Arc::clone(&tracker));
        assert_eq!(Arc::strong_count(&tracker), 2);

        map.clear();
        assert_eq!(Arc::strong_count(&tracker), 1);

        map.get_or_init(|| Arc::clone(&tracker));
        drop(map);
        assert_eq!(Arc::strong_count(&tracker), 1);
    }

    #[test]
    fn recovers_from_poisoned_lock() {
        let map = OnceTypeMap::new();
        map.get_or_init(|| 1i32);
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _guard = map.slots.write().unwrap();
            panic!("poison the lock");
        }));
        assert!(map.slots.is_poisoned());

        assert_eq!(map.get::<i32>(), Some(&1));
        assert_eq!(*map.get_or_init(|| 2u8), 2);
        assert_eq!(map.len(), 2);

        let mut map = map;
        *map.get_mut::<i32>().unwrap() += 1;
        assert_eq!(map.remove::<i32>(), Some(2));
        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn waiter_initializes_after_concurrent_initializer_panics() {
        let map = OnceTypeMap::new();
        let (started_tx, started_rx) = std::sync::mpsc::channel();

        thread::scope(|s| {
            let panicking = s.spawn(|| {
                map.get_or_init::<u32>(|| {
                    started_tx.send(()).unwrap();
                    thread::sleep(Duration::from_millis(50));
                    panic!("init failed");
                });
            });
            // Wait until the first initializer is running, so this call
            // blocks on it rather than winning the race.
            started_rx.recv().unwrap();
            assert_eq!(*map.get_or_init(|| 7u32), 7);
            assert!(panicking.join().is_err());
        });
        assert_eq!(map.get::<u32>(), Some(&7));
    }

    #[test]
    fn concurrent_initializer_runs_exactly_once() {
        const THREADS: usize = 8;
        let rounds = if cfg!(miri) { 2 } else { 100 };

        for _ in 0..rounds {
            let map = Arc::new(OnceTypeMap::new());
            let calls = Arc::new(AtomicUsize::new(0));
            let barrier = Arc::new(Barrier::new(THREADS));

            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let (map, calls, barrier) = (map.clone(), calls.clone(), barrier.clone());
                    thread::spawn(move || {
                        barrier.wait();
                        *map.get_or_init(|| {
                            calls.fetch_add(1, Ordering::SeqCst);
                            thread::sleep(Duration::from_millis(1));
                            42u64
                        })
                    })
                })
                .collect();

            for handle in handles {
                assert_eq!(handle.join().unwrap(), 42);
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn concurrent_different_types() {
        let map = Arc::new(OnceTypeMap::new());
        let handles: Vec<_> = (0..10)
            .map(|thread_id| {
                let map = Arc::clone(&map);
                thread::spawn(move || match thread_id % 5 {
                    0 => assert_eq!(*map.get_or_init(|| 42i32), 42),
                    1 => assert_eq!(map.get_or_init(|| "thread".to_string()), "thread"),
                    2 => assert_eq!(*map.get_or_init(|| 2.5f64), 2.5),
                    3 => assert!(*map.get_or_init(|| true)),
                    _ => assert_eq!(*map.get_or_init(|| vec![1, 2, 3]), vec![1, 2, 3]),
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn concurrent_readers_and_writers() {
        let map = Arc::new(OnceTypeMap::new());
        map.get_or_init(|| "shared".to_string());
        let iterations = if cfg!(miri) { 5 } else { 200 };

        let handles: Vec<_> = (0..6)
            .map(|thread_id| {
                let map = Arc::clone(&map);
                thread::spawn(move || {
                    let shared: &String = map.get().unwrap();
                    for i in 0..iterations {
                        match thread_id % 3 {
                            0 => {
                                map.get_or_init(|| i as u64);
                            }
                            1 => {
                                let _ = map.insert(i as u32);
                            }
                            _ => assert!(!map.is_empty()),
                        }
                        assert_eq!(shared, "shared");
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn debug_shows_len() {
        let map = OnceTypeMap::new();
        map.get_or_init(|| 1i32);
        assert_eq!(format!("{map:?}"), "OnceTypeMap { len: 1, .. }");
    }
}
