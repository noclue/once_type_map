# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## How we maintain this

- **One entry per release**: `## x.y.z - YYYY-MM-DD`
- **Unreleased goes first**: changes that will ship next live under `## [Unreleased]`
- **Categorize changes**:
  - **Added**: new features
  - **Changed**: behavior changes (non-breaking)
  - **Deprecated**: features slated for removal
  - **Removed**: features removed
  - **Fixed**: bug fixes
  - **Security**: security fixes
- **Call out breaking changes explicitly** (especially while pre-1.0).

## [Unreleased]

## [0.1.1] - 2026-11-13

> **Patch release.** No API or behavior changes. The map now owns its slots
> through `Arc` instead of `Box`, so its soundness no longer depends on an
> unsettled part of Rust's aliasing rules. MSRV stays **1.86**.

### Changed

- Each slot is owned through an `Arc<dyn Slot>` that is never cloned, instead
  of a `Box<dyn Slot>`. The internal hash map moves these owning pointers when
  it grows, while `&T` references into the slots may be alive. Whether moving a
  `Box` invalidates such references is still open in Rust's aliasing rules; an
  `Arc` makes no uniqueness claim, so moving it is just copying an address. The
  only `unsafe` code is still `detach` and its two call sites. Each stored type
  costs 16 more bytes for the `Arc` counts; lookups don't touch them.
- The crate docs and README explain the `Arc` choice, and the README's
  "Checking the unsafe code" section describes the current layout.

### Added

- `repository` and `documentation` links in `Cargo.toml`, so crates.io links to
  GitHub and docs.rs, and the README's relative links to `examples/` resolve on
  crates.io.
- A test that moves every slot through typed code while a `&T` is held, run
  under Miri in CI.
- This changelog.

## [0.1.0] - 2026-11-10

Initial release.

### Added

- `OnceTypeMap`: a thread-safe map holding at most one value per type. Values
  are set through `&self` and returned as plain `&T` references tied to the
  borrow of the map, with no lock guard.
- `&self` methods: `get`, `get_or_init`, `insert`, `contains`, `len`,
  `is_empty`. `&mut self` methods: `get_mut`, `remove`, `clear`.
- Documentation of initialization semantics, deadlocks (reentrant and cyclic
  initializers), empty slots left by panicking initializers, and fallible and
  async initialization patterns.
- Examples: `extensions` (extension traits on a library type), `async_extensions`
  (async setup with `tokio::sync::OnceCell`), and `fallible_extensions` (cached
  vs retried failures with `once_cell::sync::OnceCell`).
- CI on GitHub Actions: tests on Linux, macOS and Windows, MSRV, lint, and Miri
  under Stacked Borrows and Tree Borrows.

MSRV is **1.86**. No runtime dependencies.
