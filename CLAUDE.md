# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the guide for every agent in this repository (workspace map, layering, golden rules
including the asset/attribution and clean-room rules, never-crash rules, definition of done). It is
imported here so it is always in context:

@AGENTS.md

Then read `../craftrules/AGENTS.md` for the standards shared by all Crafting Apps. If `../craftrules`
is not checked out next to this repo, it is `storytold/craftrules` on GitHub. Ask before cloning it.

The notes below add to `AGENTS.md`; where they disagree, the code is the authority.

## Commands

```sh
cargo xtask ci                                  # what CI runs: fmt --check, clippy -D warnings, test --workspace, layers, wasm
cargo test -p texturecraft-engine --release --test session panic_hunt   # one test: -p <crate> --test <file in tests/> <name filter>
cargo test -p texturecraft-formats --release mesh_import                # unit tests in src/: filter by module path
cargo test -p texturecraft-engine --release --test perf -- --ignored    # ignored (heavy/GPU) tests need -- --ignored
cargo test -p texturecraft-automation --release --test perf -- --ignored --nocapture --test-threads=1   # headless bake + composite perf suite (budgets; see file header)
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only CARGO_TARGET_DIR=target/profile cargo test -p texturecraft-automation --release --test perf --no-run   # then: perf record -F 999 -- <binary> --ignored --exact <test>; perf report --no-children --sort sym|srcline
TC_UPDATE_PARITY=1 cargo test -p texturecraft-ui-egui parity            # regenerate docs/parity.md from the menu catalog
cargo run --release -p texturecraft -- --control 7878                   # desktop app + JSON control channel (screenshots)
cargo run --release -p texturecraft-cli -- mcp                          # MCP server on stdio
cd apps/texturecraft-web && trunk serve                                 # web build
```

- `cargo xtask` is a cargo alias (`.cargo/config.toml`) for `run -p xtask --`. Its subcommands are
  `layers | wasm | ci | version | ico | icns`. There is no `parity` subcommand, despite the list in
  `AGENTS.md`.
- `texturecraft-web` is not in `default-members`, so plain `cargo build`/`cargo test` skip it.
- Tests marked `#[ignore]` are the heavy or GPU-dependent ones: `engine/tests/perf.rs`,
  `render/tests/headless.rs`, `render/src/{gpu_bake,gpu_trace,ibl}.rs`, `bake/src/tests.rs`,
  `paint/src/alphas.rs`, `ui-egui/tests/ui_snapshots.rs`. `cargo xtask ci` does not run them; run
  them with `-- --ignored` when you touch those paths.
- Formatting is `max_width = 160`, `use_small_heuristics = "Max"`. Clippy allows
  unwrap/expect/panic/indexing in tests only (`clippy.toml`).
- CI pins toolchain 1.96.0 on Ubuntu and Windows via the shared `storytold/craft-common-actions`
  gate workflow. The README minimum is Rust 1.95.

## Adding a command (the usual shape of a feature)

1. Write a `CommandSpec` (`crates/engine/src/commands.rs`): `id`, `label`, `menu` path, `shortcut`
   in egui notation, a `params` doc string for agents, `enabled` (a `fn(&Session) -> Result<(), String>`,
   e.g. `has_layer`) and `run`.
2. Put it in a module's `specs()`, a new `*_cmds.rs` module when working in parallel, and register
   that with one `v.extend(crate::<module>::specs())` line in `command_specs()`.
3. Mutations go through `Session::edit` so they are atomic and undoable. Make the change visible in
   `project.inspect` / `session.inspect`.
4. If it belongs in a menu, its id must be in the catalog in `crates/ui-egui/src/menus.rs`.
   Regenerate `docs/parity.md` if menu coverage changed.
5. Add tests under `crates/engine/tests/`, and keep `panic_hunt` green.
