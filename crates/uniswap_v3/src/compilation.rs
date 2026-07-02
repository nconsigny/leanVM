//! Compilation of the Uniswap-V3 zkDSL guest into leanVM bytecode.
//!
//! Mirrors `crates/rec_aggregation/src/compilation.rs`: the `zkdsl_implem`
//! directory is embedded at compile time via `include_dir!`, compiled once, and
//! cached in a `OnceLock<Bytecode>`. Phase 0 needs no compile-time
//! `replacements`; later phases may add `<NAME>_PLACEHOLDER` substitutions here.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use lean_compiler::{CompilationFlags, ProgramSource, compile_program_with_flags};
use lean_vm::Bytecode;

/// Number of preamble cells the guest fills at the top of `main`.
///
/// MUST equal `PREAMBLE_MEMORY_LEN` in `zkdsl_implem/preamble.py`:
///   ZERO_VEC_LEN (16) + DIM (5) = 21.
pub const PREAMBLE_MEMORY_LEN: usize = 21;

static EMBEDDED_ZK_DSL: include_dir::Dir<'_> = include_dir::include_dir!("$CARGO_MANIFEST_DIR/zkdsl_implem");

static BYTECODE: OnceLock<Bytecode> = OnceLock::new();

/// Compile the guest (`main.py` entrypoint). Phase 0: empty replacements.
fn compile_guest() -> Bytecode {
    let source = ProgramSource::Embedded {
        entry: "main.py".to_string(),
        dir: &EMBEDDED_ZK_DSL,
    };
    compile_program_with_flags(
        &source,
        CompilationFlags {
            replacements: BTreeMap::new(),
        },
    )
}

/// Compile-and-cache the guest bytecode. Idempotent.
pub fn init_bytecode() {
    BYTECODE.get_or_init(compile_guest);
}

/// The cached guest bytecode. Panics if `init_bytecode()` (or `setup()`) was not called first.
pub fn get_bytecode() -> &'static Bytecode {
    BYTECODE
        .get()
        .unwrap_or_else(|| panic!("call init_bytecode() / setup() first"))
}
