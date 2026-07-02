//! Uniswap-V3-in-leanVM — Rust harness.
//!
//! Phase 1: bignum foundation + `FullMath.mulDiv`. This crate compiles the zkDSL
//! guest in `zkdsl_implem/`, builds an `ExecutionWitness` (operands + the
//! prover's q/r limb buffers), and proves+verifies it end-to-end against a
//! trusted `U256` reference. It only *consumes* the VM / compiler / prover as
//! fixed external interfaces.
//!
//! NOTE: we deliberately do NOT call `lean_multisig::setup_prover` — that
//! initializes the *aggregation* bytecode and would pull in the root package.
//! Our `setup()` replicates only the generic prover init for OUR bytecode.

#![cfg_attr(not(test), allow(unused_crate_dependencies))]

pub mod bigint;
pub mod compilation;
pub mod reference;
pub mod witness;

use backend::{ProofError, RawProof, enable_arena, parallel, precompute_dft_twiddles};
use lean_prover::prove_execution::{ExecutionProof, prove_execution};
use lean_prover::verify_execution::{ProofVerificationDetails, verify_execution};
use lean_prover::{ProverError, default_whir_config};
use lean_vm::{Bytecode, ExecutionWitness, F, PUBLIC_INPUT_LEN, RunnerError, try_execute_bytecode};

pub use bigint::{U256, U512};
pub use compilation::{PREAMBLE_MEMORY_LEN, get_bytecode, init_bytecode};
pub use reference::{
    HintBuf, I256, MAX_TICK, MIN_TICK, PRODUCT_LIMBS, PoolFixture, SqrtInputs, SqrtOp, SwapInputs,
    SwapResult, SwapStepResult, UINT256_LIMBS, compute_swap_step_inputs, emit_main_hints,
    emit_sqrt_hints, emit_swap_hints, least_significant_bit_inputs, max_sqrt_ratio, min_sqrt_ratio,
    most_significant_bit_inputs, muldiv, muldiv_ceil_u256, muldiv_floor_u256, next_init_inputs,
    ref_add_delta, ref_compressed, ref_compute_swap_step, ref_get_sqrt_ratio_at_tick,
    ref_get_tick_at_sqrt_ratio, ref_least_significant_bit, ref_most_significant_bit, ref_next_init,
    ref_pool_swap, ref_position, sqrt_op_head_scalars, sqrt_op_result, sqrt_op_swap_slots,
    sqrt_ratio_at_tick_inputs, swap_public_input_for, swap_step_result, tick_at_sqrt_ratio_inputs,
};
pub use witness::{
    build_muldiv_witness, build_sqrt_witness, build_sqrt_witness_with_bufs, build_swap_witness,
    build_swap_witness_with_bufs, build_swap_witness_with_pi, build_witness_with_bufs,
    hints_from_bufs, public_input_for, sqrt_public_input_for,
};

/// Call once before proving (or before `try_execute`/`verify` standalone).
///
/// Replicates the *generic* parts of `lean_multisig::setup_prover` for this
/// crate's bytecode: enable the arena allocator, init the parallel runtime,
/// compile+cache our guest bytecode, and precompute DFT twiddles for WHIR.
///
/// # Safety / concurrency
/// The arena allocator is a single shared per-process region. NEVER generate two
/// proofs concurrently in one process — serialize proving (e.g. with a mutex) or
/// use separate processes.
pub fn setup() {
    enable_arena();
    parallel::init();
    init_bytecode();
    precompute_dft_twiddles::<F>(1 << 24);
}

/// Run the guest without proving (fast dev loop / sanity check). Errors if any
/// assert fails or a hint is mis-sized.
pub fn try_execute(
    bytecode: &Bytecode,
    public_input: &[F; PUBLIC_INPUT_LEN],
    witness: &ExecutionWitness,
) -> Result<(), RunnerError> {
    try_execute_bytecode(bytecode, public_input, witness, false).map(|_| ())
}

/// Prove guest execution at the given WHIR rate (`log_inv_rate in {1,2,3,4}`;
/// lower = faster prover, larger proof).
pub fn prove(
    bytecode: &Bytecode,
    public_input: &[F; PUBLIC_INPUT_LEN],
    witness: &ExecutionWitness,
    log_inv_rate: usize,
) -> Result<ExecutionProof, ProverError> {
    let whir_config = default_whir_config(log_inv_rate);
    prove_execution(bytecode, public_input, witness, &whir_config, false)
}

/// Verify a guest-execution proof against the public input.
pub fn verify(
    bytecode: &Bytecode,
    public_input: &[F; PUBLIC_INPUT_LEN],
    proof: ExecutionProof,
) -> Result<(ProofVerificationDetails, RawProof<F>), ProofError> {
    verify_execution(bytecode, public_input, proof.proof)
}
