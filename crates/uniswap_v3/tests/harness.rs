//! Harness test: prove + verify the guest end-to-end (Phase-0 guarantee,
//! preserved through Phase 1).
//!
//! This de-risks the exact compile -> witness -> prove -> verify path that all
//! later (Uniswap-math) phases build on. Phase 1 evolved the guest from a no-op
//! into a real `FullMath.mulDiv`; these tests keep the original guarantees —
//! a real WHIR prove + a real verify both succeed, and a proof bound to one
//! public input does NOT verify against a tampered one — now over real mulDiv
//! I/O committed to the public input.

use backend::PrimeCharacteristicRing;
use lean_vm::F;
use std::sync::{Mutex, MutexGuard};
use uniswap_v3::{U256, build_muldiv_witness, get_bytecode, prove, setup, try_execute, verify};

// The arena allocator is a single shared per-process region; serialize any test
// that proves so two never run concurrently in the same process.
static ARENA_LOCK: Mutex<()> = Mutex::new(());

fn serialize_arena() -> MutexGuard<'static, ()> {
    ARENA_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

// See the note in tests/bignum.rs: this mulDiv guest trips a debug-only SIMD
// `debug_assert` in the prover's packed eq-MLE; it proves+verifies correctly in
// release. Run prove+verify in release; skip (not fail) in debug.
#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn harness_prove_and_verify() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Honest prover: a * b / denom with a remainder (exercises floor and ceil).
    let a = U256::from_u64(1_000_000_007);
    let b = U256::from_u64(998_244_353);
    let denom = U256::from_u64(1_000_000);
    let (public_input, witness) = build_muldiv_witness(bytecode, a, b, denom);

    // Fast sanity run first (catches an assert/hint mismatch before the heavy prove).
    try_execute(bytecode, &public_input, &witness).expect("guest execution (witness gen) must succeed");

    // Real WHIR proof at the fastest rate (log_inv_rate = 1).
    let proof = prove(bytecode, &public_input, &witness, 1).expect("prove_execution must return Ok");

    // Real verification against the same public input.
    let (_details, _raw) = verify(bytecode, &public_input, proof).expect("verify_execution must return Ok");
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn harness_verify_rejects_wrong_public_input() {
    // Soundness smoke test: a proof bound to one public input must NOT verify
    // against a different one. Confirms the binding commitment is load-bearing.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let a = U256::from_u64(123_456);
    let b = U256::from_u64(789_012);
    let denom = U256::from_u64(7);
    let (public_input, witness) = build_muldiv_witness(bytecode, a, b, denom);
    let proof = prove(bytecode, &public_input, &witness, 1).expect("prove_execution must return Ok");

    let mut tampered = public_input;
    tampered[0] += F::ONE;
    assert!(
        verify(bytecode, &tampered, proof).is_err(),
        "verification must reject a tampered public input"
    );
}
