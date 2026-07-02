//! Phase 1 — bignum + FullMath.mulDiv: differential, prove+verify, and negative
//! soundness tests.
//!
//! 1. DIFFERENTIAL: many random `(a, b, denom)` plus edge cases are run through
//!    the guest via `try_execute`. The guest's *internal* asserts (Euclidean
//!    identity over full product width, r < denom, denom != 0, all limbs < 256,
//!    quotient fits uint256) ARE the validation: an Ok result means the guest
//!    agrees with the U256 reference that produced its q/r hints.
//! 2. PROVE+VERIFY: a few representative vectors go through real WHIR
//!    prove + verify (the Phase-1 milestone).
//! 3. NEGATIVE: a WRONG q (and separately a wrong r) for a known-good vector is
//!    fed to the guest; the run MUST fail. The honest counterpart passing first
//!    makes the failure attributable to the corruption (i.e. the constraints
//!    actually bind the hint).

use backend::PrimeCharacteristicRing;
use lean_vm::F;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::sync::{Mutex, MutexGuard};
use uniswap_v3::{
    U256, build_muldiv_witness, build_witness_with_bufs, emit_main_hints, get_bytecode, muldiv_ceil_u256,
    muldiv_floor_u256, prove, setup, try_execute, verify,
};

static ARENA_LOCK: Mutex<()> = Mutex::new(());

fn serialize_arena() -> MutexGuard<'static, ()> {
    ARENA_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn random_u256(rng: &mut StdRng) -> U256 {
    let mut bytes = [0u8; 32];
    for byte in &mut bytes {
        *byte = rng.random::<u8>();
    }
    U256::from_le_bytes(&bytes)
}

/// Pick `(a, b, denom)` such that `floor(a*b/denom)` is GUARANTEED to fit in
/// uint256 (the constraint v3 enforces). Construction: `a` is a full 256-bit
/// value, `b` is at most 128 bits, so `a*b < 2^384` and `high(a*b) < 2^128`;
/// `denom` is then forced to have its top 128 bits nonzero (>= 2^128 > high), so
/// the quotient is < 2^256. denom is always nonzero. This still exercises full
/// 256-bit `a`, wide products, carries, and borrows across all limbs.
fn random_valid_triple(rng: &mut StdRng) -> (U256, U256, U256) {
    let a = random_u256(rng);
    // b: random low 128 bits only.
    let mut b = random_u256(rng);
    b.words[2] = 0;
    b.words[3] = 0;
    // denom: random, but force the top word nonzero so denom >= 2^192 > high.
    let mut denom = random_u256(rng);
    if denom.words[3] == 0 {
        denom.words[3] = 1;
    }
    (a, b, denom)
}

/// Run the guest on `(a, b, denom)` and assert it accepts (Ok). The witness's
/// q/r hints come from the trusted reference, so Ok == "guest agrees".
fn assert_guest_accepts(a: U256, b: U256, denom: U256) {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let (public_input, witness) = build_muldiv_witness(bytecode, a, b, denom);
    try_execute(bytecode, &public_input, &witness)
        .unwrap_or_else(|e| panic!("guest must accept ({a:?}, {b:?}, {denom:?}): {e:?}"));
}

#[test]
fn differential_random_vectors() {
    let mut rng = StdRng::seed_from_u64(0xA11CE);
    // Keep the count modest: each run compiles nothing (bytecode cached) but does
    // a full witness-gen pass. 64 random vectors comfortably exercise carries,
    // borrows, and the full-width identity across many limb patterns.
    for _ in 0..64 {
        let (a, b, denom) = random_valid_triple(&mut rng);
        assert_guest_accepts(a, b, denom);
    }
}

#[test]
fn differential_edge_cases() {
    let max = U256::MAX;
    let one = U256::ONE;
    let zero = U256::ZERO;

    // denom = 1, but a*b must fit uint256 => keep b = 1 so a*b = a < 2^256.
    assert_guest_accepts(max, one, one); // q = max, r = 0
    assert_guest_accepts(U256::from_u64(123_456_789), one, one);

    // r = 0 (exact division): a*b = q*denom exactly. Use a*denom / denom.
    let d = U256::from_u64(1_000_003);
    assert_guest_accepts(U256::from_u64(42).full_mul_into_u256_unchecked(d), one, d);

    // q = 0 (a*b < denom): tiny product, huge denom.
    assert_guest_accepts(U256::from_u64(2), U256::from_u64(3), max); // 6 / max = 0, r = 6
    assert_guest_accepts(zero, max, max); // 0 / max = 0

    // a = b = max within the result-fits constraint. The smallest denom for
    // which floor(max*max/denom) still fits uint256 is high(max*max)+1, which
    // equals max itself (high = 2^256-2). So the only a=b=max case that fits is
    // denom = max (=> q = max, r = 0); anything smaller overflows uint256 and v3
    // would revert. We exercise that exact boundary here.
    {
        let prod = max.full_mul(max);
        let high = U256 {
            words: [prod.words[4], prod.words[5], prod.words[6], prod.words[7]],
        };
        let (denom, _) = high.overflowing_add(U256::ONE); // == max
        assert_eq!(denom, max);
        assert_guest_accepts(max, max, denom); // q = max, r = 0
    }

    // uint160 x uint128 magnitudes (still uint256-width operands), denom mid-size.
    {
        let a = U256::ONE.shl(159); // ~uint160 top bit
        let b = U256::ONE.shl(120); // ~uint128
        let denom = U256::ONE.shl(64);
        assert_guest_accepts(a, b, denom);
    }

    // denom just below and just above the product (q = 1 and q = 0).
    {
        let a = U256::from_u64(0xDEAD_BEEF);
        let b = U256::from_u64(0xC0FFEE);
        let prod = a.full_mul(b);
        let prod256 = U256 {
            words: [prod.words[0], prod.words[1], prod.words[2], prod.words[3]],
        };
        assert_eq!(prod.words[4..], [0, 0, 0, 0]); // small enough to be 256-bit
        // denom == prod => q = 1, r = 0
        assert_guest_accepts(a, b, prod256);
        // denom == prod + 1 => q = 0, r = prod
        let (dp1, _) = prod256.overflowing_add(U256::ONE);
        assert_guest_accepts(a, b, dp1);
        // denom == prod - 1 => q = 1, r = 1
        let dm1 = prod256.wrapping_sub(U256::ONE);
        assert_guest_accepts(a, b, dm1);
    }
}

#[test]
fn rounding_up_path_accepts() {
    // mulDivRoundingUp is exercised inside main() too; assert both a rounding (r!=0)
    // and an exact (r==0) case accept, and that the reference ceil matches.
    let a = U256::from_u64(7);
    let b = U256::from_u64(3);
    let d4 = U256::from_u64(4);
    assert_eq!(muldiv_floor_u256(a, b, d4), U256::from_u64(5));
    assert_eq!(muldiv_ceil_u256(a, b, d4), U256::from_u64(6));
    assert_guest_accepts(a, b, d4);

    let d3 = U256::from_u64(3);
    assert_eq!(muldiv_ceil_u256(a, b, d3), U256::from_u64(7));
    assert_guest_accepts(a, b, d3);
}

// NOTE on debug vs release: this guest's trace shape trips a *debug-only*
// `debug_assert!(log_packing_width <= eval_points.len())` deep in the prover's
// packed eq-MLE evaluation (`backend::poly::eq_mle`), which is a SIMD edge-case
// guard the repo acknowledges (constants.rs MIN_LOG_N_ROWS_PER_TABLE comment:
// "without too much edge cases"). It is compiled OUT in release, where proving
// and verification work correctly and are fully exercised below. The no-op
// Phase-0 guest does prove in debug; this one does not — a prover-internal
// limitation, not a guest or soundness issue (release prove+verify round-trips
// and rejects tampered public inputs). We therefore run prove+verify in release
// and skip (not fail) it in debug, matching the PLAN's "use --release for
// proving". The soundness-critical differential + negative tests above run in
// BOTH debug and release.
#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn prove_and_verify_representative_vectors() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // A small set (3) of representative vectors: remainder!=0, exact, and q=0.
    let vectors = [
        (
            U256::from_u64(1_000_000_007),
            U256::from_u64(998_244_353),
            U256::from_u64(1_000_000),
        ),
        (U256::from_u64(42).full_mul_into_u256_unchecked(U256::from_u64(1_000_003)), U256::ONE, U256::from_u64(1_000_003)), // r = 0
        (U256::from_u64(5), U256::from_u64(6), U256::MAX), // q = 0
    ];

    for (a, b, denom) in vectors {
        let (public_input, witness) = build_muldiv_witness(bytecode, a, b, denom);
        try_execute(bytecode, &public_input, &witness).expect("witness gen must succeed");
        let proof = prove(bytecode, &public_input, &witness, 1).expect("prove must succeed");
        verify(bytecode, &public_input, proof).expect("verify must succeed");
    }
}

#[test]
fn negative_wrong_q_is_rejected() {
    // Critical soundness test: corrupting the prover's q for a known-good vector
    // MUST make the guest reject (the Euclidean identity no longer holds). The
    // honest run is asserted to pass first, so the rejection is attributable.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let a = U256::from_u64(1_000_000_007);
    let b = U256::from_u64(998_244_353);
    let denom = U256::from_u64(1_000_000);

    // Honest run passes.
    let honest = emit_main_hints(a, b, denom);
    let (pi, w_ok) = build_witness_with_bufs(bytecode, a, b, denom, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest witness must pass");

    // Corrupt the FIRST muldiv_q buffer's least-significant limb by +1.
    let mut corrupted = honest.clone();
    let q_idx = corrupted
        .iter()
        .position(|h| h.label == "muldiv_q")
        .expect("a muldiv_q buffer must exist");
    corrupted[q_idx].cells[0] += F::from_u32(1);
    let (_pi2, w_bad) = build_witness_with_bufs(bytecode, a, b, denom, &corrupted);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject a wrong q (constraints must bind the quotient hint)"
    );
}

#[test]
fn negative_wrong_r_is_rejected() {
    // Corrupting the prover's r MUST make the guest reject (identity breaks; and
    // if the corruption pushes r >= denom, the r < denom check also fires).
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let a = U256::from_u64(1_000_000_007);
    let b = U256::from_u64(998_244_353);
    let denom = U256::from_u64(1_000_000);

    let honest = emit_main_hints(a, b, denom);
    let (pi, w_ok) = build_witness_with_bufs(bytecode, a, b, denom, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest witness must pass");

    let mut corrupted = honest.clone();
    let r_idx = corrupted
        .iter()
        .position(|h| h.label == "muldiv_r")
        .expect("a muldiv_r buffer must exist");
    corrupted[r_idx].cells[0] += F::from_u32(1);
    let (_pi2, w_bad) = build_witness_with_bufs(bytecode, a, b, denom, &corrupted);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject a wrong r (constraints must bind the remainder hint)"
    );
}

#[test]
fn negative_non_canonical_limb_is_rejected() {
    // A limb >= 256 in the prover's q must be rejected by range_check_limbs even
    // if it would otherwise satisfy the (mod p) identity — preventing
    // multiple-representation forgery.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let a = U256::from_u64(123_456);
    let b = U256::from_u64(789_012);
    let denom = U256::from_u64(7);

    let honest = emit_main_hints(a, b, denom);
    let (pi, w_ok) = build_witness_with_bufs(bytecode, a, b, denom, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest witness must pass");

    // Set a q limb to 256 (out of [0,256)).
    let mut corrupted = honest.clone();
    let q_idx = corrupted.iter().position(|h| h.label == "muldiv_q").unwrap();
    corrupted[q_idx].cells[0] = F::from_u32(256);
    let (_pi2, w_bad) = build_witness_with_bufs(bytecode, a, b, denom, &corrupted);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject a non-canonical (>= 256) limb"
    );
}

#[test]
fn negative_qm1_rplusdenom_forgery_rejected() {
    // The classic mulDiv forgery: submit (q-1, r+denom) instead of (q, r). The
    // Euclidean identity STILL HOLDS — (q-1)*denom + (r+denom) == q*denom + r ==
    // a*b — but r+denom >= denom, so ONLY the `bignum_lt(r, denom) == 1` check
    // catches it. This proves the remainder-range check is load-bearing (not made
    // redundant by the identity). Honest run asserted to pass first.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Choose a vector with a nonzero remainder and r+denom that does not overflow
    // the r width (32 limbs): denom small, r < denom, r+denom < 2^256 trivially.
    let a = U256::from_u64(1_000_000_007);
    let b = U256::from_u64(998_244_353);
    let denom = U256::from_u64(1_000_000); // q = 998244..., r != 0

    let honest = emit_main_hints(a, b, denom);
    let (pi, w_ok) = build_witness_with_bufs(bytecode, a, b, denom, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest witness must pass");

    // Build the forged (q-1, r+denom) for BOTH muldiv pairs (mulDiv + ceil),
    // mirroring how emit_main_hints lays them out.
    let (q, r) = uniswap_v3::muldiv(a, b, denom);
    assert!(!r.is_zero(), "need a nonzero remainder for this forgery");
    // q-1 over 64 limbs (q fits 256 bits here, so q-1 is a U256 then zero-extended).
    let q256 = U256 {
        words: [q.words[0], q.words[1], q.words[2], q.words[3]],
    };
    let qm1 = q256.wrapping_sub(U256::ONE);
    let (rpd, carry) = r.overflowing_add(denom);
    assert!(!carry, "r+denom must not overflow uint256 for this vector");

    let mut forged_q = qm1.to_le_bytes().to_vec();
    forged_q.resize(64, 0);
    let forged_q_cells: Vec<F> = forged_q.iter().map(|&x| F::from_u32(x as u32)).collect();
    let forged_r_cells: Vec<F> = rpd.to_le_bytes().iter().map(|&x| F::from_u32(x as u32)).collect();

    let mut corrupted = honest.clone();
    for h in corrupted.iter_mut() {
        if h.label == "muldiv_q" {
            h.cells = forged_q_cells.clone();
        } else if h.label == "muldiv_r" {
            h.cells = forged_r_cells.clone();
        }
    }
    let (_pi2, w_bad) = build_witness_with_bufs(bytecode, a, b, denom, &corrupted);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject (q-1, r+denom): r+denom >= denom violates r < denom"
    );
}

#[test]
fn negative_quotient_above_nout_rejected() {
    // A quotient with a nonzero limb at index >= NOUT (=32) must be rejected by
    // the `q[i] == 0 for i >= NOUT` fits-uint256 check (the in-circuit analogue of
    // v3's "result fits" require). Setting limb[32] also breaks the identity, but
    // the point is the dedicated high-limb check. Honest run passes first.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let a = U256::from_u64(123_456);
    let b = U256::from_u64(789_012);
    let denom = U256::from_u64(7);

    let honest = emit_main_hints(a, b, denom);
    let (pi, w_ok) = build_witness_with_bufs(bytecode, a, b, denom, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest witness must pass");

    let mut corrupted = honest.clone();
    let q_idx = corrupted.iter().position(|h| h.label == "muldiv_q").unwrap();
    // limb index 32 is the first limb ABOVE the uint256 output width.
    corrupted[q_idx].cells[32] = F::from_u32(1);
    let (_pi2, w_bad) = build_witness_with_bufs(bytecode, a, b, denom, &corrupted);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject a quotient with a nonzero limb above NOUT"
    );
}
