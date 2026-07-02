//! Phase 3 — TickMath: differential, prove+verify, and negative soundness tests.
//!
//! 1. DIFFERENTIAL: getSqrtRatioAtTick over {0, +/-1, MIN, MAX, many random}
//!    via `try_execute`; the guest's internal asserts + commitment to the honest
//!    reference result ARE the validation (Ok == "guest agrees with v3 port").
//! 2. ROUND-TRIP: getTickAtSqrtRatio on sqrtRatio(t) recovers t; boundary cases.
//! 3. NEGATIVE: a wrong getSqrtRatioAtTick output (forged commitment) rejected;
//!    a forged tick hint outside the [ratio(tick), ratio(tick+1)) bucket rejected;
//!    abs_tick > MAX_TICK rejected — each with the honest counterpart passing.
//! 4. PROVE+VERIFY: a representative getSqrtRatioAtTick and getTickAtSqrtRatio
//!    (release only; debug trips a known prover-internal SIMD assert — see Phase 1).

use backend::PrimeCharacteristicRing;
use lean_vm::F;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::sync::{Mutex, MutexGuard};
use uniswap_v3::{
    MAX_TICK, MIN_TICK, SqrtInputs, U256, build_sqrt_witness, build_sqrt_witness_with_bufs,
    emit_sqrt_hints, get_bytecode, max_sqrt_ratio, min_sqrt_ratio, prove, ref_get_sqrt_ratio_at_tick,
    ref_get_tick_at_sqrt_ratio, setup, sqrt_ratio_at_tick_inputs, tick_at_sqrt_ratio_inputs, try_execute, verify,
};

static ARENA_LOCK: Mutex<()> = Mutex::new(());

fn serialize_arena() -> MutexGuard<'static, ()> {
    ARENA_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Run the guest on `inp` and assert it accepts. The witness comes from the
/// trusted reference, so Ok == "guest agrees with the v3 TickMath port".
fn assert_accepts(inp: &SqrtInputs) {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let (public_input, witness) = build_sqrt_witness(bytecode, inp);
    try_execute(bytecode, &public_input, &witness).unwrap_or_else(|e| panic!("guest must accept {inp:?}: {e:?}"));
}

// ---------------------------------------------------------------------------
// DIFFERENTIAL: getSqrtRatioAtTick.
// ---------------------------------------------------------------------------

#[test]
fn differential_sqrt_ratio_at_tick_boundaries() {
    for t in [0, 1, -1, 2, -2, MIN_TICK, MAX_TICK, MIN_TICK + 1, MAX_TICK - 1] {
        assert_accepts(&sqrt_ratio_at_tick_inputs(t));
    }
}

#[test]
fn differential_sqrt_ratio_at_tick_random() {
    let mut rng = StdRng::seed_from_u64(0x71CCAA);
    for _ in 0..40 {
        // uniform tick in [MIN_TICK, MAX_TICK].
        let span = (MAX_TICK as i64 - MIN_TICK as i64) as u64;
        let t = MIN_TICK as i64 + (rng.random::<u64>() % (span + 1)) as i64;
        assert_accepts(&sqrt_ratio_at_tick_inputs(t as i32));
    }
}

// ---------------------------------------------------------------------------
// ROUND-TRIP / bucket verify: getTickAtSqrtRatio.
// ---------------------------------------------------------------------------

#[test]
fn differential_tick_at_sqrt_ratio_roundtrip() {
    // For a sampled set of ticks, sqrtRatio(t) must verify back to t.
    let sample = [
        MIN_TICK,
        MIN_TICK + 1,
        -100000,
        -1,
        0,
        1,
        100000,
        500000,
        MAX_TICK - 2,
        MAX_TICK - 1, // tick+1 == MAX_TICK, representable
    ];
    for t in sample {
        let sp = ref_get_sqrt_ratio_at_tick(t);
        // sqrtRatio(t) is in [MIN_SQRT_RATIO, MAX_SQRT_RATIO]; getTickAtSqrtRatio
        // requires sqrtP < MAX_SQRT_RATIO, so skip t == MAX_TICK (sp == MAX).
        if sp >= max_sqrt_ratio() {
            continue;
        }
        let answer = ref_get_tick_at_sqrt_ratio(sp);
        assert_accepts(&tick_at_sqrt_ratio_inputs(sp, answer));
    }
}

#[test]
fn differential_tick_at_sqrt_ratio_boundary() {
    // sqrtP == MIN_SQRT_RATIO -> tick == MIN_TICK.
    let sp_min = min_sqrt_ratio();
    let t_min = ref_get_tick_at_sqrt_ratio(sp_min);
    assert_eq!(t_min, MIN_TICK);
    assert_accepts(&tick_at_sqrt_ratio_inputs(sp_min, t_min));

    // sqrtP just below MAX_SQRT_RATIO -> tick == MAX_TICK - 1.
    let sp_hi = max_sqrt_ratio().wrapping_sub(U256::ONE);
    let t_hi = ref_get_tick_at_sqrt_ratio(sp_hi);
    assert_eq!(t_hi, MAX_TICK - 1);
    assert_accepts(&tick_at_sqrt_ratio_inputs(sp_hi, t_hi));

    // an interior random sqrtP.
    let mut rng = StdRng::seed_from_u64(0xB0CCE7);
    for _ in 0..20 {
        let span = (MAX_TICK as i64 - 1 - MIN_TICK as i64) as u64;
        let t = MIN_TICK as i64 + (rng.random::<u64>() % (span + 1)) as i64;
        let sp = ref_get_sqrt_ratio_at_tick(t as i32);
        if sp < min_sqrt_ratio() || sp >= max_sqrt_ratio() {
            continue;
        }
        let answer = ref_get_tick_at_sqrt_ratio(sp);
        assert_accepts(&tick_at_sqrt_ratio_inputs(sp, answer));
    }
}

// ---------------------------------------------------------------------------
// NEGATIVE soundness tests.
// ---------------------------------------------------------------------------

#[test]
fn negative_forged_sqrt_ratio_output_rejected() {
    // A forged committed RESULT for getSqrtRatioAtTick must be rejected: the guest
    // computes the true ratio and asserts the commitment digest == public_input,
    // so pinning the public input to a WRONG result breaks the digest equality.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let inp = sqrt_ratio_at_tick_inputs(12345);
    let honest = emit_sqrt_hints(&inp);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &inp, &honest);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest getSqrtRatioAtTick must pass");

    // Pin the public input to a DIFFERENT tick's commitment (wrong result), but
    // feed the honest hints for tick 12345. The digest won't match.
    let other = sqrt_ratio_at_tick_inputs(12346);
    let (pi_wrong, _) = build_sqrt_witness(bytecode, &other);
    assert!(
        try_execute(bytecode, &pi_wrong, &w_ok).is_err(),
        "guest must reject a forged committed sqrtRatio output"
    );
}

#[test]
fn negative_forged_tick_hint_rejected() {
    // getTickAtSqrtRatio: a forged tick that does NOT land in the bucket
    // [ratio(tick), ratio(tick+1)) must be rejected by the two bucket inequalities.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Honest: sqrtP = sqrtRatio(50000) -> answer 50000.
    let true_tick = 50000;
    let sp = ref_get_sqrt_ratio_at_tick(true_tick);
    let answer = ref_get_tick_at_sqrt_ratio(sp);
    let honest = tick_at_sqrt_ratio_inputs(sp, answer);
    let h = emit_sqrt_hints(&honest);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &honest, &h);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest tick verification must pass");

    // Forge a tick two below the true tick: sqrtP >= ratio(forged+1) breaks the
    // upper bucket bound `sqrtP < ratio(tick+1)`.
    let forged = tick_at_sqrt_ratio_inputs(sp, true_tick - 2);
    let fh = emit_sqrt_hints(&forged);
    // Public input pinned to the FORGED inputs' commitment (so the digest matches
    // the forged tick) — the rejection must come from the bucket inequality, not
    // the commitment.
    let (pi_forged, w_forged) = build_sqrt_witness_with_bufs(bytecode, &forged, &fh);
    assert!(
        try_execute(bytecode, &pi_forged, &w_forged).is_err(),
        "guest must reject a tick outside its sqrt-ratio bucket"
    );

    // Forge a tick one ABOVE the true tick: ratio(forged) > sqrtP breaks the lower
    // bucket bound `sqrtP >= ratio(tick)`.
    let forged_hi = tick_at_sqrt_ratio_inputs(sp, true_tick + 1);
    let fhh = emit_sqrt_hints(&forged_hi);
    let (pi_fh, w_fh) = build_sqrt_witness_with_bufs(bytecode, &forged_hi, &fhh);
    assert!(
        try_execute(bytecode, &pi_fh, &w_fh).is_err(),
        "guest must reject a tick above its sqrt-ratio bucket"
    );
}

#[test]
fn negative_abs_tick_above_max_rejected() {
    // abs_tick > MAX_TICK must be rejected by assert_le_max_tick (the diff
    // decomposition fails). We supply abs_tick = MAX_TICK + 1 with sign 0.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Honest accepted op for attribution.
    let ok = sqrt_ratio_at_tick_inputs(MAX_TICK);
    let okh = emit_sqrt_hints(&ok);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &ok, &okh);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest MAX_TICK must pass");

    // Build a getSqrtRatioAtTick op with abs_tick out of range. The reference
    // would panic (asserts in range), so construct inputs by hand and emit only
    // the input block (the run aborts at the range check before any muldiv).
    let mut bad = sqrt_ratio_at_tick_inputs(MAX_TICK);
    bad.abs_tick = (MAX_TICK as u32) + 1; // 887273, > MAX_TICK
    let bad_inputs_only = emit_inputs_only(&bad);
    // Pin the public input to the honest case; the run must fail at the range
    // check before the commitment is reached.
    assert!(
        try_execute(bytecode, &pi_ok, &build_w(bytecode, &ok, &bad_inputs_only)).is_err(),
        "guest must reject abs_tick > MAX_TICK"
    );
}

#[test]
fn negative_wrong_reciprocal_hint_rejected() {
    // getSqrtRatioAtTick at a POSITIVE tick consumes one muldiv pair (the
    // reciprocal). Corrupting that quotient must be rejected by the Euclidean
    // identity inside bignum_muldiv.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let inp = sqrt_ratio_at_tick_inputs(200000); // positive => one reciprocal pair
    let honest = emit_sqrt_hints(&inp);
    assert!(
        honest.iter().any(|h| h.label == "muldiv_q"),
        "positive tick must emit a reciprocal muldiv pair"
    );
    let (pi, w_ok) = build_sqrt_witness_with_bufs(bytecode, &inp, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest positive-tick run must pass");

    let mut corrupt = honest.clone();
    let q_idx = corrupt.iter().position(|h| h.label == "muldiv_q").unwrap();
    corrupt[q_idx].cells[0] += F::from_u32(1);
    let (_pi2, w_bad) = build_sqrt_witness_with_bufs(bytecode, &inp, &corrupt);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject a corrupted reciprocal quotient"
    );
}

// ---------------------------------------------------------------------------
// Phase-3 audit carry-ins (applied in Phase 4).
// ---------------------------------------------------------------------------

#[test]
fn differential_sqrt_ratio_mid_tick_magic_bits() {
    // A fixed tick whose |tick| sets magic bits {4,9,10,12,13,14,17}, which the
    // MIN/MAX_TICK boundary vectors do NOT exercise (MIN/MAX = 0xd89e8). Runs both
    // op 8 (getSqrtRatioAtTick) and op 9 (round-trip) so a mistyped MAGIC[i] for
    // any of those bits would diverge from the reference and fail the commitment.
    let t = 161296; // 2^17+2^14+2^13+2^12+2^10+2^9+2^4
    assert_accepts(&sqrt_ratio_at_tick_inputs(t));
    assert_accepts(&sqrt_ratio_at_tick_inputs(-t));
    let sp = ref_get_sqrt_ratio_at_tick(t);
    assert_accepts(&tick_at_sqrt_ratio_inputs(sp, ref_get_tick_at_sqrt_ratio(sp)));
}

#[test]
fn negative_tick_canonical_zero_encoding() {
    // Phase-3 carry-in #1: the non-canonical zero (tick_sign=1, abs_tick=0) must be
    // rejected by `assert tick_sign*(1-abs_nonzero)==0`. Honest zero (sign 0) passes.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let ok = sqrt_ratio_at_tick_inputs(0);
    let okh = emit_sqrt_hints(&ok);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &ok, &okh);
    try_execute(bytecode, &pi_ok, &w_ok).expect("canonical tick 0 must pass");

    // Forge (sign=1, abs=0): pin the public input to its own commitment so the
    // rejection comes from the canonical assert, not the digest.
    let mut bad = ok;
    bad.tick_sign = true; // abs_tick stays 0
    let badh = emit_sqrt_hints(&bad);
    let (pi_bad, w_bad) = build_sqrt_witness_with_bufs(bytecode, &bad, &badh);
    assert!(
        try_execute(bytecode, &pi_bad, &w_bad).is_err(),
        "guest must reject the non-canonical zero (sign=1, abs=0)"
    );
}

#[test]
fn negative_op9_band_edges_rejected() {
    // Phase-3 carry-in #3: op 9 requires MIN_SQRT_RATIO <= P < MAX_SQRT_RATIO.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // (a) P < MIN_SQRT_RATIO must be rejected (sqrtRatio(MIN_TICK)=MIN is smallest).
    // Use P = MIN_SQRT_RATIO - 1 with the plausible tick hint MIN_TICK. The guest
    // aborts at `assert bignum_lt(P, MIN) == 0` before any muldiv pair.
    let p_below = min_sqrt_ratio().wrapping_sub(U256::ONE);
    let mut below = tick_at_sqrt_ratio_inputs(min_sqrt_ratio(), MIN_TICK);
    below.sqrt_a = p_below;
    let below_io = emit_inputs_only(&below);
    // Pin the public input to this op9's own commitment (echoes sqrt_a = P_below).
    let (pi_below, _) = build_sqrt_witness(bytecode, &below);
    assert!(
        try_execute(bytecode, &pi_below, &build_w(bytecode, &below, &below_io)).is_err(),
        "op9 must reject P < MIN_SQRT_RATIO"
    );

    // (b) P == MAX_SQRT_RATIO must be rejected (band is half-open: P < MAX).
    let mut at_max = tick_at_sqrt_ratio_inputs(max_sqrt_ratio().wrapping_sub(U256::ONE), MAX_TICK - 1);
    at_max.sqrt_a = max_sqrt_ratio();
    let at_max_io = emit_inputs_only(&at_max);
    let (pi_at_max, _) = build_sqrt_witness(bytecode, &at_max);
    assert!(
        try_execute(bytecode, &pi_at_max, &build_w(bytecode, &at_max, &at_max_io)).is_err(),
        "op9 must reject P == MAX_SQRT_RATIO"
    );
}

#[test]
fn negative_op9_abs_p1_above_max_rejected() {
    // Phase-3 carry-in #3: a direct abs_p1 <= MAX_TICK check. get_sqrt_ratio_at_tick
    // on (sign_p1, abs_p1) bounds abs_p1 via assert_le_max_tick; abs_p1 = MAX_TICK+1
    // makes the diff decomposition fail. We keep (tick, tick+1) self-consistent by
    // using tick = MAX_TICK (so tick+1 = MAX_TICK+1 is naturally out of range) and a
    // P just below MAX so the band check passes and execution reaches the tick+1
    // sqrtRatio computation.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let sp = max_sqrt_ratio().wrapping_sub(U256::ONE);
    // Honest: tick = MAX_TICK - 1, tick+1 = MAX_TICK (in range) passes.
    let ok = tick_at_sqrt_ratio_inputs(sp, MAX_TICK - 1);
    let okh = emit_sqrt_hints(&ok);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &ok, &okh);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest near-MAX op9 must pass");

    // Forge abs_p1 = MAX_TICK + 1 (sign_p1 = 0). Keep abs_tick = MAX_TICK so the
    // (tick+1 == tick succeeds) encoding holds: sv_t = MAX_TICK, sv_p1 = MAX_TICK+1.
    let mut bad = ok;
    bad.abs_tick = MAX_TICK as u32;
    bad.tick_sign = false;
    bad.sign_p1 = false;
    bad.abs_p1 = (MAX_TICK as u32) + 1;
    let bad_io = emit_inputs_only(&bad);
    let (pi_bad, _) = build_sqrt_witness(bytecode, &ok); // pin to honest digest; abort is pre-commit
    assert!(
        try_execute(bytecode, &pi_bad, &build_w(bytecode, &ok, &bad_io)).is_err(),
        "op9 must reject abs_p1 > MAX_TICK"
    );
}

// ---------------------------------------------------------------------------
// PROVE + VERIFY (release only).
// ---------------------------------------------------------------------------

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn prove_and_verify_tickmath() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // A getSqrtRatioAtTick at a positive tick (exercises the reciprocal pair).
    let inp_sqrt = sqrt_ratio_at_tick_inputs(123456);
    let (pi1, w1) = build_sqrt_witness(bytecode, &inp_sqrt);
    try_execute(bytecode, &pi1, &w1).expect("witness gen (op8) must succeed");
    let p1 = prove(bytecode, &pi1, &w1, 1).expect("prove op8 must succeed");
    verify(bytecode, &pi1, p1).expect("verify op8 must succeed");

    // A getTickAtSqrtRatio at a positive tick (two getSqrtRatioAtTick calls).
    let sp = ref_get_sqrt_ratio_at_tick(123456);
    let inp_tick = tick_at_sqrt_ratio_inputs(sp, ref_get_tick_at_sqrt_ratio(sp));
    let (pi2, w2) = build_sqrt_witness(bytecode, &inp_tick);
    try_execute(bytecode, &pi2, &w2).expect("witness gen (op9) must succeed");
    let p2 = prove(bytecode, &pi2, &w2, 1).expect("prove op9 must succeed");
    verify(bytecode, &pi2, p2).expect("verify op9 must succeed");
}

/// Phase-3 memory/trace/timing budget probe (run with `--release --nocapture`).
/// Reports total memory cells, runtime cells, executed cycles, dominant trace
/// table heights, and prove+verify wall-clock for one getSqrtRatioAtTick (op 8,
/// positive tick) and one getTickAtSqrtRatio (op 9, which calls getSqrtRatioAtTick
/// twice). We track these against the 2^26 memory budget for the Phase-5 swap loop.
#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn tickmath_budget_report() {
    use lean_vm::try_execute_bytecode;
    use std::time::Instant;

    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let cases: [(&str, SqrtInputs); 2] = [
        ("getSqrtRatioAtTick(123456)", sqrt_ratio_at_tick_inputs(123456)),
        ("getTickAtSqrtRatio(sqrtRatio(123456))", {
            let sp = ref_get_sqrt_ratio_at_tick(123456);
            tick_at_sqrt_ratio_inputs(sp, ref_get_tick_at_sqrt_ratio(sp))
        }),
    ];

    for (label, inp) in cases {
        let (pi, w) = build_sqrt_witness(bytecode, &inp);
        let res = try_execute_bytecode(bytecode, &pi, &w, false).expect("exec must succeed");
        let total_cells = res.memory.0.len();
        let runtime_cells = res.runtime_memory_size;
        let cycles = res.n_cycles();
        let log2_total = (usize::BITS - total_cells.next_power_of_two().leading_zeros() - 1) as usize;

        // dominant trace tables by row count.
        let mut tbls: Vec<(String, usize, usize)> = res
            .traces
            .iter()
            .map(|(t, tr)| (format!("{t:?}"), tr.non_padded_n_rows, tr.log_n_rows))
            .collect();
        tbls.sort_by_key(|&(_, n, _)| std::cmp::Reverse(n));

        let t0 = Instant::now();
        let proof = prove(bytecode, &pi, &w, 1).expect("prove must succeed");
        let t_prove = t0.elapsed();
        let t1 = Instant::now();
        verify(bytecode, &pi, proof).expect("verify must succeed");
        let t_verify = t1.elapsed();

        println!("\n=== {label} ===");
        println!(
            "  total memory cells = {total_cells} (~2^{log2_total}); runtime cells = {runtime_cells}; cycles = {cycles}",
        );
        println!("  prove = {t_prove:.3?}; verify = {t_verify:.3?}");
        println!("  top trace tables (non_padded_n_rows, log_n_rows):");
        for (name, n, lg) in tbls.iter().take(6) {
            println!("    {name:<28} {n:>8}  (2^{lg})");
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Build only the input-block hints for any op (no muldiv pairs). Used to test
/// guards that abort before reaching any mulDiv call.
fn emit_inputs_only(inp: &SqrtInputs) -> Vec<uniswap_v3::HintBuf> {
    use uniswap_v3::{HintBuf, UINT256_LIMBS};
    vec![
        HintBuf { label: "op", cells: vec![F::from_u32(inp.op.code())] },
        HintBuf { label: "flag_a", cells: vec![F::from_u32(inp.flag_a as u32)] },
        HintBuf { label: "sqrtA", cells: inp.sqrt_a.to_le_base256_limbs(UINT256_LIMBS) },
        HintBuf { label: "sqrtB", cells: inp.sqrt_b.to_le_base256_limbs(UINT256_LIMBS) },
        HintBuf { label: "liquidity", cells: inp.liquidity.to_le_base256_limbs(UINT256_LIMBS) },
        HintBuf { label: "amount", cells: inp.amount.to_le_base256_limbs(UINT256_LIMBS) },
        HintBuf { label: "y_sign", cells: vec![F::from_u32(inp.y_sign as u32)] },
        HintBuf { label: "tick_sign", cells: vec![F::from_u32(inp.tick_sign as u32)] },
        HintBuf { label: "abs_tick", cells: vec![F::from_u32(inp.abs_tick)] },
        HintBuf { label: "sign_p1", cells: vec![F::from_u32(inp.sign_p1 as u32)] },
        HintBuf { label: "abs_p1", cells: vec![F::from_u32(inp.abs_p1)] },
        HintBuf { label: "tick_spacing", cells: vec![F::from_u32(inp.tick_spacing)] },
        HintBuf { label: "bit_pos", cells: vec![F::from_u32(inp.bit_pos)] },
        HintBuf { label: "word", cells: inp.word.to_le_base256_limbs(UINT256_LIMBS) },
    ]
}

fn build_w(
    bytecode: &lean_vm::Bytecode,
    pin: &SqrtInputs,
    bufs: &[uniswap_v3::HintBuf],
) -> lean_vm::ExecutionWitness {
    let (_pi, w) = build_sqrt_witness_with_bufs(bytecode, pin, bufs);
    w
}
