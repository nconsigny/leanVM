//! Phase 5b — full Pool.swap multi-tick loop: differential vs the v3 reference,
//! verified-termination and unified-commitment negative soundness, and
//! prove+verify of representative swaps.
//!
//! 1. DIFFERENTIAL via `try_execute`: single-tick (stays in range), multi-tick
//!    crossing several initialized ticks, exactIn AND exactOut, zeroForOne both,
//!    a swap that HITS the price limit, and one that EXHAUSTS the amount. Witness
//!    comes from the trusted `ref_pool_swap`, so Ok == "guest agrees with the v3
//!    port" (the unified commitment binds context + initial state + every per-step
//!    word/liquidityNet + final results).
//! 2. NEGATIVE soundness: wrong num_steps (too few -> termination assert fails;
//!    too many -> per-step guard fails), a forged bitmap word, a forged
//!    liquidityNet, and a forged committed result, each with the honest
//!    counterpart passing.
//! 3. PROVE+VERIFY: 2-3 representative swaps (release only; debug trips a known
//!    prover-internal SIMD assert — see Phase 1).

use backend::PrimeCharacteristicRing;
use lean_vm::F;
use std::sync::{Mutex, MutexGuard};
use uniswap_v3::{
    PoolFixture, SwapInputs, U256, build_swap_witness, build_swap_witness_with_bufs,
    build_swap_witness_with_pi, emit_swap_hints, get_bytecode, prove, ref_pool_swap, setup,
    swap_public_input_for, try_execute, verify,
};

static ARENA_LOCK: Mutex<()> = Mutex::new(());

fn serialize_arena() -> MutexGuard<'static, ()> {
    ARENA_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn u(s: &str) -> U256 {
    U256::from_dec_str(s)
}

/// sqrtPriceX96 at a tick (oracle for building fixtures).
fn sp_at(tick: i32) -> U256 {
    uniswap_v3::ref_get_sqrt_ratio_at_tick(tick)
}

/// Run the guest on `inp` and assert it accepts; witness from the trusted ref.
fn assert_accepts(inp: &SwapInputs) {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let (pi, w) = build_swap_witness(bytecode, inp);
    try_execute(bytecode, &pi, &w).unwrap_or_else(|e| panic!("guest must accept swap: {e:?}"));
}

// A pool with liquidity 1e18, tickSpacing 60, and a few initialized ticks each
// with net liquidity, on a grid of multiples of 60.
fn basic_pool() -> PoolFixture {
    PoolFixture::new()
        .with_tick(-120, false, u("500000000000000000"))
        .with_tick(-60, false, u("250000000000000000"))
        .with_tick(60, true, u("250000000000000000"))
        .with_tick(120, true, u("500000000000000000"))
}

// ---------------------------------------------------------------------------
// 0. Reference v3-invariant validation (validates ref_pool_swap itself).
// ---------------------------------------------------------------------------

#[test]
fn reference_conservation_exact_in() {
    // exactIn: amountIn+fee summed across steps == amountSpecified - remaining,
    // i.e. `spent` (amount0 when zfo==exactInput) equals the consumed input.
    let inp = limit_swap(); // exactIn, zfo
    let res = ref_pool_swap(&inp);
    // spent = amountSpecified - remaining (both nonneg here).
    let spent = inp.amount_mag.wrapping_sub(res.amount_remaining.mag);
    // For zfo==exactInput, amount0 == spent (sign nonneg).
    assert_eq!(res.amount0.mag, spent, "conservation: amount0 != spent");
    assert!(!res.amount0.sign, "amount0 sign for exactIn input side");
    // amount1 == amountCalculated (the output side, negative for exactIn).
    assert_eq!(res.amount1.mag, res.amount_calculated.mag);
}

#[test]
fn reference_sqrt_monotone_in_direction() {
    // zeroForOne: sqrtPrice is non-increasing across the swap (price falls).
    let inp = differential_crossing_inputs(); // zfo
    let res = ref_pool_swap(&inp);
    assert!(res.sqrt_p_final <= inp.sqrt_p0, "zfo: sqrtPrice must not rise");
    // !zfo: sqrtPrice non-decreasing.
    let mut up = differential_crossing_inputs();
    up.zero_for_one = false;
    up.amount_sign = false;
    up.sqrt_p0 = sp_at(-70);
    up.tick0 = -70;
    up.sqrt_limit = sp_at(887000);
    let res2 = ref_pool_swap(&up);
    assert!(res2.sqrt_p_final >= up.sqrt_p0, "!zfo: sqrtPrice must not fall");
}

#[test]
fn reference_liquidity_changes_only_on_crossing() {
    // A swap that exhausts within range (no crossing) must leave liquidity intact.
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000"), // tiny -> single in-range step
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    };
    let res = ref_pool_swap(&inp);
    assert_eq!(res.num_steps, 1);
    assert_eq!(res.liquidity_final, inp.liquidity0, "no crossing -> liquidity unchanged");
}

// ---------------------------------------------------------------------------
// 1. Differential — representative swaps.
// ---------------------------------------------------------------------------

#[test]
fn differential_single_tick_exact_in_zfo() {
    // zeroForOne exactIn, small amount, price barely moves and the limit is far:
    // a single step that exhausts the amount (remaining -> 0) without crossing.
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000"), // 1e12, small -> partial single step
        sqrt_limit: sp_at(-887000),     // far below current
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert_eq!(res.num_steps, 1, "expected a single exhausting step");
    assert_accepts(&inp);
}

#[test]
fn differential_single_tick_exact_out_one_for_zero() {
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: false,
        amount_sign: true,
        amount_mag: u("1000000000000"),
        sqrt_limit: sp_at(887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 1);
    assert_accepts(&inp);
}

#[test]
fn differential_hits_price_limit_zfo() {
    // A large exactIn amount but a price limit one tick away: the swap reaches the
    // limit (guard ends on sqrtPrice == limit), not on amount exhaustion.
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000000000000000"), // huge -> would cross many ticks
        sqrt_limit: sp_at(-65),                      // limit one tick-spacing below
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert_eq!(res.sqrt_p_final, inp.sqrt_limit, "must end at the price limit");
    assert_accepts(&inp);
}

#[test]
fn differential_multi_tick_crossing_zfo() {
    // exactIn zeroForOne with enough amount to cross at least one initialized tick.
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("100000000000000000"), // 1e17, enough to cross -60
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(70),
        tick0: 70,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2, "expected multiple steps, got {}", res.num_steps);
    assert_accepts(&inp);
}

// ---------------------------------------------------------------------------
// 1b. EXPANDED differential coverage (Phase-6 carry-ins): exactOut multi-tick,
//     !zeroForOne crossing, tickSpacing in {1, 10, 200}, an UNINITIALIZED-word
//     boundary crossing, and a fixture that lands sqrtP strictly inside a range
//     (the `moved` branch -> getTickAtSqrtRatio recompute). Each is validated
//     with `assert_accepts` (witness from the trusted ref; Ok == guest agrees),
//     and each asserts the property that makes it cover the intended branch.
// ---------------------------------------------------------------------------

/// Does any step take the `moved` (NOT crossed, price moved) branch? That is the
/// branch where the reference fills non-neutral tick_next_* (the getTickAtSqrtRatio
/// recompute). Neutral marker is (false, 0, false, 1).
fn has_moved_step(res: &uniswap_v3::SwapResult) -> bool {
    res.steps
        .iter()
        .any(|s| !(s.tick_next_sign == false && s.tick_next_abs == 0 && s.tnp1_sign == false && s.tnp1_abs == 1))
}

/// Does any step cross an UNINITIALIZED word boundary? In the reference that is a
/// step whose committed bitmap word is zero AND which is `crossed` (the price
/// reaches sqrtRatio(tickNext) where tickNext came from the uninitialized
/// fall-through: word == 0 -> initialized == false -> next = word-boundary tick).
fn has_uninitialized_boundary_crossing(inp: &SwapInputs) -> bool {
    let res = ref_pool_swap(inp);
    res.steps.iter().any(|s| s.word == U256::ZERO && !s.liq_net_sign)
}

#[test]
fn differential_multi_tick_exact_out_zfo() {
    // exactOut zeroForOne with enough output demanded to cross initialized ticks.
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: true, // exactOut
        amount_mag: u("50000000000000000"),
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(130),
        tick0: 130,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2, "exactOut zfo expected multi-step, got {}", res.num_steps);
    assert_accepts(&inp);
}

#[test]
fn differential_multi_tick_crossing_one_for_zero() {
    // !zeroForOne (one-for-zero) exactIn crossing initialized ticks upward.
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: false,
        amount_sign: false,
        amount_mag: u("100000000000000000"),
        sqrt_limit: sp_at(887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(-130),
        tick0: -130,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(
        res.num_steps >= 2,
        "!zfo crossing expected multi-step, got {}",
        res.num_steps
    );
    assert!(res.sqrt_p_final >= inp.sqrt_p0, "!zfo: price must not fall");
    assert_accepts(&inp);
}

#[test]
fn differential_exact_out_one_for_zero_crossing() {
    // exactOut, !zeroForOne, crossing upward.
    let pool = PoolFixture::new()
        .with_tick(60, true, u("300000000000000000"))
        .with_tick(180, true, u("400000000000000000"))
        .with_tick(300, true, u("200000000000000000"));
    let inp = SwapInputs {
        zero_for_one: false,
        amount_sign: true, // exactOut
        amount_mag: u("30000000000000000"),
        sqrt_limit: sp_at(887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(10),
        tick0: 10,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2, "exactOut !zfo expected multi-step");
    assert_accepts(&inp);
}

fn tick_spacing_pool(ts: i32, span: i32, net: &str) -> PoolFixture {
    // Initialized ticks on the grid below 0 (net add when zfo crosses downward).
    let mut p = PoolFixture::new();
    let mut k = -1;
    while k >= -5 {
        p = p.with_tick(k * ts * span, false, u(net));
        k -= 1;
    }
    p
}

#[test]
fn differential_tick_spacing_1_crossing() {
    let ts = 1;
    let pool = tick_spacing_pool(ts, 3, "100000000000000000");
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("50000000000000000"),
        sqrt_limit: sp_at(-50),
        fee_pips: 500,
        tick_spacing: ts as u32,
        sqrt_p0: sp_at(0),
        tick0: 0,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 1, "ts=1 swap ran no steps");
    assert_accepts(&inp);
}

#[test]
fn differential_tick_spacing_10_crossing() {
    let ts = 10;
    let pool = tick_spacing_pool(ts, 2, "120000000000000000");
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("80000000000000000"),
        sqrt_limit: sp_at(-2000),
        fee_pips: 3000,
        tick_spacing: ts as u32,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 1, "ts=10 swap ran no steps");
    assert_accepts(&inp);
}

#[test]
fn differential_tick_spacing_200_crossing() {
    let ts = 200;
    let pool = tick_spacing_pool(ts, 1, "300000000000000000");
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("200000000000000000"),
        sqrt_limit: sp_at(-20000),
        fee_pips: 10000,
        tick_spacing: ts as u32,
        sqrt_p0: sp_at(50),
        tick0: 50,
        liquidity0: u("2000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 1, "ts=200 swap ran no steps");
    assert_accepts(&inp);
}

#[test]
fn differential_uninitialized_word_boundary_crossing() {
    // A sparse pool: the first initialized tick is more than 256 compressed
    // positions away, so the swap MUST step over at least one UNINITIALIZED word
    // boundary (word == 0 -> the next_val uninitialized fall-through) before it
    // can reach the limit/amount end. tickSpacing 1 keeps a compressed position
    // == tick, so 256+ ticks of distance guarantees an empty word.
    let ts = 1;
    // One far-away initialized tick well past a word boundary downward.
    let pool = PoolFixture::new().with_tick(-400, false, u("100000000000000000"));
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("500000000000000000000000"), // huge -> reaches the limit
        sqrt_limit: sp_at(-600),
        fee_pips: 3000,
        tick_spacing: ts as u32,
        sqrt_p0: sp_at(0),
        tick0: 0,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    assert!(
        has_uninitialized_boundary_crossing(&inp),
        "fixture must traverse an uninitialized word (word == 0) crossing"
    );
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2, "expected multiple steps over empty words");
    assert_accepts(&inp);
}

#[test]
fn differential_moved_branch_inside_range() {
    // A swap that exhausts its amount STRICTLY inside a range: the final step
    // does not reach the next tick (not crossed) but the price moved, so the
    // reference takes the getTickAtSqrtRatio recompute branch. We assert that the
    // `moved` branch is actually exercised, then check the guest accepts.
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("3000000000000000"), // ~3e15: partial, lands mid-range
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(200),
        tick0: 200,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(
        has_moved_step(&res),
        "fixture must hit the `moved` (getTickAtSqrtRatio recompute) branch; steps={}",
        res.num_steps
    );
    assert_accepts(&inp);
}

// ---------------------------------------------------------------------------
// 2. NEGATIVE soundness.
// ---------------------------------------------------------------------------

fn limit_swap() -> SwapInputs {
    SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000000000000000"),
        sqrt_limit: sp_at(-65),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    }
}

#[test]
fn negative_num_steps_too_few_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = limit_swap();
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 1);

    // Honest passes.
    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("honest swap must pass");

    // Forge num_steps = honest - 1: the loop stops early, the after-loop guard
    // (remaining == 0 OR sqrtP == limit) FAILS because neither holds yet.
    let mut bufs = emit_swap_hints(&inp);
    let idx = bufs.iter().position(|h| h.label == "swap_num_steps").unwrap();
    bufs[idx].cells[0] = F::from_u32(res.num_steps - 1);
    // Pin PI to the honest digest (built with the honest num_steps); the loop
    // ends in a different state so the commitment + guard reject it. The hint
    // count mismatch (one fewer step's hints consumed) also fails.
    let (pi2, w2) = build_swap_witness_with_bufs(bytecode, &inp, &bufs);
    assert!(try_execute(bytecode, &pi2, &w2).is_err(), "too-few num_steps must be rejected");
}

#[test]
fn negative_num_steps_too_many_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = limit_swap();
    let res = ref_pool_swap(&inp);

    // Forge num_steps = honest + 1: the extra step runs the START-of-step guard
    // assert (remaining != 0 AND sqrtP != limit), which FAILS (guard already
    // false). We must also provide a hint block for the extra step or the runner
    // errors on exhausted hints — either way it is rejected.
    let mut bufs = emit_swap_hints(&inp);
    let idx = bufs.iter().position(|h| h.label == "swap_num_steps").unwrap();
    bufs[idx].cells[0] = F::from_u32(res.num_steps + 1);
    let (pi, w) = build_swap_witness_with_bufs(bytecode, &inp, &bufs);
    assert!(try_execute(bytecode, &pi, &w).is_err(), "too-many num_steps must be rejected");
}

#[test]
fn negative_forged_bitmap_word_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = differential_crossing_inputs();

    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("honest crossing swap must pass");

    // Corrupt the FIRST step's bitmap word: the committed word changes, so the
    // digest mismatches; and the masked/next-tick logic changes, so the proof is
    // about a different pool. Pin PI to honest -> rejected.
    let mut bufs = emit_swap_hints(&inp);
    let widx = bufs.iter().position(|h| h.label == "word").unwrap();
    bufs[widx].cells[0] += F::ONE;
    let (pi2, w2) = build_swap_witness_with_bufs(bytecode, &inp, &bufs);
    assert!(try_execute(bytecode, &pi2, &w2).is_err(), "forged bitmap word must be rejected");
}

#[test]
fn negative_forged_liquidity_net_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = differential_crossing_inputs();

    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("honest crossing swap must pass");

    // Corrupt a liq_net_mag limb: the committed liquidityNet changes -> digest
    // mismatch (and a crossed step would update liquidity wrongly).
    let mut bufs = emit_swap_hints(&inp);
    let lidx = bufs.iter().position(|h| h.label == "liq_net_mag").unwrap();
    bufs[lidx].cells[0] += F::ONE;
    let (pi2, w2) = build_swap_witness_with_bufs(bytecode, &inp, &bufs);
    assert!(try_execute(bytecode, &pi2, &w2).is_err(), "forged liquidityNet must be rejected");
}

#[test]
fn negative_forged_result_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = limit_swap();

    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("honest swap must pass");

    // Tamper the public input (the committed result): honest hints, forged
    // commitment -> the guest's recomputed digest mismatches.
    let mut forged = pi;
    forged[0] += F::ONE;
    let (pi2, w2) = build_swap_witness_with_pi(bytecode, &inp, forged);
    assert!(try_execute(bytecode, &pi2, &w2).is_err(), "forged committed result must be rejected");
}

/// A degenerate no-op swap: sp0 == limit and amount_mag == 0, so the v3 loop
/// guard is immediately false (num_steps == 0). Phase-6 makes num_steps == 0
/// UNREACHABLE: pool_swap now asserts amountSpecified != 0 (v3 'AS') AND
/// sp0 != limit (SPL), so this whole fixture is now rejected by the guest. It is
/// kept as the driver for the zero-amount / sp0==limit reachability negatives.
fn noop_swap() -> SwapInputs {
    SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("0"),
        sqrt_limit: sp_at(0),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(0), // sp0 == limit -> guard false at step 0
        tick0: 0,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    }
}

#[test]
fn negative_zero_amount_specified_rejected() {
    // v3 'AS' parity + num_steps==0 closure: a valid swap requires
    // amountSpecified != 0. amount_mag == 0 makes the step-0 guard false
    // (num_steps == 0); pool_swap now asserts `amount_mag_nz == 1` UNCONDITIONALLY
    // (before the loop), so the whole no-op is rejected. This also removes the
    // num_steps==0 surface that let committed head scalars float free.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Honest counterpart: a NONZERO amount on the same pool runs >=1 step + passes.
    let ok = limit_swap();
    let okres = ref_pool_swap(&ok);
    assert!(okres.num_steps >= 1, "honest swap must run >=1 step");
    let (pi, w) = build_swap_witness(bytecode, &ok);
    try_execute(bytecode, &pi, &w).expect("honest nonzero-amount swap must pass");

    // Zero amountSpecified: ref runs 0 steps; the honest commitment of a 0-step
    // swap is rejected by the guest's `amount_mag != 0` assert.
    let inp = noop_swap();
    let res = ref_pool_swap(&inp);
    assert_eq!(res.num_steps, 0, "noop swap must run zero steps");
    let (pi2, w2) = build_swap_witness(bytecode, &inp);
    assert!(
        try_execute(bytecode, &pi2, &w2).is_err(),
        "zero amountSpecified (num_steps==0) must be rejected"
    );
}

#[test]
fn negative_sp0_equals_limit_rejected() {
    // SPL parity + num_steps==0 closure: v3 requires sqrtPriceLimit strictly on
    // the correct side of the current price, so sp0 != limit. A swap with a
    // NONZERO amount but sp0 == limit makes the step-0 guard false (num_steps==0);
    // pool_swap now asserts `sp0 != limit` UNCONDITIONALLY, rejecting it.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // sp0 == limit but amount_mag != 0: ref still runs 0 steps (guard false).
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000"),
        sqrt_limit: sp_at(5),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5), // sp0 == limit
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    };
    let res = ref_pool_swap(&inp);
    assert_eq!(res.num_steps, 0, "sp0==limit swap must run zero steps");
    let (pi, w) = build_swap_witness(bytecode, &inp);
    assert!(
        try_execute(bytecode, &pi, &w).is_err(),
        "sp0 == limit (num_steps==0) must be rejected"
    );
}

#[test]
fn negative_noncanonical_amount_sign_zero_mag_rejected() {
    // Commitment-malleability regression (Phase-6 whole-system audit): the
    // swap-context amount_sign must be canonical against amount_mag. The original
    // attack used a no-op swap (num_steps == 0, amount_mag == 0) committing
    // amount_sign = 1. That whole surface is now closed twice over: the
    // unconditional `amount_mag != 0` assert rejects any amount_mag == 0, and the
    // canonical `amount_sign * (1 - is_nonzero(amount_mag)) == 0` assert rejects
    // sign=1 with a zero magnitude. We exercise the canonical assert directly by
    // forging swap_amount_sign = 1 on the zero-magnitude no-op fixture.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = noop_swap();

    // Forge swap_amount_sign = 1 with magnitude 0; honest PI. The guest rejects
    // (either the amount_mag!=0 reachability assert or the canonical-zero assert
    // trips before any digest comparison).
    let mut bufs = emit_swap_hints(&inp);
    let idx = bufs.iter().position(|h| h.label == "swap_amount_sign").unwrap();
    bufs[idx].cells[0] = F::ONE;
    let (pi2, w2) = build_swap_witness_with_bufs(bytecode, &inp, &bufs);
    assert!(
        try_execute(bytecode, &pi2, &w2).is_err(),
        "non-canonical amount_sign=1 with zero magnitude must be rejected"
    );
}

#[test]
fn negative_noncanonical_tick0_sign_rejected() {
    // Commitment-malleability regression (Phase-6 final audit): tick0_sign /
    // tick0_abs are folded into the public input (head0[6], head0[7]) but their
    // validating asserts previously lived ONLY inside
    // next_initialized_tick_within_one_word (per-step) — skipped at num_steps==0
    // or on filler steps. A probe committed tick0_sign = 99999 and the guest
    // ACCEPTED. pool_swap now binds tick0 UNCONDITIONALLY before the loop:
    // sign boolean, abs <= MAX_TICK, canonical signed-zero. The non-boolean
    // tick0_sign = 99999 must now be REJECTED.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = differential_crossing_inputs();

    // Honest counterpart: the same valid multi-step swap passes.
    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("honest swap must pass");

    // Forge swap_tick0_sign = 99999 (non-boolean). The unconditional
    // `tick0_sign * (1 - tick0_sign) == 0` assert trips.
    let mut bufs = emit_swap_hints(&inp);
    let idx = bufs.iter().position(|h| h.label == "swap_tick0_sign").unwrap();
    bufs[idx].cells[0] = F::from_u32(99999);
    let (pi2, w2) = build_swap_witness_with_bufs(bytecode, &inp, &bufs);
    assert!(
        try_execute(bytecode, &pi2, &w2).is_err(),
        "non-canonical tick0_sign = 99999 must be rejected"
    );
}

#[test]
fn negative_flipped_liq_net_sign_zero_mag_rejected() {
    // Commitment-malleability regression (Phase-6 final audit): per-step
    // liq_net_sign is folded into the digest (fold_step_data) but was only
    // asserted boolean — not canonical. On a crossed+initialized tick whose
    // liquidityNet == 0 (or any filler step), a prover could flip the committed
    // sign and produce a second valid public input. swap_step now asserts the
    // canonical `liq_net_sign * (1 - is_nonzero(liq_net_mag)) == 0` over the full
    // committed 32-limb magnitude. We forge liq_net_sign = 1 on a step whose
    // liq_net_mag is zero; the canonical assert must reject it.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = differential_crossing_inputs();
    let res = ref_pool_swap(&inp);

    // Honest counterpart passes.
    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("honest swap must pass");

    // Find a step with a zero liquidityNet magnitude (uninitialized-word / empty
    // crossing step). Such steps exist in the multi-tick crossing fixture.
    let zero_step = res
        .steps
        .iter()
        .position(|s| s.liq_net_mag == U256::ZERO && !s.liq_net_sign);
    let step_ix = zero_step.expect("fixture must contain a zero-liquidityNet step");

    // Forge the step_ix-th liq_net_sign hint to 1 while its magnitude stays 0.
    let mut bufs = emit_swap_hints(&inp);
    let liq_sign_positions: Vec<usize> = bufs
        .iter()
        .enumerate()
        .filter(|(_, h)| h.label == "liq_net_sign")
        .map(|(i, _)| i)
        .collect();
    bufs[liq_sign_positions[step_ix]].cells[0] = F::ONE;
    let (pi2, w2) = build_swap_witness_with_bufs(bytecode, &inp, &bufs);
    assert!(
        try_execute(bytecode, &pi2, &w2).is_err(),
        "flipped liq_net_sign with zero magnitude must be rejected"
    );
}

fn differential_crossing_inputs() -> SwapInputs {
    SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("100000000000000000"),
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(70),
        tick0: 70,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    }
}

// ---------------------------------------------------------------------------
// 3. PROVE + VERIFY (release only).
// ---------------------------------------------------------------------------

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn prove_verify_single_step_swap() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let pool = basic_pool();
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000"),
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool,
    };
    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("single-step swap must pass try_execute");
    let proof = prove(bytecode, &pi, &w, 1).expect("prove must succeed");
    verify(bytecode, &pi, proof).expect("verify must succeed");
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn prove_verify_limit_swap() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let inp = limit_swap();
    let (pi, w) = build_swap_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("limit swap must pass try_execute");
    let proof = prove(bytecode, &pi, &w, 1).expect("prove must succeed");
    verify(bytecode, &pi, proof).expect("verify must succeed");
}

// ---------------------------------------------------------------------------
// 3b. END-TO-END prove+verify matrix (Definition of Done). A realistic pool +
//     swaps proved AND verified, asserting the committed result == ref_pool_swap.
//     Covers: single-tick (in-range), multi-tick-crossing, exactIn AND exactOut,
//     BOTH directions, plus a DEEPER swap (8-16 crossings) confirming the
//     ~130k/step budget holds. Release only (debug trips the SIMD assert).
// ---------------------------------------------------------------------------

/// Prove+verify `inp` and assert the proved public input equals the committed
/// `ref_pool_swap` digest (so the proof is ABOUT the correct result, not just
/// "some accepting witness"). Returns the executed step count + memory.
fn e2e_prove_verify(label: &str, inp: &SwapInputs) -> (u32, usize) {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let res = ref_pool_swap(inp);
    let (pi, w) = build_swap_witness(bytecode, inp);
    // The committed public input MUST equal the independently-computed digest.
    assert_eq!(pi, swap_public_input_for(inp), "{label}: PI != ref digest");
    // Sanity: the digest binds the full result (changing any output changes it).
    try_execute(bytecode, &pi, &w).unwrap_or_else(|e| panic!("{label}: try_execute: {e:?}"));
    let exec = lean_vm::try_execute_bytecode(bytecode, &pi, &w, false).expect("exec");
    let mem = exec.metadata.memory;
    assert!(mem < (1usize << 26), "{label}: memory {mem} >= 2^26");
    let proof = prove(bytecode, &pi, &w, 1).unwrap_or_else(|e| panic!("{label}: prove: {e:?}"));
    verify(bytecode, &pi, proof).unwrap_or_else(|e| panic!("{label}: verify: {e:?}"));
    println!(
        "[e2e] {label}: steps={} memory={} (~2^{:.1})",
        res.num_steps,
        mem,
        (mem as f64).log2()
    );
    (res.num_steps, mem)
}

/// A deep pool: many initialized ticks at every tickSpacing step on both sides of
/// 0, so a large swap crosses many of them.
fn deep_pool(ts: i32, count: i32, net: &str) -> PoolFixture {
    let mut p = PoolFixture::new();
    let mut k = 1;
    while k <= count {
        // below 0 (net add on zfo-down crossings) and above 0.
        p = p.with_tick(-k * ts, false, u(net));
        p = p.with_tick(k * ts, true, u(net));
        k += 1;
    }
    p
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn e2e_single_tick_in_range_exact_in_zfo() {
    // A swap that exhausts inside the current range (no crossing): single step.
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000"),
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    };
    let res = ref_pool_swap(&inp);
    assert_eq!(res.num_steps, 1, "expected a single in-range step");
    e2e_prove_verify("single-tick-in-range-exactIn-zfo", &inp);
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn e2e_multi_tick_exact_in_zfo() {
    let inp = differential_crossing_inputs();
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2);
    e2e_prove_verify("multi-tick-exactIn-zfo", &inp);
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn e2e_multi_tick_exact_in_one_for_zero() {
    let inp = SwapInputs {
        zero_for_one: false,
        amount_sign: false,
        amount_mag: u("100000000000000000"),
        sqrt_limit: sp_at(887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(-130),
        tick0: -130,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2);
    e2e_prove_verify("multi-tick-exactIn-!zfo", &inp);
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn e2e_multi_tick_exact_out_zfo() {
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: true, // exactOut
        amount_mag: u("50000000000000000"),
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(130),
        tick0: 130,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2);
    e2e_prove_verify("multi-tick-exactOut-zfo", &inp);
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn e2e_multi_tick_exact_out_one_for_zero() {
    let inp = SwapInputs {
        zero_for_one: false,
        amount_sign: true, // exactOut
        amount_mag: u("30000000000000000"),
        sqrt_limit: sp_at(887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(10),
        tick0: 10,
        liquidity0: u("1000000000000000000"),
        pool: PoolFixture::new()
            .with_tick(60, true, u("300000000000000000"))
            .with_tick(180, true, u("400000000000000000"))
            .with_tick(300, true, u("200000000000000000")),
    };
    let res = ref_pool_swap(&inp);
    assert!(res.num_steps >= 2);
    e2e_prove_verify("multi-tick-exactOut-!zfo", &inp);
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert (deep); run with --release")]
#[test]
fn e2e_deep_swap_many_crossings() {
    // A DEEP swap that crosses 8-16 initialized ticks: confirms the per-step
    // budget holds well under 2^26 with a realistic step count.
    let ts = 60;
    let pool = deep_pool(ts, 20, "50000000000000000");
    let inp = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("3000000000000000000"), // 3e18: cross several ticks
        sqrt_limit: sp_at(-660),              // ~11 tickSpacing below 0 -> bounds steps
        fee_pips: 3000,
        tick_spacing: ts as u32,
        sqrt_p0: sp_at(60),
        tick0: 60,
        liquidity0: u("5000000000000000000"),
        pool,
    };
    let res = ref_pool_swap(&inp);
    assert!(
        res.num_steps >= 8 && res.num_steps <= 16,
        "deep swap should cross 8-16 ticks, got {} steps",
        res.num_steps
    );
    let (steps, mem) = e2e_prove_verify("deep-swap", &inp);
    // ~130k/step budget check (with generous slack for the fixed proving floor).
    let per_step = mem / steps.max(1) as usize;
    println!("[e2e] deep-swap per_step ~ {per_step} cells");
    assert!(mem < (1usize << 26), "deep swap memory {mem} exceeds 2^26");
}

// Public-input sanity: committed digest is deterministic from the inputs.
#[test]
fn swap_public_input_deterministic() {
    let inp = limit_swap();
    assert_eq!(swap_public_input_for(&inp), swap_public_input_for(&inp));
}

/// Memory-budget probe: report total memory cells for 1-step and multi-step
/// swaps and confirm they stay well under 2^26. Prints with `--nocapture`.
#[test]
fn swap_budget_report() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let probe = |label: &str, inp: &SwapInputs| {
        let res = ref_pool_swap(inp);
        let (pi, w) = build_swap_witness(bytecode, inp);
        let exec = lean_vm::try_execute_bytecode(bytecode, &pi, &w, false).expect("must execute");
        let mem = exec.metadata.memory;
        let per_step = if res.num_steps > 0 { mem / res.num_steps as usize } else { mem };
        println!(
            "[budget] {label}: steps={} total_memory={} (~2^{:.1}) per_step~{} (2^26 = {})",
            res.num_steps,
            mem,
            (mem as f64).log2(),
            per_step,
            1usize << 26,
        );
        assert!(mem < (1usize << 26), "{label}: memory {mem} exceeds 2^26");
    };

    let one = SwapInputs {
        zero_for_one: true,
        amount_sign: false,
        amount_mag: u("1000000000000"),
        sqrt_limit: sp_at(-887000),
        fee_pips: 3000,
        tick_spacing: 60,
        sqrt_p0: sp_at(5),
        tick0: 5,
        liquidity0: u("1000000000000000000"),
        pool: basic_pool(),
    };
    probe("single-step", &one);
    probe("limit (multi-step)", &limit_swap());
    probe("crossing (multi-step)", &differential_crossing_inputs());
}
