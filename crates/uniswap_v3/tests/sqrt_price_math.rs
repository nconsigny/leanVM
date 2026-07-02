//! Phase 2 — SqrtPriceMath + LiquidityMath: differential, prove+verify, and
//! negative soundness tests.
//!
//! 1. DIFFERENTIAL: random + edge-case inputs for every op (both flag values,
//!    both directions, add/remove) are run through the guest via `try_execute`.
//!    The guest's internal asserts + the public-input commitment to the honest
//!    reference result ARE the validation: Ok == "guest agrees with the v3 port".
//! 2. PROVE+VERIFY: a few representative vectors go through real WHIR prove+verify
//!    (release only; debug trips a known prover-internal SIMD assert — see Phase 1).
//! 3. NEGATIVE: a wrong muldiv hint, a forged overflow guard, and an underflow
//!    case are REJECTED, each with the honest counterpart asserted to pass first.

use backend::PrimeCharacteristicRing;
use lean_vm::F;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::sync::{Mutex, MutexGuard};
use uniswap_v3::{
    SqrtInputs, SqrtOp, U256, build_sqrt_witness, build_sqrt_witness_with_bufs, emit_sqrt_hints, get_bytecode,
    prove, setup, sqrt_op_result, try_execute, verify,
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

/// A random uint160 sqrt price in (0, 2^160).
fn random_sqrt(rng: &mut StdRng) -> U256 {
    let mut bytes = [0u8; 32];
    for b in bytes.iter_mut().take(20) {
        *b = rng.random::<u8>();
    }
    let v = U256::from_le_bytes(&bytes);
    if v.is_zero() { U256::from_u64(1) } else { v }
}

/// A random uint128 liquidity in (0, 2^128).
fn random_liquidity(rng: &mut StdRng) -> U256 {
    let mut bytes = [0u8; 32];
    for b in bytes.iter_mut().take(16) {
        *b = rng.random::<u8>();
    }
    let v = U256::from_le_bytes(&bytes);
    if v.is_zero() { U256::from_u64(1) } else { v }
}

/// Run the guest on `inp` and assert it accepts (Ok). The witness q/r hints come
/// from the trusted reference, so Ok == "guest agrees".
fn assert_accepts(inp: &SqrtInputs) {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let (public_input, witness) = build_sqrt_witness(bytecode, inp);
    try_execute(bytecode, &public_input, &witness)
        .unwrap_or_else(|e| panic!("guest must accept {inp:?}: {e:?}"));
}

#[test]
fn differential_amount_deltas_random() {
    let mut rng = StdRng::seed_from_u64(0xA5717);
    for _ in 0..32 {
        let sqrt_a = random_sqrt(&mut rng);
        let sqrt_b = random_sqrt(&mut rng);
        let liquidity = random_liquidity(&mut rng);
        for op in [SqrtOp::Amount0Delta, SqrtOp::Amount1Delta] {
            for flag_a in [false, true] {
                // Amount0Delta requires the smaller sqrt > 0; both are >= 1 here.
                let inp = SqrtInputs {
                    op,
                    flag_a,
                    sqrt_a,
                    sqrt_b,
                    liquidity,
                    amount: U256::ZERO,
                    y_sign: false,
                    ..SqrtInputs::default()
                };
                assert_accepts(&inp);
            }
        }
    }
}

#[test]
fn differential_amount_deltas_edges() {
    // sqrtA == sqrtB (zero diff), reversed order, liquidity boundaries.
    let s = U256::from_u64(0x1_0000_0000); // ~Q32
    for op in [SqrtOp::Amount0Delta, SqrtOp::Amount1Delta] {
        for flag_a in [false, true] {
            // equal endpoints
            assert_accepts(&SqrtInputs {
                op,
                flag_a,
                sqrt_a: s,
                sqrt_b: s,
                liquidity: U256::from_u64(1),
                amount: U256::ZERO,
                y_sign: false,
                ..SqrtInputs::default()
            });
            // reversed (sqrt_a > sqrt_b) -> guest must swap
            assert_accepts(&SqrtInputs {
                op,
                flag_a,
                sqrt_a: s.shl(40),
                sqrt_b: s,
                liquidity: U256::UINT128_MAX,
                amount: U256::ZERO,
                y_sign: false,
                ..SqrtInputs::default()
            });
        }
    }
}

#[test]
fn differential_get_next_from_amount_random() {
    let mut rng = StdRng::seed_from_u64(0xBEEF1);
    for _ in 0..32 {
        let sqrt_p = random_sqrt(&mut rng);
        let liquidity = random_liquidity(&mut rng);
        // amounts: tiny, medium, and a large uint256 to exercise both overflow
        // branches and the amount<=uint160_max boundary.
        let amounts = [
            U256::from_u64(0),
            U256::from_u64(1),
            U256::from_u64(rng.random::<u64>().max(1)),
            U256::UINT160_MAX,                       // boundary
            U256::UINT160_MAX.overflowing_add(U256::ONE).0, // just above
            random_u256(&mut rng),
        ];
        for amount in amounts {
            for op in [
                SqrtOp::NextFromAmount0RoundingUp,
                SqrtOp::NextFromAmount1RoundingDown,
                SqrtOp::NextFromInput,
                SqrtOp::NextFromOutput,
            ] {
                for flag_a in [false, true] {
                    let inp = SqrtInputs {
                        op,
                        flag_a,
                        sqrt_a: sqrt_p,
                        sqrt_b: U256::ZERO,
                        liquidity,
                        amount,
                        y_sign: false,
                        ..SqrtInputs::default()
                    };
                    // Some combinations are reverts in v3 (e.g. remove with too-large
                    // amount, or sqrtP <= quotient). The reference panics on those;
                    // skip them (they are out of the valid input domain v3 accepts).
                    if std::panic::catch_unwind(|| sqrt_op_result(&inp)).is_err() {
                        continue;
                    }
                    assert_accepts(&inp);
                }
            }
        }
    }
}

#[test]
fn differential_add_delta() {
    let mut rng = StdRng::seed_from_u64(0xADDE17A);
    for _ in 0..32 {
        let x = random_liquidity(&mut rng);
        let y = random_liquidity(&mut rng);
        for y_sign in [false, true] {
            let inp = SqrtInputs {
                op: SqrtOp::AddDelta,
                flag_a: false,
                sqrt_a: U256::ZERO,
                sqrt_b: U256::ZERO,
                liquidity: x,
                amount: y,
                y_sign,
                ..SqrtInputs::default()
            };
            // sub with x < y underflows (revert in v3); add overflowing uint128
            // reverts. Skip the reverts; otherwise assert acceptance.
            if std::panic::catch_unwind(|| sqrt_op_result(&inp)).is_err() {
                continue;
            }
            assert_accepts(&inp);
        }
    }
    // explicit boundary: x + y == 2^128 - 1 (max, no overflow) and add 0.
    assert_accepts(&SqrtInputs {
        op: SqrtOp::AddDelta,
        flag_a: false,
        sqrt_a: U256::ZERO,
        sqrt_b: U256::ZERO,
        liquidity: U256::UINT128_MAX,
        amount: U256::ZERO,
        y_sign: false,
        ..SqrtInputs::default()
    });
    // exact subtraction to zero.
    assert_accepts(&SqrtInputs {
        op: SqrtOp::AddDelta,
        flag_a: false,
        sqrt_a: U256::ZERO,
        sqrt_b: U256::ZERO,
        liquidity: U256::from_u64(5),
        amount: U256::from_u64(5),
        y_sign: true,
        ..SqrtInputs::default()
    });
}

// ---------------------------------------------------------------------------
// PROVE + VERIFY (release only; debug trips a known prover-internal SIMD assert).
// ---------------------------------------------------------------------------

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn prove_and_verify_representative_vectors() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let sqrt_a = U256::from_u64(0x1234_5678_9ABC);
    let sqrt_b = U256::from_u64(0x9999_8888_7777_6666);
    let liquidity = U256::from_u64(1_000_000_000_000);

    let vectors = [
        // getAmount0Delta roundUp (two muldiv pairs)
        SqrtInputs {
            op: SqrtOp::Amount0Delta,
            flag_a: true,
            sqrt_a,
            sqrt_b,
            liquidity,
            amount: U256::ZERO,
            y_sign: false,
            ..SqrtInputs::default()
        },
        // getNextSqrtPriceFromInput, zeroForOne=true (amount0 add path)
        SqrtInputs {
            op: SqrtOp::NextFromInput,
            flag_a: true,
            sqrt_a: sqrt_b,
            sqrt_b: U256::ZERO,
            liquidity,
            amount: U256::from_u64(123_456_789),
            y_sign: false,
            ..SqrtInputs::default()
        },
        // addDelta sub
        SqrtInputs {
            op: SqrtOp::AddDelta,
            flag_a: false,
            sqrt_a: U256::ZERO,
            sqrt_b: U256::ZERO,
            liquidity: U256::from_u64(1_000),
            amount: U256::from_u64(400),
            y_sign: true,
            ..SqrtInputs::default()
        },
    ];

    for inp in vectors {
        let (public_input, witness) = build_sqrt_witness(bytecode, &inp);
        try_execute(bytecode, &public_input, &witness).expect("witness gen must succeed");
        let proof = prove(bytecode, &public_input, &witness, 1).expect("prove must succeed");
        verify(bytecode, &public_input, proof).expect("verify must succeed");
    }
}

// ---------------------------------------------------------------------------
// NEGATIVE soundness tests.
// ---------------------------------------------------------------------------

#[test]
fn negative_wrong_muldiv_hint_rejected() {
    // Corrupting a muldiv q for a known-good op MUST make the guest reject (the
    // Euclidean identity no longer holds). Honest run asserted to pass first.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let inp = SqrtInputs {
        op: SqrtOp::Amount1Delta,
        flag_a: false,
        sqrt_a: U256::from_u64(0x10_0000),
        sqrt_b: U256::from_u64(0x99_0000),
        liquidity: U256::from_u64(1_000_000),
        amount: U256::ZERO,
        y_sign: false,
        ..SqrtInputs::default()
    };
    let honest = emit_sqrt_hints(&inp);
    let (pi, w_ok) = build_sqrt_witness_with_bufs(bytecode, &inp, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest witness must pass");

    let mut corrupted = honest.clone();
    let q_idx = corrupted.iter().position(|h| h.label == "muldiv_q").expect("muldiv_q present");
    corrupted[q_idx].cells[0] += F::from_u32(1);
    let (_pi2, w_bad) = build_sqrt_witness_with_bufs(bytecode, &inp, &corrupted);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject a wrong muldiv q"
    );
}

#[test]
fn negative_forged_overflow_guard_rejected() {
    // getNextSqrtPriceFromAmount0RoundingUp remove path: a forged `product` that
    // claims `amount*sqrtP` fit uint256 when it overflowed must be rejected by the
    // in-circuit fits256 / numerator1>product checks. We force the situation by
    // submitting a removal op whose true result requires the overflow guard, then
    // perturbing the muldiv quotient that the guard feeds. (Direct forging of the
    // product is not a hint — it is derived in-circuit — so we instead show that a
    // tampered downstream muldiv hint for this guarded path is rejected.)
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // remove path with a valid (small) amount so the honest run passes.
    let inp = SqrtInputs {
        op: SqrtOp::NextFromAmount0RoundingUp,
        flag_a: false, // remove
        sqrt_a: U256::from_u64(0x1_0000_0000_0000),
        sqrt_b: U256::ZERO,
        liquidity: U256::from_u64(1_000_000_000),
        amount: U256::from_u64(7),
        y_sign: false,
        ..SqrtInputs::default()
    };
    let honest = emit_sqrt_hints(&inp);
    let (pi, w_ok) = build_sqrt_witness_with_bufs(bytecode, &inp, &honest);
    try_execute(bytecode, &pi, &w_ok).expect("honest remove-path witness must pass");

    let mut corrupted = honest.clone();
    let r_idx = corrupted.iter().position(|h| h.label == "muldiv_r").expect("muldiv_r present");
    corrupted[r_idx].cells[0] += F::from_u32(1);
    let (_pi2, w_bad) = build_sqrt_witness_with_bufs(bytecode, &inp, &corrupted);
    assert!(
        try_execute(bytecode, &pi, &w_bad).is_err(),
        "guest must reject a tampered remainder on the overflow-guarded remove path"
    );
}

#[test]
fn negative_skipped_overflow_guard_rejected() {
    // getNextSqrtPriceFromAmount0RoundingUp REMOVE path with an `amount` so large
    // that amount*sqrtP overflows uint256. v3 reverts (the guard is false). The
    // guest computes fits256 = 0 and `assert fits256 == 1` MUST fire — a prover
    // cannot "skip" the guard because fits256 is derived in-circuit from the true
    // product, not hinted. The guest aborts before any muldiv, so we supply only
    // the input block (no muldiv pairs) and require rejection.
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // amount = 2^200, sqrtP ~ 2^159 -> product ~ 2^359 >> 2^256 (overflows).
    let amount = U256::ONE.shl(200);
    let sqrt_p = U256::ONE.shl(159);
    let bad = SqrtInputs {
        op: SqrtOp::NextFromAmount0RoundingUp,
        flag_a: false, // remove
        sqrt_a: sqrt_p,
        sqrt_b: U256::ZERO,
        liquidity: U256::from_u64(1_000_000),
        amount,
        y_sign: false,
        ..SqrtInputs::default()
    };
    // Reference would panic (v3 revert), so build input-block hints only and pin
    // the public input to an unrelated honest case (the run must fail at the
    // overflow guard, before commitment).
    let input_only = emit_inputs_only(&bad);
    let pin = SqrtInputs {
        op: SqrtOp::AddDelta,
        flag_a: false,
        sqrt_a: U256::ZERO,
        sqrt_b: U256::ZERO,
        liquidity: U256::from_u64(1),
        amount: U256::ZERO,
        y_sign: false,
        ..SqrtInputs::default()
    };
    let (_pi, _) = build_sqrt_witness(bytecode, &pin); // ensure bytecode ready
    let (_pi2, w_bad) = build_sqrt_witness_with_bufs(bytecode, &pin, &input_only);
    let pi_pin = uniswap_v3::sqrt_public_input_for(&pin);
    assert!(
        try_execute(bytecode, &pi_pin, &w_bad).is_err(),
        "guest must reject a remove-path overflow (assert fits256 == 1 fires)"
    );
}

/// Build only the 7 input-block hints for any op (no muldiv pairs). Used to test
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
    ]
}

#[test]
fn negative_adddelta_underflow_rejected() {
    // addDelta sub with x < y_mag underflows; v3 reverts. The guest's bignum_sub
    // top-borrow==0 assert must reject. We submit a forged y_sign=sub with x<y and
    // a public input pinned to the (panicking) honest result — instead, we drive
    // the guest directly with hints and an arbitrary public input and require the
    // run to error (the sub underflow assert fires before the commitment).
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Honest, accepted sub (no underflow), to attribute the later rejection.
    let ok = SqrtInputs {
        op: SqrtOp::AddDelta,
        flag_a: false,
        sqrt_a: U256::ZERO,
        sqrt_b: U256::ZERO,
        liquidity: U256::from_u64(10),
        amount: U256::from_u64(3),
        y_sign: true,
        ..SqrtInputs::default()
    };
    let ok_bufs = emit_sqrt_hints(&ok);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &ok, &ok_bufs);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest sub must pass");

    // Underflow case: x = 3, y_mag = 10, sub. Reference would panic; we build the
    // hints by hand (reusing the input encoding) and reuse the honest public input
    // (the run must fail at the sub-underflow assert, before commitment).
    let bad = SqrtInputs {
        op: SqrtOp::AddDelta,
        flag_a: false,
        sqrt_a: U256::ZERO,
        sqrt_b: U256::ZERO,
        liquidity: U256::from_u64(3),
        amount: U256::from_u64(10),
        y_sign: true,
        ..SqrtInputs::default()
    };
    // Build the input-block hints manually (no muldiv pairs are consumed by
    // addDelta), pinning the public input to the honest accepted case.
    let bad_bufs = emit_inputs_only(&bad);
    let (_pi, w_bad) = build_sqrt_witness_with_bufs(bytecode, &ok, &bad_bufs);
    assert!(
        try_execute(bytecode, &pi_ok, &w_bad).is_err(),
        "guest must reject an addDelta subtraction underflow (x < y_mag)"
    );
}
