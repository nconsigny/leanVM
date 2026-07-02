//! Trusted Rust reference for FullMath, plus the hint-emission helpers that feed
//! the guest the prover-supplied values it consumes.
//!
//! ------------------------------------------------------------------------
//! HINT CONTRACT (must match zkdsl_implem/bignum.py + main.py EXACTLY)
//! ------------------------------------------------------------------------
//! The guest `main` consumes hints in this order:
//!   "a"     : one 32-cell buffer  (uint256 a, little-endian base-256 limbs)
//!   "b"     : one 32-cell buffer  (uint256 b)
//!   "denom" : one 32-cell buffer  (uint256 denom)
//!   "muldiv_q" : one (NA+NB)-cell buffer per bignum_muldiv call, in call order
//!   "muldiv_r" : one ND-cell buffer per bignum_muldiv call, SAME order
//! `main` calls mulDiv (pair #0) then mulDivRoundingUp (pair #1), so two q/r
//! pairs are queued, both for the SAME (a, b, denom). Each is the floor quotient
//! and remainder of `a*b / denom` at uint256 width (NA=NB=ND=32 => q width 64,
//! r width 32). FIFO per label; the runner errors unless every queued buffer is
//! consumed.

use backend::PrimeCharacteristicRing;
use lean_vm::{F, PUBLIC_INPUT_LEN};

use crate::bigint::{U256, U512};

/// uint256 width in base-256 limbs.
pub const UINT256_LIMBS: usize = 32;
/// Product (a*b) width in base-256 limbs.
pub const PRODUCT_LIMBS: usize = 2 * UINT256_LIMBS; // 64

/// `floor(a*b / denom)` and `a*b mod denom`. `denom` must be nonzero.
///
/// Returned as `(q, r)`: `q` is the full 512-bit quotient (`U512`; its high
/// words are zero whenever the quotient fits in 256 bits, which it always does
/// for `denom >= 1` since `q <= a*b < 2^512` and in our callers `q < 2^256`),
/// `r` is the `U256` remainder with `r < denom`.
pub fn muldiv(a: U256, b: U256, denom: U256) -> (U512, U256) {
    assert!(!denom.is_zero(), "muldiv: denom == 0");
    let prod = a.full_mul(b);
    prod.divmod_u256(denom)
}

/// Convenience: the floor quotient as a `U256` (asserts it fits in 256 bits).
pub fn muldiv_floor_u256(a: U256, b: U256, denom: U256) -> U256 {
    let (q, _) = muldiv(a, b, denom);
    assert_eq!(q.words[4..], [0, 0, 0, 0], "muldiv quotient exceeds uint256");
    U256 {
        words: [q.words[0], q.words[1], q.words[2], q.words[3]],
    }
}

/// Convenience: the ceil quotient as a `U256` (`mulDivRoundingUp`).
/// Asserts it fits in 256 bits (true whenever floor+1 does not overflow, which
/// the Phase-2 callers guarantee).
pub fn muldiv_ceil_u256(a: U256, b: U256, denom: U256) -> U256 {
    let (q, r) = muldiv(a, b, denom);
    assert_eq!(q.words[4..], [0, 0, 0, 0], "muldiv quotient exceeds uint256");
    let q256 = U256 {
        words: [q.words[0], q.words[1], q.words[2], q.words[3]],
    };
    if r.is_zero() {
        q256
    } else {
        let (c, carry) = q256.overflowing_add(U256::ONE);
        assert!(!carry, "mulDivRoundingUp result overflows uint256");
        c
    }
}

/// One queued `hint_witness` buffer (label + cells), in guest consumption order.
#[derive(Clone, Debug)]
pub struct HintBuf {
    pub label: &'static str,
    pub cells: Vec<F>,
}

/// Emit the operand hints `a`, `b`, `denom` (one 32-limb buffer each), in the
/// order `main` reads them.
pub fn emit_operand_hints(a: U256, b: U256, denom: U256) -> Vec<HintBuf> {
    vec![
        HintBuf {
            label: "a",
            cells: a.to_le_base256_limbs(UINT256_LIMBS),
        },
        HintBuf {
            label: "b",
            cells: b.to_le_base256_limbs(UINT256_LIMBS),
        },
        HintBuf {
            label: "denom",
            cells: denom.to_le_base256_limbs(UINT256_LIMBS),
        },
    ]
}

/// Emit ONE `(muldiv_q, muldiv_r)` hint pair for a single `bignum_muldiv`
/// (uint256 width): `q` as `PRODUCT_LIMBS` cells, `r` as `UINT256_LIMBS` cells.
pub fn emit_one_muldiv_pair(a: U256, b: U256, denom: U256) -> Vec<HintBuf> {
    let (q, r) = muldiv(a, b, denom);
    vec![
        HintBuf {
            label: "muldiv_q",
            cells: q.to_le_base256_limbs(PRODUCT_LIMBS),
        },
        HintBuf {
            label: "muldiv_r",
            cells: r.to_le_base256_limbs(UINT256_LIMBS),
        },
    ]
}

/// Full hint sequence for `main`: operands, then the two `(q, r)` pairs that
/// mulDiv and mulDivRoundingUp consume (same `(a, b, denom)`, same floor q/r).
///
/// This is the canonical "honest prover" emission. Negative tests perturb the
/// returned buffers (e.g. corrupt a `muldiv_q` cell) to confirm the guest's
/// constraints reject a wrong witness.
///
/// Phase-1 bignum forgery-test surface: maps `(a, b, denom)` onto the Phase-2
/// driver's `RawMulDiv` op (op 7), which calls `mulDiv` then `mulDivRoundingUp`
/// on arbitrary uint256 operands — exactly the Phase-1 behaviour, now driven
/// through the unified `main.py`. The op-7 input mapping is:
///   sqrtA = a, sqrtB = b, amount = denom, liquidity = 0 (unused).
pub fn raw_muldiv_inputs(a: U256, b: U256, denom: U256) -> SqrtInputs {
    SqrtInputs {
        op: SqrtOp::RawMulDiv,
        sqrt_a: a,
        sqrt_b: b,
        amount: denom,
        ..SqrtInputs::default()
    }
}

/// Honest hint emission for a Phase-1 `(a, b, denom)` mulDiv, via op 7.
pub fn emit_main_hints(a: U256, b: U256, denom: U256) -> Vec<HintBuf> {
    emit_sqrt_hints(&raw_muldiv_inputs(a, b, denom))
}

// ===========================================================================
// Phase 2: SqrtPriceMath + LiquidityMath — faithful v3-core Rust port, plus the
// ordered hint emitter the guest consumes.
// ===========================================================================
//
// The reference computes each op exactly as v3-core Solidity does, AND replays
// the SAME branch decisions so it can push the per-mulDiv (q, r) hint pairs in
// the EXACT order the guest calls mulDiv / mulDivRoundingUp. A small recorder
// (`MdRecorder`) both computes a muldiv result and appends its (q, r) buffers.

/// Result of every `floor`/`ceil` division: the value plus the appended hints.
struct MdRecorder {
    bufs: Vec<HintBuf>,
}

impl MdRecorder {
    fn new() -> Self {
        MdRecorder { bufs: Vec::new() }
    }

    /// `floor(a*b/denom)`, recording one (q, r) pair (uint256 width).
    fn mul_div(&mut self, a: U256, b: U256, denom: U256) -> U256 {
        self.bufs.extend(emit_one_muldiv_pair(a, b, denom));
        muldiv_floor_u256(a, b, denom)
    }

    /// `ceil(a*b/denom)`, recording one (q, r) pair (uint256 width). The guest's
    /// `mulDivRoundingUp` performs exactly one `bignum_muldiv`, so one pair.
    fn mul_div_rounding_up(&mut self, a: U256, b: U256, denom: U256) -> U256 {
        self.bufs.extend(emit_one_muldiv_pair(a, b, denom));
        muldiv_ceil_u256(a, b, denom)
    }

    fn div_floor(&mut self, x: U256, y: U256) -> U256 {
        self.mul_div(x, U256::ONE, y)
    }

    fn div_rounding_up(&mut self, x: U256, y: U256) -> U256 {
        self.mul_div_rounding_up(x, U256::ONE, y)
    }
}

/// `uint256(liquidity) << 96` (liquidity is a uint128; result fits uint256).
fn numerator1(liquidity: U256) -> U256 {
    assert!(liquidity.fits_uint128(), "liquidity must be uint128");
    // liquidity < 2^128 => liquidity<<96 < 2^224 < 2^256; no overflow.
    liquidity.shl(96)
}

/// v3 SqrtPriceMath.getAmount0Delta (faithful), recording muldiv hints.
fn ref_get_amount0_delta(rec: &mut MdRecorder, sqrt_a: U256, sqrt_b: U256, liquidity: U256, round_up: bool) -> U256 {
    let (lo, hi) = if sqrt_a > sqrt_b { (sqrt_b, sqrt_a) } else { (sqrt_a, sqrt_b) };
    assert!(!lo.is_zero(), "require(sqrtRatioAX96 > 0)");
    let num1 = numerator1(liquidity);
    let num2 = hi.wrapping_sub(lo);
    if round_up {
        let inner = rec.mul_div_rounding_up(num1, num2, hi);
        rec.div_rounding_up(inner, lo)
    } else {
        let inner = rec.mul_div(num1, num2, hi);
        rec.div_floor(inner, lo)
    }
}

/// v3 SqrtPriceMath.getAmount1Delta (faithful), recording muldiv hints.
fn ref_get_amount1_delta(rec: &mut MdRecorder, sqrt_a: U256, sqrt_b: U256, liquidity: U256, round_up: bool) -> U256 {
    let (lo, hi) = if sqrt_a > sqrt_b { (sqrt_b, sqrt_a) } else { (sqrt_a, sqrt_b) };
    let diff = hi.wrapping_sub(lo);
    if round_up {
        rec.mul_div_rounding_up(liquidity, diff, U256::Q96)
    } else {
        rec.mul_div(liquidity, diff, U256::Q96)
    }
}

/// v3 SqrtPriceMath.getNextSqrtPriceFromAmount0RoundingUp (faithful).
fn ref_next_from_amount0(rec: &mut MdRecorder, sqrt_p: U256, liquidity: U256, amount: U256, add: bool) -> U256 {
    if amount.is_zero() {
        assert!(sqrt_p.fits_uint160(), "sqrtP must be uint160");
        return sqrt_p;
    }
    let num1 = numerator1(liquidity);
    if add {
        let prod512 = amount.full_mul(sqrt_p);
        let fits256 = prod512.words[4..] == [0, 0, 0, 0];
        if fits256 {
            let product = U256 {
                words: [prod512.words[0], prod512.words[1], prod512.words[2], prod512.words[3]],
            };
            let (denominator, carry) = num1.overflowing_add(product);
            if !carry {
                // denominator >= num1
                let res = rec.mul_div_rounding_up(num1, sqrt_p, denominator);
                assert!(res.fits_uint160(), "uint160(...)");
                return res;
            }
        }
        // fallback: divRoundingUp(num1, (num1 / sqrtP) + amount)
        let n1_div_p = rec.div_floor(num1, sqrt_p);
        let (denom, carry) = n1_div_p.overflowing_add(amount);
        assert!(!carry, "SafeMath .add() overflow");
        let res = rec.div_rounding_up(num1, denom);
        assert!(res.fits_uint160(), "uint160(...)");
        res
    } else {
        let prod512 = amount.full_mul(sqrt_p);
        let fits256 = prod512.words[4..] == [0, 0, 0, 0];
        assert!(fits256, "(amount*sqrtP)/amount == sqrtP");
        let product = U256 {
            words: [prod512.words[0], prod512.words[1], prod512.words[2], prod512.words[3]],
        };
        assert!(num1 > product, "numerator1 > product");
        let denominator = num1.wrapping_sub(product);
        let res = rec.mul_div_rounding_up(num1, sqrt_p, denominator);
        assert!(res.fits_uint160(), "toUint160()");
        res
    }
}

/// v3 SqrtPriceMath.getNextSqrtPriceFromAmount1RoundingDown (faithful to the
/// *collapsed* form — see the guest docstring; result is bit-identical to v3).
fn ref_next_from_amount1(rec: &mut MdRecorder, sqrt_p: U256, liquidity: U256, amount: U256, add: bool) -> U256 {
    if add {
        let quotient = rec.mul_div(amount, U256::Q96, liquidity);
        let (sum, carry) = sqrt_p.overflowing_add(quotient);
        assert!(!carry, "SafeMath .add() overflow");
        assert!(sum.fits_uint160(), "toUint160()");
        sum
    } else {
        let quotient = rec.mul_div_rounding_up(amount, U256::Q96, liquidity);
        assert!(sqrt_p > quotient, "require(sqrtP > quotient)");
        let diff = sqrt_p.wrapping_sub(quotient);
        assert!(diff.fits_uint160(), "uint160(...)");
        diff
    }
}

fn ref_next_from_input(rec: &mut MdRecorder, sqrt_p: U256, liquidity: U256, amount_in: U256, zero_for_one: bool) -> U256 {
    assert!(!sqrt_p.is_zero() && !liquidity.is_zero());
    if zero_for_one {
        ref_next_from_amount0(rec, sqrt_p, liquidity, amount_in, true)
    } else {
        ref_next_from_amount1(rec, sqrt_p, liquidity, amount_in, true)
    }
}

fn ref_next_from_output(rec: &mut MdRecorder, sqrt_p: U256, liquidity: U256, amount_out: U256, zero_for_one: bool) -> U256 {
    assert!(!sqrt_p.is_zero() && !liquidity.is_zero());
    if zero_for_one {
        ref_next_from_amount1(rec, sqrt_p, liquidity, amount_out, false)
    } else {
        ref_next_from_amount0(rec, sqrt_p, liquidity, amount_out, false)
    }
}

/// v3 LiquidityMath.addDelta (faithful). `y_sign` 0 = add, 1 = sub; `y_mag` is
/// |y| (uint128). Returns the uint128 result.
pub fn ref_add_delta(x: U256, y_sign: bool, y_mag: U256) -> U256 {
    assert!(x.fits_uint128() && y_mag.fits_uint128());
    if y_sign {
        // sub: require x >= y_mag.
        assert!(x >= y_mag, "LS underflow");
        x.wrapping_sub(y_mag)
    } else {
        // add: require x + y_mag fits uint128.
        let (z, _) = x.overflowing_add(y_mag);
        assert!(z.fits_uint128(), "LA overflow");
        z
    }
}

// ===========================================================================
// Phase 3: TickMath — faithful v3-core port.
// ===========================================================================
//
// The 20 magic Q128.128 multipliers, taken VERBATIM from v3-core TickMath.sol
// as little-endian [u64; 4] words (each fits a uint128, so the high two words
// are zero). MAGIC[i] gates bit i of |tick|. These MUST equal the guest's MAGIC
// (tick_math.py); the boundary validators in tests catch any divergence.
const MAGIC: [[u64; 4]; 20] = [
    [0xaa2d162d1a594001, 0xfffcb933bd6fad37, 0x0000000000000000, 0x0000000000000000],
    [0x59a46990580e213a, 0xfff97272373d4132, 0x0000000000000000, 0x0000000000000000],
    [0xef12357cf3c7fdcc, 0xfff2e50f5f656932, 0x0000000000000000, 0x0000000000000000],
    [0x1c3624eaa0941cd0, 0xffe5caca7e10e4e6, 0x0000000000000000, 0x0000000000000000],
    [0xc9db58835c926644, 0xffcb9843d60f6159, 0x0000000000000000, 0x0000000000000000],
    [0x472e6896dfb254c0, 0xff973b41fa98c081, 0x0000000000000000, 0x0000000000000000],
    [0x43ec78b326b52861, 0xff2ea16466c96a38, 0x0000000000000000, 0x0000000000000000],
    [0x11c461f1969c3053, 0xfe5dee046a99a2a8, 0x0000000000000000, 0x0000000000000000],
    [0xdcffc83b479aa3a4, 0xfcbe86c7900a88ae, 0x0000000000000000, 0x0000000000000000],
    [0x6f2b074cf7815e54, 0xf987a7253ac41317, 0x0000000000000000, 0x0000000000000000],
    [0x940c7a398e4b70f3, 0xf3392b0822b70005, 0x0000000000000000, 0x0000000000000000],
    [0x43b29c7fa6e889d9, 0xe7159475a2c29b74, 0x0000000000000000, 0x0000000000000000],
    [0x845ad8f792aa5825, 0xd097f3bdfd2022b8, 0x0000000000000000, 0x0000000000000000],
    [0x8a65dc1f90e061e5, 0xa9f746462d870fdf, 0x0000000000000000, 0x0000000000000000],
    [0x90bb3df62baf32f7, 0x70d869a156d2a1b8, 0x0000000000000000, 0x0000000000000000],
    [0x81231505542fcfa6, 0x31be135f97d08fd9, 0x0000000000000000, 0x0000000000000000],
    [0xc677de54f3e99bc9, 0x09aa508b5b7a84e1, 0x0000000000000000, 0x0000000000000000],
    [0x6699c329225ee604, 0x005d6af8dedb8119, 0x0000000000000000, 0x0000000000000000],
    [0x1ea926041bedfe98, 0x00002216e584f5fa, 0x0000000000000000, 0x0000000000000000],
    [0x91f7dc42444e8fa2, 0x00000000048a1703, 0x0000000000000000, 0x0000000000000000],
];

/// v3-core MAX_TICK.
pub const MAX_TICK: i32 = 887272;
/// v3-core MIN_TICK = -MAX_TICK.
pub const MIN_TICK: i32 = -887272;

/// MIN_SQRT_RATIO = getSqrtRatioAtTick(MIN_TICK).
pub fn min_sqrt_ratio() -> U256 {
    U256::from_u64(4_295_128_739)
}

/// MAX_SQRT_RATIO = getSqrtRatioAtTick(MAX_TICK).
pub fn max_sqrt_ratio() -> U256 {
    // 1461446703485210103287273052203988822378723970342 (160-bit)
    U256::from_le_bytes(&{
        let mut z = [0u8; 32];
        let hexbe: [u8; 20] = [
            0xFF, 0xFD, 0x89, 0x63, 0xEF, 0xD1, 0xFC, 0x6A, 0x50, 0x64, 0x88, 0x49, 0x5D, 0x95, 0x1D, 0x52, 0x63,
            0x98, 0x8D, 0x26,
        ];
        for (i, b) in hexbe.iter().rev().enumerate() {
            z[i] = *b;
        }
        z
    })
}

/// v3-core TickMath.getSqrtRatioAtTick (EXACT). `tick` is the signed tick.
/// Returns the uint160 sqrtPriceX96.
pub fn ref_get_sqrt_ratio_at_tick(tick: i32) -> U256 {
    assert!((MIN_TICK..=MAX_TICK).contains(&tick), "tick out of range");
    let abs_tick = tick.unsigned_abs();
    let magic = |i: usize| U256 {
        words: MAGIC[i],
    };
    let two_pow_128 = U256::ONE.shl(128);

    let mut ratio = if abs_tick & 0x1 != 0 { magic(0) } else { two_pow_128 };
    for i in 1..20usize {
        if abs_tick & (1u32 << i) != 0 {
            // (ratio * magic) >> 128, exact via the full 512-bit product.
            let prod = ratio.full_mul(magic(i));
            ratio = prod.shr_to_u256(128);
        }
    }
    if tick > 0 {
        // type(uint256).max / ratio (floor).
        let (q, _) = U256::MAX.to_u512().divmod_u256(ratio);
        ratio = U256 {
            words: [q.words[0], q.words[1], q.words[2], q.words[3]],
        };
    }
    // (ratio >> 32) + (ratio % 2^32 != 0 ? 1 : 0).
    let low32_nonzero = (ratio.words[0] & 0xFFFF_FFFF) != 0;
    let shifted = ratio.shr(32);
    let res = if low32_nonzero { shifted.overflowing_add(U256::ONE).0 } else { shifted };
    assert!(res.fits_uint160(), "sqrtPriceX96 must fit uint160");
    res
}

/// v3-core TickMath.getTickAtSqrtRatio (faithful tick discovery). Returns the
/// unique signed tick `t` with sqrtRatio(t) <= sqrtP < sqrtRatio(t+1).
pub fn ref_get_tick_at_sqrt_ratio(sqrt_p: U256) -> i32 {
    assert!(sqrt_p >= min_sqrt_ratio() && sqrt_p < max_sqrt_ratio(), "sqrtP out of band");
    // Binary search over [MIN_TICK, MAX_TICK-1] for the bucket. Monotone, so the
    // largest t with sqrtRatio(t) <= sqrtP is the answer.
    let mut lo = MIN_TICK;
    let mut hi = MAX_TICK - 1;
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if ref_get_sqrt_ratio_at_tick(mid) <= sqrt_p {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Split a signed tick into the guest's (sign, abs) encoding.
fn tick_to_sign_abs(tick: i32) -> (bool, u32) {
    (tick < 0, tick.unsigned_abs())
}

/// getSqrtRatioAtTick, recording the SAME muldiv pair the guest emits: exactly
/// one pair iff `tick > 0` (the `type(uint256).max / ratio` reciprocal, which the
/// guest computes as `div_floor_u256(umax, ratio)` = `mulDiv(umax, 1, ratio)`).
fn rec_sqrt_ratio_at_tick(rec: &mut MdRecorder, tick: i32) -> U256 {
    assert!((MIN_TICK..=MAX_TICK).contains(&tick), "tick out of range");
    let abs_tick = tick.unsigned_abs();
    let magic = |i: usize| U256 { words: MAGIC[i] };
    let two_pow_128 = U256::ONE.shl(128);
    let mut ratio = if abs_tick & 0x1 != 0 { magic(0) } else { two_pow_128 };
    for i in 1..20usize {
        if abs_tick & (1u32 << i) != 0 {
            ratio = ratio.full_mul(magic(i)).shr_to_u256(128);
        }
    }
    if tick > 0 {
        // guest: div_floor_u256(umax, ratio) = mulDiv(umax, 1, ratio) -> ONE pair.
        ratio = rec.mul_div(U256::MAX, U256::ONE, ratio);
    }
    let low32_nonzero = (ratio.words[0] & 0xFFFF_FFFF) != 0;
    let shifted = ratio.shr(32);
    let res = if low32_nonzero { shifted.overflowing_add(U256::ONE).0 } else { shifted };
    assert!(res.fits_uint160(), "sqrtPriceX96 must fit uint160");
    res
}

// ===========================================================================
// Phase 4: BitMath + TickBitmap — faithful v3-core port.
// ===========================================================================

/// v3-core BitMath.mostSignificantBit. `x` must be nonzero. Returns r in
/// [0,255] with `2^r <= x < 2^(r+1)` (the index of the highest set bit).
pub fn ref_most_significant_bit(x: U256) -> u32 {
    assert!(!x.is_zero(), "BitMath.mostSignificantBit: x == 0");
    // Highest nonzero word, then its highest set bit.
    for w in (0..4).rev() {
        if x.words[w] != 0 {
            return (w as u32) * 64 + (63 - x.words[w].leading_zeros());
        }
    }
    unreachable!("x != 0")
}

/// v3-core BitMath.leastSignificantBit. `x` must be nonzero. Returns r in
/// [0,255]: bit r set and all lower bits zero (the index of the lowest set bit).
pub fn ref_least_significant_bit(x: U256) -> u32 {
    assert!(!x.is_zero(), "BitMath.leastSignificantBit: x == 0");
    for w in 0..4 {
        if x.words[w] != 0 {
            return (w as u32) * 64 + x.words[w].trailing_zeros();
        }
    }
    unreachable!("x != 0")
}

/// `2^k` as a `U256`, for k in [0,256). (k == 256 would be 2^256, not a U256.)
fn two_pow(k: u32) -> U256 {
    assert!(k < 256, "two_pow: k >= 256");
    U256::ONE.shl(k)
}

/// Keep the low `k` bits of `x` (`x mod 2^k`), for k in [0,256].
fn keep_low(x: U256, k: u32) -> U256 {
    if k >= 256 {
        return x;
    }
    if k == 0 {
        return U256::ZERO;
    }
    // mask = 2^k - 1.
    let mask = two_pow(k).wrapping_sub(U256::ONE);
    U256 {
        words: [
            x.words[0] & mask.words[0],
            x.words[1] & mask.words[1],
            x.words[2] & mask.words[2],
            x.words[3] & mask.words[3],
        ],
    }
}

/// Signed floor division `floor(tick / tick_spacing)` (= v3's `compressed`).
/// v3: `compressed = tick / tickSpacing; if (tick<0 && tick%tickSpacing != 0) compressed--`.
pub fn ref_compressed(tick: i32, tick_spacing: u32) -> i32 {
    assert!(tick_spacing >= 1, "tick_spacing must be >= 1");
    let ts = tick_spacing as i32;
    let mut compressed = tick / ts; // truncates toward zero
    if tick < 0 && tick % ts != 0 {
        compressed -= 1;
    }
    compressed
}

/// `position(t)`: wordPos = t >> 8 (arithmetic floor shift), bitPos = t mod 256
/// (euclidean, always in [0,255]). Returns `(wordPos, bitPos)`.
pub fn ref_position(t: i32) -> (i32, u32) {
    let word_pos = t >> 8; // arithmetic shift = floor(t / 256)
    let bit_pos = t.rem_euclid(256) as u32; // nonnegative remainder, [0,255]
    (word_pos, bit_pos)
}

/// One faithful evaluation of `nextInitializedTickWithinOneWord`. `word` is the
/// bitmap at the relevant wordPos (the caller supplies it; the reference also
/// returns which wordPos was relevant so the witness can confirm it). Returns
/// `(next, initialized, bit_pos, masked, word_pos)`:
///   next        : signed next tick,
///   initialized : masked != 0,
///   bit_pos     : v3 bitPos (the masking position; ALSO the position the guest
///                 commits in the head),
///   masked      : the masked word (for the msb/lsb hint),
///   word_pos    : the relevant wordPos (lte: compressed; gt: compressed+1).
pub fn ref_next_init(tick: i32, tick_spacing: u32, lte: bool, word: U256) -> (i32, bool, u32, U256, i32) {
    let compressed = ref_compressed(tick, tick_spacing);
    let ts = tick_spacing as i32;
    if lte {
        let (word_pos, bit_pos) = ref_position(compressed);
        // mask = low (bitPos + 1) bits.
        let masked = keep_low(word, bit_pos + 1);
        let initialized = !masked.is_zero();
        let next = if initialized {
            let msb = ref_most_significant_bit(masked) as i32;
            (compressed - (bit_pos as i32 - msb)) * ts
        } else {
            (compressed - bit_pos as i32) * ts
        };
        (next, initialized, bit_pos, masked, word_pos)
    } else {
        let (word_pos, bit_pos) = ref_position(compressed + 1);
        // mask = bits >= bitPos  =>  masked = word with low bitPos bits cleared.
        let masked = word.wrapping_sub(keep_low(word, bit_pos));
        let initialized = !masked.is_zero();
        let next = if initialized {
            let lsb = ref_least_significant_bit(masked) as i32;
            (compressed + 1 + (lsb - bit_pos as i32)) * ts
        } else {
            (compressed + 1 + (255 - bit_pos as i32)) * ts
        };
        (next, initialized, bit_pos, masked, word_pos)
    }
}

// ===========================================================================
// Phase 5a: SwapMath.computeSwapStep — faithful v3-core port.
// ===========================================================================
//
// `ref_compute_swap_step` mirrors the GUEST's control flow (swap_math.py)
// site-by-site, calling the SAME ref_* recorder helpers in the SAME order so the
// muldiv (q, r) hint pairs the host queues match the guest's taken path EXACTLY.
// Every branch flag (zeroForOne, exactIn, reach, max) is derived from the inputs,
// identical to the guest's in-circuit derivation. See swap_math.py for the v3
// source and the per-call pair-count table.

/// The four outputs of one `computeSwapStep`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapStepResult {
    pub sqrt_ratio_next: U256,
    pub amount_in: U256,
    pub amount_out: U256,
    pub fee_amount: U256,
}

/// v3-core `FullMath.mulDiv(x, y, denom)` floor, recording one (q, r) pair (the
/// guest's `mulDiv`). Thin wrapper so the SwapMath port reads like the guest.
fn rec_mul_div(rec: &mut MdRecorder, x: U256, y: U256, denom: U256) -> U256 {
    rec.mul_div(x, y, denom)
}

/// v3-core SwapMath.computeSwapStep (EXACT), recorder-free wrapper. Builds a
/// fresh recorder so the muldiv pairs are discarded; for the witness path use
/// `compute_op` (which threads the shared recorder). This is the test-facing
/// oracle: `ref_compute_swap_step(...)` returns the four outputs directly.
pub fn ref_compute_swap_step(
    current: U256,
    target: U256,
    liquidity: U256,
    amount_sign: bool,
    magnitude: U256,
    fee_pips: u32,
) -> SwapStepResult {
    let mut rec = MdRecorder::new();
    rec_compute_swap_step(&mut rec, current, target, liquidity, amount_sign, magnitude, fee_pips)
}

/// The recorder-threaded implementation: appends the muldiv (q, r) pairs for each
/// FullMath / SqrtPriceMath call in the guest's exact branch order.
///
/// `current` / `target` are the uint160 sqrt prices; `liquidity` a uint128;
/// `amount_sign` the sign of `amountRemaining` (false = `>= 0` = exactIn, true =
/// `< 0` = exactOut); `magnitude` = `|amountRemaining|`; `fee_pips < 1_000_000`.
fn rec_compute_swap_step(
    rec: &mut MdRecorder,
    current: U256,
    target: U256,
    liquidity: U256,
    amount_sign: bool,
    magnitude: U256,
    fee_pips: u32,
) -> SwapStepResult {
    const FEE_DENOM: u32 = 1_000_000;
    assert!(fee_pips < FEE_DENOM, "feePips must be < 1e6");
    assert!(current.fits_uint160() && target.fits_uint160(), "sqrt prices must be uint160");
    assert!(liquidity.fits_uint128(), "liquidity must be uint128");

    let zero_for_one = current >= target;
    let exact_in = !amount_sign;

    let fee_denom_u = U256::from_u64(FEE_DENOM as u64);
    let denom_comp_u = U256::from_u64((FEE_DENOM - fee_pips) as u64);
    let fee_pips_u = U256::from_u64(fee_pips as u64);

    let sqrt_ratio_next: U256;
    let amount_in: U256;
    let amount_out: U256;
    let fee_amount: U256;

    if exact_in {
        // amountRemainingLessFee = mulDiv(magnitude, FEE_DENOM - feePips, FEE_DENOM)  [1 pair]
        let arlf = rec_mul_div(rec, magnitude, denom_comp_u, fee_denom_u);

        // stage-1 amountIn (roundUp).
        let amount_in_s1 = if zero_for_one {
            ref_get_amount0_delta(rec, target, current, liquidity, true) // 2 pairs
        } else {
            ref_get_amount1_delta(rec, current, target, liquidity, true) // 1 pair
        };

        // reach = arlf >= amountIn_s1.
        let reach = arlf >= amount_in_s1;

        // sqrtRatioNext = reach ? target : getNextSqrtPriceFromInput(...).
        let next = if reach {
            target // 0 pairs
        } else {
            ref_next_from_input(rec, current, liquidity, arlf, zero_for_one)
        };

        let max_reached = target == next;

        // stage-2 amountIn: exactIn => max ? amountIn_s1 : recompute at next.
        let ai_final = if max_reached {
            amount_in_s1 // 0 pairs
        } else if zero_for_one {
            ref_get_amount0_delta(rec, next, current, liquidity, true) // 2 pairs
        } else {
            ref_get_amount1_delta(rec, current, next, liquidity, true) // 1 pair
        };

        // stage-2 amountOut: exactIn => ALWAYS recompute.
        let ao_final = if zero_for_one {
            ref_get_amount1_delta(rec, next, current, liquidity, false) // 1 pair
        } else {
            ref_get_amount0_delta(rec, current, next, liquidity, false) // 2 pairs
        };

        // fee: next != target ? magnitude - amountIn : mulDivRoundingUp(amountIn, feePips, FEE_DENOM-feePips).
        let fee_final = if !max_reached {
            // v3 invariant: amountIn <= magnitude on this path (else v3 reverts).
            assert!(magnitude >= ai_final, "computeSwapStep: fee subtraction would underflow");
            magnitude.wrapping_sub(ai_final) // 0 pairs
        } else {
            rec.mul_div_rounding_up(ai_final, fee_pips_u, denom_comp_u) // 1 pair
        };

        sqrt_ratio_next = next;
        amount_in = ai_final;
        amount_out = ao_final;
        fee_amount = fee_final;
    } else {
        // exactOut.
        // stage-1 amountOut (roundDown).
        let amount_out_s1 = if zero_for_one {
            ref_get_amount1_delta(rec, target, current, liquidity, false) // 1 pair
        } else {
            ref_get_amount0_delta(rec, current, target, liquidity, false) // 2 pairs
        };

        // reach = magnitude >= amountOut_s1.
        let reach = magnitude >= amount_out_s1;

        // sqrtRatioNext = reach ? target : getNextSqrtPriceFromOutput(...).
        let next = if reach {
            target // 0 pairs
        } else {
            ref_next_from_output(rec, current, liquidity, magnitude, zero_for_one)
        };

        let max_reached = target == next;

        // stage-2 amountIn: exactOut => ALWAYS recompute.
        let ai_final = if zero_for_one {
            ref_get_amount0_delta(rec, next, current, liquidity, true) // 2 pairs
        } else {
            ref_get_amount1_delta(rec, current, next, liquidity, true) // 1 pair
        };

        // stage-2 amountOut: exactOut => max ? amountOut_s1 : recompute at next.
        let ao_recomp = if max_reached {
            amount_out_s1 // 0 pairs
        } else if zero_for_one {
            ref_get_amount1_delta(rec, next, current, liquidity, false) // 1 pair
        } else {
            ref_get_amount0_delta(rec, current, next, liquidity, false) // 2 pairs
        };

        // cap output: if amountOut > magnitude { amountOut = magnitude }.
        let ao_final = if ao_recomp > magnitude { magnitude } else { ao_recomp };

        // fee: exactIn false => mulDivRoundingUp(amountIn, feePips, FEE_DENOM-feePips).  [1 pair]
        let fee_final = rec.mul_div_rounding_up(ai_final, fee_pips_u, denom_comp_u);

        sqrt_ratio_next = next;
        amount_in = ai_final;
        amount_out = ao_final;
        fee_amount = fee_final;
    }

    SwapStepResult {
        sqrt_ratio_next,
        amount_in,
        amount_out,
        fee_amount,
    }
}

/// The Phase-2/3/4 operation selector. Mirrors the OP table in `main.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqrtOp {
    Amount0Delta,  // 0
    Amount1Delta,  // 1
    NextFromAmount0RoundingUp, // 2
    NextFromAmount1RoundingDown, // 3
    NextFromInput, // 4
    NextFromOutput, // 5
    AddDelta,      // 6
    RawMulDiv,     // 7 — direct FullMath on arbitrary uint256 (Phase-1 surface)
    GetSqrtRatioAtTick, // 8
    GetTickAtSqrtRatio, // 9
    MostSignificantBit, // 10
    LeastSignificantBit, // 11
    NextInitializedTickWithinOneWord, // 12
    ComputeSwapStep, // 13
}

impl Default for SqrtOp {
    /// Neutral default for `..Default::default()` spreads in tests that don't
    /// exercise a specific op (they always set `op` explicitly).
    fn default() -> Self {
        SqrtOp::Amount0Delta
    }
}

impl SqrtOp {
    pub fn code(self) -> u32 {
        match self {
            SqrtOp::Amount0Delta => 0,
            SqrtOp::Amount1Delta => 1,
            SqrtOp::NextFromAmount0RoundingUp => 2,
            SqrtOp::NextFromAmount1RoundingDown => 3,
            SqrtOp::NextFromInput => 4,
            SqrtOp::NextFromOutput => 5,
            SqrtOp::AddDelta => 6,
            SqrtOp::RawMulDiv => 7,
            SqrtOp::GetSqrtRatioAtTick => 8,
            SqrtOp::GetTickAtSqrtRatio => 9,
            SqrtOp::MostSignificantBit => 10,
            SqrtOp::LeastSignificantBit => 11,
            SqrtOp::NextInitializedTickWithinOneWord => 12,
            SqrtOp::ComputeSwapStep => 13,
        }
    }
}

/// All inputs to one driver invocation, matching `main.py`'s input block. The
/// `tick_*` / `*_p1` fields are only meaningful for ops 8/9 (TickMath); other
/// ops set them to neutral values (committed but unused).
#[derive(Clone, Copy, Debug)]
pub struct SqrtInputs {
    pub op: SqrtOp,
    pub flag_a: bool,
    pub sqrt_a: U256,
    pub sqrt_b: U256,
    pub liquidity: U256, // uint128 value, uint256-encoded
    pub amount: U256,
    pub y_sign: bool,
    /// TickMath: sign bit of the tick (0 = tick>=0, 1 = tick<0).
    pub tick_sign: bool,
    /// TickMath: |tick| in [0, MAX_TICK].
    pub abs_tick: u32,
    /// TickMath (op 9): sign bit of tick+1.
    pub sign_p1: bool,
    /// TickMath (op 9): |tick+1| in [0, MAX_TICK].
    pub abs_p1: u32,
    /// TickBitmap (op 12): tick spacing (positive int, < 2^16).
    pub tick_spacing: u32,
    /// BitMath (ops 10/11) / TickBitmap (op 12): the hinted bit position. For
    /// MSB/LSB it is the answer; for NextInit it is msb/lsb of `masked` (or 0
    /// when uninitialized). Committed in the head.
    pub bit_pos: u32,
    /// TickBitmap (op 12): the 256-bit bitmap word at the relevant wordPos. A
    /// COMMITTED input.
    pub word: U256,
    /// SwapMath (op 13): the fee in pips (uint24, < 1_000_000). Committed in the
    /// head (head[11]).
    pub fee_pips: u32,
    /// SwapMath (op 13): sign of amountRemaining (false = >= 0 = exactIn, true =
    /// < 0 = exactOut). `amount` holds |amountRemaining|. Committed in head[12].
    pub amount_sign: bool,
}

impl Default for SqrtInputs {
    /// Neutral defaults so non-TickMath tests can `..Default::default()` the
    /// tick fields (committed but unused for ops 0..7). `abs_p1 = 1` keeps the
    /// (tick, tick+1) encoding self-consistent if ever inspected. `tick_spacing
    /// = 1` keeps the signed-floor-div self-consistent if ever inspected.
    fn default() -> Self {
        SqrtInputs {
            op: SqrtOp::default(),
            flag_a: false,
            sqrt_a: U256::ZERO,
            sqrt_b: U256::ZERO,
            liquidity: U256::ZERO,
            amount: U256::ZERO,
            y_sign: false,
            tick_sign: false,
            abs_tick: 0,
            sign_p1: false,
            abs_p1: 1,
            tick_spacing: 1,
            bit_pos: 0,
            word: U256::ZERO,
            fee_pips: 0,
            amount_sign: false,
        }
    }
}

/// The full output of one op evaluation: the uint256-encoded `result` (the slot
/// committed for ops 0..9 and op 13's sqrtRatioNext), the muldiv hint `pairs` the
/// guest consumes, the Phase-4 head scalars `next_val` / `initialized`, and the
/// Phase-5a swap-output slots (op 13's amountIn / amountOut / feeAmount; zero for
/// all other ops).
struct OpOutput {
    result: U256,
    next_val: F,
    initialized: u32,
    swap_in: U256,
    swap_out: U256,
    fee: U256,
    pairs: Vec<HintBuf>,
}

/// Encode a signed `i32` as a field element (matching the guest's signed field
/// arithmetic: a negative value is `p - |v|`).
fn signed_to_f(v: i32) -> F {
    if v >= 0 {
        F::from_u32(v as u32)
    } else {
        -F::from_u32((-(v as i64)) as u32)
    }
}

/// Compute the op result, the ordered muldiv hint pairs, and the Phase-4 head
/// scalars (next_val, initialized).
fn compute_op(inp: &SqrtInputs) -> OpOutput {
    let mut rec = MdRecorder::new();
    let mut next_val = F::ZERO;
    let mut initialized = 0u32;
    let mut swap_in = U256::ZERO;
    let mut swap_out = U256::ZERO;
    let mut fee = U256::ZERO;
    let result = match inp.op {
        SqrtOp::Amount0Delta => ref_get_amount0_delta(&mut rec, inp.sqrt_a, inp.sqrt_b, inp.liquidity, inp.flag_a),
        SqrtOp::Amount1Delta => ref_get_amount1_delta(&mut rec, inp.sqrt_a, inp.sqrt_b, inp.liquidity, inp.flag_a),
        SqrtOp::NextFromAmount0RoundingUp => {
            ref_next_from_amount0(&mut rec, inp.sqrt_a, inp.liquidity, inp.amount, inp.flag_a)
        }
        SqrtOp::NextFromAmount1RoundingDown => {
            ref_next_from_amount1(&mut rec, inp.sqrt_a, inp.liquidity, inp.amount, inp.flag_a)
        }
        SqrtOp::NextFromInput => ref_next_from_input(&mut rec, inp.sqrt_a, inp.liquidity, inp.amount, inp.flag_a),
        SqrtOp::NextFromOutput => ref_next_from_output(&mut rec, inp.sqrt_a, inp.liquidity, inp.amount, inp.flag_a),
        SqrtOp::AddDelta => {
            // x = low uint128 of liquidity, y_mag = low uint128 of amount.
            let x = U256 {
                words: [inp.liquidity.words[0], inp.liquidity.words[1], 0, 0],
            };
            let y = U256 {
                words: [inp.amount.words[0], inp.amount.words[1], 0, 0],
            };
            ref_add_delta(x, inp.y_sign, y)
        }
        SqrtOp::RawMulDiv => {
            // a = sqrtA, b = sqrtB, denom = amount. Floor committed; ceil consumes
            // a second pair (order: floor pair, then ceil pair) matching the guest.
            let q = rec.mul_div(inp.sqrt_a, inp.sqrt_b, inp.amount);
            let _ceil = rec.mul_div_rounding_up(inp.sqrt_a, inp.sqrt_b, inp.amount);
            q
        }
        SqrtOp::GetSqrtRatioAtTick => {
            // tick reconstructed from (tick_sign, abs_tick). Emits one muldiv pair
            // iff tick > 0. Result = uint160 sqrtPriceX96.
            let tick = signed_tick(inp.tick_sign, inp.abs_tick);
            rec_sqrt_ratio_at_tick(&mut rec, tick)
        }
        SqrtOp::GetTickAtSqrtRatio => {
            // Verify sqrtA's tick: the guest recomputes sqrtRatio(tick) and
            // sqrtRatio(tick+1), each emitting one pair iff its tick > 0 — IN THAT
            // ORDER. The committed result echoes the input sqrtPriceX96 (sqrtA).
            let tick = signed_tick(inp.tick_sign, inp.abs_tick);
            let tick_p1 = signed_tick(inp.sign_p1, inp.abs_p1);
            let _r0 = rec_sqrt_ratio_at_tick(&mut rec, tick);
            let _r1 = rec_sqrt_ratio_at_tick(&mut rec, tick_p1);
            inp.sqrt_a
        }
        SqrtOp::MostSignificantBit | SqrtOp::LeastSignificantBit => {
            // x = sqrtA; the answer is committed via the head `bit_pos`. No muldiv
            // pairs (BitMath uses no FullMath). result slot = 0.
            U256::ZERO
        }
        SqrtOp::NextInitializedTickWithinOneWord => {
            // next_initialized_tick_within_one_word(tick, tick_spacing, lte=flag_a,
            // word). Commits (next_val, initialized) in the head; no muldiv pairs.
            let tick = signed_tick(inp.tick_sign, inp.abs_tick);
            let (next, init, _bit_pos, _masked, _word_pos) =
                ref_next_init(tick, inp.tick_spacing, inp.flag_a, inp.word);
            next_val = signed_to_f(next);
            initialized = init as u32;
            U256::ZERO
        }
        SqrtOp::ComputeSwapStep => {
            // current = sqrtA, target = sqrtB, magnitude = amount; (amount_sign,
            // fee_pips) scalars. result = sqrtRatioNext; swap_in/out/fee = the
            // other three outputs. Pairs recorded by ref_compute_swap_step in the
            // guest's exact branch order.
            let r = rec_compute_swap_step(
                &mut rec,
                inp.sqrt_a,
                inp.sqrt_b,
                inp.liquidity,
                inp.amount_sign,
                inp.amount,
                inp.fee_pips,
            );
            swap_in = r.amount_in;
            swap_out = r.amount_out;
            fee = r.fee_amount;
            r.sqrt_ratio_next
        }
    };
    OpOutput {
        result,
        next_val,
        initialized,
        swap_in,
        swap_out,
        fee,
        pairs: rec.bufs,
    }
}

/// Reconstruct a signed tick from the guest's (sign, abs) encoding.
fn signed_tick(sign: bool, abs: u32) -> i32 {
    if sign { -(abs as i32) } else { abs as i32 }
}

/// The public-input-committed result of a Phase-2 invocation (uint256-encoded).
pub fn sqrt_op_result(inp: &SqrtInputs) -> U256 {
    compute_op(inp).result
}

/// The Phase-4 head scalars committed alongside the result: `(next_val,
/// initialized)` (next_val as the signed-tick `F` encoding). Zero for ops 0..11.
pub fn sqrt_op_head_scalars(inp: &SqrtInputs) -> (F, u32) {
    let out = compute_op(inp);
    (out.next_val, out.initialized)
}

/// The Phase-5a swap-output commit slots `(amountIn, amountOut, feeAmount)`
/// committed alongside `result` (= sqrtRatioNext). All zero for ops 0..12.
pub fn sqrt_op_swap_slots(inp: &SqrtInputs) -> (U256, U256, U256) {
    let out = compute_op(inp);
    (out.swap_in, out.swap_out, out.fee)
}

/// The full SwapMath result for op 13 (sqrtRatioNext / amountIn / amountOut /
/// feeAmount), as the differential oracle for tests.
pub fn swap_step_result(inp: &SqrtInputs) -> SwapStepResult {
    let out = compute_op(inp);
    SwapStepResult {
        sqrt_ratio_next: out.result,
        amount_in: out.swap_in,
        amount_out: out.swap_out,
        fee_amount: out.fee,
    }
}

/// Emit the full ordered hint sequence for the driver `main`: the input block
/// (op, flag_a, sqrtA, sqrtB, liquidity, amount, y_sign, tick scalars,
/// tick_spacing, bit_pos, word) followed by the muldiv (q, r) pairs in the order
/// the dispatched op calls them. Order MUST match main.py's hint reads.
pub fn emit_sqrt_hints(inp: &SqrtInputs) -> Vec<HintBuf> {
    let mut bufs = vec![
        HintBuf {
            label: "mode",
            cells: vec![F::ZERO], // 0 = op-dispatch mode (the unit-test driver)
        },
        HintBuf {
            label: "op",
            cells: vec![F::from_u32(inp.op.code())],
        },
        HintBuf {
            label: "flag_a",
            cells: vec![F::from_u32(inp.flag_a as u32)],
        },
        HintBuf {
            label: "sqrtA",
            cells: inp.sqrt_a.to_le_base256_limbs(UINT256_LIMBS),
        },
        HintBuf {
            label: "sqrtB",
            cells: inp.sqrt_b.to_le_base256_limbs(UINT256_LIMBS),
        },
        HintBuf {
            label: "liquidity",
            cells: inp.liquidity.to_le_base256_limbs(UINT256_LIMBS),
        },
        HintBuf {
            label: "amount",
            cells: inp.amount.to_le_base256_limbs(UINT256_LIMBS),
        },
        HintBuf {
            label: "y_sign",
            cells: vec![F::from_u32(inp.y_sign as u32)],
        },
        HintBuf {
            label: "tick_sign",
            cells: vec![F::from_u32(inp.tick_sign as u32)],
        },
        HintBuf {
            label: "abs_tick",
            cells: vec![F::from_u32(inp.abs_tick)],
        },
        HintBuf {
            label: "sign_p1",
            cells: vec![F::from_u32(inp.sign_p1 as u32)],
        },
        HintBuf {
            label: "abs_p1",
            cells: vec![F::from_u32(inp.abs_p1)],
        },
        HintBuf {
            label: "tick_spacing",
            cells: vec![F::from_u32(inp.tick_spacing)],
        },
        HintBuf {
            label: "bit_pos",
            cells: vec![F::from_u32(inp.bit_pos)],
        },
        HintBuf {
            label: "word",
            cells: inp.word.to_le_base256_limbs(UINT256_LIMBS),
        },
        HintBuf {
            label: "fee_pips",
            cells: vec![F::from_u32(inp.fee_pips)],
        },
        HintBuf {
            label: "amount_sign",
            cells: vec![F::from_u32(inp.amount_sign as u32)],
        },
    ];
    let pairs = compute_op(inp).pairs;
    bufs.extend(pairs);
    bufs
}

/// Build `SqrtInputs` for a SwapMath computeSwapStep op (op 13). `current` /
/// `target` are uint160 sqrt prices; `liquidity` a uint128; `amount_remaining`
/// the signed `int256` (sign + magnitude); `fee_pips < 1_000_000`.
pub fn compute_swap_step_inputs(
    current: U256,
    target: U256,
    liquidity: U256,
    amount_sign: bool,
    magnitude: U256,
    fee_pips: u32,
) -> SqrtInputs {
    SqrtInputs {
        op: SqrtOp::ComputeSwapStep,
        sqrt_a: current,
        sqrt_b: target,
        liquidity,
        amount: magnitude,
        amount_sign,
        fee_pips,
        ..SqrtInputs::default()
    }
}

/// Build `SqrtInputs` for a TickMath getSqrtRatioAtTick op (op 8).
pub fn sqrt_ratio_at_tick_inputs(tick: i32) -> SqrtInputs {
    let (tick_sign, abs_tick) = tick_to_sign_abs(tick);
    SqrtInputs {
        op: SqrtOp::GetSqrtRatioAtTick,
        tick_sign,
        abs_tick,
        ..SqrtInputs::default()
    }
}

/// Build `SqrtInputs` for a TickMath getTickAtSqrtRatio op (op 9). `tick` is the
/// honest answer for `sqrt_p`; tick+1 is derived. `sqrt_p` is the uint160 input.
pub fn tick_at_sqrt_ratio_inputs(sqrt_p: U256, tick: i32) -> SqrtInputs {
    let (tick_sign, abs_tick) = tick_to_sign_abs(tick);
    let (sign_p1, abs_p1) = tick_to_sign_abs(tick + 1);
    SqrtInputs {
        op: SqrtOp::GetTickAtSqrtRatio,
        sqrt_a: sqrt_p,
        tick_sign,
        abs_tick,
        sign_p1,
        abs_p1,
        ..SqrtInputs::default()
    }
}

/// Build `SqrtInputs` for a BitMath mostSignificantBit op (op 10). `x` must be
/// nonzero; `bit_pos` is set to the true MSB (the host hint the guest verifies).
pub fn most_significant_bit_inputs(x: U256) -> SqrtInputs {
    SqrtInputs {
        op: SqrtOp::MostSignificantBit,
        sqrt_a: x,
        bit_pos: ref_most_significant_bit(x),
        ..SqrtInputs::default()
    }
}

/// Build `SqrtInputs` for a BitMath leastSignificantBit op (op 11). `x` must be
/// nonzero; `bit_pos` is set to the true LSB.
pub fn least_significant_bit_inputs(x: U256) -> SqrtInputs {
    SqrtInputs {
        op: SqrtOp::LeastSignificantBit,
        sqrt_a: x,
        bit_pos: ref_least_significant_bit(x),
        ..SqrtInputs::default()
    }
}

/// Build `SqrtInputs` for a TickBitmap nextInitializedTickWithinOneWord op (op
/// 12). `tick`, `tick_spacing`, `lte`, and the bitmap `word` are inputs; the
/// `bit_pos` hint is set to msb/lsb of the masked word (or 0 when uninitialized).
pub fn next_init_inputs(tick: i32, tick_spacing: u32, lte: bool, word: U256) -> SqrtInputs {
    let (tick_sign, abs_tick) = tick_to_sign_abs(tick);
    let (_next, initialized, _bit_pos, masked, _word_pos) = ref_next_init(tick, tick_spacing, lte, word);
    // The guest hints msb(masked) (lte) or lsb(masked) (gt) and binds it inside
    // BitMath, but ONLY on the initialized branch. When uninitialized, the hint
    // is unused (the BitMath call is not reached) — emit 0.
    let bit_pos_hint = if initialized {
        if lte {
            ref_most_significant_bit(masked)
        } else {
            ref_least_significant_bit(masked)
        }
    } else {
        0
    };
    SqrtInputs {
        op: SqrtOp::NextInitializedTickWithinOneWord,
        flag_a: lte,
        tick_sign,
        abs_tick,
        tick_spacing,
        bit_pos: bit_pos_hint,
        word,
        ..SqrtInputs::default()
    }
}

// ===========================================================================
// Phase 5b: Pool.swap multi-tick loop — faithful v3-core port + ordered hint
// emission across ALL steps (one threaded MdRecorder), plus a pool fixture.
// ===========================================================================
//
// PER-STEP HINT CONSUMPTION ORDER (must match swap.py swap_step EXACTLY):
//   input block (one buffer each, FIFO per label):
//     "word"(32), "bit_pos"(1), "tick_next_sign"(1), "tick_next_abs"(1),
//     "tnp1_sign"(1), "tnp1_abs"(1), "liq_net_sign"(1), "liq_net_mag"(32),
//   then the step's muldiv (q, r) pairs IN THIS CALL ORDER:
//     1. get_sqrt_ratio_at_tick(clamped tickNext)   -> 1 pair iff tickNext > 0
//     2. compute_swap_step(...)                      -> data-dependent
//     3. (only if NOT crossed AND moved) get_tick_at_sqrt_ratio(new_sp) ->
//        sqrtRatio(tick) then sqrtRatio(tick+1), each 1 pair iff its tick > 0.
//
// The SWAP-MODE context hint block (read by main.run_swap, BEFORE the per-step
// blocks): "mode"(1, ==1), "swap_zero_for_one"(1), "swap_amount_sign"(1),
// "swap_fee_pips"(1), "swap_tick_spacing"(1), "swap_num_steps"(1),
// "swap_tick0_sign"(1), "swap_tick0_abs"(1), "swap_sp0"(32), "swap_liq0"(32),
// "swap_amount_mag"(32), "swap_limit"(32).

/// Signed int256 (sign + magnitude); the reference mirror of the guest's
/// amountSpecifiedRemaining / amountCalculated / amount0 / amount1 encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I256 {
    /// false = nonnegative, true = negative. Canonical: zero magnitude -> false.
    pub sign: bool,
    pub mag: U256,
}

impl I256 {
    pub const ZERO: I256 = I256 {
        sign: false,
        mag: U256::ZERO,
    };

    pub fn nonneg(mag: U256) -> I256 {
        I256 { sign: false, mag }
    }

    /// `self + other` over the integers, canonical result.
    fn add(self, other: I256) -> I256 {
        if self.sign == other.sign {
            let (s, c) = self.mag.overflowing_add(other.mag);
            assert!(!c, "I256::add magnitude overflow");
            I256 {
                sign: self.sign && !s.is_zero(),
                mag: s,
            }
        } else if self.mag >= other.mag {
            let m = self.mag.wrapping_sub(other.mag);
            I256 {
                sign: self.sign && !m.is_zero(),
                mag: m,
            }
        } else {
            let m = other.mag.wrapping_sub(self.mag);
            I256 {
                sign: other.sign && !m.is_zero(),
                mag: m,
            }
        }
    }

    fn neg(self) -> I256 {
        I256 {
            sign: !self.sign && !self.mag.is_zero(),
            mag: self.mag,
        }
    }

    /// Encode as (sign cell, 32 magnitude cells) for hint/commit emission.
    fn to_le_limbs(self) -> Vec<F> {
        self.mag.to_le_base256_limbs(UINT256_LIMBS)
    }
}

/// getTickAtSqrtRatio, recording the SAME pairs the guest emits: the two internal
/// getSqrtRatioAtTick(tick) / (tick+1) calls, in that order. Returns the tick.
fn rec_tick_at_sqrt_ratio(rec: &mut MdRecorder, sqrt_p: U256) -> i32 {
    let tick = ref_get_tick_at_sqrt_ratio(sqrt_p);
    let _r0 = rec_sqrt_ratio_at_tick(rec, tick);
    let _r1 = rec_sqrt_ratio_at_tick(rec, tick + 1);
    tick
}

/// All pool/swap inputs for one full `Pool.swap`.
#[derive(Clone, Debug)]
pub struct SwapInputs {
    pub zero_for_one: bool,
    /// amountSpecified (signed): false = exactIn, true = exactOut.
    pub amount_sign: bool,
    pub amount_mag: U256,
    pub sqrt_limit: U256,
    pub fee_pips: u32,
    pub tick_spacing: u32,
    pub sqrt_p0: U256,
    pub tick0: i32,
    pub liquidity0: U256,
    /// The pool fixture: a map tick -> (bitmap word at that tick's wordPos,
    /// liquidityNet). The reference reads the relevant word/liquidityNet per step.
    pub pool: PoolFixture,
}

/// A minimal pool fixture: the set of initialized ticks with their liquidityNet,
/// from which the per-step bitmap word and crossed-tick net are derived.
#[derive(Clone, Debug, Default)]
pub struct PoolFixture {
    /// initialized tick -> liquidityNet (int128, sign + magnitude).
    pub ticks: std::collections::BTreeMap<i32, (bool, U256)>,
}

impl PoolFixture {
    pub fn new() -> Self {
        PoolFixture {
            ticks: std::collections::BTreeMap::new(),
        }
    }

    /// Add an initialized tick with net liquidity `net` (signed int128).
    pub fn with_tick(mut self, tick: i32, net_sign: bool, net_mag: U256) -> Self {
        self.ticks.insert(tick, (net_sign, net_mag));
        self
    }

    /// The 256-bit bitmap word for the wordPos that `position(base)` lands in.
    /// `base` is `compressed` (lte) or `compressed+1` (gt). Bit `bitPos` is set iff
    /// the tick `(wordPos*256 + bitPos) * tickSpacing` is an initialized tick.
    fn word_for(&self, base: i32, tick_spacing: i32) -> U256 {
        let word_pos = base >> 8; // floor(base/256)
        let mut w = U256::ZERO;
        for (&t, _) in self.ticks.iter() {
            // t must be a multiple of tick_spacing to live in the compressed grid.
            if t % tick_spacing != 0 {
                continue;
            }
            let compressed = ref_compressed(t, tick_spacing as u32);
            let (wp, bp) = ref_position(compressed);
            if wp == word_pos {
                w = w.overflowing_add(U256::ONE.shl(bp)).0;
            }
        }
        w
    }

    /// liquidityNet at a crossed tick (0 if the tick is not initialized).
    fn net_at(&self, tick: i32) -> (bool, U256) {
        self.ticks.get(&tick).copied().unwrap_or((false, U256::ZERO))
    }
}

/// The full result of one `Pool.swap`, with the per-step ordered hints.
#[derive(Clone, Debug)]
pub struct SwapResult {
    pub sqrt_p_final: U256,
    pub tick_final: i32,
    pub liquidity_final: U256,
    pub amount_remaining: I256,
    pub amount_calculated: I256,
    pub amount0: I256,
    pub amount1: I256,
    pub num_steps: u32,
    /// Per-step prover data (committed), in step order.
    pub steps: Vec<StepData>,
    /// The muldiv (q, r) hint pairs across ALL steps, in guest consumption order.
    pub muldiv_pairs: Vec<HintBuf>,
}

/// The prover-controlled per-step data the guest folds into the commitment, plus
/// the hinted scalars the step consumes.
#[derive(Clone, Debug)]
pub struct StepData {
    pub word: U256,
    pub bit_pos: u32,
    pub tick_next_sign: bool, // getTickAtSqrtRatio answer (moved branch)
    pub tick_next_abs: u32,
    pub tnp1_sign: bool,
    pub tnp1_abs: u32,
    pub liq_net_sign: bool, // crossed-tick liquidityNet (committed)
    pub liq_net_mag: U256,
}

/// Faithful v3 Pool.swap core loop. Computes the result and emits EVERY hint
/// across ALL steps in exact guest consumption order, through ONE recorder.
pub fn ref_pool_swap(inp: &SwapInputs) -> SwapResult {
    const FEE_DENOM: u32 = 1_000_000;
    assert!(inp.fee_pips < FEE_DENOM, "fee_pips must be < 1e6");
    assert!(inp.tick_spacing >= 1, "tick_spacing >= 1");
    assert!(inp.sqrt_p0.fits_uint160() && inp.sqrt_limit.fits_uint160());
    assert!(inp.liquidity0.fits_uint128());

    let ts = inp.tick_spacing as i32;
    let exact_input = !inp.amount_sign;
    let zfo = inp.zero_for_one;

    let mut rec = MdRecorder::new();
    let mut steps: Vec<StepData> = Vec::new();

    let mut sqrt_p = inp.sqrt_p0;
    let mut tick = inp.tick0;
    let mut liquidity = inp.liquidity0;
    let mut remaining = I256 {
        sign: inp.amount_sign && !inp.amount_mag.is_zero(),
        mag: inp.amount_mag,
    };
    let mut calculated = I256::ZERO;

    let guard = |rem: &I256, sp: &U256| !rem.mag.is_zero() && *sp != inp.sqrt_limit;

    let mut num_steps = 0u32;
    while guard(&remaining, &sqrt_p) {
        let start_sp = sqrt_p;

        // --- nextInitializedTickWithinOneWord(tick, ts, lte = zfo, word) ---
        // The bitmap word is derived from the fixture at the relevant wordPos.
        let compressed = ref_compressed(tick, inp.tick_spacing);
        let base = if zfo { compressed } else { compressed + 1 };
        let word = inp.pool.word_for(base, ts);
        let (next_val, initialized, _bp, masked, _wp) = ref_next_init(tick, inp.tick_spacing, zfo, word);
        let bit_pos = if initialized {
            if zfo {
                ref_most_significant_bit(masked)
            } else {
                ref_least_significant_bit(masked)
            }
        } else {
            0
        };

        // --- clamp tickNext into [MIN_TICK, MAX_TICK] ---
        let tick_next = next_val.clamp(MIN_TICK, MAX_TICK);

        // --- sqrtPriceNext = getSqrtRatioAtTick(tickNext) (records a pair iff >0) ---
        let sqrt_next = rec_sqrt_ratio_at_tick(&mut rec, tick_next);

        // --- target = zfo ? max(sqrt_next, limit) : min(sqrt_next, limit) ---
        let target = if zfo {
            sqrt_next.max(inp.sqrt_limit)
        } else {
            sqrt_next.min(inp.sqrt_limit)
        };

        // --- computeSwapStep (records its data-dependent pairs) ---
        let step = rec_compute_swap_step(
            &mut rec,
            sqrt_p,
            target,
            liquidity,
            remaining.sign,
            remaining.mag,
            inp.fee_pips,
        );
        sqrt_p = step.sqrt_ratio_next;

        // --- signed amount accumulation ---
        if exact_input {
            let (in_fee, c) = step.amount_in.overflowing_add(step.fee_amount);
            assert!(!c, "amountIn+fee overflow");
            remaining = remaining.add(I256::nonneg(in_fee).neg());
            calculated = calculated.add(I256::nonneg(step.amount_out).neg());
        } else {
            remaining = remaining.add(I256::nonneg(step.amount_out));
            let (in_fee, c) = step.amount_in.overflowing_add(step.fee_amount);
            assert!(!c, "amountIn+fee overflow");
            calculated = calculated.add(I256::nonneg(in_fee));
        }

        // --- tick transition + per-step hint capture ---
        let crossed = sqrt_p == sqrt_next;
        let moved = sqrt_p != start_sp;

        // liquidityNet of the crossed tick (committed; the guest folds it every
        // step regardless, so capture it every step — 0 when not crossing).
        let (net_sign, net_mag) = if crossed && initialized {
            inp.pool.net_at(tick_next)
        } else {
            // The guest still consumes (and commits) a liq_net hint every step; on a
            // non-updating step it folds a canonical zero. We emit (false, 0) and the
            // guest's eff_net_sign zeroes it (do_update == 0).
            (false, U256::ZERO)
        };

        if crossed {
            if initialized {
                let mut net = I256 {
                    sign: net_sign && !net_mag.is_zero(),
                    mag: net_mag,
                };
                if zfo {
                    net = net.neg();
                }
                liquidity = ref_add_delta(liquidity, net.sign, net.mag);
            }
            tick = if zfo { tick_next - 1 } else { tick_next };
        } else if moved {
            tick = rec_tick_at_sqrt_ratio(&mut rec, sqrt_p);
        }
        // else: tick unchanged.

        // The (tick_next_*, tnp1_*) hints are the getTickAtSqrtRatio answer used on
        // the `moved` branch; emit the honest tick/tick+1 there, neutral otherwise.
        let (tn_sign, tn_abs, tnp1_sign, tnp1_abs) = if !crossed && moved {
            let tt = tick; // = getTickAtSqrtRatio(sqrt_p)
            let (s, a) = tick_to_sign_abs(tt);
            let (s1, a1) = tick_to_sign_abs(tt + 1);
            (s, a, s1, a1)
        } else {
            // unused; emit a self-consistent (0, 0, +1) pair so the guest's
            // tick+1 binding (sv_t + 1 == sv_p1) holds even though it is not used.
            (false, 0u32, false, 1u32)
        };

        steps.push(StepData {
            word,
            bit_pos,
            tick_next_sign: tn_sign,
            tick_next_abs: tn_abs,
            tnp1_sign,
            tnp1_abs,
            liq_net_sign: net_sign && !net_mag.is_zero(),
            liq_net_mag: net_mag,
        });

        num_steps += 1;
        assert!(num_steps <= 100_000, "swap did not terminate (fixture bug?)");
    }

    // --- amount0 / amount1 ---
    let spent = I256 {
        sign: inp.amount_sign && !inp.amount_mag.is_zero(),
        mag: inp.amount_mag,
    }
    .add(remaining.neg());
    let (amount0, amount1) = if zfo == exact_input {
        (spent, calculated)
    } else {
        (calculated, spent)
    };

    SwapResult {
        sqrt_p_final: sqrt_p,
        tick_final: tick,
        liquidity_final: liquidity,
        amount_remaining: remaining,
        amount_calculated: calculated,
        amount0,
        amount1,
        num_steps,
        steps,
        muldiv_pairs: rec.bufs,
    }
}

/// Emit the FULL ordered hint sequence for SWAP mode (`mode == 1`): the context
/// block, then per-step (input block + that step's muldiv pairs), in the order
/// the guest consumes them. The per-step muldiv pairs are interleaved with the
/// per-step input blocks, but since they use distinct labels, FIFO-per-label
/// ordering is preserved by appending each step's input scalars then its pairs.
pub fn emit_swap_hints(inp: &SwapInputs) -> Vec<HintBuf> {
    let res = ref_pool_swap(inp);
    let mut bufs: Vec<HintBuf> = vec![HintBuf {
        label: "mode",
        cells: vec![F::ONE],
    }];
    let push_scalar = |bufs: &mut Vec<HintBuf>, label: &'static str, v: u32| {
        bufs.push(HintBuf {
            label,
            cells: vec![F::from_u32(v)],
        });
    };
    push_scalar(&mut bufs, "swap_zero_for_one", inp.zero_for_one as u32);
    push_scalar(&mut bufs, "swap_amount_sign", inp.amount_sign as u32);
    push_scalar(&mut bufs, "swap_fee_pips", inp.fee_pips);
    push_scalar(&mut bufs, "swap_tick_spacing", inp.tick_spacing);
    push_scalar(&mut bufs, "swap_num_steps", res.num_steps);
    let (t0s, t0a) = tick_to_sign_abs(inp.tick0);
    push_scalar(&mut bufs, "swap_tick0_sign", t0s as u32);
    push_scalar(&mut bufs, "swap_tick0_abs", t0a);
    bufs.push(HintBuf {
        label: "swap_sp0",
        cells: inp.sqrt_p0.to_le_base256_limbs(UINT256_LIMBS),
    });
    bufs.push(HintBuf {
        label: "swap_liq0",
        cells: inp.liquidity0.to_le_base256_limbs(UINT256_LIMBS),
    });
    bufs.push(HintBuf {
        label: "swap_amount_mag",
        cells: inp.amount_mag.to_le_base256_limbs(UINT256_LIMBS),
    });
    bufs.push(HintBuf {
        label: "swap_limit",
        cells: inp.sqrt_limit.to_le_base256_limbs(UINT256_LIMBS),
    });

    // Per-step input blocks. The muldiv pairs (distinct labels) are appended once
    // at the end in their global order; FIFO-per-label keeps them aligned with the
    // guest's per-step consumption (the guest pops muldiv_q/_r in the same global
    // order this recorder produced them).
    for s in &res.steps {
        bufs.push(HintBuf {
            label: "word",
            cells: s.word.to_le_base256_limbs(UINT256_LIMBS),
        });
        push_scalar(&mut bufs, "bit_pos", s.bit_pos);
        push_scalar(&mut bufs, "tick_next_sign", s.tick_next_sign as u32);
        push_scalar(&mut bufs, "tick_next_abs", s.tick_next_abs);
        push_scalar(&mut bufs, "tnp1_sign", s.tnp1_sign as u32);
        push_scalar(&mut bufs, "tnp1_abs", s.tnp1_abs);
        push_scalar(&mut bufs, "liq_net_sign", s.liq_net_sign as u32);
        bufs.push(HintBuf {
            label: "liq_net_mag",
            cells: s.liq_net_mag.to_le_base256_limbs(UINT256_LIMBS),
        });
    }
    bufs.extend(res.muldiv_pairs.clone());
    bufs
}

/// The 8-cell public-input digest for a SWAP-mode run, computed with the SAME
/// Poseidon left-fold the guest performs (main.run_swap + swap.fold_step_data).
pub fn swap_public_input_for(inp: &SwapInputs) -> [F; PUBLIC_INPUT_LEN] {
    use backend::poseidon16_compress_pair;
    let res = ref_pool_swap(inp);

    let fold8 = |acc: [F; 8], chunk: &[F]| -> [F; 8] {
        let c: [F; 8] = chunk.try_into().unwrap();
        poseidon16_compress_pair(&acc, &c)
    };
    let fold_u256 = |mut acc: [F; 8], v: U256| -> [F; 8] {
        let limbs = v.to_le_base256_limbs(UINT256_LIMBS);
        for c in 0..4 {
            acc = fold8(acc, &limbs[c * 8..c * 8 + 8]);
        }
        acc
    };

    // dig0 = head0, then sp0, liq0, amount_mag, limit.
    let (t0s, t0a) = tick_to_sign_abs(inp.tick0);
    let mut acc: [F; 8] = [
        F::ONE,
        F::from_u32(inp.zero_for_one as u32),
        F::from_u32(inp.amount_sign as u32),
        F::from_u32(inp.fee_pips),
        F::from_u32(inp.tick_spacing),
        F::from_u32(res.num_steps),
        F::from_u32(t0s as u32),
        F::from_u32(t0a),
    ];
    acc = fold_u256(acc, inp.sqrt_p0);
    acc = fold_u256(acc, inp.liquidity0);
    acc = fold_u256(acc, inp.amount_mag);
    acc = fold_u256(acc, inp.sqrt_limit);

    // Per-step fold: word (4 chunks) then [liq_net_sign | liq_net_mag(32) | 0*7]
    // (5 chunks), matching swap.fold_step_data.
    for s in &res.steps {
        acc = fold_u256(acc, s.word);
        let mut lblock = vec![F::ZERO; 40];
        lblock[0] = F::from_u32(s.liq_net_sign as u32);
        let mag = s.liq_net_mag.to_le_base256_limbs(UINT256_LIMBS);
        lblock[1..33].copy_from_slice(&mag);
        for c in 0..5 {
            acc = fold8(acc, &lblock[c * 8..c * 8 + 8]);
        }
    }

    // Final block: fhead, then final_sp, final_liq, rem_mag, calc_mag, a0_mag, a1_mag.
    let (fts, fta) = tick_to_sign_abs(res.tick_final);
    let fhead: [F; 8] = [
        F::from_u32(fts as u32),
        F::from_u32(fta),
        F::from_u32(res.amount_remaining.sign as u32),
        F::from_u32(res.amount_calculated.sign as u32),
        F::from_u32(res.amount0.sign as u32),
        F::from_u32(res.amount1.sign as u32),
        F::ZERO,
        F::ZERO,
    ];
    acc = fold8(acc, &fhead);
    acc = fold_u256(acc, res.sqrt_p_final);
    acc = fold_u256(acc, res.liquidity_final);
    acc = fold_u256(acc, res.amount_remaining.mag);
    acc = fold_u256(acc, res.amount_calculated.mag);
    acc = fold_u256(acc, res.amount0.mag);
    acc = fold_u256(acc, res.amount1.mag);
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify q*d + r == prod over the full 512-bit width (q may exceed 256 bits).
    #[allow(clippy::needless_range_loop)] // indexed schoolbook accumulation reads clearer
    fn check_euclid(prod: U512, q: U512, d: U256, r: U256) {
        let mut w = [0u128; 9];
        for i in 0..8 {
            for j in 0..4 {
                if i + j < 8 {
                    w[i + j] += q.words[i] as u128 * d.words[j] as u128;
                } else {
                    assert_eq!(q.words[i] as u128 * d.words[j] as u128, 0);
                }
            }
        }
        for j in 0..4 {
            w[j] += r.words[j] as u128;
        }
        let mut out = [0u64; 8];
        let mut carry = 0u128;
        for i in 0..8 {
            let s = w[i] + carry;
            out[i] = s as u64;
            carry = s >> 64;
        }
        carry += w[8];
        assert_eq!(carry, 0);
        assert_eq!(U512 { words: out }, prod);
    }

    #[test]
    fn muldiv_identity_holds() {
        let a = U256::from_le_bytes(&[0x9F; 32]);
        let b = U256::from_le_bytes(&[0x37; 32]);
        let mut dz = [0u8; 32];
        dz[0] = 0xFD;
        dz[1] = 0xAC;
        dz[5] = 0x11;
        let d = U256::from_le_bytes(&dz);

        let (q, r) = muldiv(a, b, d);
        assert!(r < d);
        check_euclid(a.full_mul(b), q, d, r);
    }

    #[test]
    fn rounding_up_matches_definition() {
        let a = U256::from_u64(7);
        let b = U256::from_u64(3);
        let d = U256::from_u64(4); // 21/4 = 5.25 -> floor 5, ceil 6
        assert_eq!(muldiv_floor_u256(a, b, d), U256::from_u64(5));
        assert_eq!(muldiv_ceil_u256(a, b, d), U256::from_u64(6));
        // exact division: ceil == floor
        let d2 = U256::from_u64(3); // 21/3 = 7
        assert_eq!(muldiv_floor_u256(a, b, d2), U256::from_u64(7));
        assert_eq!(muldiv_ceil_u256(a, b, d2), U256::from_u64(7));
    }

    #[test]
    fn emit_order_and_widths() {
        // Phase-1 mulDiv routed through op 7 (RawMulDiv): input block, then floor
        // and ceil muldiv pairs (FIFO per label).
        let a = U256::from_u64(123);
        let b = U256::from_u64(456);
        let d = U256::from_u64(7);
        let bufs = emit_main_hints(a, b, d);
        let labels: Vec<&str> = bufs.iter().map(|h| h.label).collect();
        assert_eq!(
            labels,
            vec![
                "mode", "op", "flag_a", "sqrtA", "sqrtB", "liquidity", "amount", "y_sign", "tick_sign", "abs_tick",
                "sign_p1", "abs_p1", "tick_spacing", "bit_pos", "word", "fee_pips", "amount_sign", "muldiv_q",
                "muldiv_r", "muldiv_q", "muldiv_r"
            ]
        );
        // sqrtA/sqrtB/liquidity/amount are 32 cells; muldiv_q is 64, muldiv_r 32.
        let sqrt_a_idx = bufs.iter().position(|h| h.label == "sqrtA").unwrap();
        assert_eq!(bufs[sqrt_a_idx].cells.len(), UINT256_LIMBS);
        let q_idx = bufs.iter().position(|h| h.label == "muldiv_q").unwrap();
        assert_eq!(bufs[q_idx].cells.len(), PRODUCT_LIMBS);
        assert_eq!(bufs[q_idx + 1].cells.len(), UINT256_LIMBS);
    }

    fn inputs(op: SqrtOp, flag_a: bool, sqrt_a: U256, sqrt_b: U256, liquidity: U256, amount: U256) -> SqrtInputs {
        SqrtInputs {
            op,
            flag_a,
            sqrt_a,
            sqrt_b,
            liquidity,
            amount,
            ..SqrtInputs::default()
        }
    }

    #[test]
    fn invariant_roundup_ge_floor() {
        // ceil result >= floor result for both getAmount deltas.
        let a = U256::from_u64(0x1_0000_0000);
        let b = U256::from_u64(0x9_8765_4321);
        let liq = U256::from_u64(1_000_003);
        for op in [SqrtOp::Amount0Delta, SqrtOp::Amount1Delta] {
            let floor = sqrt_op_result(&inputs(op, false, a, b, liq, U256::ZERO));
            let ceil = sqrt_op_result(&inputs(op, true, a, b, liq, U256::ZERO));
            assert!(ceil >= floor, "{op:?}: ceil {ceil:?} < floor {floor:?}");
            assert!(ceil <= floor.overflowing_add(U256::ONE).0 || floor == ceil);
        }
    }

    #[test]
    fn invariant_amount1_monotone_in_liquidity() {
        // getAmount1Delta is non-decreasing in liquidity (linear in it).
        let a = U256::from_u64(0x1_0000_0000);
        let b = U256::from_u64(0x5_0000_0000);
        let mut prev = U256::ZERO;
        for k in 1..20u64 {
            let liq = U256::from_u64(k * 1_000_000);
            let v = sqrt_op_result(&inputs(SqrtOp::Amount1Delta, false, a, b, liq, U256::ZERO));
            assert!(v >= prev, "not monotone at k={k}");
            prev = v;
        }
    }

    #[test]
    fn invariant_getnext_stays_in_sqrt_bounds() {
        // From a price inside [MIN, MAX], a single input/output step stays inside
        // (MIN, MAX] for typical (non-reverting) parameters.
        let _lo = min_sqrt_ratio();
        let hi = max_sqrt_ratio();
        let sqrt_p = U256::from_u64(0x1_0000_0000_0000); // well inside the band
        let liq = U256::from_u64(1_000_000_000_000);
        let amount = U256::from_u64(1_000_000);
        for zfo in [false, true] {
            let next_in = sqrt_op_result(&inputs(SqrtOp::NextFromInput, zfo, sqrt_p, U256::ZERO, liq, amount));
            // Stays uint160 and below MAX_SQRT_RATIO; nonzero (price never zero).
            assert!(next_in.fits_uint160() && next_in <= hi, "next_in out of band: {next_in:?}");
            assert!(!next_in.is_zero(), "next price must be positive");
        }
    }

    #[test]
    fn invariant_adddelta_roundtrip() {
        // (x + d) - d == x for add then sub of the same magnitude.
        let x = U256::from_u64(1_000_000);
        let d = U256::from_u64(250_000);
        let added = ref_add_delta(x, false, d);
        let back = ref_add_delta(added, true, d);
        assert_eq!(back, x);
    }

    #[test]
    fn sqrt_op_codes_contiguous() {
        let ops = [
            SqrtOp::Amount0Delta,
            SqrtOp::Amount1Delta,
            SqrtOp::NextFromAmount0RoundingUp,
            SqrtOp::NextFromAmount1RoundingDown,
            SqrtOp::NextFromInput,
            SqrtOp::NextFromOutput,
            SqrtOp::AddDelta,
            SqrtOp::RawMulDiv,
            SqrtOp::GetSqrtRatioAtTick,
            SqrtOp::GetTickAtSqrtRatio,
            SqrtOp::MostSignificantBit,
            SqrtOp::LeastSignificantBit,
            SqrtOp::NextInitializedTickWithinOneWord,
            SqrtOp::ComputeSwapStep,
        ];
        for (i, op) in ops.iter().enumerate() {
            assert_eq!(op.code() as usize, i);
        }
    }

    #[test]
    fn swap_step_pair_counts_per_branch() {
        // The number of muldiv pairs op 13 emits is data-dependent but DETERMINED
        // by the derived branch flags; the guest consumes exactly this many on the
        // taken path (try_execute would error otherwise). Here we pin the expected
        // count for representative branches as a regression guard.
        fn pair_count(inp: &SqrtInputs) -> usize {
            emit_sqrt_hints(inp).iter().filter(|h| h.label == "muldiv_q").count()
        }
        let q96 = U256::Q96;
        // Build prices via Q96 * num / den.
        let price = |num: u64, den: u64| -> U256 {
            let (q, _r) = q96.full_mul(U256::from_u64(num)).divmod_u256(U256::from_u64(den));
            q.low_u256()
        };
        let hi = price(1, 1);
        let low = price(9, 10);
        let l = U256::from_dec_str("1000000000000000000");
        let small = U256::from_dec_str("1000000000000000"); // partial
        let huge = U256::from_dec_str("1000000000000000000000000000000"); // reaches target

        // exactIn / zfo / partial: mulDiv(fee) 1 + getAmount0Delta(stage1) 2 +
        // getNextFromInput(amount0RoundingUp,!reach) 1 + getAmount0Delta(stage2 in) 2 +
        // getAmount1Delta(stage2 out) 1 + fee=sub 0  = 7.
        assert_eq!(pair_count(&compute_swap_step_inputs(hi, low, l, false, small, 3000)), 7);

        // exactIn / zfo / max (huge): mulDiv(fee) 1 + getAmount0Delta(stage1) 2 +
        // reach(next=target) 0 + amountIn=stage1 (max) 0 + getAmount1Delta(stage2 out) 1 +
        // fee=mulDivRoundingUp 1 = 5.
        assert_eq!(pair_count(&compute_swap_step_inputs(hi, low, l, false, huge, 3000)), 5);

        // exactOut / !zfo / partial: getAmount0Delta(stage1 out) 2 +
        // getNextFromOutput(amount0RoundingUp,!reach) 1 + getAmount1Delta(stage2 in) 1 +
        // getAmount0Delta(stage2 out) 2 + fee=mulDivRoundingUp 1 = 7.
        assert_eq!(pair_count(&compute_swap_step_inputs(low, hi, l, true, small, 3000)), 7);
    }

    #[test]
    fn phase4_ops_emit_no_muldiv_pairs() {
        // BitMath / TickBitmap use no FullMath, so ops 10/11/12 emit 0 muldiv pairs.
        let msb = most_significant_bit_inputs(U256::ONE.shl(100));
        assert_eq!(emit_sqrt_hints(&msb).iter().filter(|h| h.label == "muldiv_q").count(), 0);
        let lsb = least_significant_bit_inputs(U256::ONE.shl(100));
        assert_eq!(emit_sqrt_hints(&lsb).iter().filter(|h| h.label == "muldiv_q").count(), 0);
        let ni = next_init_inputs(-61, 60, false, U256::ONE.shl(130));
        assert_eq!(emit_sqrt_hints(&ni).iter().filter(|h| h.label == "muldiv_q").count(), 0);
        // The new input-block labels appear exactly once each, in order.
        let labels: Vec<&str> = emit_sqrt_hints(&ni).iter().map(|h| h.label).collect();
        assert_eq!(labels.iter().filter(|&&l| l == "tick_spacing").count(), 1);
        assert_eq!(labels.iter().filter(|&&l| l == "bit_pos").count(), 1);
        assert_eq!(labels.iter().filter(|&&l| l == "word").count(), 1);
    }

    #[test]
    fn ref_compressed_floor_semantics() {
        // floor division toward -inf (v3 compressed).
        assert_eq!(ref_compressed(100, 10), 10);
        assert_eq!(ref_compressed(105, 10), 10);
        assert_eq!(ref_compressed(-100, 10), -10);
        assert_eq!(ref_compressed(-105, 10), -11); // -10.5 -> floor -11
        assert_eq!(ref_compressed(-1, 60), -1); // -0.0166 -> floor -1
        assert_eq!(ref_compressed(-60, 60), -1);
        assert_eq!(ref_compressed(-61, 60), -2);
        assert_eq!(ref_compressed(0, 7), 0);
    }

    #[test]
    fn ref_next_init_matches_v3_position_choice() {
        // gt uses position(compressed+1); lte uses position(compressed). Confirm the
        // returned bit_pos matches the v3 position() of the right argument.
        let word = U256::MAX;
        for tick in [0i32, 1, -1, 255, 256, -256, -257, 12345, -12345] {
            for ts in [1u32, 10, 60] {
                let c = ref_compressed(tick, ts);
                let (_n, _i, bp_lte, _m, wp_lte) = ref_next_init(tick, ts, true, word);
                assert_eq!((wp_lte, bp_lte), ref_position(c), "lte position mismatch tick={tick} ts={ts}");
                let (_n, _i, bp_gt, _m, wp_gt) = ref_next_init(tick, ts, false, word);
                assert_eq!((wp_gt, bp_gt), ref_position(c + 1), "gt position mismatch tick={tick} ts={ts}");
            }
        }
    }

    // -------- TickMath reference validation (the magic-constant safety net) ----

    #[test]
    fn tickmath_boundary_vectors() {
        // getSqrtRatioAtTick(0) == 2^96.
        assert_eq!(ref_get_sqrt_ratio_at_tick(0), U256::ONE.shl(96), "sqrtRatio(0) != 2^96");
        // getSqrtRatioAtTick(MIN_TICK) == MIN_SQRT_RATIO.
        assert_eq!(ref_get_sqrt_ratio_at_tick(MIN_TICK), min_sqrt_ratio(), "sqrtRatio(MIN) != MIN_SQRT_RATIO");
        // getSqrtRatioAtTick(MAX_TICK) == MAX_SQRT_RATIO.
        assert_eq!(ref_get_sqrt_ratio_at_tick(MAX_TICK), max_sqrt_ratio(), "sqrtRatio(MAX) != MAX_SQRT_RATIO");
    }

    #[test]
    fn tickmath_monotone_and_roundtrip() {
        // Strictly increasing in tick, and getTickAtSqrtRatio inverts it.
        let mut prev = U256::ZERO;
        let mut t = MIN_TICK;
        while t <= MAX_TICK {
            let r = ref_get_sqrt_ratio_at_tick(t);
            if t > MIN_TICK {
                assert!(r > prev, "not strictly increasing at tick {t}");
            }
            prev = r;
            // round-trip on a sampled subset (full scan is ~1.7M ticks).
            if t % 7919 == 0 || t.abs() < 3 {
                let rt = ref_get_tick_at_sqrt_ratio(r);
                assert_eq!(rt, t, "round-trip failed at tick {t}");
            }
            t += if t.abs() < 16 { 1 } else { 521 };
        }
    }

    #[test]
    fn tickmath_hint_emission_pair_counts() {
        // op 8 with tick <= 0 emits 0 muldiv pairs; tick > 0 emits 1.
        let neg = sqrt_ratio_at_tick_inputs(-12345);
        assert_eq!(emit_sqrt_hints(&neg).iter().filter(|h| h.label == "muldiv_q").count(), 0);
        let zero = sqrt_ratio_at_tick_inputs(0);
        assert_eq!(emit_sqrt_hints(&zero).iter().filter(|h| h.label == "muldiv_q").count(), 0);
        let pos = sqrt_ratio_at_tick_inputs(12345);
        assert_eq!(emit_sqrt_hints(&pos).iter().filter(|h| h.label == "muldiv_q").count(), 1);
        // op 9 at a positive tick: sqrtRatio(tick) and sqrtRatio(tick+1) both >0 => 2.
        let sp = ref_get_sqrt_ratio_at_tick(50000);
        let inp9 = tick_at_sqrt_ratio_inputs(sp, ref_get_tick_at_sqrt_ratio(sp));
        assert_eq!(emit_sqrt_hints(&inp9).iter().filter(|h| h.label == "muldiv_q").count(), 2);
        // op 9 at tick = -1: sqrtRatio(-1) tick<=0 (0 pairs), sqrtRatio(0) (0) => 0.
        let sp_neg = ref_get_sqrt_ratio_at_tick(-1);
        let inp9n = tick_at_sqrt_ratio_inputs(sp_neg, ref_get_tick_at_sqrt_ratio(sp_neg));
        assert_eq!(emit_sqrt_hints(&inp9n).iter().filter(|h| h.label == "muldiv_q").count(), 0);
    }
}
