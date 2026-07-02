from snark_lib import *
from bignum import *
from full_math import *

# ============================================================================
# Uniswap V3 SqrtPriceMath, mirrored from v3-core SqrtPriceMath.sol.
# ============================================================================
#
# Sources of truth (the exact v3 formula is reproduced in each function's
# docstring). Widths (base 2^8 little-endian limbs):
#     sqrtPX96 : uint160 = 20 limbs        (UINT160_LIMBS)
#     liquidity: uint128 = 16 limbs        (UINT128_LIMBS)
#     amount   : uint256 = 32 limbs        (UINT256_LIMBS)
#     result   : uint160 (asserted to fit) for getNextSqrtPrice*
#                uint256 for getAmount0/1Delta
#     Q96 = 2^96  (a uint256 constant: limb[12] = 1)
#     type(uint160).max = 2^160 - 1
#
# MULDIV WIDTH POLICY (keeps the hint contract uniform):
#   Every FullMath.mulDiv / mulDivRoundingUp call operates at uint256 width
#   (NA = NB = ND = NOUT = 32). Operands are widened to 32 limbs first; results
#   that v3 narrows with `toUint160()` are asserted to fit uint160 in-circuit.
#   Consequently EVERY mulDiv/mulDivRoundingUp consumes exactly ONE
#   ("muldiv_q" 64-limb, "muldiv_r" 32-limb) hint pair, in guest call order,
#   FIFO per label — identical to FullMath (see bignum.py HINT CONTRACT).
#
# ----------------------------------------------------------------------------
# HINT CONTRACT for the Phase-2 driver (must match src/reference.rs EXACTLY)
# ----------------------------------------------------------------------------
#   The Phase-2 main.py reads its scalar inputs via these labels (one buffer
#   each, in this order):
#       "op"        : 1 cell   (operation selector, see main.py OP_* table)
#       "flag_a"    : 1 cell   (roundUp / add / zeroForOne, per op)
#       "sqrtA"     : 32 cells (uint256-encoded uint160)
#       "sqrtB"     : 32 cells (uint256-encoded uint160)
#       "liquidity" : 32 cells (uint256-encoded uint128)
#       "amount"    : 32 cells (uint256)
#       "y_sign"    : 1 cell   (LiquidityMath addDelta sign bit)
#   Then, during the dispatched computation, each mulDiv / mulDivRoundingUp
#   consumes one ("muldiv_q","muldiv_r") pair in the exact order the guest calls
#   them. The reference (reference.rs) replays the SAME branch decisions so the
#   number and order of emitted pairs matches the guest's taken path.
# ============================================================================

# Note: UINT128_LIMBS lives in liquidity_math.py (single definition; duplicate
# top-level constant names across `import *` are a compile error).
UINT160_LIMBS = 20
SQRT_RESULT_LIMBS = 20  # uint160


# ----------------------------------------------------------------------------
# uint256 constants built at runtime (not in the preamble; pure base-field).
# ----------------------------------------------------------------------------


def make_one_u256():
    # The uint256 constant 1, as 32 base-256 limbs.
    one = Array(UINT256_LIMBS)
    one[0] = 1
    for i in unroll(1, UINT256_LIMBS):
        one[i] = 0
    return one


def make_q96_u256():
    # The uint256 constant Q96 = 2^96. At base 2^8, 2^96 = 256^12, so limb[12]=1.
    # (Write each cell exactly once: limb[12] = 1, all others 0 — write-once.)
    q = Array(UINT256_LIMBS)
    for i in unroll(0, SHIFT_96_LIMBS):
        q[i] = 0
    q[SHIFT_96_LIMBS] = 1  # SHIFT_96_LIMBS == 12
    for i in unroll(SHIFT_96_LIMBS + 1, UINT256_LIMBS):
        q[i] = 0
    return q


def make_uint160_max():
    # type(uint160).max = 2^160 - 1, as a uint256 (32 limbs): limbs 0..19 = 255,
    # limbs 20..31 = 0. (Provided for completeness / TickMath reuse; the Phase-2
    # SqrtPriceMath collapses the amount<=uint160_max branch — see
    # get_next_sqrt_price_from_amount1_rounding_down.)
    m = Array(UINT256_LIMBS)
    for i in unroll(0, UINT160_LIMBS):
        m[i] = LIMB_BASE - 1  # 255
    for i in unroll(UINT160_LIMBS, UINT256_LIMBS):
        m[i] = 0
    return m


# ----------------------------------------------------------------------------
# Thin uint256-width division helpers built on FullMath.
# ----------------------------------------------------------------------------


def div_floor_u256(x, y):
    # floor(x / y), all uint256. = mulDiv(x, 1, y). y must be nonzero (asserted
    # inside bignum_muldiv). Consumes one muldiv pair.
    one = make_one_u256()
    return mulDiv(x, one, y)


def div_rounding_up_u256(x, y):
    # ceil(x / y) = floor(x/y) + (x mod y != 0), all uint256. Implemented soundly
    # as mulDivRoundingUp(x, 1, y) (v3's UnsafeMath.divRoundingUp result, with the
    # full Euclidean identity enforcing the +1). y must be nonzero. Consumes one
    # muldiv pair.
    one = make_one_u256()
    return mulDivRoundingUp(x, one, y)


def liquidity_shl_96_u256(liquidity32):
    # numerator1 = uint256(liquidity) << 96. liquidity32 is the uint128 widened to
    # 32 limbs. shl_96 over 32 limbs yields 44 limbs; cast back to 32 (the top 12
    # limbs are provably zero because liquidity<<96 < 2^224 < 2^256).
    shifted = shl_96(liquidity32, UINT256_LIMBS)  # 44 limbs
    return bignum_cast(shifted, UINT256_LIMBS + SHIFT_96_LIMBS, UINT256_LIMBS)


def order_pair_u256(a, b):
    # Return (lo, hi) with lo = min(a,b), hi = max(a,b). a_lt_b in {0,1} is the
    # proven comparison; we select limbwise so the swap is fully constrained and
    # no control-flow branch is needed. Covers equality: if a == b, a_lt_b == 0 so
    # lo = b, hi = a == b. lo <= hi is then a proven fact downstream.
    range_check_limbs(a, UINT256_LIMBS)
    range_check_limbs(b, UINT256_LIMBS)
    a_lt_b = bignum_lt(a, b, UINT256_LIMBS)  # 1 iff a < b
    lo = Array(UINT256_LIMBS)
    hi = Array(UINT256_LIMBS)
    for i in unroll(0, UINT256_LIMBS):
        lo[i] = a_lt_b * a[i] + (1 - a_lt_b) * b[i]
        hi[i] = a_lt_b * b[i] + (1 - a_lt_b) * a[i]
    return lo, hi


# ----------------------------------------------------------------------------
# getAmount0Delta / getAmount1Delta
# ----------------------------------------------------------------------------


def get_amount0_delta(sqrtA, sqrtB, liquidity32, round_up):
    # v3-core:
    #   if (sqrtA > sqrtB) (sqrtA, sqrtB) = (sqrtB, sqrtA);
    #   uint256 numerator1 = uint256(liquidity) << RESOLUTION;   // <<96
    #   uint256 numerator2 = sqrtB - sqrtA;
    #   require(sqrtA > 0);
    #   return roundUp
    #     ? divRoundingUp(mulDivRoundingUp(numerator1, numerator2, sqrtB), sqrtA)
    #     : mulDiv(numerator1, numerator2, sqrtB) / sqrtA;
    #
    # sqrtA, sqrtB are uint256-encoded uint160 (32 limbs); result is uint256.
    assert round_up * (1 - round_up) == 0

    lo, hi = order_pair_u256(sqrtA, sqrtB)
    # require(sqrtA > 0): the smaller of the two (lo) must be positive.
    assert bignum_is_zero(lo, UINT256_LIMBS) == 0

    numerator1 = liquidity_shl_96_u256(liquidity32)
    numerator2 = Array(UINT256_LIMBS)
    bignum_sub(hi, lo, numerator2, UINT256_LIMBS)  # hi - lo, hi >= lo guaranteed

    result = Array(UINT256_LIMBS)
    if round_up == 1:
        inner = mulDivRoundingUp(numerator1, numerator2, hi)
        ru = div_rounding_up_u256(inner, lo)
        for i in unroll(0, UINT256_LIMBS):
            result[i] = ru[i]
    else:
        inner = mulDiv(numerator1, numerator2, hi)
        fl = div_floor_u256(inner, lo)
        for i in unroll(0, UINT256_LIMBS):
            result[i] = fl[i]
    return result


def get_amount1_delta(sqrtA, sqrtB, liquidity32, round_up):
    # v3-core:
    #   if (sqrtA > sqrtB) (sqrtA, sqrtB) = (sqrtB, sqrtA);
    #   return roundUp
    #     ? mulDivRoundingUp(liquidity, sqrtB - sqrtA, Q96)
    #     : mulDiv(liquidity, sqrtB - sqrtA, Q96);
    # Result is uint256.
    assert round_up * (1 - round_up) == 0
    lo, hi = order_pair_u256(sqrtA, sqrtB)

    diff = Array(UINT256_LIMBS)
    bignum_sub(hi, lo, diff, UINT256_LIMBS)
    q96 = make_q96_u256()

    result = Array(UINT256_LIMBS)
    if round_up == 1:
        ru = mulDivRoundingUp(liquidity32, diff, q96)
        for i in unroll(0, UINT256_LIMBS):
            result[i] = ru[i]
    else:
        fl = mulDiv(liquidity32, diff, q96)
        for i in unroll(0, UINT256_LIMBS):
            result[i] = fl[i]
    return result


# ----------------------------------------------------------------------------
# getNextSqrtPriceFromAmount0RoundingUp + its divRoundingUp fallback
# ----------------------------------------------------------------------------


def _amount0_add_fallback(numerator1, sqrtP, amount):
    # return uint160(divRoundingUp(numerator1, (numerator1 / sqrtPX96) + amount))
    # The inner add is v3's checked SafeMath `.add()` (reverts on overflow): we
    # assert no uint256 overflow. Returns SQRT_RESULT_LIMBS (uint160), asserted
    # to fit. Asserts sqrtP > 0 so the division is defined (caller invariant,
    # re-asserted here for isolation).
    assert bignum_is_zero(sqrtP, UINT256_LIMBS) == 0
    n1_div_p = div_floor_u256(numerator1, sqrtP)
    denom, no_ovf = bignum_add_fits(n1_div_p, amount, UINT256_LIMBS)
    assert no_ovf == 1  # SafeMath .add() would revert on overflow
    res = div_rounding_up_u256(numerator1, denom)
    bignum_assert_fits(res, UINT256_LIMBS, SQRT_RESULT_LIMBS)
    out = Array(SQRT_RESULT_LIMBS)
    for i in unroll(0, SQRT_RESULT_LIMBS):
        out[i] = res[i]
    return out


def get_next_sqrt_price_from_amount0_rounding_up(sqrtP, liquidity32, amount, add):
    # v3-core:
    #   if (amount == 0) return sqrtPX96;
    #   uint256 numerator1 = uint256(liquidity) << RESOLUTION;
    #   if (add) {
    #     uint256 product;
    #     if ((product = amount * sqrtPX96) / amount == sqrtPX96) {
    #       uint256 denominator = numerator1 + product;
    #       if (denominator >= numerator1)
    #         return uint160(mulDivRoundingUp(numerator1, sqrtPX96, denominator));
    #     }
    #     return uint160(divRoundingUp(numerator1, (numerator1 / sqrtPX96) + amount));
    #   } else {
    #     uint256 product;
    #     require((product = amount * sqrtPX96) / amount == sqrtPX96 && numerator1 > product);
    #     uint256 denominator = numerator1 - product;
    #     return mulDivRoundingUp(numerator1, sqrtPX96, denominator).toUint160();
    #   }
    #
    # IN-CIRCUIT TRANSLATION of v3's overflow guard `(amount*sqrtP)/amount==sqrtP`:
    #   true iff amount*sqrtP did NOT overflow uint256, i.e. iff the full 64-limb
    #   product has all limbs at index >= 32 equal to zero. We compute the full
    #   product (no truncation), test `fits256`, and only then take the truncated
    #   32-limb product. This is a REAL in-circuit assert on the remove path and a
    #   real branch on the add path (matching v3's `if`).
    assert add * (1 - add) == 0
    range_check_limbs(amount, UINT256_LIMBS)

    amount_zero = bignum_is_zero(amount, UINT256_LIMBS)
    numerator1 = liquidity_shl_96_u256(liquidity32)

    result = Array(SQRT_RESULT_LIMBS)
    if amount_zero == 1:
        # return sqrtPX96 unchanged (assert it fits uint160, return low 20 limbs).
        bignum_assert_fits(sqrtP, UINT256_LIMBS, SQRT_RESULT_LIMBS)
        for i in unroll(0, SQRT_RESULT_LIMBS):
            result[i] = sqrtP[i]
    else:
        prod = bignum_mul_schoolbook(amount, sqrtP, UINT256_LIMBS, UINT256_LIMBS)  # 64 limbs
        fits256 = bignum_high_is_zero(prod, 2 * UINT256_LIMBS, UINT256_LIMBS)

        if add == 1:
            inner_ru = Array(SQRT_RESULT_LIMBS)
            if fits256 == 1:
                product = bignum_cast(prod, 2 * UINT256_LIMBS, UINT256_LIMBS)  # asserts fit (holds)
                denominator, no_overflow = bignum_add_fits(numerator1, product, UINT256_LIMBS)
                if no_overflow == 1:
                    res = mulDivRoundingUp(numerator1, sqrtP, denominator)
                    bignum_assert_fits(res, UINT256_LIMBS, SQRT_RESULT_LIMBS)
                    for i in unroll(0, SQRT_RESULT_LIMBS):
                        inner_ru[i] = res[i]
                else:
                    fb = _amount0_add_fallback(numerator1, sqrtP, amount)
                    for i in unroll(0, SQRT_RESULT_LIMBS):
                        inner_ru[i] = fb[i]
            else:
                fb = _amount0_add_fallback(numerator1, sqrtP, amount)
                for i in unroll(0, SQRT_RESULT_LIMBS):
                    inner_ru[i] = fb[i]
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = inner_ru[i]
        else:
            # remove: require no overflow AND numerator1 > product.
            assert fits256 == 1  # (amount*sqrtP)/amount == sqrtP
            product = bignum_cast(prod, 2 * UINT256_LIMBS, UINT256_LIMBS)
            assert bignum_lt(product, numerator1, UINT256_LIMBS) == 1  # numerator1 > product
            denominator = Array(UINT256_LIMBS)
            bignum_sub(numerator1, product, denominator, UINT256_LIMBS)
            res = mulDivRoundingUp(numerator1, sqrtP, denominator)
            bignum_assert_fits(res, UINT256_LIMBS, SQRT_RESULT_LIMBS)  # toUint160()
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = res[i]
    return result


# ----------------------------------------------------------------------------
# getNextSqrtPriceFromAmount1RoundingDown
# ----------------------------------------------------------------------------


def get_next_sqrt_price_from_amount1_rounding_down(sqrtP, liquidity32, amount, add):
    # v3-core:
    #   if (add) {
    #     uint256 quotient = amount <= type(uint160).max
    #         ? (amount << RESOLUTION) / liquidity
    #         : mulDiv(amount, Q96, liquidity);
    #     return uint256(sqrtPX96).add(quotient).toUint160();
    #   } else {
    #     uint256 quotient = amount <= type(uint160).max
    #         ? divRoundingUp(amount << RESOLUTION, liquidity)
    #         : mulDivRoundingUp(amount, Q96, liquidity);
    #     require(sqrtPX96 > quotient);
    #     return uint160(sqrtPX96 - quotient);
    #   }
    #
    # FAITHFUL SIMPLIFICATION of the `amount <= uint160.max` branch:
    #   That branch is taken only when amount <= 2^160-1, so `amount << 96 < 2^256`
    #   and `(amount<<96)/liquidity` == `mulDiv(amount, Q96, liquidity)` EXACTLY
    #   (the rounding-up form == `mulDivRoundingUp(amount, Q96, liquidity)`). Our
    #   mulDiv carries the full 512-bit product without truncation, so it is also
    #   correct for amount > 2^160-1. The two v3 arms therefore collapse to a single
    #   mulDiv/mulDivRoundingUp(amount, Q96, liquidity) for ALL amounts; the result
    #   is bit-identical to v3 (economic parity preserved), only the gas-motivated
    #   branch is removed.
    assert add * (1 - add) == 0
    q96 = make_q96_u256()

    result = Array(SQRT_RESULT_LIMBS)
    if add == 1:
        quotient = mulDiv(amount, q96, liquidity32)
        sumv, no_ovf = bignum_add_fits(sqrtP, quotient, UINT256_LIMBS)  # .add()
        assert no_ovf == 1
        bignum_assert_fits(sumv, UINT256_LIMBS, SQRT_RESULT_LIMBS)  # toUint160()
        for i in unroll(0, SQRT_RESULT_LIMBS):
            result[i] = sumv[i]
    else:
        quotient = mulDivRoundingUp(amount, q96, liquidity32)
        assert bignum_lt(quotient, sqrtP, UINT256_LIMBS) == 1  # require(sqrtP > quotient)
        diff = Array(UINT256_LIMBS)
        bignum_sub(sqrtP, quotient, diff, UINT256_LIMBS)
        bignum_assert_fits(diff, UINT256_LIMBS, SQRT_RESULT_LIMBS)  # uint160(...)
        for i in unroll(0, SQRT_RESULT_LIMBS):
            result[i] = diff[i]
    return result


# ----------------------------------------------------------------------------
# getNextSqrtPriceFromInput / getNextSqrtPriceFromOutput
# ----------------------------------------------------------------------------


def get_next_sqrt_price_from_input(sqrtP, liquidity32, amount_in, zero_for_one):
    # v3-core:
    #   require(sqrtPX96 > 0); require(liquidity > 0);
    #   return zeroForOne
    #     ? getNextSqrtPriceFromAmount0RoundingUp(sqrtPX96, liquidity, amountIn, true)
    #     : getNextSqrtPriceFromAmount1RoundingDown(sqrtPX96, liquidity, amountIn, true);
    assert zero_for_one * (1 - zero_for_one) == 0
    assert bignum_is_zero(sqrtP, UINT256_LIMBS) == 0
    assert bignum_is_zero(liquidity32, UINT256_LIMBS) == 0

    result = Array(SQRT_RESULT_LIMBS)
    if zero_for_one == 1:
        r = get_next_sqrt_price_from_amount0_rounding_up(sqrtP, liquidity32, amount_in, 1)
        for i in unroll(0, SQRT_RESULT_LIMBS):
            result[i] = r[i]
    else:
        r = get_next_sqrt_price_from_amount1_rounding_down(sqrtP, liquidity32, amount_in, 1)
        for i in unroll(0, SQRT_RESULT_LIMBS):
            result[i] = r[i]
    return result


def get_next_sqrt_price_from_output(sqrtP, liquidity32, amount_out, zero_for_one):
    # v3-core:
    #   require(sqrtPX96 > 0); require(liquidity > 0);
    #   return zeroForOne
    #     ? getNextSqrtPriceFromAmount1RoundingDown(sqrtPX96, liquidity, amountOut, false)
    #     : getNextSqrtPriceFromAmount0RoundingUp(sqrtPX96, liquidity, amountOut, false);
    assert zero_for_one * (1 - zero_for_one) == 0
    assert bignum_is_zero(sqrtP, UINT256_LIMBS) == 0
    assert bignum_is_zero(liquidity32, UINT256_LIMBS) == 0

    result = Array(SQRT_RESULT_LIMBS)
    if zero_for_one == 1:
        r = get_next_sqrt_price_from_amount1_rounding_down(sqrtP, liquidity32, amount_out, 0)
        for i in unroll(0, SQRT_RESULT_LIMBS):
            result[i] = r[i]
    else:
        r = get_next_sqrt_price_from_amount0_rounding_up(sqrtP, liquidity32, amount_out, 0)
        for i in unroll(0, SQRT_RESULT_LIMBS):
            result[i] = r[i]
    return result
