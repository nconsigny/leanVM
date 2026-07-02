from snark_lib import *
from bignum import *
from full_math import *
from sqrt_price_math import *
from helpers import *

# ============================================================================
# Uniswap V3 SwapMath.computeSwapStep, mirrored from v3-core SwapMath.sol (EXACT).
# ============================================================================
#
# v3-core SwapMath.sol (verbatim):
#
#   function computeSwapStep(
#       uint160 sqrtRatioCurrentX96,
#       uint160 sqrtRatioTargetX96,
#       uint128 liquidity,
#       int256 amountRemaining,
#       uint24 feePips
#   ) internal pure returns (
#       uint160 sqrtRatioNextX96, uint256 amountIn, uint256 amountOut, uint256 feeAmount
#   ) {
#       bool zeroForOne = sqrtRatioCurrentX96 >= sqrtRatioTargetX96;
#       bool exactIn = amountRemaining >= 0;
#
#       if (exactIn) {
#           uint256 amountRemainingLessFee = FullMath.mulDiv(uint256(amountRemaining), 1e6 - feePips, 1e6);
#           amountIn = zeroForOne
#               ? SqrtPriceMath.getAmount0Delta(sqrtRatioTargetX96, sqrtRatioCurrentX96, liquidity, true)
#               : SqrtPriceMath.getAmount1Delta(sqrtRatioCurrentX96, sqrtRatioTargetX96, liquidity, true);
#           if (amountRemainingLessFee >= amountIn) sqrtRatioNextX96 = sqrtRatioTargetX96;
#           else
#               sqrtRatioNextX96 = SqrtPriceMath.getNextSqrtPriceFromInput(
#                   sqrtRatioCurrentX96, liquidity, amountRemainingLessFee, zeroForOne
#               );
#       } else {
#           amountOut = zeroForOne
#               ? SqrtPriceMath.getAmount1Delta(sqrtRatioTargetX96, sqrtRatioCurrentX96, liquidity, false)
#               : SqrtPriceMath.getAmount0Delta(sqrtRatioCurrentX96, sqrtRatioTargetX96, liquidity, false);
#           if (uint256(-amountRemaining) >= amountOut) sqrtRatioNextX96 = sqrtRatioTargetX96;
#           else
#               sqrtRatioNextX96 = SqrtPriceMath.getNextSqrtPriceFromOutput(
#                   sqrtRatioCurrentX96, liquidity, uint256(-amountRemaining), zeroForOne
#               );
#       }
#
#       bool max = sqrtRatioTargetX96 == sqrtRatioNextX96;
#
#       // get the input/output amounts
#       if (zeroForOne) {
#           amountIn = max && exactIn
#               ? amountIn
#               : SqrtPriceMath.getAmount0Delta(sqrtRatioNextX96, sqrtRatioCurrentX96, liquidity, true);
#           amountOut = max && !exactIn
#               ? amountOut
#               : SqrtPriceMath.getAmount1Delta(sqrtRatioNextX96, sqrtRatioCurrentX96, liquidity, false);
#       } else {
#           amountIn = max && exactIn
#               ? amountIn
#               : SqrtPriceMath.getAmount1Delta(sqrtRatioCurrentX96, sqrtRatioNextX96, liquidity, true);
#           amountOut = max && !exactIn
#               ? amountOut
#               : SqrtPriceMath.getAmount0Delta(sqrtRatioCurrentX96, sqrtRatioNextX96, liquidity, false);
#       }
#
#       // cap the output amount to not exceed the remaining output amount
#       if (!exactIn && amountOut > uint256(-amountRemaining)) {
#           amountOut = uint256(-amountRemaining);
#       }
#
#       if (exactIn && sqrtRatioNextX96 != sqrtRatioTargetX96) {
#           // we didn't reach the target, so take the remainder of the maximum input as fee
#           feeAmount = uint256(amountRemaining) - amountIn;
#       } else {
#           feeAmount = FullMath.mulDivRoundingUp(amountIn, feePips, 1e6 - feePips);
#       }
#   }
#
# ----------------------------------------------------------------------------
# IN-CIRCUIT TRANSLATION — every branch condition is DERIVED, never hinted
# ----------------------------------------------------------------------------
#   zeroForOne = (current >= target)         = 1 - bignum_lt(current, target)
#   exactIn    = (amountRemaining >= 0)      = 1 - amount_sign   (sign in {0,1})
#   reach      = (budget >= stage1 amount)   = bignum_le_flag(stage1, budget)
#   max        = (target == sqrtRatioNext)   = bignum_eq_flag(target, next)
#   cap        = (amountOut > magnitude)     = bignum_lt(magnitude, amountOut)
# All five are proof-enforced booleans computed from the committed inputs; a
# malicious prover cannot forge a branch outcome (e.g. claim `max` when
# target != next) because bignum_eq_flag / bignum_lt are AIR-constrained.
#
# CONTROL FLOW & HINT CONSUMPTION (the load-bearing soundness invariant):
#   Each FullMath.mulDiv / SqrtPriceMath call consumes a deterministic number of
#   ("muldiv_q","muldiv_r") hint pairs (see sqrt_price_math.py MULDIV WIDTH
#   POLICY). We gate every hint-consuming call behind REAL VM control flow
#   (`if` over a derived flag), so ONLY the taken path consumes pairs. The Rust
#   reference (reference.rs ref_compute_swap_step) replays the IDENTICAL flag
#   decisions through the same ref_* recorder functions, so the number and order
#   of pairs the host queues matches the guest exactly, per branch.
#
#   Per-call pair counts (from sqrt_price_math.py):
#     get_amount0_delta            -> 2  (mulDiv(Up) inner + div(Up) by lo)
#     get_amount1_delta            -> 1
#     get_next_sqrt_price_from_input / _from_output -> data-dependent (0/1/2),
#       handled identically on both sides because the same sub-flags are derived.
#     mulDiv / mulDivRoundingUp    -> 1
#
# ----------------------------------------------------------------------------
# OP-13 INPUT / COMMIT CONTRACT (must match main.py + src/reference.rs EXACTLY)
# ----------------------------------------------------------------------------
#   New hint labels read by main.py's input block (one cell each, in this order,
#   AFTER "word", BEFORE the muldiv pairs):
#       "fee_pips"    : 1 cell   (uint24 fee, 0 <= fee_pips < FEE_DENOM)
#       "amount_sign" : 1 cell   (sign of amountRemaining: 0 = >=0, 1 = <0)
#   Reused existing slots for op 13:
#       sqrtA  = sqrtRatioCurrentX96 (uint160, uint256-encoded)
#       sqrtB  = sqrtRatioTargetX96  (uint160, uint256-encoded)
#       liquidity = uint128 liquidity (uint256-encoded)
#       amount = |amountRemaining| (uint256 magnitude)
#   Committed OUTPUTS (new u256 commit slots, see main.py COMMIT layout):
#       result   = sqrtRatioNextX96 (uint160, zero-extended to u256)
#       swap_in  = amountIn  (u256)
#       swap_out = amountOut (u256)
#       fee      = feeAmount (u256)
#   Committed head scalars: head[11] = fee_pips, head[12] = amount_sign.
# ============================================================================

FEE_DENOM = 1000000  # 1e6, the v3 fee denominator
FEE_PIPS_BITS = 20   # FEE_DENOM = 1000000 < 2^20 = 1048576 (fee fits in 20 bits)


def make_fee_denom_u256():
    # FEE_DENOM = 1000000 = 0xF4240, as 32 base-256 little-endian limbs.
    # LE bytes: 64 (0x40), 66 (0x42), 15 (0x0F). Write each cell exactly once.
    d = Array(UINT256_LIMBS)
    d[0] = 64
    d[1] = 66
    d[2] = 15
    for i in unroll(3, UINT256_LIMBS):
        d[i] = 0
    return d


def small_value_to_u256(v, n_bits: Const):
    # Build the uint256 (32 base-256 limbs) equal to the field value `v`, where v
    # is known to satisfy 0 <= v < 2^n_bits (n_bits <= 24). The decomposition both
    # RANGE-CHECKS v (binds 0 <= v < 2^n_bits) and constructs the bignum from the
    # proven bits, so the result is unforgeable: limb j = sum_{k} bit[8j+k]*2^k.
    #
    # checked_decompose_bits_small_value_const returns MSB-first bits
    # (bits[0] = bit n_bits-1, bits[n_bits-1] = bit 0) and asserts they recompose
    # to v, pinning every bit. We regroup them into base-256 limbs. n_bits <= 24
    # so only the low 3 limbs are nonzero; the rest are zeroed (write-once).
    if 24 < n_bits:
        assert False, "small_value_to_u256 supports n_bits <= 24"
    bits = checked_decompose_bits_small_value_const(v, n_bits)
    out = Array(UINT256_LIMBS)
    n_limbs = div_ceil(n_bits, 8)  # compile-time: number of nonzero limbs
    for j in unroll(0, n_limbs):
        limb: Mut = 0
        # bit (8j + k) has value 2^(8j+k); its MSB-first index is n_bits-1-(8j+k).
        for k in unroll(0, 8):
            bit_value_index = 8 * j + k
            if bit_value_index < n_bits:
                msb_idx = n_bits - 1 - bit_value_index
                limb = limb + bits[msb_idx] * (2 ** k)
        out[j] = limb
    for j in unroll(n_limbs, UINT256_LIMBS):
        out[j] = 0
    return out


def assert_fee_pips_lt_denom(fee_pips):
    # Enforce 0 <= fee_pips < FEE_DENOM soundly. FEE_DENOM = 1000000 > 2^16, so a
    # direct `assert fee_pips < FEE_DENOM` is NOT a valid range-check. Instead:
    #   * fee_pips_u256 = small_value_to_u256(fee_pips, 20): binds 0 <= fee_pips < 2^20.
    #   * decompose (FEE_DENOM - 1) - fee_pips into 20 bits: succeeds iff
    #     fee_pips <= FEE_DENOM - 1, i.e. fee_pips < FEE_DENOM. If fee_pips >=
    #     FEE_DENOM the field value wraps near p and its 20-bit decomposition fails.
    # Returns (fee_pips_u256, denom_complement_u256) where denom_complement =
    # FEE_DENOM - fee_pips in [1, FEE_DENOM]; both are uint256 bignums for FullMath.
    fee_pips_u256 = small_value_to_u256(fee_pips, FEE_PIPS_BITS)
    _ = checked_decompose_bits_small_value_const(FEE_DENOM - 1 - fee_pips, FEE_PIPS_BITS)
    denom_complement = FEE_DENOM - fee_pips  # field, in [1, FEE_DENOM] (no wrap)
    denom_complement_u256 = small_value_to_u256(denom_complement, FEE_PIPS_BITS)
    return fee_pips_u256, denom_complement_u256


def _copy_u256(x):
    # Fresh 32-limb copy of a 32-limb bignum (range-checked source).
    range_check_limbs(x, UINT256_LIMBS)
    out = Array(UINT256_LIMBS)
    for i in unroll(0, UINT256_LIMBS):
        out[i] = x[i]
    return out


def _select_u256(cond, a, b):
    # Limbwise select: cond ? a : b, both 32-limb. cond is a proven boolean. No
    # hints consumed (cheap arithmetic), so safe to use when both a and b are
    # already-computed buffers. Every produced limb stays < 256 (it equals a[i]
    # or b[i], both range-checked).
    range_check_limbs(a, UINT256_LIMBS)
    range_check_limbs(b, UINT256_LIMBS)
    out = Array(UINT256_LIMBS)
    for i in unroll(0, UINT256_LIMBS):
        out[i] = cond * a[i] + (1 - cond) * b[i]
    return out


def compute_swap_step(current32, target32, liquidity32, amount_sign, magnitude32, fee_pips):
    # EXACT port of v3 SwapMath.computeSwapStep. See the module docstring for the
    # v3 source and the in-circuit branch derivations.
    #
    # current32 / target32 : uint160 sqrt prices (uint256-encoded, 32 limbs, the
    #     caller asserts they fit uint160). liquidity32: uint256-encoded uint128.
    #     amount_sign in {0,1}; magnitude32: uint256 = |amountRemaining|.
    #     fee_pips: a field value asserted < FEE_DENOM by the caller helper.
    # Returns (next32, amountIn32, amountOut32, feeAmount32), all 32-limb uint256
    # (next32 is the uint160 sqrtRatioNext zero-extended).
    assert amount_sign * (1 - amount_sign) == 0  # sign boolean

    fee_pips_u256, denom_comp_u256 = assert_fee_pips_lt_denom(fee_pips)
    fee_denom_u256 = make_fee_denom_u256()

    range_check_limbs(current32, UINT256_LIMBS)
    range_check_limbs(target32, UINT256_LIMBS)
    range_check_limbs(magnitude32, UINT256_LIMBS)

    # CANONICAL signed-zero for amountRemaining (Phase-5a carry-in, mirrors the
    # tick==0 canonicalization): a magnitude of 0 forces amount_sign == 0, so the
    # value 0 has a single committed form. Otherwise (sign=1, mag=0) and
    # (sign=0, mag=0) both decode to 0 -> commitment malleability; the swap loop
    # must decode amountRemaining = (1 - 2*sign)*magnitude, never compare raw
    # (sign, magnitude). _is_bignum_nonzero(magnitude) is the proven nonzero flag
    # (count in [0, 32] < 2^16, so the constraint is sound).
    mag_nonzero = checked_less_than(0, _is_bignum_nonzero(magnitude32, UINT256_LIMBS))
    assert amount_sign * (1 - mag_nonzero) == 0

    # zeroForOne = current >= target ; exactIn = amount_sign == 0.
    zero_for_one = 1 - bignum_lt(current32, target32, UINT256_LIMBS)
    exact_in = 1 - amount_sign

    # Output buffers (written exactly once, on the taken top-level branch).
    next_out = Array(UINT256_LIMBS)
    in_out = Array(UINT256_LIMBS)
    out_out = Array(UINT256_LIMBS)
    fee_out = Array(UINT256_LIMBS)

    if exact_in == 1:
        # ---- exactIn branch ----
        # amountRemainingLessFee = mulDiv(magnitude, FEE_DENOM - feePips, FEE_DENOM)
        arlf = mulDiv(magnitude32, denom_comp_u256, fee_denom_u256)  # 1 pair

        # stage-1 amountIn (roundUp). zfo: getAmount0Delta(target,current); else getAmount1Delta(current,target).
        amount_in_s1: Imm
        if zero_for_one == 1:
            amount_in_s1 = get_amount0_delta(target32, current32, liquidity32, 1)  # 2 pairs
        else:
            amount_in_s1 = get_amount1_delta(current32, target32, liquidity32, 1)  # 1 pair

        # reach = arlf >= amountIn_s1.
        reach = bignum_le_flag(amount_in_s1, arlf, UINT256_LIMBS)

        # sqrtRatioNext = reach ? target : getNextSqrtPriceFromInput(current, L, arlf, zfo).
        next32: Imm
        if reach == 1:
            next32 = _copy_u256(target32)  # 0 pairs
        else:
            np20 = get_next_sqrt_price_from_input(current32, liquidity32, arlf, zero_for_one)
            next32 = bignum_cast(np20, SQRT_RESULT_LIMBS, UINT256_LIMBS)

        # max = target == sqrtRatioNext.
        max_reached = bignum_eq_flag(target32, next32, UINT256_LIMBS)

        # stage-2 amountIn: exactIn => (max ? amountIn_s1 : recompute at next).
        amount_in_final: Imm
        if max_reached == 1:
            amount_in_final = _copy_u256(amount_in_s1)  # 0 pairs
        else:
            if zero_for_one == 1:
                amount_in_final = get_amount0_delta(next32, current32, liquidity32, 1)  # 2 pairs
            else:
                amount_in_final = get_amount1_delta(current32, next32, liquidity32, 1)  # 1 pair

        # stage-2 amountOut: exactIn => (max && !exactIn) is false => ALWAYS recompute.
        amount_out_final: Imm
        if zero_for_one == 1:
            amount_out_final = get_amount1_delta(next32, current32, liquidity32, 0)  # 1 pair
        else:
            amount_out_final = get_amount0_delta(current32, next32, liquidity32, 0)  # 2 pairs

        # No output cap on the exactIn path.
        # Fee (v3: uint256(amountRemaining) - amountIn; on exactIn amountRemaining
        # == magnitude >= 0, so this is magnitude - amountIn):
        #   (next != target) ? magnitude - amountIn
        #                    : mulDivRoundingUp(amountIn, feePips, FEE_DENOM-feePips).
        # next != target  <=>  max_reached == 0.
        fee_final: Imm
        if max_reached == 0:
            # magnitude - amountIn ; bignum_sub asserts magnitude >= amountIn (top
            # borrow 0) — the in-circuit form of v3's checked subtraction, which
            # would revert if amountIn > magnitude (v3 invariant on this path).
            sub = Array(UINT256_LIMBS)
            bignum_sub(magnitude32, amount_in_final, sub, UINT256_LIMBS)  # 0 pairs
            fee_final = sub
        else:
            fee_final = mulDivRoundingUp(amount_in_final, fee_pips_u256, denom_comp_u256)  # 1 pair

        for i in unroll(0, UINT256_LIMBS):
            next_out[i] = next32[i]
            in_out[i] = amount_in_final[i]
            out_out[i] = amount_out_final[i]
            fee_out[i] = fee_final[i]
    else:
        # ---- exactOut branch (amount_sign == 1) ----
        # stage-1 amountOut (roundDown). zfo: getAmount1Delta(target,current); else getAmount0Delta(current,target).
        amount_out_s1: Imm
        if zero_for_one == 1:
            amount_out_s1 = get_amount1_delta(target32, current32, liquidity32, 0)  # 1 pair
        else:
            amount_out_s1 = get_amount0_delta(current32, target32, liquidity32, 0)  # 2 pairs

        # reach = magnitude >= amountOut_s1.
        reach = bignum_le_flag(amount_out_s1, magnitude32, UINT256_LIMBS)

        # sqrtRatioNext = reach ? target : getNextSqrtPriceFromOutput(current, L, magnitude, zfo).
        next32: Imm
        if reach == 1:
            next32 = _copy_u256(target32)  # 0 pairs
        else:
            np20 = get_next_sqrt_price_from_output(current32, liquidity32, magnitude32, zero_for_one)
            next32 = bignum_cast(np20, SQRT_RESULT_LIMBS, UINT256_LIMBS)

        # max = target == sqrtRatioNext.
        max_reached = bignum_eq_flag(target32, next32, UINT256_LIMBS)

        # stage-2 amountIn: exactOut => (max && exactIn) is false => ALWAYS recompute.
        amount_in_final: Imm
        if zero_for_one == 1:
            amount_in_final = get_amount0_delta(next32, current32, liquidity32, 1)  # 2 pairs
        else:
            amount_in_final = get_amount1_delta(current32, next32, liquidity32, 1)  # 1 pair

        # stage-2 amountOut: exactOut => (max ? amountOut_s1 : recompute at next).
        amount_out_recomp: Imm
        if max_reached == 1:
            amount_out_recomp = _copy_u256(amount_out_s1)  # 0 pairs
        else:
            if zero_for_one == 1:
                amount_out_recomp = get_amount1_delta(next32, current32, liquidity32, 0)  # 1 pair
            else:
                amount_out_recomp = get_amount0_delta(current32, next32, liquidity32, 0)  # 2 pairs

        # Cap output: if amountOut > magnitude: amountOut = magnitude. (exactOut path)
        # cap = magnitude < amountOut  (1 iff amountOut > magnitude). Limbwise select.
        cap = bignum_lt(magnitude32, amount_out_recomp, UINT256_LIMBS)
        amount_out_final = _select_u256(cap, magnitude32, amount_out_recomp)  # 0 pairs

        # Fee: exactIn false => feeAmount = mulDivRoundingUp(amountIn, feePips, FEE_DENOM-feePips).
        fee_final = mulDivRoundingUp(amount_in_final, fee_pips_u256, denom_comp_u256)  # 1 pair

        for i in unroll(0, UINT256_LIMBS):
            next_out[i] = next32[i]
            in_out[i] = amount_in_final[i]
            out_out[i] = amount_out_final[i]
            fee_out[i] = fee_final[i]

    return next_out, in_out, out_out, fee_out
