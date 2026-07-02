from snark_lib import *
from bignum import *
from full_math import *
from sqrt_price_math import *
from liquidity_math import *
from tick_math import *
from bit_math import *
from tick_bitmap import *
from swap_math import *

# ============================================================================
# Uniswap V3 Pool.swap core loop, mirrored from v3-core UniswapV3Pool.sol (EXACT).
# ============================================================================
#
# v3-core UniswapV3Pool.swap (the core loop, verbatim shape):
#
#   while (state.amountSpecifiedRemaining != 0 && state.sqrtPriceX96 != sqrtPriceLimitX96) {
#       StepComputations memory step;
#       step.sqrtPriceStartX96 = state.sqrtPriceX96;
#       (step.tickNext, step.initialized) = tickBitmap.nextInitializedTickWithinOneWord(
#           state.tick, tickSpacing, zeroForOne);
#       if (step.tickNext < MIN_TICK)      step.tickNext = MIN_TICK;
#       else if (step.tickNext > MAX_TICK) step.tickNext = MAX_TICK;
#       step.sqrtPriceNextX96 = TickMath.getSqrtRatioAtTick(step.tickNext);
#       (state.sqrtPriceX96, step.amountIn, step.amountOut, step.feeAmount) =
#           SwapMath.computeSwapStep(
#               state.sqrtPriceX96,
#               (zeroForOne ? step.sqrtPriceNextX96 < sqrtPriceLimitX96
#                           : step.sqrtPriceNextX96 > sqrtPriceLimitX96)
#                   ? sqrtPriceLimitX96 : step.sqrtPriceNextX96,
#               state.liquidity, state.amountSpecifiedRemaining, fee);
#       if (exactInput) {
#           state.amountSpecifiedRemaining -= (step.amountIn + step.feeAmount).toInt256();
#           state.amountCalculated = state.amountCalculated.sub(step.amountOut.toInt256());
#       } else {
#           state.amountSpecifiedRemaining += step.amountOut.toInt256();
#           state.amountCalculated = state.amountCalculated.add(
#               (step.amountIn + step.feeAmount).toInt256());
#       }
#       if (state.sqrtPriceX96 == step.sqrtPriceNextX96) {
#           if (step.initialized) {
#               int128 liquidityNet = ticks.cross(step.tickNext, ...);   // tick data
#               if (zeroForOne) liquidityNet = -liquidityNet;
#               state.liquidity = LiquidityMath.addDelta(state.liquidity, liquidityNet);
#           }
#           state.tick = zeroForOne ? step.tickNext - 1 : step.tickNext;
#       } else if (state.sqrtPriceX96 != step.sqrtPriceStartX96) {
#           state.tick = TickMath.getTickAtSqrtRatio(state.sqrtPriceX96);
#       }
#   }
#   (amount0, amount1) = zeroForOne == exactInput
#       ? (amountSpecified - state.amountSpecifiedRemaining, state.amountCalculated)
#       : (state.amountCalculated, amountSpecified - state.amountSpecifiedRemaining);
#
# ----------------------------------------------------------------------------
# IN-CIRCUIT TRANSLATION — soundness summary (the load-bearing invariants)
# ----------------------------------------------------------------------------
#
# CARRIED STATE (write-once buffers threaded across the runtime `range` loop):
#   A runtime `range` body cannot mutate outer variables and cannot `return`. We
#   thread every carried state variable through a read-only buffer indexed by the
#   step number: state_at_step[i] -> state_at_step[i+1]. The carried state is:
#     sqrtPriceX96  (uint160, uint256-encoded 32 limbs)   -> SP_BUF
#     tick          (sign+abs: 1 + 1 cells)               -> TICK_SIGN_BUF/TICK_ABS_BUF
#     liquidity     (uint128, uint256-encoded 32 limbs)   -> LIQ_BUF
#     amountSpecifiedRemaining (int256 sign+mag: 1 + 32)  -> REM_SIGN_BUF/REM_MAG_BUF
#     amountCalculated         (int256 sign+mag: 1 + 32)  -> CALC_SIGN_BUF/CALC_MAG_BUF
#     commitment digest accumulator (8 cells)             -> DIG_BUF
#
# VERIFIED TERMINATION (binds the prover-supplied num_steps EXACTLY):
#   * At the START of each EXECUTED step we assert the v3 loop guard held:
#         amountSpecifiedRemaining != 0  AND  sqrtPriceX96 != sqrtPriceLimitX96
#     (so no step ran past the natural end of the loop — no OVER-run).
#   * AFTER the loop we assert the guard is now FALSE:
#         amountSpecifiedRemaining == 0  OR  sqrtPriceX96 == sqrtPriceLimitX96
#     (so the loop did not stop EARLY — no UNDER-run).
#   Together these pin num_steps to the real v3 iteration count; a malicious
#   prover cannot lengthen or shorten the loop.
#
# UNIFIED POOL-STATE COMMITMENT (binds everything the prover could vary):
#   Each per-step `word[i]` (the bitmap word fed to nextInitializedTick) and
#   `liquidityNet[i]` (the crossed-tick net liquidity) are HINTED and therefore
#   prover-controlled. We Poseidon-fold each into a running digest as it is
#   consumed, so the final public input is about a SPECIFIC pool/bitmap/tick-data.
#   The swap context, initial state, and final results are folded by the caller
#   (main.py SWAP mode). An unbound per-step word/liquidityNet would let the
#   prover invent pool state -> CRITICAL, hence the per-step fold here.
#
# SIGNED INT256 ACCUMULATION (sign+magnitude, see signed_add_u256):
#   amountSpecifiedRemaining and amountCalculated are int256 in sign+magnitude
#   form. Add/subtract are done by signed_add_u256, which compares magnitudes and
#   does the proper bignum add or sub, tracking the result sign canonically
#   (zero -> sign 0).
#
# ----------------------------------------------------------------------------
# PER-STEP HINT CONSUMPTION ORDER (must match src/reference.rs swap emitter)
# ----------------------------------------------------------------------------
#   Read AT THE TOP of each step body, BEFORE any sub-op:
#     "word"          : 32 cells  (bitmap word for nextInitializedTick; COMMITTED)
#     "bit_pos"       : 1 cell    (msb/lsb hint for nextInitializedTick's BitMath)
#     "tick_next_sign": 1 cell    (sign of getTickAtSqrtRatio answer, used on the
#                                  "moved within range" tick-recompute branch)
#     "tick_next_abs" : 1 cell    (|tick| of that answer)
#     "tnp1_sign"     : 1 cell    (sign of tick+1 for the getTickAtSqrtRatio bucket)
#     "tnp1_abs"      : 1 cell    (|tick+1|)
#     "liq_net_sign"  : 1 cell    (sign of crossed-tick liquidityNet; COMMITTED)
#     "liq_net_mag"   : 32 cells  (|liquidityNet| as uint128/uint256; COMMITTED)
#   then the muldiv (q, r) pairs the step's computeSwapStep + getSqrtRatioAtTick +
#   getTickAtSqrtRatio sub-ops consume, FIFO per label, in guest call order.
#   (Per-step muldiv counts are data-dependent but flag-determined; the reference
#   replays the SAME derived flags so host/guest pair order and count match.)
# ============================================================================

# Encoding sizes for the carried state.
SP_LIMBS = 32          # sqrtPriceX96 as uint256
LIQ_LIMBS = 32         # liquidity as uint256 (uint128 value)
AMT_LIMBS = 32         # int256 magnitudes as uint256
DIGEST_LEN_SWAP = 8

# MIN_TICK is -MAX_TICK (TICK encoding: sign 1, abs MAX_TICK).
MIN_TICK_ABS = MAX_TICK  # MAX_TICK is imported from tick_math

# next_val (signed-field tick from nextInitializedTick) decode window.
NEXT_VAL_BITS = 25       # |next_val| < 2^25 (tick_bitmap bound)
NEXT_VAL_OFF = 33554432  # 2^25: shift window so next_val + OFF in (0, 2^26)
NEXT_VAL_OFF_BITS = 26


def _copy_limbs(src, n: Const):
    # Fresh n-limb copy of a (range-checked) bignum.
    range_check_limbs(src, n)
    out = Array(n)
    for i in unroll(0, n):
        out[i] = src[i]
    return out


def _is_u256_nonzero(x):
    # Boolean: 1 iff the 32-limb x is nonzero. Sound (count in [0,32] < 2^16).
    return checked_less_than(0, _is_bignum_nonzero(x, UINT256_LIMBS))


# ----------------------------------------------------------------------------
# Signed int256 add on sign+magnitude encoding (both magnitudes uint256).
# ----------------------------------------------------------------------------


def signed_add_u256(sa, ma, sb, mb):
    # Return (sr, mr) = (sa,ma) + (sb,mb) as int256 sign+magnitude.
    #   sa/sb in {0,1} (0 = nonnegative, 1 = negative); ma/mb uint256 magnitudes.
    # Canonical result: zero magnitude -> sign 0.
    #
    # Same sign: mr = ma + mb (no overflow allowed -> carry-out 0), sr = sa.
    # Diff sign: let big = max(ma,mb), small = min, mr = big - small,
    #            sr = (sign of the larger magnitude) unless mr == 0 (then 0).
    # All branches are derived booleans (no hints): a_lt_b decides which is larger.
    range_check_limbs(ma, UINT256_LIMBS)
    range_check_limbs(mb, UINT256_LIMBS)
    assert sa * (1 - sa) == 0
    assert sb * (1 - sb) == 0

    same_sign = _is_eq_small(sa, sb)  # 1 iff sa == sb

    mr = Array(UINT256_LIMBS)
    sr: Imm
    if same_sign == 1:
        # mr = ma + mb, require no uint256 overflow (carry-out 0).
        carry = bignum_add(ma, mb, mr, UINT256_LIMBS)
        assert carry == 0
        sr = sa
    else:
        # opposite signs: subtract the smaller magnitude from the larger.
        a_lt_b = bignum_lt(ma, mb, UINT256_LIMBS)  # 1 iff ma < mb
        if a_lt_b == 1:
            # mb > ma: mr = mb - ma, sign follows b.
            bignum_sub(mb, ma, mr, UINT256_LIMBS)
            mr_nz = _is_u256_nonzero(mr)
            sr = sb * mr_nz  # canonical zero -> sign 0
        else:
            # ma >= mb: mr = ma - mb, sign follows a.
            bignum_sub(ma, mb, mr, UINT256_LIMBS)
            mr_nz = _is_u256_nonzero(mr)
            sr = sa * mr_nz
    return sr, mr


def _add_in_fee(amount_in32, fee32):
    # amountIn + feeAmount as a uint256 (require no overflow). Both nonnegative.
    s = Array(UINT256_LIMBS)
    carry = bignum_add(amount_in32, fee32, s, UINT256_LIMBS)
    assert carry == 0
    return s


# ----------------------------------------------------------------------------
# Per-step commitment fold: absorb a 32-limb word (4 eight-cell chunks) into the
# running 8-cell digest, then the liquidityNet sign+mag (1+32 cells, padded to 40
# = 5 chunks). Uses the SAME poseidon16_compress_half left-fold as commit_fold.
# ----------------------------------------------------------------------------


def _fold8(acc, chunk8):
    # acc' = poseidon16_compress_half(acc, chunk8) -> fresh 8-cell digest.
    out = Array(DIGEST_LEN_SWAP)
    poseidon16_compress_half(acc, chunk8, out)
    return out


def fold_step_data(acc_in, word, liq_net_sign, liq_net_mag):
    # Fold the per-step prover-controlled data into the digest accumulator.
    # Layout absorbed (every consumed value is bound):
    #   word (32 cells = 4 chunks), then [liq_net_sign | liq_net_mag(32) | 0*7]
    #   (40 cells = 5 chunks; the 8-cell-aligned padding keeps the fold canonical).
    # Returns the new 8-cell accumulator.
    range_check_limbs(word, UINT256_LIMBS)
    acc: Mut = acc_in
    # 4 word chunks.
    for c in unroll(0, 4):
        chunk = word + c * DIGEST_LEN_SWAP
        acc = _fold8(acc, chunk)
    # liquidityNet block: [sign, mag[0..32], 0*7] = 40 cells = 5 chunks.
    lblock = Array(5 * DIGEST_LEN_SWAP)
    lblock[0] = liq_net_sign
    for i in unroll(0, UINT256_LIMBS):
        lblock[1 + i] = liq_net_mag[i]
    for i in unroll(1 + UINT256_LIMBS, 5 * DIGEST_LEN_SWAP):
        lblock[i] = 0
    for c in unroll(0, 5):
        chunk = lblock + c * DIGEST_LEN_SWAP
        acc = _fold8(acc, chunk)
    return acc


# ----------------------------------------------------------------------------
# bignum max / min via bignum_lt-select (for the price-limit clamp).
# ----------------------------------------------------------------------------


def _select32(cond, a, b):
    # cond ? a : b, limbwise, both 32-limb (range-checked). cond a proven boolean.
    range_check_limbs(a, UINT256_LIMBS)
    range_check_limbs(b, UINT256_LIMBS)
    out = Array(UINT256_LIMBS)
    for i in unroll(0, UINT256_LIMBS):
        out[i] = cond * a[i] + (1 - cond) * b[i]
    return out


def bignum_max(a, b):
    # max(a, b) as 32-limb. a_lt_b == 1 -> b, else a.
    a_lt_b = bignum_lt(a, b, UINT256_LIMBS)
    return _select32(a_lt_b, b, a)


def bignum_min(a, b):
    # min(a, b) as 32-limb. a_lt_b == 1 -> a, else b.
    a_lt_b = bignum_lt(a, b, UINT256_LIMBS)
    return _select32(a_lt_b, a, b)


# ----------------------------------------------------------------------------
# Signed-field tick decode and tickNext clamp to [MIN_TICK, MAX_TICK].
# ----------------------------------------------------------------------------
#
# next_val (from nextInitializedTick) is a SIGNED field element: nonnegative ticks
# are their own value, negative ticks are p - |tick|. tick_bitmap guarantees
# |next_val| < 2^25 (= NEXT_VAL_BITS bits). We first decode next_val into a sign
# and a 25-bit magnitude, then clamp to [MIN_TICK, MAX_TICK].

def clamp_tick_next(next_val):
    # v3:  if tickNext < MIN_TICK: MIN_TICK; elif tickNext > MAX_TICK: MAX_TICK.
    # Returns the clamped tick as (sign, abs) with 0 <= abs <= MAX_TICK, canonical.
    #
    # Strategy: shift next_val into the nonnegative window by +2^25, bit-decompose
    # (binds |next_val| < 2^25 AND nonnegativity of the shifted value), then derive
    #   below_min = shifted < (2^25 + MIN_TICK)  = (2^25 - MAX_TICK)
    #   above_max = shifted > (2^25 + MAX_TICK)
    # from compile-time bound comparisons against the 26-bit shifted value. Every
    # comparison is done via a fresh 26-bit gap decomposition (the only sound way
    # to compare values > 2^16). The clamped signed tick is then assembled and the
    # in-range case decoded into (sign, abs).
    shifted = next_val + NEXT_VAL_OFF  # honest value in (2^25 - (MAX_TICK+ε), 2^25 + (MAX_TICK+ε))
    _sbits = checked_decompose_bits_small_value_const(shifted, NEXT_VAL_OFF_BITS)  # 0 <= shifted < 2^26

    lo_img = NEXT_VAL_OFF - MAX_TICK   # image of MIN_TICK; in (0, 2^26)
    hi_img = NEXT_VAL_OFF + MAX_TICK   # image of MAX_TICK; in (0, 2^26)

    below_min = _wide_lt(shifted, lo_img)        # shifted <  lo_img
    above_max = _wide_lt(hi_img + 1, shifted + 1)  # hi_img < shifted  <=>  hi_img+1 < shifted+1
    assert below_min * above_max == 0  # mutually exclusive (lo_img <= hi_img)
    in_range = (1 - below_min) * (1 - above_max)

    # In-range decode of next_val. GUARDED: assert_le_max_tick(abs_v) would fail
    # for an out-of-range tick, so decode only on the in_range arm; on the clamped
    # arms (sign, abs) come from the MIN/MAX_TICK constants.
    clamped_sign: Imm
    clamped_abs: Imm
    if in_range == 1:
        nv_sign, nv_abs = decode_in_range_tick(next_val, shifted)
        clamped_sign = nv_sign
        clamped_abs = nv_abs
    else:
        # below_min -> MIN_TICK (sign 1) ; above_max -> MAX_TICK (sign 0).
        clamped_sign = below_min
        clamped_abs = MAX_TICK
    return clamped_sign, clamped_abs


def _wide_lt(a, b):
    # 1 iff a < b, for a, b nonnegative field values < 2^26. Hint the boolean and
    # bind it via a 26-bit decomposition of the nonnegative gap on the taken side.
    res: Imm
    hint_less_than(a, b, res)  # UNCONSTRAINED boolean (operands may exceed 2^16)
    assert res * (1 - res) == 0
    if res == 1:
        _ = checked_decompose_bits_small_value_const(b - 1 - a, NEXT_VAL_OFF_BITS)  # a < b => b-1-a >= 0, < 2^26
    else:
        _ = checked_decompose_bits_small_value_const(a - b, NEXT_VAL_OFF_BITS)      # a >= b => a-b >= 0, < 2^26
    return res


def decode_in_range_tick(next_val, shifted):
    # Decode next_val (a valid tick in [MIN_TICK, MAX_TICK] on the in-range branch)
    # into (sign, abs) with abs <= MAX_TICK, canonical, and next_val == abs -
    # 2*sign*abs. `shifted = next_val + 2^25` is already bound to 26 bits by the
    # caller. We hint (sign, abs) and bind: abs <= MAX_TICK (bit-decomp) + the
    # equality next_val == abs - 2*sign*abs (integer-exact, |next_val| < 2^20 << p).
    sign: Imm
    abs_v: Imm
    # Derive sign/abs from a div hint on `shifted`: q = floor(shifted / 2^25) in
    # {0,1} for shifted in [2^25 - MAX_TICK, 2^25 + MAX_TICK]; q==1 -> next_val>=0.
    q: Imm
    r: Imm
    hint_div_floor(shifted, NEXT_VAL_OFF, q, r)
    assert q * (1 - q) == 0
    _ = checked_decompose_bits_small_value_const(r, NEXT_VAL_BITS)  # 0 <= r < 2^25 (sound; r is the remainder)
    assert q * NEXT_VAL_OFF + r == shifted  # binds q, r (shifted < 2^26 < p)
    sign = 1 - q
    abs_v = q * r + (1 - q) * (NEXT_VAL_OFF - r)  # q==1: next_val=r ; q==0: |next_val|=2^25-r
    bits = assert_le_max_tick(abs_v)  # 0 <= abs_v <= MAX_TICK
    any_bit: Mut = 0
    for i in unroll(0, TICK_ABS_BITS):
        any_bit = any_bit + bits[i]
    abs_nonzero = checked_less_than(0, any_bit)
    assert sign * (1 - abs_nonzero) == 0  # canonical zero
    assert next_val == abs_v - 2 * sign * abs_v  # bind (sign, abs) to next_val
    return sign, abs_v


# ----------------------------------------------------------------------------
# One swap step (the body of v3's while loop). Reads carried state from the
# buffers at index `i`, consumes the per-step hints, writes carried state at
# index `i+1`. Returns nothing (all output via the write-once buffers).
# ----------------------------------------------------------------------------


def _read32(buf, i):
    # Read the 32-limb slice buf[i*32 .. i*32+32] as a fresh range-checked bignum.
    out = Array(UINT256_LIMBS)
    base = i * UINT256_LIMBS
    for k in unroll(0, UINT256_LIMBS):
        out[k] = buf[base + k]
    range_check_limbs(out, UINT256_LIMBS)
    return out


def _write32(buf, i, val):
    base = i * UINT256_LIMBS
    for k in unroll(0, UINT256_LIMBS):
        buf[base + k] = val[k]
    return


def swap_step(
    i,
    sp_buf, tsign_buf, tabs_buf, liq_buf,
    rem_sign_buf, rem_mag_buf, calc_sign_buf, calc_mag_buf, dig_buf,
    zero_for_one, amount_sign, fee_pips, tick_spacing, sqrt_limit32,
):
    # --- carried state at step i ---
    sqrt_p = _read32(sp_buf, i)
    tick_sign = tsign_buf[i]
    tick_abs = tabs_buf[i]
    liquidity = _read32(liq_buf, i)
    rem_sign = rem_sign_buf[i]
    rem_mag = _read32(rem_mag_buf, i)
    calc_sign = calc_sign_buf[i]
    calc_mag = _read32(calc_mag_buf, i)
    acc_in = Array(DIGEST_LEN_SWAP)
    for k in unroll(0, DIGEST_LEN_SWAP):
        acc_in[k] = dig_buf[i * DIGEST_LEN_SWAP + k]

    # --- VERIFIED TERMINATION (guard held at the START of this executed step) ---
    #   amountSpecifiedRemaining != 0  AND  sqrtPriceX96 != sqrtPriceLimitX96.
    rem_nz = _is_u256_nonzero(rem_mag)  # remaining != 0 iff its magnitude != 0
    assert rem_nz == 1
    sp_eq_limit = bignum_eq_flag(sqrt_p, sqrt_limit32, UINT256_LIMBS)
    assert sp_eq_limit == 0  # sqrtPriceX96 != limit

    # --- per-step prover hints (consumed in a FIXED order; see module docstring) ---
    word = Array(UINT256_LIMBS)
    hint_witness("word", word)
    bit_pos_buf = Array(1)
    hint_witness("bit_pos", bit_pos_buf)
    bit_pos = bit_pos_buf[0]
    tn_sign_buf = Array(1)
    hint_witness("tick_next_sign", tn_sign_buf)
    tn_sign = tn_sign_buf[0]
    tn_abs_buf = Array(1)
    hint_witness("tick_next_abs", tn_abs_buf)
    tn_abs = tn_abs_buf[0]
    tnp1_sign_buf = Array(1)
    hint_witness("tnp1_sign", tnp1_sign_buf)
    tnp1_sign = tnp1_sign_buf[0]
    tnp1_abs_buf = Array(1)
    hint_witness("tnp1_abs", tnp1_abs_buf)
    tnp1_abs = tnp1_abs_buf[0]
    liq_net_sign_buf = Array(1)
    hint_witness("liq_net_sign", liq_net_sign_buf)
    liq_net_sign = liq_net_sign_buf[0]
    liq_net_mag = Array(UINT256_LIMBS)
    hint_witness("liq_net_mag", liq_net_mag)

    # --- fold per-step prover data into the running commitment digest ---
    acc_out = fold_step_data(acc_in, word, liq_net_sign, liq_net_mag)
    for k in unroll(0, DIGEST_LEN_SWAP):
        dig_buf[(i + 1) * DIGEST_LEN_SWAP + k] = acc_out[k]

    # --- step.tickNext / initialized = nextInitializedTickWithinOneWord(...) ---
    # lte == zeroForOne (v3 calls it with lte = zeroForOne).
    next_val, initialized = next_initialized_tick_within_one_word(
        tick_sign, tick_abs, tick_spacing, zero_for_one, word, bit_pos
    )

    # --- clamp tickNext into [MIN_TICK, MAX_TICK] (returns sign, abs) ---
    cl_sign, cl_abs = clamp_tick_next(next_val)

    # --- sqrtPriceNext = getSqrtRatioAtTick(tickNext_clamped) -> uint160 ---
    sp_next20 = get_sqrt_ratio_at_tick(cl_sign, cl_abs)
    sp_next32 = bignum_cast(sp_next20, SQRT_RESULT_LIMBS, UINT256_LIMBS)

    # --- target = zeroForOne ? max(sp_next, limit) : min(sp_next, limit) ---
    target32: Imm
    if zero_for_one == 1:
        target32 = bignum_max(sp_next32, sqrt_limit32)
    else:
        target32 = bignum_min(sp_next32, sqrt_limit32)

    # --- computeSwapStep(current, target, liquidity, amountRemaining, fee) ---
    # amountRemaining = (rem_sign, rem_mag) sign+magnitude (already canonical).
    new_sp32, amount_in, amount_out, fee_amount = compute_swap_step(
        sqrt_p, target32, liquidity, rem_sign, rem_mag, fee_pips
    )

    # --- signed amount accumulation ---
    exact_input = 1 - amount_sign
    new_rem_sign: Imm
    new_calc_sign: Imm
    new_rem_mag = Array(UINT256_LIMBS)
    new_calc_mag = Array(UINT256_LIMBS)
    if exact_input == 1:
        # amountSpecifiedRemaining -= (amountIn + feeAmount); amountCalculated -= amountOut.
        spent = _add_in_fee(amount_in, fee_amount)
        rs, rm = signed_add_u256(rem_sign, rem_mag, 1, spent)  # remaining + (-(in+fee))
        cs, cm = signed_add_u256(calc_sign, calc_mag, 1, amount_out)  # calc + (-out)
        new_rem_sign = rs
        new_calc_sign = cs
        for k in unroll(0, UINT256_LIMBS):
            new_rem_mag[k] = rm[k]
            new_calc_mag[k] = cm[k]
    else:
        # amountSpecifiedRemaining += amountOut; amountCalculated += (amountIn + feeAmount).
        gained = _add_in_fee(amount_in, fee_amount)
        rs, rm = signed_add_u256(rem_sign, rem_mag, 0, amount_out)  # remaining + (+out)
        cs, cm = signed_add_u256(calc_sign, calc_mag, 0, gained)    # calc + (+(in+fee))
        new_rem_sign = rs
        new_calc_sign = cs
        for k in unroll(0, UINT256_LIMBS):
            new_rem_mag[k] = rm[k]
            new_calc_mag[k] = cm[k]

    # --- tick transition ---
    # crossed = (new_sp == sp_next) ; moved = (new_sp != start_sp).
    crossed = bignum_eq_flag(new_sp32, sp_next32, UINT256_LIMBS)
    moved = 1 - bignum_eq_flag(new_sp32, sqrt_p, UINT256_LIMBS)

    # liquidity update: only on crossed && initialized. liquidityNet (hinted,
    # committed) is an int128; negate it when zeroForOne, then addDelta.
    # add_delta works on uint128 (16-limb) operands; bind liq_net_mag fits uint128.
    bignum_assert_fits(liq_net_mag, UINT256_LIMBS, UINT128_LIMBS)
    assert liq_net_sign * (1 - liq_net_sign) == 0
    # CANONICAL SIGNED-ZERO for the committed liquidityNet. liq_net_sign and the
    # full 32-limb liq_net_mag are folded into the public input (fold_step_data),
    # but the verifier never re-derives the sign. Without this a prover could flip
    # the committed sign on a crossed+initialized tick whose liquidityNet == 0
    # (or on any filler step) -> a second valid public input for the same swap.
    # Pin sign 0 whenever the (committed, 32-limb) magnitude is zero. This binds
    # the SIGN AS COMMITTED, independent of the do_update / net16 path below.
    liq_net_mag_nz = _is_u256_nonzero(liq_net_mag)
    assert liq_net_sign * (1 - liq_net_mag_nz) == 0
    liq16 = Array(UINT128_LIMBS)
    net16 = Array(UINT128_LIMBS)
    for k in unroll(0, UINT128_LIMBS):
        liq16[k] = liquidity[k]
        net16[k] = liq_net_mag[k]
    # effective sign after the zeroForOne negation: zfo flips the sign. A flip of
    # a zero net stays zero -> keep canonical (sign 0 when |net| == 0).
    flipped_sign = (1 - zero_for_one) * liq_net_sign + zero_for_one * (1 - liq_net_sign)
    net_mag_nz = checked_less_than(0, _is_bignum_nonzero(net16, UINT128_LIMBS))
    eff_net_sign = flipped_sign * net_mag_nz

    do_update = crossed * initialized
    new_liq16: Imm
    if do_update == 1:
        new_liq16 = add_delta(liq16, eff_net_sign, net16)
    else:
        new_liq16 = liq16
    new_liq32 = Array(UINT256_LIMBS)
    for k in unroll(0, UINT128_LIMBS):
        new_liq32[k] = new_liq16[k]
    for k in unroll(UINT128_LIMBS, UINT256_LIMBS):
        new_liq32[k] = 0

    # new tick:
    #   crossed     -> zeroForOne ? tickNext - 1 : tickNext   (clamped tick)
    #   moved (!crossed && new_sp != start) -> getTickAtSqrtRatio(new_sp)
    #   else        -> tick unchanged
    new_tick_sign: Imm
    new_tick_abs: Imm
    if crossed == 1:
        # tickNext is (cl_sign, cl_abs) signed value cl. zfo -> cl - 1, else cl.
        cl_signed = cl_abs - 2 * cl_sign * cl_abs           # signed field value of tickNext
        new_signed = cl_signed - zero_for_one * 1            # zfo ? cl-1 : cl
        # decode new_signed into (sign, abs); |new_signed| <= MAX_TICK+1 < 2^25.
        #
        # KNOWN EDGE — the `MIN_TICK - 1` case (DOCUMENTED, intentionally rejected):
        #   v3 stores `state.tick = tickNext - 1` which, when `tickNext == MIN_TICK`
        #   on a zeroForOne crossing, would be `MIN_TICK - 1` (|tick| = MAX_TICK+1).
        #   decode_in_range_tick asserts `abs_v <= MAX_TICK`, so it REJECTS that
        #   value. This is sound (no honest valid swap can reach it):
        #     * `crossed` (new_sp == sp_next) at `tickNext == MIN_TICK` requires
        #       new_sp == getSqrtRatioAtTick(MIN_TICK) == MIN_SQRT_RATIO.
        #     * A valid v3 swap requires `sqrtPriceLimit > MIN_SQRT_RATIO` (strict;
        #       v3 reverts SPL otherwise), so `target` is clamped to >= limit >
        #       MIN_SQRT_RATIO and `new_sp` (between current and target) can never
        #       equal MIN_SQRT_RATIO. Hence `tickNext == MIN_TICK` cannot be a
        #       crossed step, and `MIN_TICK - 1` is unreachable.
        #   We keep the strict decode (rather than widening abs to MAX_TICK+1)
        #   because the widened tick is never consumed: it would only feed back into
        #   nextInitializedTick / getSqrtRatioAtTick on a SUBSEQUENT step, but the
        #   swap has already terminated at the limit (guard false) by then. Widening
        #   would add an unreachable, harder-to-audit code path for no behavioural
        #   gain. If a future caller needs MIN_TICK-1 storage, widen here AND in
        #   decode_in_range_tick's bound, and re-derive the >=/clamp feed-back proof.
        ds, da = decode_in_range_tick(new_signed, new_signed + NEXT_VAL_OFF)
        new_tick_sign = ds
        new_tick_abs = da
    else:
        if moved == 1:
            # getTickAtSqrtRatio(new_sp): new_sp is uint160. Verify hinted tickNext
            # (tn_sign, tn_abs) is its tick via the bucket inequality. Bind tick+1.
            bignum_assert_fits(new_sp32, UINT256_LIMBS, SQRT_RESULT_LIMBS)
            sp20 = Array(SQRT_RESULT_LIMBS)
            for k in unroll(0, SQRT_RESULT_LIMBS):
                sp20[k] = new_sp32[k]
            # bind tick+1 to tick: signed(t)+1 == signed(t+1).
            sv_t = tn_abs - 2 * tn_sign * tn_abs
            sv_p1 = tnp1_abs - 2 * tnp1_sign * tnp1_abs
            assert sv_t + 1 == sv_p1
            assert tn_sign * (1 - tn_sign) == 0
            assert tnp1_sign * (1 - tnp1_sign) == 0
            _t, _a = get_tick_at_sqrt_ratio(sp20, tn_sign, tn_abs, tnp1_sign, tnp1_abs)
            new_tick_sign = tn_sign
            new_tick_abs = tn_abs
        else:
            new_tick_sign = tick_sign
            new_tick_abs = tick_abs

    # --- write carried state at i+1 ---
    _write32(sp_buf, i + 1, new_sp32)
    tsign_buf[i + 1] = new_tick_sign
    tabs_buf[i + 1] = new_tick_abs
    _write32(liq_buf, i + 1, new_liq32)
    rem_sign_buf[i + 1] = new_rem_sign
    _write32(rem_mag_buf, i + 1, new_rem_mag)
    calc_sign_buf[i + 1] = new_calc_sign
    _write32(calc_mag_buf, i + 1, new_calc_mag)
    return


def pool_swap(
    sp0_32, tick0_sign, tick0_abs, liq0_32,
    amount_sign, amount_mag32, sqrt_limit32, fee_pips, tick_spacing,
    zero_for_one, num_steps, dig0,
):
    # Run the v3 Pool.swap core loop for exactly `num_steps` iterations.
    #
    # Inputs (the swap context + initial pool state, all committed by the caller):
    #   sp0_32      : initial sqrtPriceX96 (uint160, uint256-encoded)
    #   tick0_*     : initial tick (sign, abs)
    #   liq0_32     : initial liquidity (uint128, uint256-encoded)
    #   amount_sign : sign of amountSpecified (0 = exactIn, 1 = exactOut)
    #   amount_mag32: |amountSpecified| (uint256)
    #   sqrt_limit32: sqrtPriceLimitX96 (uint160, uint256-encoded)
    #   fee_pips    : the pool fee (< FEE_DENOM)
    #   tick_spacing: tickSpacing (1 <= ts < 2^16)
    #   zero_for_one: swap direction (boolean)
    #   num_steps   : prover-supplied loop bound (VERIFIED to be the real count)
    #   dig0        : initial 8-cell commitment digest accumulator (from the caller,
    #                 already binding the context + initial state)
    # Returns (final_sp32, final_tick_sign, final_tick_abs, final_liq32,
    #          rem_sign, rem_mag32, calc_sign, calc_mag32, amount0_*, amount1_*,
    #          final_digest8): the final state, the signed leftovers, the resulting
    #          amount0/amount1 (sign+mag), and the final commitment digest.
    assert zero_for_one * (1 - zero_for_one) == 0
    assert amount_sign * (1 - amount_sign) == 0
    # CANONICAL SIGNED-ZERO for amountSpecified (mirrors swap_math.py Phase-5a):
    # the verifier never re-derives amount_sign, so without this a prover could
    # commit amount_sign=1 with amount_mag==0 (a no-op swap, num_steps==0), yielding
    # a second valid public input for the same semantic swap -> commitment
    # malleability. Pin sign 0 whenever the magnitude is zero.
    amount_mag_nz = _is_u256_nonzero(amount_mag32)
    assert amount_sign * (1 - amount_mag_nz) == 0

    # ------------------------------------------------------------------------
    # UNCONDITIONAL COMMITTED-METADATA BINDING (Phase-6 malleability fix).
    # Every field folded into the public input (in main.run_swap head0 + here)
    # MUST be fully constrained at or before the point it enters the commitment,
    # NEVER gated by the per-step loop body (which is skipped at num_steps == 0
    # or on filler steps a prover could try to shorten). The per-step sub-ops
    # (next_initialized_tick_within_one_word, compute_swap_step, ...) do validate
    # tick0/fee_pips/tick_spacing, but ONLY when at least one step executes; a
    # num_steps == 0 (no-op) swap would otherwise let a prover commit arbitrary
    # values for these head scalars -> a forgeable second public input.
    #
    # tick0 (head0[6]=tick0_sign, head0[7]=tick0_abs):
    #   * sign boolean,
    #   * 0 <= abs <= MAX_TICK (decompose-the-diff; MAX_TICK > 2^16 so a direct
    #     range-check is unsound — assert_le_max_tick does the bit-decomp),
    #   * canonical signed-zero (abs == 0 -> sign == 0).
    assert tick0_sign * (1 - tick0_sign) == 0
    t0_bits = assert_le_max_tick(tick0_abs)
    t0_any: Mut = 0
    for i in unroll(0, TICK_ABS_BITS):
        t0_any = t0_any + t0_bits[i]
    tick0_abs_nz = checked_less_than(0, t0_any)
    assert tick0_sign * (1 - tick0_abs_nz) == 0

    # fee_pips (head0[3]): 0 <= fee_pips < FEE_DENOM (1e6 > 2^16; decompose-the-gap).
    _fp_u, _dc_u = assert_fee_pips_lt_denom(fee_pips)

    # tick_spacing (head0[4]): 1 <= tick_spacing < 2^16.
    check_tick_spacing(tick_spacing)

    # ------------------------------------------------------------------------
    # num_steps >= 1 reachability (v3 entry-condition parity + defense-in-depth).
    # A valid v3 swap always executes >= 1 step. v3 reverts before the loop on:
    #   * require(amountSpecified != 0, 'AS')   -> amount_mag must be nonzero,
    #   * the SPL check: sqrtPriceLimit is STRICTLY on the correct side of the
    #     current price for the direction, so sp0 != limit.
    # Both make the step-0 loop guard (remaining != 0 AND sqrtP != limit) TRUE,
    # so num_steps == 0 is unreachable for a valid swap. The unconditional tick0
    # binding above is the primary malleability fix; these add v3 parity and
    # close the num_steps == 0 no-op as an attack surface entirely.
    assert amount_mag_nz == 1  # v3 'AS': amountSpecified != 0
    sp0_eq_limit = bignum_eq_flag(sp0_32, sqrt_limit32, UINT256_LIMBS)
    assert sp0_eq_limit == 0  # SPL: limit strictly on the correct side -> sp0 != limit

    n1 = num_steps + 1
    # Carried-state buffers (write-once; index 0 = initial, index num_steps = final).
    sp_buf = Array(n1 * UINT256_LIMBS)
    tsign_buf = Array(n1)
    tabs_buf = Array(n1)
    liq_buf = Array(n1 * UINT256_LIMBS)
    rem_sign_buf = Array(n1)
    rem_mag_buf = Array(n1 * UINT256_LIMBS)
    calc_sign_buf = Array(n1)
    calc_mag_buf = Array(n1 * UINT256_LIMBS)
    dig_buf = Array(n1 * DIGEST_LEN_SWAP)

    # --- initial state at index 0 ---
    _write32(sp_buf, 0, sp0_32)
    tsign_buf[0] = tick0_sign
    tabs_buf[0] = tick0_abs
    _write32(liq_buf, 0, liq0_32)
    # amountSpecifiedRemaining_0 = amountSpecified.
    rem_sign_buf[0] = amount_sign
    _write32(rem_mag_buf, 0, amount_mag32)
    # amountCalculated_0 = 0.
    calc_sign_buf[0] = 0
    zero32 = Array(UINT256_LIMBS)
    for k in unroll(0, UINT256_LIMBS):
        zero32[k] = 0
    _write32(calc_mag_buf, 0, zero32)
    for k in unroll(0, DIGEST_LEN_SWAP):
        dig_buf[k] = dig0[k]

    # --- the loop (runtime range; num_steps verified below) ---
    for i in range(0, num_steps):
        swap_step(
            i,
            sp_buf, tsign_buf, tabs_buf, liq_buf,
            rem_sign_buf, rem_mag_buf, calc_sign_buf, calc_mag_buf, dig_buf,
            zero_for_one, amount_sign, fee_pips, tick_spacing, sqrt_limit32,
        )

    # --- final state at index num_steps ---
    final_sp32 = _read32(sp_buf, num_steps)
    final_tsign = tsign_buf[num_steps]
    final_tabs = tabs_buf[num_steps]
    final_liq32 = _read32(liq_buf, num_steps)
    rem_sign = rem_sign_buf[num_steps]
    rem_mag32 = _read32(rem_mag_buf, num_steps)
    calc_sign = calc_sign_buf[num_steps]
    calc_mag32 = _read32(calc_mag_buf, num_steps)
    final_dig = Array(DIGEST_LEN_SWAP)
    for k in unroll(0, DIGEST_LEN_SWAP):
        final_dig[k] = dig_buf[num_steps * DIGEST_LEN_SWAP + k]

    # --- VERIFIED TERMINATION (guard FALSE after the loop) ---
    #   amountSpecifiedRemaining == 0  OR  sqrtPriceX96 == sqrtPriceLimitX96.
    rem_zero = 1 - _is_u256_nonzero(rem_mag32)
    sp_eq_limit = bignum_eq_flag(final_sp32, sqrt_limit32, UINT256_LIMBS)
    # OR of two booleans: at least one must hold -> their (1-a)*(1-b) == 0.
    assert (1 - rem_zero) * (1 - sp_eq_limit) == 0

    # --- amount0 / amount1 ---
    # spent = amountSpecified - amountSpecifiedRemaining (signed int256)
    #       = (amount_sign, amount_mag) + (-(rem_sign, rem_mag)).
    # The negated operand sign must be CANONICAL: -R is (neg_sign, rem_mag) with
    # neg_sign = (1 - rem_sign) * (rem_mag != 0)  (negating 0 stays sign 0).
    rem_nz = _is_u256_nonzero(rem_mag32)
    neg_rem_sign = (1 - rem_sign) * rem_nz
    spent_sign, spent_mag = signed_add_u256(amount_sign, amount_mag32, neg_rem_sign, rem_mag32)

    # v3: (amount0, amount1) = zeroForOne == exactInput
    #       ? (spent, amountCalculated) : (amountCalculated, spent).
    exact_input = 1 - amount_sign
    take_spent_as_a0 = _is_eq_small(zero_for_one, exact_input)  # 1 iff zfo == exactInput
    a0_sign = take_spent_as_a0 * spent_sign + (1 - take_spent_as_a0) * calc_sign
    a1_sign = take_spent_as_a0 * calc_sign + (1 - take_spent_as_a0) * spent_sign
    a0_mag = Array(UINT256_LIMBS)
    a1_mag = Array(UINT256_LIMBS)
    for k in unroll(0, UINT256_LIMBS):
        a0_mag[k] = take_spent_as_a0 * spent_mag[k] + (1 - take_spent_as_a0) * calc_mag32[k]
        a1_mag[k] = take_spent_as_a0 * calc_mag32[k] + (1 - take_spent_as_a0) * spent_mag[k]
        assert a0_mag[k] < LIMB_BASE
        assert a1_mag[k] < LIMB_BASE

    return (
        final_sp32, final_tsign, final_tabs, final_liq32,
        rem_sign, rem_mag32, calc_sign, calc_mag32,
        a0_sign, a0_mag, a1_sign, a1_mag, final_dig,
    )
