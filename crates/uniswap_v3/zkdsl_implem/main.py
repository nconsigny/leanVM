from snark_lib import *
from preamble import *
from full_math import *
from sqrt_price_math import *
from liquidity_math import *
from tick_math import *
from bit_math import *
from tick_bitmap import *
from swap_math import *
from swap import *

# ============================================================================
# Phase 2/3/4/5a/5b driver — SqrtPriceMath + LiquidityMath + TickMath + BitMath +
# TickBitmap + SwapMath (op-dispatch mode, ops 0..13, NUM_OPS = 14), plus the
# full Pool.swap multi-tick loop (SWAP mode, mode == 1), end-to-end.
# ============================================================================
#
# The guest:
#   1. fills the preamble,
#   2. reads a single operation selector `op` and a flag `flag_a`, plus the
#      operands sqrtA, sqrtB, liquidity, amount (each uint256 = 32 base-256
#      limbs), the LiquidityMath sign bit `y_sign`, the TickMath scalars, and the
#      Phase-4 scalars (tick_spacing, bit_pos) + bitmap `word`, all as host hints,
#   3. range-checks every input limb (< 256) so they are canonical,
#   4. asserts `op` is in range and dispatches (real VM control flow) to the
#      corresponding v3 function, whose internal asserts are the proven
#      statement,
#   5. binds (head | sqrtA | sqrtB | liquidity | amount | result | word) to the
#      8-cell public input via the same Poseidon fold as Phase 1.
#
# RESULT ENCODING: every op produces a uint256-wide (32-limb) result buffer
# (narrower results are zero-extended), so the commitment layout is fixed. The
# Phase-4 ops also commit scalar answers in the head: MSB/LSB return p (committed
# as `bit_pos`); NextInit returns (next_val signed, initialized) (committed as
# head cells next_val / initialized).
#
# OP TABLE (op selector value -> function; flag_a meaning):
#   0  get_amount0_delta(sqrtA, sqrtB, liquidity, roundUp=flag_a)        -> u256
#   1  get_amount1_delta(sqrtA, sqrtB, liquidity, roundUp=flag_a)        -> u256
#   2  get_next_sqrt_price_from_amount0_rounding_up(sqrtA, liq, amount, add=flag_a) -> u160
#   3  get_next_sqrt_price_from_amount1_rounding_down(sqrtA, liq, amount, add=flag_a) -> u160
#   4  get_next_sqrt_price_from_input(sqrtA, liq, amount, zeroForOne=flag_a)  -> u160
#   5  get_next_sqrt_price_from_output(sqrtA, liq, amount, zeroForOne=flag_a) -> u160
#   6  add_delta(x=liquidity[:16], y_sign, y_mag=amount[:16])            -> u128
#   7  RawMulDiv: floor = mulDiv(sqrtA, sqrtB, amount); also ceil (2nd pair) -> u256
#   8  get_sqrt_ratio_at_tick(tick_sign, abs_tick)                       -> u160
#   9  get_tick_at_sqrt_ratio(sqrtA[:20], tick_sign, abs_tick, sign_p1, abs_p1)
#        -> tick (committed via head cells); result slot echoes input sqrtA.
#   10 most_significant_bit(x=sqrtA, p=bit_pos)   -> p committed in head bit_pos
#   11 least_significant_bit(x=sqrtA, p=bit_pos)  -> p committed in head bit_pos
#   12 next_initialized_tick_within_one_word(tick_sign, abs_tick, tick_spacing,
#        lte=flag_a, word, bit_pos) -> (next_val, initialized) in head
#   13 compute_swap_step(current=sqrtA, target=sqrtB, liquidity, amount_sign,
#        magnitude=amount, fee_pips) -> (sqrtRatioNext=result, amountIn=swap_in,
#        amountOut=swap_out, feeAmount=fee). fee_pips/amount_sign in head[11]/[12].
#
# HINT CONSUMPTION ORDER (must match src/reference.rs):
#   "op", "flag_a"      : 1 cell each
#   "sqrtA","sqrtB","liquidity","amount" : 32 cells each
#   "y_sign"            : 1 cell
#   "tick_sign","abs_tick","sign_p1","abs_p1" : 1 cell each  (TickMath ops 8/9)
#   "tick_spacing"      : 1 cell             (TickBitmap op 12)
#   "bit_pos"           : 1 cell             (BitMath ops 10/11; NextInit msb/lsb)
#   "word"              : 32 cells           (TickBitmap op 12 bitmap; committed)
#   "fee_pips"          : 1 cell             (SwapMath op 13 fee, < FEE_DENOM)
#   "amount_sign"       : 1 cell             (SwapMath op 13 sign of amountRemaining)
#   then "muldiv_q"/"muldiv_r" pairs in the order the dispatched op calls
#   mulDiv/mulDivRoundingUp (FIFO per label). Ops 8/9 emit 0..2 pairs; ops
#   10/11/12 emit ZERO muldiv pairs; op 13 emits a data-dependent count (the
#   reference replays the SAME derived branch flags so the counts/order match).
#
# NUM_OPS = 14: a hard `assert op < 14` (14 <= 2^16, sound range-check) gates the
# match to ops {0..13}, so no out-of-range match UB on a prover-controlled op.

NUM_OPS = 14
# Phase 5b swap-loop max step count (bounds the runtime range loop and the
# carried-state buffers; <= 2^16 so `assert num_steps < SWAP_MAX_STEPS` is sound).
# Chosen to keep total memory well under 2^26 (one step ~ 2^17 cells).
SWAP_MAX_STEPS = 64
NUM_CTX_U256 = 4    # swap: sp0, liq0, amount_mag, limit
NUM_FINAL_U256 = 6  # swap: final_sp, final_liq, rem_mag, calc_mag, a0_mag, a1_mag
COMMIT_HEAD = 16       # op, flag_a, y_sign, tick_sign, abs_tick, sign_p1, abs_p1,
#                        tick_spacing, bit_pos, next_val, initialized,
#                        fee_pips, amount_sign, 0,0,0
COMMIT_U256_SLOTS = 9  # sqrtA|sqrtB|liq|amount|result|word|swap_in|swap_out|fee
COMMIT_CELLS = COMMIT_HEAD + COMMIT_U256_SLOTS * UINT256_LIMBS
DIGEST_LEN = 8


def commit_fold(data, n_chunks: Const):
    # Left-fold a length-(8*n_chunks) buffer into an 8-cell digest (identical to
    # Phase 1): acc_0 = data[0..8]; acc_k+1 = poseidon16_compress_half(acc_k, chunk_k).
    debug_assert(1 <= n_chunks)
    acc = Array(DIGEST_LEN)
    for k in unroll(0, DIGEST_LEN):
        acc[k] = data[k]
    out_buf = Array((n_chunks - 1) * DIGEST_LEN)
    src: Mut = acc
    for c in unroll(1, n_chunks):
        dst = out_buf + (c - 1) * DIGEST_LEN
        poseidon16_compress_half(src, data + c * DIGEST_LEN, dst)
        src = dst
    return src


# ============================================================================
# Phase 5b — SWAP mode (mode == 1): the full Pool.swap multi-tick loop with a
# unified pool-state commitment.
# ============================================================================
#
# SWAP-MODE HINT BLOCK (FIFO, read in this order; must match src/reference.rs):
#   "mode"            : 1 cell  (== 1; read in main before dispatch)
#   "swap_zero_for_one": 1 cell (boolean)
#   "swap_amount_sign" : 1 cell (sign of amountSpecified)
#   "swap_fee_pips"    : 1 cell (< FEE_DENOM)
#   "swap_tick_spacing": 1 cell (1 <= ts < 2^16)
#   "swap_num_steps"   : 1 cell (loop bound; VERIFIED to be the real count)
#   "swap_tick0_sign"  : 1 cell
#   "swap_tick0_abs"   : 1 cell
#   "swap_sp0"         : 32 cells (initial sqrtPriceX96, uint256-encoded uint160)
#   "swap_liq0"        : 32 cells (initial liquidity, uint256-encoded uint128)
#   "swap_amount_mag"  : 32 cells (|amountSpecified|, uint256)
#   "swap_limit"       : 32 cells (sqrtPriceLimitX96, uint256-encoded uint160)
#   then, PER STEP (i = 0 .. num_steps-1), the swap.py step hint block:
#     "word"(32), "bit_pos"(1), "tick_next_sign"(1), "tick_next_abs"(1),
#     "tnp1_sign"(1), "tnp1_abs"(1), "liq_net_sign"(1), "liq_net_mag"(32),
#     then the step's muldiv (q, r) pairs in call order.
#
# COMMITMENT (the 8-cell public input is the Poseidon left-fold of):
#   CONTEXT + INITIAL chunk 0 (head): [1, zfo, amount_sign, fee_pips,
#     tick_spacing, num_steps, tick0_sign, tick0_abs]   (8 cells = chunk 0 seed)
#   then sp0(32) | liq0(32) | amount_mag(32) | limit(32)  (4 u256 = 16 chunks)
#   -> this is `dig0`, passed to pool_swap, which folds EVERY per-step word +
#      liquidityNet (4 + 5 chunks per step) as they are consumed,
#   then the FINAL block:
#     head: [final_tick_sign, final_tick_abs, rem_sign, calc_sign, a0_sign,
#            a1_sign, 0, 0]  (1 chunk)
#     final_sp(32) | final_liq(32) | rem_mag(32) | calc_mag(32) | a0_mag(32) |
#       a1_mag(32)  (6 u256 = 24 chunks)
#   The final accumulator must equal memory[0..8]. Binding the per-step words +
#   liquidityNets is what stops a prover from inventing pool state.

def _fold_block8(acc, chunk8):
    out = Array(DIGEST_LEN)
    poseidon16_compress_half(acc, chunk8, out)
    return out


def _fold_u256_into(acc, val):
    # Fold a 32-limb value (4 eight-cell chunks) into the digest accumulator.
    a: Mut = acc
    for c in unroll(0, 4):
        a = _fold_block8(a, val + c * DIGEST_LEN)
    return a


def run_swap(public_input):
    # --- swap context + initial state (committed input block) ---
    zfo_buf = Array(1)
    hint_witness("swap_zero_for_one", zfo_buf)
    zfo = zfo_buf[0]
    amount_sign_buf = Array(1)
    hint_witness("swap_amount_sign", amount_sign_buf)
    amount_sign = amount_sign_buf[0]
    fee_pips_buf = Array(1)
    hint_witness("swap_fee_pips", fee_pips_buf)
    fee_pips = fee_pips_buf[0]
    tick_spacing_buf = Array(1)
    hint_witness("swap_tick_spacing", tick_spacing_buf)
    tick_spacing = tick_spacing_buf[0]
    num_steps_buf = Array(1)
    hint_witness("swap_num_steps", num_steps_buf)
    num_steps = num_steps_buf[0]
    tick0_sign_buf = Array(1)
    hint_witness("swap_tick0_sign", tick0_sign_buf)
    tick0_sign = tick0_sign_buf[0]
    tick0_abs_buf = Array(1)
    hint_witness("swap_tick0_abs", tick0_abs_buf)
    tick0_abs = tick0_abs_buf[0]
    sp0 = Array(UINT256_LIMBS)
    hint_witness("swap_sp0", sp0)
    range_check_limbs(sp0, UINT256_LIMBS)
    liq0 = Array(UINT256_LIMBS)
    hint_witness("swap_liq0", liq0)
    range_check_limbs(liq0, UINT256_LIMBS)
    amount_mag = Array(UINT256_LIMBS)
    hint_witness("swap_amount_mag", amount_mag)
    range_check_limbs(amount_mag, UINT256_LIMBS)
    limit = Array(UINT256_LIMBS)
    hint_witness("swap_limit", limit)
    range_check_limbs(limit, UINT256_LIMBS)

    # Bound num_steps so it can size buffers and the range loop is safe. We allow
    # up to SWAP_MAX_STEPS; the verified-termination asserts pin it to the real
    # count, but the range loop needs start <= end (0 <= num_steps) and a sane
    # upper bound to keep memory < 2^26.
    assert num_steps < SWAP_MAX_STEPS  # SWAP_MAX_STEPS <= 2^16: sound range-check

    # uint160 / uint128 width invariants on the committed operands.
    bignum_assert_fits(sp0, UINT256_LIMBS, SQRT_RESULT_LIMBS)
    bignum_assert_fits(limit, UINT256_LIMBS, SQRT_RESULT_LIMBS)
    bignum_assert_fits(liq0, UINT256_LIMBS, UINT128_LIMBS)

    # --- build dig0 = fold(context head, sp0, liq0, amount_mag, limit) ---
    head0 = Array(DIGEST_LEN)
    head0[0] = 1            # mode == 1 (swap)
    head0[1] = zfo
    head0[2] = amount_sign
    head0[3] = fee_pips
    head0[4] = tick_spacing
    head0[5] = num_steps
    head0[6] = tick0_sign
    head0[7] = tick0_abs
    dig0: Mut = head0
    dig0 = _fold_u256_into(dig0, sp0)
    dig0 = _fold_u256_into(dig0, liq0)
    dig0 = _fold_u256_into(dig0, amount_mag)
    dig0 = _fold_u256_into(dig0, limit)

    # --- run the swap loop ---
    (
        final_sp, final_tsign, final_tabs, final_liq,
        rem_sign, rem_mag, calc_sign, calc_mag,
        a0_sign, a0_mag, a1_sign, a1_mag, dig_after,
    ) = pool_swap(
        sp0, tick0_sign, tick0_abs, liq0,
        amount_sign, amount_mag, limit, fee_pips, tick_spacing,
        zfo, num_steps, dig0,
    )

    # --- fold the final results into the accumulator ---
    fhead = Array(DIGEST_LEN)
    fhead[0] = final_tsign
    fhead[1] = final_tabs
    fhead[2] = rem_sign
    fhead[3] = calc_sign
    fhead[4] = a0_sign
    fhead[5] = a1_sign
    fhead[6] = 0
    fhead[7] = 0
    dig: Mut = _fold_block8(dig_after, fhead)
    dig = _fold_u256_into(dig, final_sp)
    dig = _fold_u256_into(dig, final_liq)
    dig = _fold_u256_into(dig, rem_mag)
    dig = _fold_u256_into(dig, calc_mag)
    dig = _fold_u256_into(dig, a0_mag)
    dig = _fold_u256_into(dig, a1_mag)

    for i in unroll(0, DIGEST_LEN):
        assert dig[i] == public_input[i]
    return


def main():
    build_preamble_memory()
    public_input = 0  # memory[0..8]

    # --- top-level mode selector: 0 = unit-test op-dispatcher (ops 0..13),
    # 1 = full Pool.swap multi-tick loop (Phase 5b). UNCONSTRAINED until the
    # boolean assert below. ---
    mode_buf = Array(1)
    hint_witness("mode", mode_buf)
    mode = mode_buf[0]
    assert mode * (1 - mode) == 0  # mode in {0,1} (sound: 1 <= 2^16)

    if mode == 1:
        run_swap(public_input)
        return

    # --- host-supplied selector + flag (UNCONSTRAINED until range-checked) ---
    op_buf = Array(1)
    hint_witness("op", op_buf)
    op = op_buf[0]
    flag_buf = Array(1)
    hint_witness("flag_a", flag_buf)
    flag_a = flag_buf[0]
    y_sign_buf = Array(1)
    # placed last in the input block but read here; see HINT ORDER above.

    # --- operands ---
    sqrtA = Array(UINT256_LIMBS)
    hint_witness("sqrtA", sqrtA)
    sqrtB = Array(UINT256_LIMBS)
    hint_witness("sqrtB", sqrtB)
    liquidity = Array(UINT256_LIMBS)
    hint_witness("liquidity", liquidity)
    amount = Array(UINT256_LIMBS)
    hint_witness("amount", amount)
    hint_witness("y_sign", y_sign_buf)
    y_sign = y_sign_buf[0]

    # --- TickMath scalars (ops 8/9). Read here so the input block is fixed for
    # every op; only the dispatched arm interprets them. ---
    tick_sign_buf = Array(1)
    hint_witness("tick_sign", tick_sign_buf)
    tick_sign = tick_sign_buf[0]
    abs_tick_buf = Array(1)
    hint_witness("abs_tick", abs_tick_buf)
    abs_tick = abs_tick_buf[0]
    sign_p1_buf = Array(1)
    hint_witness("sign_p1", sign_p1_buf)
    sign_p1 = sign_p1_buf[0]
    abs_p1_buf = Array(1)
    hint_witness("abs_p1", abs_p1_buf)
    abs_p1 = abs_p1_buf[0]

    # --- Phase-4 scalars (ops 10/11/12). Read here so the input block is fixed
    # for every op; only the dispatched arm interprets them. ---
    tick_spacing_buf = Array(1)
    hint_witness("tick_spacing", tick_spacing_buf)
    tick_spacing = tick_spacing_buf[0]
    bit_pos_buf = Array(1)
    hint_witness("bit_pos", bit_pos_buf)
    bit_pos = bit_pos_buf[0]
    # --- the bitmap word (TickBitmap op 12). A COMMITTED input (32 limbs). ---
    word = Array(UINT256_LIMBS)
    hint_witness("word", word)

    # --- SwapMath scalars (op 13). Read here so the input block is fixed for
    # every op; only the dispatched arm interprets them. ---
    fee_pips_buf = Array(1)
    hint_witness("fee_pips", fee_pips_buf)
    fee_pips = fee_pips_buf[0]
    amount_sign_buf = Array(1)
    hint_witness("amount_sign", amount_sign_buf)
    amount_sign = amount_sign_buf[0]

    # Inputs must be canonical base-256 before arithmetic / commitment relies on
    # them (a non-canonical limb is a multiple-representation forgery).
    range_check_limbs(sqrtA, UINT256_LIMBS)
    range_check_limbs(sqrtB, UINT256_LIMBS)
    range_check_limbs(liquidity, UINT256_LIMBS)
    range_check_limbs(amount, UINT256_LIMBS)
    range_check_limbs(word, UINT256_LIMBS)

    # liquidity is a uint128: assert its high 16 limbs are zero.
    bignum_assert_fits(liquidity, UINT256_LIMBS, UINT128_LIMBS)

    # op MUST be in [0, NUM_OPS) before any match dispatch (prover-controlled).
    # NUM_OPS = 14 <= 2^16, so this is a sound range-check; it forces op in {0..13}.
    assert op < NUM_OPS
    assert flag_a * (1 - flag_a) == 0  # flag is boolean for every op that uses it
    assert y_sign * (1 - y_sign) == 0

    # --- Phase-4 result scalars committed in the head; neutral for ops 0..11. ---
    next_val: Mut = 0
    initialized: Mut = 0

    # --- Phase-5a swap-output commit slots. Default to a shared zero buffer for
    # ops 0..12 (no write-once conflict — the pointer is only RESEATED, never
    # overwritten); op 13 reseats each to its computed result. Mirrors the
    # next_val/initialized Mut-reseat pattern. ---
    swap_zero = Array(UINT256_LIMBS)
    for i in unroll(0, UINT256_LIMBS):
        swap_zero[i] = 0
    swap_in_p: Mut = swap_zero
    swap_out_p: Mut = swap_zero
    fee_p: Mut = swap_zero

    # --- dispatch (real VM control flow; only the taken arm runs/consumes hints) ---
    result = Array(UINT256_LIMBS)
    match op:
        case 0:
            r = get_amount0_delta(sqrtA, sqrtB, liquidity, flag_a)
            for i in unroll(0, UINT256_LIMBS):
                result[i] = r[i]
        case 1:
            r = get_amount1_delta(sqrtA, sqrtB, liquidity, flag_a)
            for i in unroll(0, UINT256_LIMBS):
                result[i] = r[i]
        case 2:
            r = get_next_sqrt_price_from_amount0_rounding_up(sqrtA, liquidity, amount, flag_a)
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = r[i]
            for i in unroll(SQRT_RESULT_LIMBS, UINT256_LIMBS):
                result[i] = 0
        case 3:
            r = get_next_sqrt_price_from_amount1_rounding_down(sqrtA, liquidity, amount, flag_a)
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = r[i]
            for i in unroll(SQRT_RESULT_LIMBS, UINT256_LIMBS):
                result[i] = 0
        case 4:
            r = get_next_sqrt_price_from_input(sqrtA, liquidity, amount, flag_a)
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = r[i]
            for i in unroll(SQRT_RESULT_LIMBS, UINT256_LIMBS):
                result[i] = 0
        case 5:
            r = get_next_sqrt_price_from_output(sqrtA, liquidity, amount, flag_a)
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = r[i]
            for i in unroll(SQRT_RESULT_LIMBS, UINT256_LIMBS):
                result[i] = 0
        case 6:
            # add_delta on uint128 operands: x = liquidity[:16], y_mag = amount[:16].
            # Assert amount's high limbs are zero so y_mag is a genuine uint128.
            bignum_assert_fits(amount, UINT256_LIMBS, UINT128_LIMBS)
            x16 = Array(UINT128_LIMBS)
            ymag16 = Array(UINT128_LIMBS)
            for i in unroll(0, UINT128_LIMBS):
                x16[i] = liquidity[i]
                ymag16[i] = amount[i]
            z = add_delta(x16, y_sign, ymag16)
            for i in unroll(0, UINT128_LIMBS):
                result[i] = z[i]
            for i in unroll(UINT128_LIMBS, UINT256_LIMBS):
                result[i] = 0
        case 7:
            # RawMulDiv: exercise FullMath directly on arbitrary uint256 operands.
            # a = sqrtA, b = sqrtB, denom = amount. Computes floor (committed as
            # result) and ceil (consumes a 2nd muldiv pair; its correctness is
            # enforced by mulDivRoundingUp's internal asserts). This is the
            # Phase-1 direct-mulDiv driver, folded in so the bignum layer keeps a
            # direct differential/forgery test surface.
            q = mulDiv(sqrtA, sqrtB, amount)
            # The ceil call is consumed for its hint pair + internal asserts (the
            # second pair Phase-1 emits); its result is discarded (floor is the
            # committed result). Discarding a single return value with `_` keeps
            # the side-effecting call (hint consumption + Euclidean asserts).
            _ = mulDivRoundingUp(sqrtA, sqrtB, amount)
            for i in unroll(0, UINT256_LIMBS):
                result[i] = q[i]
        case 8:
            # get_sqrt_ratio_at_tick(tick_sign, abs_tick) -> uint160. abs_tick is
            # bound + range-checked to [0, MAX_TICK] inside; tick_sign asserted
            # boolean inside. The committed tick (head cells) plus the result bind
            # this uniquely. At most one muldiv pair (the tick>0 reciprocal).
            sp = get_sqrt_ratio_at_tick(tick_sign, abs_tick)
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = sp[i]
            for i in unroll(SQRT_RESULT_LIMBS, UINT256_LIMBS):
                result[i] = 0
        case 9:
            # get_tick_at_sqrt_ratio: hint-and-verify the tick for sqrtA[:20].
            # Bind tick+1 to tick over the signed integers (no wrap: magnitudes
            # < 2^20, field ~ 2^31). signed(t) = abs_t - 2*sign_t*abs_t.
            sv_t = abs_tick - 2 * tick_sign * abs_tick
            sv_p1 = abs_p1 - 2 * sign_p1 * abs_p1
            assert sv_t + 1 == sv_p1  # tick+1 really succeeds tick
            assert sign_p1 * (1 - sign_p1) == 0  # boolean (abs ranges checked inside)
            # sqrtA holds the uint160 sqrtPriceX96 (its high 12 limbs asserted 0).
            bignum_assert_fits(sqrtA, UINT256_LIMBS, SQRT_RESULT_LIMBS)
            sp20 = Array(SQRT_RESULT_LIMBS)
            for i in unroll(0, SQRT_RESULT_LIMBS):
                sp20[i] = sqrtA[i]
            _ts, _at = get_tick_at_sqrt_ratio(sp20, tick_sign, abs_tick, sign_p1, abs_p1)
            # Result slot echoes the input sqrtPriceX96 (the tick itself is bound
            # via the committed head cells tick_sign/abs_tick).
            for i in unroll(0, SQRT_RESULT_LIMBS):
                result[i] = sqrtA[i]
            for i in unroll(SQRT_RESULT_LIMBS, UINT256_LIMBS):
                result[i] = 0
        case 10:
            # most_significant_bit(x = sqrtA, p = bit_pos). Verifies bit_pos is the
            # MSB of sqrtA (2^p <= x < 2^(p+1)); requires x != 0 (asserted inside).
            # The committed answer is bit_pos (already in the head). result = 0.
            _p = most_significant_bit(sqrtA, bit_pos)
            for i in unroll(0, UINT256_LIMBS):
                result[i] = 0
        case 11:
            # least_significant_bit(x = sqrtA, p = bit_pos). Verifies bit_pos is the
            # LSB of sqrtA (keep_low(x, p+1) == 2^p); requires x != 0. result = 0.
            _p = least_significant_bit(sqrtA, bit_pos)
            for i in unroll(0, UINT256_LIMBS):
                result[i] = 0
        case 12:
            # next_initialized_tick_within_one_word(tick, tick_spacing, lte=flag_a,
            # word, msb/lsb hint = bit_pos). Returns (next_val signed, initialized);
            # both committed in the head. result = 0 (word is committed separately).
            nv, init = next_initialized_tick_within_one_word(
                tick_sign, abs_tick, tick_spacing, flag_a, word, bit_pos
            )
            next_val = nv
            initialized = init
            for i in unroll(0, UINT256_LIMBS):
                result[i] = 0
        case 13:
            # compute_swap_step: current = sqrtA, target = sqrtB (both asserted to
            # fit uint160), liquidity (uint128), amount_sign + magnitude = amount,
            # fee_pips. Returns (sqrtRatioNext, amountIn, amountOut, feeAmount):
            #   result   = sqrtRatioNext (uint160, zero-extended)
            #   swap_in  = amountIn ; swap_out = amountOut ; fee = feeAmount.
            # fee_pips / amount_sign committed in head[11]/[12].
            bignum_assert_fits(sqrtA, UINT256_LIMBS, SQRT_RESULT_LIMBS)  # current uint160
            bignum_assert_fits(sqrtB, UINT256_LIMBS, SQRT_RESULT_LIMBS)  # target uint160
            nx, ai, ao, fa = compute_swap_step(sqrtA, sqrtB, liquidity, amount_sign, amount, fee_pips)
            for i in unroll(0, UINT256_LIMBS):
                result[i] = nx[i]
            swap_in_p = ai
            swap_out_p = ao
            fee_p = fa

    # --- bind (head | sqrtA|sqrtB|liq|amount|result|word|swap_in|swap_out|fee) ---
    commit = Array(COMMIT_CELLS)
    commit[0] = op
    commit[1] = flag_a
    commit[2] = y_sign
    commit[3] = tick_sign
    commit[4] = abs_tick
    commit[5] = sign_p1
    commit[6] = abs_p1
    commit[7] = tick_spacing
    commit[8] = bit_pos
    commit[9] = next_val
    commit[10] = initialized
    commit[11] = fee_pips      # SwapMath op 13
    commit[12] = amount_sign   # SwapMath op 13
    for i in unroll(13, COMMIT_HEAD):
        commit[i] = 0
    for i in unroll(0, UINT256_LIMBS):
        commit[COMMIT_HEAD + i] = sqrtA[i]
        commit[COMMIT_HEAD + UINT256_LIMBS + i] = sqrtB[i]
        commit[COMMIT_HEAD + 2 * UINT256_LIMBS + i] = liquidity[i]
        commit[COMMIT_HEAD + 3 * UINT256_LIMBS + i] = amount[i]
        commit[COMMIT_HEAD + 4 * UINT256_LIMBS + i] = result[i]
        commit[COMMIT_HEAD + 5 * UINT256_LIMBS + i] = word[i]
        commit[COMMIT_HEAD + 6 * UINT256_LIMBS + i] = swap_in_p[i]
        commit[COMMIT_HEAD + 7 * UINT256_LIMBS + i] = swap_out_p[i]
        commit[COMMIT_HEAD + 8 * UINT256_LIMBS + i] = fee_p[i]

    n_chunks = div_floor(COMMIT_CELLS, DIGEST_LEN)
    digest = commit_fold(commit, n_chunks)
    for i in unroll(0, DIGEST_LEN):
        assert digest[i] == public_input[i]
    return
