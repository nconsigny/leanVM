from snark_lib import *
from bignum import *
from full_math import *
from tick_math import *
from bit_math import *

# ============================================================================
# Uniswap V3 TickBitmap.nextInitializedTickWithinOneWord, mirrored from
# v3-core TickBitmap.sol (EXACT).
# ============================================================================
#
# v3-core:
#   int24 compressed = tick / tickSpacing;
#   if (tick < 0 && tick % tickSpacing != 0) compressed--;     // floor division
#   if (lte) {
#       (int16 wordPos, uint8 bitPos) = position(compressed);
#       uint256 mask = (1 << bitPos) - 1 + (1 << bitPos);       // low bitPos+1 bits
#       uint256 masked = self[wordPos] & mask;
#       initialized = masked != 0;
#       next = initialized
#           ? (compressed - int24(bitPos - mostSignificantBit(masked))) * tickSpacing
#           : (compressed - int24(bitPos)) * tickSpacing;
#   } else {
#       (int16 wordPos, uint8 bitPos) = position(compressed + 1);
#       uint256 mask = ~((1 << bitPos) - 1);                    // bits >= bitPos
#       uint256 masked = self[wordPos] & mask;
#       initialized = masked != 0;
#       next = initialized
#           ? (compressed + 1 + int24(leastSignificantBit(masked) - bitPos)) * tickSpacing
#           : (compressed + 1 + int24(type(uint8).max - bitPos)) * tickSpacing;
#   }
#   where position(t): wordPos = t >> 8 (arithmetic), bitPos = uint8(uint24(t) % 256).
#
# KEY INSIGHTS THAT MAKE THIS SOUND & CHEAP:
#
# 1. SIGNED FLOOR DIVISION without wide bignums. |tick| <= MAX_TICK < 2^20, so
#    q = floor(|tick|/ts) < 2^20 and ts in [1, 2^16). The single field product
#    q*ts could overflow the field (q < 2^20, ts < 2^16 => up to 2^36 > p), so we
#    split q into two 10-bit halves and bound the high partial product, keeping
#    every accumulator < p (see signed_floor_compressed). Then compressed is the
#    true floor: tick>=0 -> q; tick<0 -> -(q + (r != 0)).
#
# 2. position() via euclidean div on a SHIFTED value. The masks are computed from
#    `base` = compressed (lte) or compressed+1 (gt). bitPos = base mod 256 and
#    wordPos = floor(base/256) (both for signed base). We shift base by
#    OFFSET = 2^20 (a multiple of 256, > max |base|) into the nonnegative range,
#    then do a plain euclidean div by 256: bitPos in [0,255] is unchanged by the
#    shift (256 | OFFSET), and every term stays < p (see bit_pos_of).
#
# 3. The v3 masks are CONTIGUOUS, so no general bitwise AND is needed:
#      lte: masked = low (bitPos+1) bits of word          = keep_low(word, bitPos+1)
#      gt:  masked = word with low bitPos bits cleared     = clear_low(word, bitPos)
#    Both come from bit_math.keep_low_split (kept / cleared), fully constrained.
#
# 4. next = compressed_adjusted * tickSpacing, where compressed_adjusted is a
#    signed tick index with |compressed_adjusted| <= |compressed| + 256. Since
#    compressed and ts are both already pinned to their honest values, the field
#    product equals the integer next exactly (|next| < 2^25 < p), so committing
#    `next` (a signed field value) binds the answer.
#
# The bitmap WORD is a COMMITTED input (read under hint label "word", committed in
# main.py's public input): at this phase the word IS an input to the contract,
# exactly like sqrtP. The swap-loop phase will additionally bind words to the
# unified pool-state commitment.
#
# Returns (next_val, initialized): next_val is the signed next tick as a field
# element; initialized in {0,1}.
# ============================================================================

TICK_OFFSET = 1048576  # 2^20: a multiple of 256, strictly greater than MAX_TICK
TICK_OFFSET_WORD_BITS = 21  # base + OFFSET in [0, 2^21): woff = floor(.../256) < 2^13
WOFF_BITS = 13
Q_HALF_BITS = 10  # split q (< 2^20) into two 10-bit halves
Q_HALF_BASE = 1024  # 2^10
P_HI_BOUND = 1024  # honest high partial product q_hi*ts < 2^10 (see soundness note)
TS_MAX_BITS = 16  # tick_spacing in [1, 2^16)
UINT8_MAX = 255  # type(uint8).max


def check_tick_spacing(ts):
    # Enforce 1 <= ts < 2^16 soundly. Decompose ts into 16 bits -> 0 <= ts < 2^16
    # (so `assert r < ts` and `checked_less_than(_, ts)` are valid range-checks),
    # then assert ts != 0 from the proven bits (OR of bits, count <= 16 < 2^16).
    bits = checked_decompose_bits_small_value_const(ts, TS_MAX_BITS)
    any_bit: Mut = 0
    for i in unroll(0, TS_MAX_BITS):
        any_bit = any_bit + bits[i]
    assert checked_less_than(0, any_bit) == 1  # ts != 0
    return


def signed_floor_compressed(tick_sign, abs_tick, ts):
    # compressed = floor(signed_tick / ts), returned as a signed field value.
    # Preconditions (caller-enforced): tick_sign boolean & canonical (abs==0 =>
    # sign==0); 0 <= abs_tick <= MAX_TICK; 1 <= ts < 2^16.
    #
    # q, r = floor(abs_tick / ts), abs_tick mod ts (VM hint, then constrained so
    # q is provably the true quotient — the 10+10 split keeps q*ts exact in-field).
    q: Imm
    r: Imm
    hint_div_floor(abs_tick, ts, q, r)
    assert r < ts  # r < ts < 2^16: sound; remainder canonical

    # Bind q to the true quotient: q < 2^20, then q*ts + r == abs_tick over the
    # integers. Split q = q_hi*2^10 + q_lo so each half*ts stays < 2^26 < p, and
    # bound the high partial product so q_hi*ts*2^10 < 2^20 < p (no field wrap).
    q_bits = checked_decompose_bits_small_value_const(q, 2 * Q_HALF_BITS)  # 0 <= q < 2^20
    # q_bits is MSB-first (q_bits[0] = bit 19, q_bits[19] = bit 0).
    q_hi: Mut = 0
    for i in unroll(0, Q_HALF_BITS):
        q_hi = q_hi * 2 + q_bits[i]  # bits 19..10 -> high half
    q_lo: Mut = 0
    for i in unroll(Q_HALF_BITS, 2 * Q_HALF_BITS):
        q_lo = q_lo * 2 + q_bits[i]  # bits 9..0 -> low half
    p_hi = q_hi * ts  # < 2^10 * 2^16 = 2^26 < p: exact
    p_lo = q_lo * ts  # < 2^26 < p: exact
    assert p_hi < P_HI_BOUND  # honest q_hi*ts < 2^10 (q*ts <= abs_tick < 2^20); sound
    assert p_hi * Q_HALF_BASE + p_lo + r == abs_tick  # < 2^27 < p: integer-exact

    # r != 0 (for the negative-tick decrement).
    r_nonzero = checked_less_than(0, r)  # r < ts < 2^16: sound boolean

    # tick >= 0 -> compressed = q ; tick < 0 -> compressed = -(q + r_nonzero).
    #   (1 - tick_sign)*q - tick_sign*(q + r_nonzero)
    compressed = (1 - tick_sign) * q - tick_sign * (q + r_nonzero)
    return compressed


def bit_pos_of(base):
    # bitPos = base mod 256 for a SIGNED `base` (= compressed or compressed+1),
    # |base| <= MAX_TICK + 1 < 2^20. Shift into the nonnegative range by
    # OFFSET = 2^20 (multiple of 256, so base mod 256 is preserved), then plain
    # euclidean div by 256. Every term stays < p.
    base_off = base + TICK_OFFSET  # in [2^20 - (MAX_TICK+1), 2^20 + (MAX_TICK+1)] subset [0, 2^21)
    woff: Imm
    bit_pos: Imm
    hint_div_floor(base_off, LIMB_BASE, woff, bit_pos)  # LIMB_BASE == 256
    assert bit_pos < LIMB_BASE  # 256 <= 2^16: sound
    assert woff < 2 ** WOFF_BITS  # 2^13 <= 2^16: sound (woff = floor(base_off/256) < 2^13)
    assert woff * LIMB_BASE + bit_pos == base_off  # woff*256 < 2^21 < p: integer-exact
    return bit_pos


def next_initialized_tick_within_one_word(tick_sign, abs_tick, ts, lte, word, bit_pos_hint):
    # EXACT port of v3 TickBitmap.nextInitializedTickWithinOneWord.
    #   tick = signed (tick_sign, abs_tick); ts = tickSpacing; lte boolean;
    #   word = the 256-bit bitmap at the relevant wordPos (committed input);
    #   bit_pos_hint = host-hinted MSB/LSB of `masked` (bound inside BitMath, used
    #     only on the initialized branch).
    # Returns (next_val signed-field, initialized 0/1).
    assert lte * (1 - lte) == 0  # lte boolean
    # tick encoding: boolean & canonical (abs==0 => sign==0) and 0 <= abs <= MAX_TICK.
    assert tick_sign * (1 - tick_sign) == 0
    abs_bits = assert_le_max_tick(abs_tick)  # 0 <= abs_tick <= MAX_TICK; returns 20 bits
    abs_any: Mut = 0
    for i in unroll(0, TICK_ABS_BITS):
        abs_any = abs_any + abs_bits[i]
    abs_nonzero = checked_less_than(0, abs_any)  # abs_any <= 20 < 2^16: sound
    assert tick_sign * (1 - abs_nonzero) == 0  # canonical: abs == 0 forces sign == 0

    check_tick_spacing(ts)  # 1 <= ts < 2^16
    range_check_limbs(word, UINT256_LIMBS)  # committed input must be canonical limbs

    compressed = signed_floor_compressed(tick_sign, abs_tick, ts)

    # base = compressed (lte) or compressed + 1 (gt); bitPos = base mod 256.
    base = (1 - lte) * (compressed + 1) + lte * compressed
    bit_pos = bit_pos_of(base)

    # masked: lte keeps the low (bitPos+1) bits; gt clears the low bitPos bits.
    kept_le, _cl_le = keep_low_split(word, bit_pos + 1)
    _kept_gt, cleared_gt = keep_low_split(word, bit_pos)
    masked = Array(UINT256_LIMBS)
    for i in unroll(0, UINT256_LIMBS):
        masked[i] = lte * kept_le[i] + (1 - lte) * cleared_gt[i]
        assert masked[i] < LIMB_BASE

    initialized = 1 - bignum_is_zero(masked, UINT256_LIMBS)  # masked != 0

    # next_val = compressed_adjusted * ts (signed field). The four v3 cases:
    #   lte  init   : (compressed - (bitPos - msb)) * ts = (base - bitPos + msb)*ts
    #   lte  uninit : (compressed - bitPos)        * ts = (base - bitPos)*ts
    #   gt   init   : (compressed+1 + (lsb - bitPos)) * ts = (base + lsb - bitPos)*ts
    #   gt   uninit : (compressed+1 + (255 - bitPos)) * ts = (base + 255 - bitPos)*ts
    # (base == compressed for lte, compressed+1 for gt; substitution above.)
    #
    # msb / lsb of `masked` (host-hinted position bound here). Computed in BOTH
    # branches' taken arm only via the `if initialized`; on the uninitialized arm
    # masked == 0 so BitMath's `x != 0` requirement would (correctly) fail — we
    # must therefore only call them when initialized. The hint bit_pos_hint is
    # consumed by the caller regardless; it is simply unused when not initialized.
    next_val: Imm
    if initialized == 1:
        if lte == 1:
            msb = most_significant_bit(masked, bit_pos_hint)
            comp_adj = base - bit_pos + msb
            next_val = comp_adj * ts
        else:
            lsb = least_significant_bit(masked, bit_pos_hint)
            comp_adj = base + lsb - bit_pos
            next_val = comp_adj * ts
    else:
        if lte == 1:
            comp_adj = base - bit_pos
            next_val = comp_adj * ts
        else:
            comp_adj = base + UINT8_MAX - bit_pos
            next_val = comp_adj * ts
    return next_val, initialized
