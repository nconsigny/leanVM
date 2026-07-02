from snark_lib import *
from bignum import *
from full_math import *
from sqrt_price_math import *

# ============================================================================
# Uniswap V3 TickMath, mirrored from v3-core TickMath.sol.
# ============================================================================
#
# Two functions:
#   get_sqrt_ratio_at_tick  — EXACT, fully in-circuit (no division-as-hint
#                             except the single tick>0 reciprocal). The bit-
#                             selected product of the 19 magic Q128.128
#                             multipliers, using real VM control flow so untaken
#                             multiplies are skipped.
#   get_tick_at_sqrt_ratio  — hint-and-verify: the host supplies the tick; we
#                             range-check it and verify it lands in the unique
#                             [sqrtRatio(tick), sqrtRatio(tick+1)) bucket.
#
# TICK ENCODING (matching reference.rs / main.py):
#   A signed tick is (tick_sign, abs_tick): tick_sign in {0,1} (0 = tick >= 0,
#   1 = tick < 0), abs_tick = |tick| as a plain field value in [0, MAX_TICK].
#   tick > 0  <=>  tick_sign == 0 AND abs_tick != 0.
#
# MAGIC CONSTANTS — SOURCE OF TRUTH:
#   The 20 entries of MAGIC below are the v3-core TickMath.sol Q128.128
#   multipliers, taken VERBATIM and reproduced as base-256 little-endian uint128
#   limb arrays (16 limbs each). MAGIC[i] gates bit i of abs_tick. The Rust
#   reference (reference.rs) carries the SAME 20 hex constants; the boundary
#   validators (getSqrtRatioAtTick(0) == 2^96, (MIN_TICK) == MIN_SQRT_RATIO,
#   (MAX_TICK) == MAX_SQRT_RATIO) plus a fixed mid-tick vector and the
#   strict-monotonicity scan in tests are the safety net for a mistyped limb.
#   (Note: the three boundary vectors alone do NOT exercise every magic bit —
#   MIN/MAX_TICK = 0xd89e8 share the same bit pattern, so e.g. bits {4,9,10,12,
#   13,14,17} are uncovered by boundaries; the differential mid-tick vector and
#   the random/monotone differential tests cover the rest.)
#
# WIDTHS (base 2^8 little-endian limbs):
#   magic       : uint128 = 16 limbs
#   ratio       : uint256 = 32 limbs
#   ratio*magic : 48 limbs (32 + 16); >> 128 drops the low 16 limbs -> 32 limbs
#   sqrtPriceX96: uint160 = 20 limbs (asserted to fit)
# ============================================================================

TICK_ABS_BITS = 20  # 2^20 = 1048576 > MAX_TICK; MAX_TICK = 0xd89e8 (20 bits)
MAX_TICK = 887272
MAGIC_FRAC_LIMBS = 16  # the >>128 shift = drop 16 base-256 limbs

# The 20 v3-core TickMath magic multipliers (Q128.128), each as 16 base-256 LE
# limbs. Generated from the verbatim hex constants; validated by the boundary
# tests. MAGIC[i] is applied when bit i of abs_tick is set.
MAGIC = [
    [1, 64, 89, 26, 45, 22, 45, 170, 55, 173, 111, 189, 51, 185, 252, 255],
    [58, 33, 14, 88, 144, 105, 164, 89, 50, 65, 61, 55, 114, 114, 249, 255],
    [204, 253, 199, 243, 124, 53, 18, 239, 50, 105, 101, 95, 15, 229, 242, 255],
    [208, 28, 148, 160, 234, 36, 54, 28, 230, 228, 16, 126, 202, 202, 229, 255],
    [68, 102, 146, 92, 131, 88, 219, 201, 89, 97, 15, 214, 67, 152, 203, 255],
    [192, 84, 178, 223, 150, 104, 46, 71, 129, 192, 152, 250, 65, 59, 151, 255],
    [97, 40, 181, 38, 179, 120, 236, 67, 56, 106, 201, 102, 100, 161, 46, 255],
    [83, 48, 156, 150, 241, 97, 196, 17, 168, 162, 153, 106, 4, 238, 93, 254],
    [164, 163, 154, 71, 59, 200, 255, 220, 174, 136, 10, 144, 199, 134, 190, 252],
    [84, 94, 129, 247, 76, 7, 43, 111, 23, 19, 196, 58, 37, 167, 135, 249],
    [243, 112, 75, 142, 57, 122, 12, 148, 5, 0, 183, 34, 8, 43, 57, 243],
    [217, 137, 232, 166, 127, 156, 178, 67, 116, 155, 194, 162, 117, 148, 21, 231],
    [37, 88, 170, 146, 247, 216, 90, 132, 184, 34, 32, 253, 189, 243, 151, 208],
    [229, 97, 224, 144, 31, 220, 101, 138, 223, 15, 135, 45, 70, 70, 247, 169],
    [247, 50, 175, 43, 246, 61, 187, 144, 184, 161, 210, 86, 161, 105, 216, 112],
    [166, 207, 47, 84, 5, 21, 35, 129, 217, 143, 208, 151, 95, 19, 190, 49],
    [201, 155, 233, 243, 84, 222, 119, 198, 225, 132, 122, 91, 139, 80, 170, 9],
    [4, 230, 94, 34, 41, 195, 153, 102, 25, 129, 219, 222, 248, 106, 93, 0],
    [152, 254, 237, 27, 4, 38, 169, 30, 250, 245, 132, 229, 22, 34, 0, 0],
    [162, 143, 78, 68, 66, 220, 247, 145, 3, 23, 138, 4, 0, 0, 0, 0],
]


def make_two_pow_128_u256():
    # 2^128 as a uint256 (32 base-256 limbs): limb[16] = 1, all others 0.
    # (Write each cell exactly once — write-once discipline.)
    r = Array(UINT256_LIMBS)
    for i in unroll(0, MAGIC_FRAC_LIMBS):
        r[i] = 0
    r[MAGIC_FRAC_LIMBS] = 1  # MAGIC_FRAC_LIMBS == 16, i.e. 2^(8*16) = 2^128
    for i in unroll(MAGIC_FRAC_LIMBS + 1, UINT256_LIMBS):
        r[i] = 0
    return r


def make_uint256_max():
    # type(uint256).max = 2^256 - 1, as 32 base-256 limbs (all 255).
    m = Array(UINT256_LIMBS)
    for i in unroll(0, UINT256_LIMBS):
        m[i] = LIMB_BASE - 1  # 255
    return m


def load_magic_const(i: Const):
    # Copy the compile-time MAGIC[i] (16-limb uint128) into a fresh runtime
    # buffer. i is compile-time so the ragged-array index is constant.
    m = Array(MAGIC_FRAC_LIMBS)
    for k in unroll(0, MAGIC_FRAC_LIMBS):
        m[k] = MAGIC[i][k]
    return m


def ratio_mul_magic_shr128(ratio, i: Const):
    # ratio' = (ratio * MAGIC[i]) >> 128, all exact.
    # ratio is uint256 (32 limbs); MAGIC[i] is uint128 (16 limbs); the product is
    # 48 limbs. `>> 128` is an exact floor shift = drop the low 16 base-256 limbs,
    # so the result is the 32 high limbs (indices 16..48). This matches v3's
    # `(ratio * magic) >> 128`. No hint, no rounding — pure schoolbook + slice.
    magic = load_magic_const(i)
    prod = bignum_mul_schoolbook(ratio, magic, UINT256_LIMBS, MAGIC_FRAC_LIMBS)  # 48 limbs
    out = Array(UINT256_LIMBS)
    for k in unroll(0, UINT256_LIMBS):
        out[k] = prod[MAGIC_FRAC_LIMBS + k]
    return out


def assert_le_max_tick(abs_tick):
    # Enforce 0 <= abs_tick <= MAX_TICK soundly without any `assert a <= b` whose
    # bound exceeds 2^16. Returns abs_tick's 20-bit big-endian decomposition
    # (bound to abs_tick), which get_sqrt_ratio_at_tick reuses to gate the magics.
    #
    # Step 1: decompose abs_tick into 20 bits -> binds 0 <= abs_tick < 2^20.
    bits = checked_decompose_bits_small_value_const(abs_tick, TICK_ABS_BITS)
    # Step 2: decompose diff = MAX_TICK - abs_tick into 20 bits -> binds
    #   0 <= diff < 2^20, and the sum identity proves abs_tick <= MAX_TICK.
    # We compute diff as a field value; if abs_tick > MAX_TICK it would be a huge
    # field element whose 20-bit decomposition fails (recomposition != diff), so
    # the prover cannot satisfy it. MAX_TICK (887272) < 2^20 < p, no wrap.
    diff = MAX_TICK - abs_tick
    _ = checked_decompose_bits_small_value_const(diff, TICK_ABS_BITS)
    # abs_tick + diff == MAX_TICK holds by construction of diff; the decomposition
    # asserts diff is a canonical nonnegative < 2^20 value, so abs_tick <= MAX_TICK.
    return bits


def get_sqrt_ratio_at_tick(tick_sign, abs_tick):
    # v3-core getSqrtRatioAtTick (EXACT). tick_sign in {0,1}, abs_tick = |tick|.
    #
    #   require(absTick <= uint256(MAX_TICK));
    #   uint256 ratio = absTick & 0x1 != 0
    #       ? 0xfffcb933bd6fad37aa2d162d1a594001 : 0x100000000000000000000000000000000;
    #   if (absTick & 0x2 != 0)     ratio = (ratio * 0xfff9...) >> 128;
    #   ... (bits 0x4 .. 0x80000) ...
    #   if (tick > 0) ratio = type(uint256).max / ratio;
    #   sqrtPriceX96 = uint160((ratio >> 32) + (ratio % (1 << 32) == 0 ? 0 : 1));
    #
    # Returned: uint160 sqrtPriceX96 as SQRT_RESULT_LIMBS (20) base-256 limbs.
    assert tick_sign * (1 - tick_sign) == 0  # boolean sign

    # Bound abs_tick to [0, MAX_TICK] and get its 20 big-endian bits (bit[19]=LSB
    # since checked_decompose_bits is MSB-first: bits[0]=MSB=bit19, bits[19]=bit0).
    bits = assert_le_max_tick(abs_tick)

    # bit i of abs_tick (value 2^i) is bits[TICK_ABS_BITS - 1 - i].
    bit0 = bits[TICK_ABS_BITS - 1]  # 2^0

    # ratio = bit0 ? MAGIC[0] (uint128, zero-extended to uint256) : 2^128.
    # WRITE-ONCE: `ratio` is a Mut POINTER reseated to a fresh array each step;
    # we never overwrite a written cell (that would be a write-once conflict).
    ratio_init: Imm
    if bit0 == 1:
        magic0 = load_magic_const(0)
        ratio_init = bignum_cast(magic0, MAGIC_FRAC_LIMBS, UINT256_LIMBS)  # 16 -> 32 limbs
    else:
        ratio_init = make_two_pow_128_u256()
    ratio: Mut = ratio_init

    # Bits 1..19: conditional multiply-then-shift. The `if bit == 1` is REAL VM
    # control flow, so an untaken multiply (a full 32x16 schoolbook) is skipped —
    # the dominant cost of this function scales with the number of SET bits. Each
    # iteration reseats `ratio` to either a fresh product array or the same
    # pointer (no allocation / no multiply on the untaken path).
    for i in unroll(1, TICK_ABS_BITS):
        bit_i = bits[TICK_ABS_BITS - 1 - i]
        next_ratio: Imm
        if bit_i == 1:
            next_ratio = ratio_mul_magic_shr128(ratio, i)
        else:
            next_ratio = ratio
        ratio = next_ratio

    # tick > 0  <=>  tick_sign == 0 AND abs_tick != 0.
    # abs_tick <= MAX_TICK = 887272 > 2^16, so checked_less_than(0, abs_tick) is
    # NOT directly sound (operand > 2^16). Use the proven bits instead: abs_tick
    # is nonzero iff any of its 20 bits is set (any_bit in [0, 20] is < 2^16).
    any_bit: Mut = 0
    for i in unroll(0, TICK_ABS_BITS):
        any_bit = any_bit + bits[i]
    abs_nonzero = checked_less_than(0, any_bit)  # any_bit <= 20 < 2^16: sound

    # CANONICAL tick==0 encoding (Phase-3 audit carry-in): abs_tick == 0 forces
    # tick_sign == 0, so zero has a single committed form. Without this,
    # (sign=0, abs=0) and (sign=1, abs=0) both decode to tick 0 -> commitment
    # malleability. (Both get_sqrt_ratio_at_tick callers — ops 8 and 9 — inherit
    # this; any consumer must still decode to abs - 2*sign*abs, never compare raw.)
    assert tick_sign * (1 - abs_nonzero) == 0

    tick_pos = (1 - tick_sign) * abs_nonzero  # 1 iff tick > 0

    # if tick > 0: ratio = floor(UINT256_MAX / ratio). One hinted division.
    inverted: Imm
    if tick_pos == 1:
        umax = make_uint256_max()
        inverted = div_floor_u256(umax, ratio)  # one muldiv pair
    else:
        inverted = ratio

    # sqrtPriceX96 = (ratio >> 32) + (low 32 bits of ratio nonzero ? 1 : 0).
    # >> 32 = drop the low 4 base-256 limbs. The ceil-adjust bit is derived: 1 iff
    # any of the dropped low 4 limbs is nonzero (NOT a hint).
    SHIFT_32_LIMBS = 4
    shifted = Array(UINT256_LIMBS)
    for k in unroll(0, UINT256_LIMBS - SHIFT_32_LIMBS):
        shifted[k] = inverted[k + SHIFT_32_LIMBS]
    for k in unroll(UINT256_LIMBS - SHIFT_32_LIMBS, UINT256_LIMBS):
        shifted[k] = 0

    low4 = Array(SHIFT_32_LIMBS)
    for k in unroll(0, SHIFT_32_LIMBS):
        low4[k] = inverted[k]
    round_bit = _is_bignum_nonzero(low4, SHIFT_32_LIMBS)  # in [0,4]
    round_bit_bool = checked_less_than(0, round_bit)  # boolean: low 32 bits != 0

    one_vec = Array(UINT256_LIMBS)
    one_vec[0] = round_bit_bool
    for k in unroll(1, UINT256_LIMBS):
        one_vec[k] = 0
    sqrtP = Array(UINT256_LIMBS)
    _ = bignum_add(shifted, one_vec, sqrtP, UINT256_LIMBS)

    # uint160(...) : assert the result fits 20 limbs, return them.
    bignum_assert_fits(sqrtP, UINT256_LIMBS, SQRT_RESULT_LIMBS)
    out = Array(SQRT_RESULT_LIMBS)
    for k in unroll(0, SQRT_RESULT_LIMBS):
        out[k] = sqrtP[k]
    return out


def _u160_to_u256(x20):
    # Zero-extend a uint160 (20 limbs) to a uint256 (32 limbs), range-checked.
    return bignum_cast(x20, SQRT_RESULT_LIMBS, UINT256_LIMBS)


def get_tick_at_sqrt_ratio(sqrtP20, tick_sign, abs_tick, sign_p1, abs_p1):
    # v3-core getTickAtSqrtRatio, hint-and-verify form.
    #
    #   require(sqrtPriceX96 >= MIN_SQRT_RATIO && sqrtPriceX96 < MAX_SQRT_RATIO);
    #   ... (in v3, a log2/log-sqrt computation derives the tick) ...
    #
    # We instead HINT the tick (tick_sign, abs_tick) from the host and PROVE it is
    # the unique answer by verifying the bucket inequality
    #     sqrtRatio(tick) <= sqrtP < sqrtRatio(tick+1)
    # Monotonicity of getSqrtRatioAtTick over [MIN_TICK, MAX_TICK] makes the tick
    # satisfying this UNIQUE, so the two comparisons bind it (no log needed).
    #
    # sqrtP20 is the uint160 input (20 limbs). The hinted tick is (tick_sign,
    # abs_tick); (tick+1) is supplied separately as (sign_p1, abs_p1) — the caller
    # (main.py) range-checks BOTH and binds tick+1 to tick via the encoding (see
    # main.py). Returns (tick_sign, abs_tick) echoed (already bound).
    range_check_limbs(sqrtP20, SQRT_RESULT_LIMBS)
    sqrtP = _u160_to_u256(sqrtP20)

    # --- require MIN_SQRT_RATIO <= sqrtP < MAX_SQRT_RATIO ---
    # MIN_SQRT_RATIO = 4295128739, MAX_SQRT_RATIO = sqrtRatio(MAX_TICK). Build both
    # as uint256 via the exact in-circuit getSqrtRatioAtTick of the boundary ticks
    # is overkill; we build MIN as a constant and reuse sqrtRatio(tick)/(tick+1).
    min_ratio = _make_min_sqrt_ratio()
    assert bignum_lt(sqrtP, min_ratio, UINT256_LIMBS) == 0  # sqrtP >= MIN
    max_ratio = _make_max_sqrt_ratio()
    assert bignum_lt(sqrtP, max_ratio, UINT256_LIMBS) == 1  # sqrtP < MAX

    # --- ratio(tick) and ratio(tick+1), recomputed EXACTLY in-circuit ---
    ratio_t20 = get_sqrt_ratio_at_tick(tick_sign, abs_tick)
    ratio_t1_20 = get_sqrt_ratio_at_tick(sign_p1, abs_p1)
    ratio_t = _u160_to_u256(ratio_t20)
    ratio_t1 = _u160_to_u256(ratio_t1_20)

    # --- bucket: ratio(tick) <= sqrtP < ratio(tick+1) ---
    assert bignum_lt(sqrtP, ratio_t, UINT256_LIMBS) == 0  # sqrtP >= ratio(tick)
    assert bignum_lt(sqrtP, ratio_t1, UINT256_LIMBS) == 1  # sqrtP <  ratio(tick+1)
    return tick_sign, abs_tick


def _make_min_sqrt_ratio():
    # MIN_SQRT_RATIO = 4295128739 = getSqrtRatioAtTick(MIN_TICK), as a uint256.
    # 4295128739 = 0x1_000276A3 -> LE base-256 bytes (decimal): 163,118,2,0,1.
    # (The compiler accepts decimal integer literals only — no 0x hex literals.)
    m = Array(UINT256_LIMBS)
    m[0] = 163  # 0xA3
    m[1] = 118  # 0x76
    m[2] = 2    # 0x02
    m[3] = 0    # 0x00
    m[4] = 1    # 0x01
    for i in unroll(5, UINT256_LIMBS):
        m[i] = 0
    return m


def _make_max_sqrt_ratio():
    # MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342
    #   = 0xFFFD8963 EFD1FC6A 5064 8849 5D951D52 63988D26 (160-bit, fits uint160).
    # The 20 low limbs are the LE base-256 byte values in decimal (the compiler
    # accepts decimal integer literals only). Validated against the reference:
    # this equals MAX_SQRT_RATIO. Each cell is written exactly once (write-once).
    m = Array(UINT256_LIMBS)
    m[0] = 38    # 0x26
    m[1] = 141   # 0x8D
    m[2] = 152   # 0x98
    m[3] = 99    # 0x63
    m[4] = 82    # 0x52
    m[5] = 29    # 0x1D
    m[6] = 149   # 0x95
    m[7] = 93    # 0x5D
    m[8] = 73    # 0x49
    m[9] = 136   # 0x88
    m[10] = 100  # 0x64
    m[11] = 80   # 0x50
    m[12] = 106  # 0x6A
    m[13] = 252  # 0xFC
    m[14] = 209  # 0xD1
    m[15] = 239  # 0xEF
    m[16] = 99   # 0x63
    m[17] = 137  # 0x89
    m[18] = 253  # 0xFD
    m[19] = 255  # 0xFF
    for i in unroll(SQRT_RESULT_LIMBS, UINT256_LIMBS):
        m[i] = 0
    return m
