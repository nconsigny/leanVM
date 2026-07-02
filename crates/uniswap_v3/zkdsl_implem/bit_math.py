from snark_lib import *
from bignum import *
from full_math import *

# ============================================================================
# Uniswap V3 BitMath, mirrored from v3-core BitMath.sol.
# ============================================================================
#
# Two functions, both REQUIRE x != 0 (asserted via _is_bignum_nonzero):
#
#   most_significant_bit(x)  -> p in [0,255] with 2^p <= x < 2^(p+1)
#       (the index of the highest set bit). v3: BitMath.mostSignificantBit.
#   least_significant_bit(x) -> p in [0,255] with bit p set AND all bits below
#       p zero, i.e. x mod 2^(p+1) == 2^p. v3: BitMath.leastSignificantBit.
#
# HINT-AND-VERIFY shape (the only sound way without a 256-way search in-circuit):
#   The position p is a host-supplied hint (read by the CALLER via hint_witness
#   under label "bit_pos" and passed in here) — UNCONSTRAINED until this function
#   binds it. We build the power-of-two 2^p from the runtime p and PROVE the
#   extremal property with bignum comparisons:
#     * MSB: 2^p <= x  (bignum_lt(x, 2^p) == 0)  AND  x < 2^(p+1)
#            (the upper bound uses a 33-limb 2^(p+1) so p == 255 -> 2^256 is
#             representable and x < 2^256 holds vacuously).
#     * LSB: keep_low(x, p+1) == 2^p  (one equality: bit p set, lower bits zero).
#   p is range-checked < 256 so the runtime 2^p builder (and any `match_range`
#   over p%8 / p//8) stays in range.
#
# 2^p / keep-low constructions are write-once-correct and every produced limb is
# range-checked < 256 (a non-canonical limb would be a multiple-representation
# forgery feeding bignum_lt / bignum_eq).
#
# Widths (base 2^8 little-endian limbs): x is uint256 = 32 limbs; the 2^(p+1)
# upper-bound power and the zero-extended x use 33 limbs.
# ============================================================================

POW_WIDTH = UINT256_LIMBS + 1  # 33 limbs: holds 2^256 (= limb[32]=1) for the p==255 upper bound
BYTE_BITS = 8
MSB_LSB_MAX = 256  # positions are in [0, 256); 256 <= 2^16 so `assert p < 256` is sound


def _pow2_byte(bit_in):
    # 2^bit_in for runtime bit_in in [0, 8). Dispatched const -> exact power.
    # bit_in is range-checked by the caller (it is a remainder mod 8). The
    # debug_assert documents the precondition the caller maintains.
    # NB: `match_range` is only expanded by the compiler as the RHS of an
    # assignment (not a bare `return`), so bind to `res` first.
    debug_assert(bit_in < BYTE_BITS)
    res = match_range(bit_in, range(0, BYTE_BITS + 1), lambda i: 2 ** i)
    return res


def _is_eq_small(a, b):
    # 1 iff a == b, else 0 (proof-enforced). Both operands must be <= 2^16
    # (callers pass limb indices <= 33). a == b  <=>  not (a < b) and not (b < a).
    a_lt_b = checked_less_than(a, b)
    b_lt_a = checked_less_than(b, a)
    return 1 - (a_lt_b + b_lt_a)


def make_two_pow(p, width: Const):
    # Build 2^p as `width` base-256 little-endian limbs, for RUNTIME p in
    # [0, 8*width). Sound construction:
    #   limb_idx = p // 8, bit_in = p % 8  (via hint_div_floor, then constrained);
    #   limb[limb_idx] = 2^bit_in, all other limbs 0.
    # p is the caller's responsibility to keep < 8*width (asserted via limb_idx).
    # WRITE-ONCE: each cell out[j] is written exactly once as eq(j,limb_idx)*2^bit_in;
    # exactly one j matches, so out[limb_idx] = 2^bit_in and the rest are 0. Each
    # produced limb is < 256 (2^bit_in <= 128, others 0), so canonical by build.
    limb_idx: Imm
    bit_in: Imm
    hint_div_floor(p, BYTE_BITS, limb_idx, bit_in)
    assert bit_in < BYTE_BITS  # 8 <= 2^16: sound range-check
    assert limb_idx < width  # 2^p must fit `width` limbs (width is small, <= 33 < 2^16)
    assert limb_idx * BYTE_BITS + bit_in == p  # p == 8*limb_idx + bit_in (8*limb_idx < 264 < p)
    pow_bit = _pow2_byte(bit_in)  # 2^bit_in in {1,2,...,128}
    out = Array(width)
    for j in unroll(0, width):
        eq = _is_eq_small(j, limb_idx)  # j, limb_idx <= 33 < 2^16: sound
        out[j] = eq * pow_bit  # 1 cell write per j; 0 unless j == limb_idx
    return out


def keep_low_split(word, k):
    # Split a 32-limb word into (kept, cleared) where:
    #   kept    = word mod 2^k   (the low k bits)
    #   cleared = word - kept    (the bits >= k)
    # for RUNTIME k in [0, 256]. Both are fresh 32-limb bignums; kept[j]+cleared[j]
    # == word[j] per limb by construction, every produced limb range-checked < 256.
    #
    # k = 8*limb_idx + bit_in. For limb j:
    #   j <  limb_idx -> kept[j] = word[j],            cleared[j] = 0
    #   j == limb_idx -> kept[j] = word[j] mod 2^bit_in, cleared[j] = word[j] - that
    #   j >  limb_idx -> kept[j] = 0,                  cleared[j] = word[j]
    # selected limbwise via proven booleans (no runtime-indexed array access).
    limb_idx: Imm
    bit_in: Imm
    hint_div_floor(k, BYTE_BITS, limb_idx, bit_in)
    assert bit_in < BYTE_BITS
    assert limb_idx < UINT256_LIMBS + 1  # k <= 256 -> limb_idx in [0,32]; 33 < 2^16
    assert limb_idx * BYTE_BITS + bit_in == k  # 8*limb_idx < 264 < p
    pow_bit = _pow2_byte(bit_in)  # 2^bit_in in {1,...,128}

    kept = Array(UINT256_LIMBS)
    cleared = Array(UINT256_LIMBS)
    for j in unroll(0, UINT256_LIMBS):
        lt = checked_less_than(j, limb_idx)  # 1 iff j < limb_idx (operands <= 32)
        eq = _is_eq_small(j, limb_idx)  # 1 iff j == limb_idx
        # Partial split of word[j] at the bit boundary: word[j] = hi*2^bit_in + lo,
        # lo < 2^bit_in (only used when j == limb_idx, but constrained for every j).
        hi: Imm
        lo: Imm
        hint_div_floor(word[j], pow_bit, hi, lo)
        assert lo < pow_bit  # pow_bit <= 128 <= 2^16: sound
        assert hi < LIMB_BASE  # word[j] < 256 => hi < 256; 256 <= 2^16
        assert hi * pow_bit + lo == word[j]  # hi*pow_bit + lo < 256 < p: exact
        # kept low part:
        kept[j] = lt * word[j] + eq * lo
        # cleared high part = word[j] - kept[j]:
        cleared[j] = (1 - lt) * word[j] - eq * lo
        assert kept[j] < LIMB_BASE
        assert cleared[j] < LIMB_BASE
    return kept, cleared


def most_significant_bit(x, p):
    # v3 BitMath.mostSignificantBit. x is a 32-limb uint256, REQUIRES x != 0.
    # p (host-hinted by the caller) is VERIFIED to satisfy 2^p <= x < 2^(p+1).
    range_check_limbs(x, UINT256_LIMBS)
    assert _is_bignum_nonzero(x, UINT256_LIMBS) != 0  # require(x > 0)
    assert p < MSB_LSB_MAX  # p in [0,256): sound range-check (256 <= 2^16)

    # 2^p <= x  <=>  not (x < 2^p).
    pow_p = make_two_pow(p, UINT256_LIMBS)  # 32 limbs (p <= 255 -> fits)
    assert bignum_lt(x, pow_p, UINT256_LIMBS) == 0

    # x < 2^(p+1). Use 33 limbs so p == 255 -> 2^256 is representable and the
    # comparison holds vacuously (every uint256 < 2^256).
    pow_p1 = make_two_pow(p + 1, POW_WIDTH)  # 33 limbs (p+1 in [1,256])
    x_ext = Array(POW_WIDTH)
    for i in unroll(0, UINT256_LIMBS):
        x_ext[i] = x[i]
    x_ext[UINT256_LIMBS] = 0
    assert bignum_lt(x_ext, pow_p1, POW_WIDTH) == 1
    return p


def least_significant_bit(x, p):
    # v3 BitMath.leastSignificantBit. x is a 32-limb uint256, REQUIRES x != 0.
    # p (host-hinted) is VERIFIED: bit p set AND all lower bits zero, captured by
    # the single equality  keep_low(x, p+1) == 2^p.
    range_check_limbs(x, UINT256_LIMBS)
    assert _is_bignum_nonzero(x, UINT256_LIMBS) != 0  # require(x > 0)
    assert p < MSB_LSB_MAX  # p in [0,256): sound

    kept, _cleared = keep_low_split(x, p + 1)  # low (p+1) bits of x
    pow_p = make_two_pow(p, UINT256_LIMBS)  # 2^p
    bignum_eq(kept, pow_p, UINT256_LIMBS)  # low (p+1) bits of x == 2^p exactly
    return p
