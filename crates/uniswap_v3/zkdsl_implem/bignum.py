from snark_lib import *
from helpers import *

# ============================================================================
# Fixed-width bignum arithmetic over base B = 2^8 (little-endian limbs).
# ============================================================================
#
# REPRESENTATION
#   A width-N bignum is a pointer to N field cells, each holding one base-256
#   "limb" in [0, 256). Little-endian: limb[0] is least significant. With base
#   256 a single field cell (KoalaBear, p ~ 2^31) holds a limb with enormous
#   headroom, so column sums in schoolbook multiplication never overflow the
#   field (see the no-overflow argument on bignum_mul_schoolbook).
#
#   Widths used by Uniswap V3:
#     uint128 = 16 limbs,  uint160 = 20,  uint256 = 32,  product up to 64.
#   Every width is a COMPILE-TIME constant (`: Const`) so loops `unroll`, arrays
#   size statically, and each call pays only for its true width.
#
# SOUNDNESS DISCIPLINE (every function upholds these; adversarial-audited)
#   * Every limb that is hinted or derived is range-checked < 256 — an
#     unconstrained limb is a multiple-representation forgery.
#   * Every hint (carry/borrow boolean, muldiv q/r) is bound by an assert.
#   * No field accumulator is ever allowed to reach p.
#   * No `assert a < b` is emitted with b > 2^16 (we only ever compare limbs,
#     which are < 256, or the constant 256, or q < 2^16 at the boundary).
#
# ----------------------------------------------------------------------------
# HINT CONTRACT (must match src/reference.rs EXACTLY — order is load-bearing)
# ----------------------------------------------------------------------------
#   Label "muldiv_q": one buffer per bignum_muldiv call, in guest call order.
#                     Buffer = q as NA+NB little-endian base-256 limbs (cells).
#   Label "muldiv_r": one buffer per bignum_muldiv call, SAME order.
#                     Buffer = r as ND little-endian base-256 limbs (cells).
#   The N-th bignum_muldiv the guest executes pops the N-th buffer from each
#   label (FIFO per label). bignum_muldiv_rounding_up performs exactly one
#   bignum_muldiv, so it consumes one (q, r) pair too. Keep the host emission in
#   src/reference.rs (emit_one_muldiv_pair / emit_main_hints) in the same
#   traversal order as the guest's bignum_muldiv calls.
# ============================================================================

LIMB_BASE = 256
SHIFT_96_LIMBS = 12  # 96 bits / 8 bits-per-limb (base 2^8)


# ----------------------------------------------------------------------------
# Range / equality / comparison
# ----------------------------------------------------------------------------


def range_check_limbs(x, N: Const):
    # Assert every limb of the width-N bignum x is a valid base-256 limb (< 256).
    # 256 <= 2^16, so each `assert x[i] < 256` is a sound range-check.
    for i in unroll(0, N):
        assert x[i] < LIMB_BASE
    return


def bignum_eq(a, b, N: Const):
    # Assert two width-N bignums are equal, limb by limb.
    for i in unroll(0, N):
        assert a[i] == b[i]
    return


def bignum_lt(a, b, N: Const):
    # Return 1 if a < b else 0 (proof-enforced), comparing most-significant limb
    # first. Once a limb pair decides the order, later limbs are ignored.
    #
    # N is compile-time, so we `unroll` and carry the (decided, result) state in
    # plain mutables. Each limb comparison uses checked_less_than on 8-bit limbs
    # (< 256 <= 2^16), which is a sound range-check. Correctness: scanning from
    # the top, the first index where a[i] != b[i] fixes the result; equal-prefix
    # cells leave `decided == 0` and the result stays 0 (a == b => not <).
    decided: Mut = 0
    result: Mut = 0
    for k in unroll(0, N):
        i = N - 1 - k  # most significant limb first
        a_lt_b = checked_less_than(a[i], b[i])
        b_lt_a = checked_less_than(b[i], a[i])
        # If still undecided and this limb differs, record the verdict.
        newly_lt = (1 - decided) * a_lt_b
        differs = a_lt_b + b_lt_a  # 0 if equal, 1 if differing (mutually exclusive)
        result = result + newly_lt
        decided = decided + (1 - decided) * differs
    return result


def bignum_eq_flag(a, b, N: Const):
    # Return 1 if a == b else 0 (proof-enforced), without asserting equality.
    # a == b  <=>  not (a < b) and not (b < a). bignum_lt range-checks limbs
    # internally; a_lt_b and b_lt_a are mutually-exclusive booleans, so
    # 1 - a_lt_b - b_lt_a is in {0,1} and equals 1 exactly when a == b. Used by
    # SwapMath's `max = (target == sqrtRatioNext)` decision (a derived flag, not an
    # assertion). Inputs are range-checked inside bignum_lt.
    a_lt_b = bignum_lt(a, b, N)
    b_lt_a = bignum_lt(b, a, N)
    return 1 - a_lt_b - b_lt_a


def bignum_le_flag(a, b, N: Const):
    # Return 1 if a <= b else 0 (proof-enforced). a <= b  <=>  not (b < a).
    return 1 - bignum_lt(b, a, N)


# ----------------------------------------------------------------------------
# Addition / subtraction
# ----------------------------------------------------------------------------


def bignum_add(a, b, out, N: Const):
    # out = (a + b) mod 256^N ; returns the final carry-out (0/1).
    # Per limb:  a[i] + b[i] + carry_in == out[i] + carry_out*256
    # The host-independent carry is recovered with a boolean hint and pinned by
    # `out[i] < 256`: since s = a[i]+b[i]+carry_in is in [0, 511] (limbs < 256),
    # only the true carry keeps out[i] in [0, 256). A lying carry_out forces
    # out[i] out of range and the assert fires. No accumulator exceeds 511 < p.
    #
    # SOUNDNESS: range-check BOTH inputs first. The carry-pinning argument
    # (s in [0, 511] and `out[i] < 256` uniquely fixing carry_out) only holds when
    # every a[i], b[i] is a canonical base-256 limb (< 256). A non-canonical limb
    # in a prover-supplied operand would let s exceed 511 and break the unique-carry
    # argument — a multiple-representation forgery vector. (Matches bignum_sub /
    # bignum_mul_schoolbook, which already self-range-check; also makes the
    # LiquidityMath.add_delta docstring's "both operands range-checked" claim true.)
    range_check_limbs(a, N)
    range_check_limbs(b, N)
    carry: Mut = 0
    for i in unroll(0, N):
        s = a[i] + b[i] + carry
        carry_out: Imm
        hint_less_than(255, s, carry_out)  # 1 iff s >= 256
        assert carry_out * (1 - carry_out) == 0
        out[i] = s - carry_out * LIMB_BASE
        assert out[i] < LIMB_BASE
        carry = carry_out
    return carry


def bignum_sub(a, b, out, N: Const):
    # out = a - b, REQUIRES a >= b (documented precondition). Asserts the top
    # borrow is 0, which proves a >= b for valid limb inputs.
    # Per limb:  a[i] + borrow_out*256 == b[i] + borrow_in + out[i]
    #   => out[i] = a[i] + borrow_out*256 - b[i] - borrow_in
    # borrow_out is a boolean hint; `out[i] < 256` pins it (only the true borrow
    # keeps out[i] in range). Final `assert borrow == 0` enforces a >= b.
    # Accumulators: a[i] + 256 <= 511 < p; the subtraction is exact in-field
    # because out[i] is forced into [0, 256).
    #
    # SOUNDNESS: range-check BOTH inputs first. The borrow-pinning argument
    # (`a[i] + 256 <= 511` and `out[i] < 256` uniquely fixing borrow_out) only
    # holds when every a[i], b[i] is a canonical base-256 limb (< 256). bignum_sub
    # gets its first real callers in SqrtPriceMath (subtracting hinted sqrt
    # prices); without this, a non-canonical limb in a prover-supplied operand is
    # a forgery vector. (Cost: a couple of cheap `< 256` asserts per limb.)
    range_check_limbs(a, N)
    range_check_limbs(b, N)
    borrow: Mut = 0
    for i in unroll(0, N):
        borrow_out: Imm
        # borrow needed iff a[i] < b[i] + borrow_in.
        hint_less_than(a[i], b[i] + borrow, borrow_out)
        assert borrow_out * (1 - borrow_out) == 0
        out[i] = a[i] + borrow_out * LIMB_BASE - b[i] - borrow
        assert out[i] < LIMB_BASE
        borrow = borrow_out
    assert borrow == 0  # precondition a >= b
    return


# ----------------------------------------------------------------------------
# Schoolbook multiplication
# ----------------------------------------------------------------------------


def bignum_mul_schoolbook(x, y, NX: Const, NY: Const):
    # Return a freshly allocated width-(NX+NY) bignum equal to x * y.
    #
    # NO-OVERFLOW ARGUMENT (the load-bearing soundness claim):
    #   Each input limb < 2^8, so each partial product x[i]*y[j] < 2^16.
    #   Column c accumulates the products with i+j==c; there are at most
    #   min(NX, NY) <= 32 such pairs, so the product sum is < 32*2^16 = 2^21.
    #   We carry a `carry` into the next column; after extracting limb < 256,
    #   carry = floor(total/256) < 2^21/256 = 2^13. Hence each column total is
    #     total = (sum of products) + carry < 2^21 + 2^13 < 2^22 << p (~2^31).
    #   Every field accumulator therefore stays < 2^22 < p: no field overflow.
    #   Each emitted limb is range-checked < 256 and bound by the column identity
    #   total == limb + carry*256, so the representation is unique.
    #
    # The no-overflow bound REQUIRES every input limb < 256 (else a product could
    # exceed 2^16 and a column could exceed p). We therefore range-check the
    # inputs here rather than trust the caller — this keeps the primitive sound in
    # isolation (the redundant re-check of an already-checked operand is cheap).
    range_check_limbs(x, NX)
    range_check_limbs(y, NY)
    NZ = NX + NY
    out = Array(NZ)
    carry_buf = Array(NZ + 1)  # carry_buf[c] = carry flowing into column c
    carry_buf[0] = 0
    for c in unroll(0, NZ):
        # Accumulate all partial products landing in column c.
        acc: Mut = carry_buf[c]
        # Compile-time index window: i in [max(0, c-(NY-1)), min(c+1, NX)).
        # min(u, v) = u - saturating_sub(u, v); all values are compile-time.
        i_lo = saturating_sub(c, NY - 1)
        i_hi = (c + 1) - saturating_sub(c + 1, NX)
        for i in unroll(i_lo, i_hi):
            j = c - i
            acc += x[i] * y[j]
        # Carry-extract: total = limb + carry*256, with limb < 256.
        # euclidean_div_runtime asserts r < 256, q < 2^16, q*256 + r == acc.
        # acc < 2^22, so q < 2^14 < 2^16 (the q < 2^16 bound is never tight, and
        # q*256 < 2^24 < p, so the identity holds over the integers => unique).
        limb: Mut
        carry_next: Mut
        carry_next, limb = euclidean_div_runtime(acc, LIMB_BASE)
        out[c] = limb
        carry_buf[c + 1] = carry_next
    # Top carry must vanish: the product of an NX- and an NY-limb number fits in
    # NX+NY limbs exactly, so the carry out of the last column is 0.
    assert carry_buf[NZ] == 0
    return out


# ----------------------------------------------------------------------------
# Fixed shifts by 96 bits (= 12 limbs at base 2^8). Used by SqrtPriceMath.
# ----------------------------------------------------------------------------


def shl_96(x, N: Const):
    # Return x << 96 as a width-(N + 12) bignum (exact, no truncation): the low
    # 12 limbs are zero, the rest are x's limbs shifted up by 12 positions.
    NZ = N + SHIFT_96_LIMBS
    out = Array(NZ)
    for i in unroll(0, SHIFT_96_LIMBS):
        out[i] = 0
    for i in unroll(0, N):
        out[i + SHIFT_96_LIMBS] = x[i]
    return out


def shr_96(x, N: Const):
    # Return floor(x / 2^96) as a width-(N - 12) bignum: drop the low 12 limbs.
    # REQUIRES N > 12 (compile-time). The discarded low limbs are the remainder;
    # this is a pure floor shift (no rounding), matching v3's `>> 96`.
    if N <= SHIFT_96_LIMBS:
        assert False, "shr_96 requires width > 12 limbs"
    NZ = N - SHIFT_96_LIMBS
    out = Array(NZ)
    for i in unroll(0, NZ):
        out[i] = x[i + SHIFT_96_LIMBS]
    return out


# ----------------------------------------------------------------------------
# The crux: mulDiv = floor(a * b / denom), fully constrained.
# ----------------------------------------------------------------------------


def _is_bignum_nonzero(x, N: Const):
    # Return 1 if any limb of x is nonzero, else 0 (proof-enforced). Uses an
    # Inverse-style witness: for each limb we OR a per-limb nonzero flag.
    # Implementation: accumulate sum of "limb != 0" indicators via checked
    # comparison against 0. A limb is nonzero iff 0 < limb; checked_less_than is
    # sound for limbs < 256. We then OR the flags (their sum, then clamp to 0/1
    # is unnecessary — we only test the final against 0).
    any_nz: Mut = 0
    for i in unroll(0, N):
        is_pos = checked_less_than(0, x[i])  # 1 iff x[i] != 0 (limbs are >= 0, < 256)
        any_nz = any_nz + is_pos
    return any_nz


# ----------------------------------------------------------------------------
# Width casts / fits-checks / zero tests used by SqrtPriceMath & LiquidityMath.
# ----------------------------------------------------------------------------


def bignum_cast(x, NIN: Const, NOUT: Const):
    # Return a width-NOUT copy of the width-NIN bignum x. If widening
    # (NOUT >= NIN) the new high limbs are zero. If narrowing (NOUT < NIN) the
    # dropped high limbs are ASSERTED zero, so the value provably fits in NOUT
    # limbs (the in-circuit analogue of Solidity's `uintK(...)` cast on a value
    # known to fit). Inputs are range-checked so the copy is canonical.
    range_check_limbs(x, NIN)
    out = Array(NOUT)
    if NOUT <= NIN:
        for i in unroll(0, NOUT):
            out[i] = x[i]
        for i in unroll(NOUT, NIN):
            assert x[i] == 0
    else:
        for i in unroll(0, NIN):
            out[i] = x[i]
        for i in unroll(NIN, NOUT):
            out[i] = 0
    return out


def bignum_assert_fits(x, N: Const, NFIT: Const):
    # Assert the width-N bignum x fits in NFIT limbs (every limb at index
    # >= NFIT is zero). Used to enforce v3's `toUint160()` / `uint160(...)`
    # result-width requirements as a real in-circuit constraint.
    for i in unroll(NFIT, N):
        assert x[i] == 0
    return


def bignum_is_zero(x, N: Const):
    # Return 1 if the width-N bignum x is zero, else 0 (proof-enforced). The
    # complement of _is_bignum_nonzero, clamped to a boolean.
    # _is_bignum_nonzero returns a count in [0, N] (N <= 64 < 256), so
    # checked_less_than(0, count) is a sound 8-bit comparison and yields the
    # boolean "x != 0"; its complement is "x == 0". Inputs must be range-checked.
    nz_count = _is_bignum_nonzero(x, N)
    return 1 - checked_less_than(0, nz_count)


def bignum_high_is_zero(x, N: Const, NLOW: Const):
    # Return 1 if every limb of x at index >= NLOW is zero (i.e. x fits in NLOW
    # limbs), else 0 (proof-enforced). Used as the in-circuit equivalent of v3's
    # uint256-overflow guard `(amount*sqrtP)/amount == sqrtP`: that guard is true
    # iff the true product amount*sqrtP did not overflow uint256, i.e. iff its
    # limbs above index 32 are all zero. Inputs must be range-checked.
    if NLOW <= 0:
        assert False, "bignum_high_is_zero requires NLOW > 0"
    if N <= NLOW:
        # No high limbs => always fits.
        return 1
    high = Array(N - NLOW)
    for i in unroll(NLOW, N):
        high[i - NLOW] = x[i]
    return bignum_is_zero(high, N - NLOW)


def bignum_add_fits(a, b, N: Const):
    # Return (out, no_overflow): out = (a + b) mod 256^N as a fresh width-N
    # bignum, no_overflow = 1 iff the true sum a + b fits in N limbs (carry-out
    # 0), else 0. no_overflow is the proven carry-out boolean, so when it is 1
    # `out` equals the true integer sum exactly. Mirrors v3's overflow-detecting
    # `denominator = numerator1 + product; if (denominator >= numerator1)`.
    out = Array(N)
    carry = bignum_add(a, b, out, N)  # carry in {0,1}, boolean-pinned inside
    no_overflow = 1 - carry
    return out, no_overflow


def bignum_muldiv(a, b, denom, NA: Const, NB: Const, ND: Const, NOUT: Const):
    # Return floor(a * b / denom) as a width-NOUT bignum.
    #
    # The prover supplies q (full NA+NB width) and r (ND width); we enforce the
    # Euclidean identity
    #       a * b == q * denom + r        (over the FULL product width)
    #       0 <= r < denom
    #       denom != 0
    # plus range-checks on EVERY hinted limb of q and r, plus q's limbs above the
    # requested output width NOUT must be 0 (so the caller's result really fits in
    # NOUT limbs — this is the in-circuit analogue of v3's `require(result fits)`;
    # a quotient that overflows NOUT cannot satisfy it). Together these pin (q, r)
    # to the unique quotient/remainder, so the returned width-NOUT prefix is
    # exactly floor(a*b/denom). q and r unconstrained would be a forgery; every
    # clause below is why that forgery fails.
    NQ = NA + NB  # quotient hint width (true quotient <= a*b fits in NA+NB limbs)
    NAB = NA + NB  # a*b width
    NFULL = NQ + ND  # q*denom width = full comparison width

    # --- consume the prover's q and r (UNCONSTRAINED until bound below) ---
    q = Array(NQ)
    hint_witness("muldiv_q", q)
    r = Array(ND)
    hint_witness("muldiv_r", r)

    # --- bind every limb to base-256 (no multiple representations) ---
    # denom is range-checked here too: the comparisons below (checked_less_than on
    # denom limbs) are only sound for operands < 256, and the no-overflow argument
    # of the q*denom multiply needs canonical denom limbs.
    range_check_limbs(q, NQ)
    range_check_limbs(r, ND)
    range_check_limbs(denom, ND)

    # --- denom must be nonzero (else division is undefined / forgeable) ---
    assert _is_bignum_nonzero(denom, ND) != 0

    # --- r < denom (remainder is canonical) ---
    assert bignum_lt(r, denom, ND) == 1

    # --- compute a*b and q*denom over full width, then a*b == q*denom + r ---
    ab = bignum_mul_schoolbook(a, b, NA, NB)  # width NAB
    qd = bignum_mul_schoolbook(q, denom, NQ, ND)  # width NFULL

    # rhs = q*denom + r, both zero-extended to NFULL.
    ab_ext = Array(NFULL)
    for i in unroll(0, NAB):
        ab_ext[i] = ab[i]
    for i in unroll(NAB, NFULL):
        ab_ext[i] = 0

    r_ext = Array(NFULL)
    for i in unroll(0, ND):
        r_ext[i] = r[i]
    for i in unroll(ND, NFULL):
        r_ext[i] = 0

    rhs = Array(NFULL)
    _ = bignum_add(qd, r_ext, rhs, NFULL)  # carry-out is 0 for a valid identity

    # FULL-width equality: a*b == q*denom + r exactly (not truncated).
    bignum_eq(ab_ext, rhs, NFULL)

    # The result must fit in NOUT limbs: every higher limb of q is 0.
    for i in unroll(NOUT, NQ):
        assert q[i] == 0

    # Return the width-NOUT prefix (the limbs above it are proven zero).
    out = Array(NOUT)
    for i in unroll(0, NOUT):
        out[i] = q[i]
    return out


def bignum_muldiv_rounding_up(a, b, denom, NA: Const, NB: Const, ND: Const, NOUT: Const):
    # Return ceil(a * b / denom) as a width-NOUT bignum: floor result, plus 1 iff
    # the remainder r != 0.
    #
    # Same constrained floor quotient as bignum_muldiv, then a constrained boolean
    # `r_nonzero` is added to it. The +1 is a real width-NOUT bignum addition so a
    # carry can ripple. The result must fit in NOUT limbs both before AND after the
    # rounding step, so we assert the carry-out of the +1 is 0 (no overflow past
    # NOUT — the in-circuit analogue of Solidity's `require(result + 1 != 0)`).
    # The rounding boolean is bound to the proven r (re-derived, limbs OR-ed) so
    # it cannot be forged independently of the proven remainder.
    NQ = NA + NB
    NAB = NA + NB  # a*b width (== NQ, but used distinctly for clarity/safety)
    NFULL = NQ + ND

    q = Array(NQ)
    hint_witness("muldiv_q", q)
    r = Array(ND)
    hint_witness("muldiv_r", r)

    range_check_limbs(q, NQ)
    range_check_limbs(r, ND)
    range_check_limbs(denom, ND)
    assert _is_bignum_nonzero(denom, ND) != 0
    assert bignum_lt(r, denom, ND) == 1

    ab = bignum_mul_schoolbook(a, b, NA, NB)
    qd = bignum_mul_schoolbook(q, denom, NQ, ND)

    # Copy the product over its true width NAB (== NA+NB), then zero-extend.
    ab_ext = Array(NFULL)
    for i in unroll(0, NAB):
        ab_ext[i] = ab[i]
    for i in unroll(NAB, NFULL):
        ab_ext[i] = 0
    r_ext = Array(NFULL)
    for i in unroll(0, ND):
        r_ext[i] = r[i]
    for i in unroll(ND, NFULL):
        r_ext[i] = 0
    rhs = Array(NFULL)
    _ = bignum_add(qd, r_ext, rhs, NFULL)
    bignum_eq(ab_ext, rhs, NFULL)

    # Floor quotient must fit in NOUT limbs.
    for i in unroll(NOUT, NQ):
        assert q[i] == 0
    q_out = Array(NOUT)
    for i in unroll(0, NOUT):
        q_out[i] = q[i]

    # r_nonzero in {0,1}, bound to the proven r (limbs all < 256).
    r_nonzero = _is_bignum_nonzero(r, ND)
    round_bit: Imm
    if r_nonzero == 0:
        round_bit = 0
    else:
        round_bit = 1

    # result = q_out + round_bit (width-NOUT add of [round_bit, 0, ...]); the
    # ceil must still fit in NOUT limbs, so the final carry-out must be 0.
    one_vec = Array(NOUT)
    one_vec[0] = round_bit
    for i in unroll(1, NOUT):
        one_vec[i] = 0
    result = Array(NOUT)
    carry_out = bignum_add(q_out, one_vec, result, NOUT)
    assert carry_out == 0
    return result
