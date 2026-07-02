from snark_lib import *

# Local copies of the small constrained-arithmetic idioms the bignum layer needs.
#
# These mirror the rec_aggregation helpers verbatim in spirit (same hint-then-bind
# shape); they are copied here rather than imported because the embedded zkdsl dir
# resolves imports within itself, and Phase 1 must not depend on rec_aggregation's
# preamble offsets. Each is soundness-critical: a hint result is unconstrained until
# the asserts below bind it to the real input.


@inline
def checked_less_than(a, b):
    # Returns 1 if a < b else 0, as a proof-enforced boolean.
    #
    # PRECONDITION (range-check soundness): both operands must be <= 2^16. Every
    # caller in bignum.py feeds 8-bit limbs (< 256), which satisfies this. The two
    # branch asserts re-derive the comparison so the hinted boolean cannot lie:
    #   res == 1  =>  a < b      (requires b <= 2^16)
    #   res == 0  =>  b <= a     (requires a <  2^16)
    res: Imm
    hint_less_than(a, b, res)
    assert res * (1 - res) == 0
    if res == 1:
        assert a < b
    else:
        assert b <= a
    return res


def checked_decompose_bits_small_value_const(to_decompose, n_bits: Const):
    # Decompose `to_decompose` into `n_bits` big-endian boolean cells (bits[0] = MSB)
    # and assert they recompose to the value. Each bit is boolean-constrained and the
    # final equality binds the whole decomposition. n_bits must be < 31 (field width)
    # and is compile-time (so the unroll bound is constant).
    bits = Array(n_bits)
    hint_decompose_bits(to_decompose, bits, n_bits)
    sum: Mut = bits[n_bits - 1]
    assert sum * (1 - sum) == 0
    for i in unroll(1, n_bits):
        b = bits[n_bits - 1 - i]
        assert b * (1 - b) == 0
        sum += b * 2**i
    assert to_decompose == sum
    return bits


@inline
def euclidean_div_runtime(a, b):
    # Returns (q, r) with q = floor(a / b), r = a mod b, proof-enforced.
    #
    # PRECONDITIONS (soundness): caller must guarantee
    #   1 <= b  and  b <= 2^16          (so `assert r < b` is a valid range-check)
    #   q < 2^16                         (asserted; so q*b + r stays < p, no wrap)
    # Used only with the column-carry base b = 256 in bignum, where these hold.
    q: Imm
    r: Imm
    hint_div_floor(a, b, q, r)
    assert r < b
    assert q < 2 ** 16
    assert q * b + r == a
    return q, r
