from snark_lib import *
from bignum import *

# Uniswap V3 LiquidityMath, mirrored from v3-core LiquidityMath.sol.
#
# v3-core (exact):
#     function addDelta(uint128 x, int128 y) internal pure returns (uint128 z) {
#         if (y < 0) {
#             require((z = x - uint128(-y)) < x, 'LS');
#         } else {
#             require((z = x + uint128(y)) >= x, 'LA');
#         }
#     }
#
# zkDSL representation of the int128 `y`: a sign bit `y_sign` in {0, 1}
# (0 = nonnegative / add, 1 = negative / subtract) and a uint128 magnitude
# `y_mag` (= |y|, 16 base-256 limbs). The caller guarantees y_mag is a valid
# uint128 (its limbs are range-checked here as part of bignum_add/bignum_sub).
#
# Widths: x, y_mag, z are uint128 = 16 limbs (UINT128_LIMBS).
#
# Soundness:
#   * y_sign is asserted boolean (a non-boolean sign would let a prover pick a
#     value outside {add, sub} and dodge both bounds).
#   * add: z = x + y_mag at width 16; the bignum_add carry-out must be 0
#     (z < 2^128), the in-circuit form of v3's `require(z >= x)` overflow guard.
#   * sub: bignum_sub requires x >= y_mag (asserts top borrow 0), the in-circuit
#     form of v3's `require(z < x)` underflow guard. (Note: v3's `< x` is
#     equivalent to "no underflow" given uint128 wraparound semantics; our
#     borrow-0 check is the faithful no-underflow assertion.)
#   * Both operands are range-checked < 256 inside bignum_add / bignum_sub.

UINT128_LIMBS = 16


def add_delta(x, y_sign, y_mag):
    # Return z = x + y (uint128), where y is the signed int128 (y_sign, y_mag).
    # x, y_mag, z are width-16 (uint128). Asserts the uint128 width invariant.
    assert y_sign * (1 - y_sign) == 0  # y_sign boolean

    z = Array(UINT128_LIMBS)
    if y_sign == 0:
        # add: z = x + y_mag, require result fits uint128 (carry-out == 0).
        carry = bignum_add(x, y_mag, z, UINT128_LIMBS)
        assert carry == 0
    else:
        # sub: z = x - y_mag, require x >= y_mag (bignum_sub asserts top borrow 0).
        bignum_sub(x, y_mag, z, UINT128_LIMBS)
    return z
