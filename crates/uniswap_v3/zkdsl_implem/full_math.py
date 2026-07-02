from snark_lib import *
from bignum import *

# Uniswap V3 FullMath, as thin uint256-width wrappers over the bignum layer.
#
# v3-core FullMath operates on uint256 operands and a uint256 denominator; the
# 512-bit product is handled with the full-width Euclidean identity in
# bignum_muldiv (no 512-bit intermediate is ever truncated). Widths:
#   a, b, denom : uint256 = 32 limbs (base 2^8)
#   product a*b : 64 limbs (handled inside bignum_muldiv)
#   result      : 32 limbs (the quotient; callers ensure it fits uint256)
#
# Rounding parity with v3 is economic-soundness critical:
#   mulDiv          -> floor(a*b / denom)   (FullMath.mulDiv)
#   mulDivRoundingUp-> ceil (a*b / denom)   (FullMath.mulDivRoundingUp)
#
# Each call consumes exactly one ("muldiv_q", "muldiv_r") hint pair (see the
# HINT CONTRACT in bignum.py / src/reference.rs).

UINT256_LIMBS = 32


def mulDiv(a, b, denom):
    # floor(a * b / denom), a, b, denom width-32 (uint256). Returns width-32 q.
    # NOUT = 32: the quotient must fit in uint256 (asserted), matching v3.
    return bignum_muldiv(a, b, denom, UINT256_LIMBS, UINT256_LIMBS, UINT256_LIMBS, UINT256_LIMBS)


def mulDivRoundingUp(a, b, denom):
    # ceil(a * b / denom), a, b, denom width-32 (uint256). Returns width-32 q.
    return bignum_muldiv_rounding_up(a, b, denom, UINT256_LIMBS, UINT256_LIMBS, UINT256_LIMBS, UINT256_LIMBS)
