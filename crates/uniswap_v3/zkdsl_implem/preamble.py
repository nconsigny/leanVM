from snark_lib import *

# Preamble memory layout for the Uniswap-V3 guest.
#
# The runner lays out memory as:
#     [ public_input (PUBLIC_INPUT_LEN cells) | preamble_memory | runtime ]
# The `public_input` region (memory[0..8]) is the 8-cell Poseidon digest the
# host commits to. The `preamble_memory` region is RESERVED but left
# UNINITIALIZED by the runner; `build_preamble_memory()` must write every cell
# it declares here, at the very top of `main`, before anything reads it.
#
# Offsets start at PUBLIC_INPUT_LEN (= DIGEST_LEN = 8) and are laid out
# contiguously. The total cell count is PREAMBLE_MEMORY_LEN, which MUST equal
# the `preamble_memory_len` passed in the Rust `ExecutionWitness`.
#
# Mirrors the convention in rec_aggregation/zkdsl_implem/utils.py
# (build_preamble_memory): the names/offsets of preamble constants are a
# repo-level convention, not part of the language.

PUBLIC_INPUT_LEN = 8  # == DIGEST_LEN, fixed by the VM
DIM = 5  # extension-field degree (cells per EF element)

# A run of zero cells. Handy as a neutral source for copies / padding and as a
# zero extension-field element (its first DIM cells). Phase 1 bignum/EF helpers
# reuse this the same way utils.py does.
ZERO_VEC_LEN = 16
ZERO_VEC_PTR = PUBLIC_INPUT_LEN  # = 8

# The extension-field constant one: [1, 0, 0, 0, 0]. Used by the EF-copy idiom
# `dot_product_ee(x, ONE_EF_PTR, dst)` and as the multiplicative identity.
ONE_EF_PTR = ZERO_VEC_PTR + ZERO_VEC_LEN  # = 24

# Total cells reserved for the preamble (must match the Rust harness constant).
PREAMBLE_MEMORY_LEN = ZERO_VEC_LEN + DIM  # = 21


@inline
def build_preamble_memory():
    # Zero vector.
    zero_vec_w = ZERO_VEC_PTR
    for i in unroll(0, ZERO_VEC_LEN):
        zero_vec_w[i] = 0

    # One in the extension field: [1, 0, 0, 0, 0].
    one_ef_w = ONE_EF_PTR
    one_ef_w[0] = 1
    for i in unroll(1, DIM):
        one_ef_w[i] = 0
    return
