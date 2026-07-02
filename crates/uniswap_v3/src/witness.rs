//! Build the `(public_input, ExecutionWitness)` pair for the Phase-1 guest.
//!
//! The witness queues the host hints (operands + the prover's q/r pairs) under
//! their labels in guest consumption order, and reserves `PREAMBLE_MEMORY_LEN`
//! preamble cells. The public input is the 8-cell Poseidon fold of the
//! committed data `[a | b | denom | q | q_up]`, computed here with the same
//! left-fold the guest performs in `commit_fold` (so a valid witness exists iff
//! the committed I/O matches).

use backend::{ArenaVec, PrimeCharacteristicRing, arena_vec, poseidon16_compress_pair};
use lean_vm::{Bytecode, ExecutionWitness, F, Hints, PUBLIC_INPUT_LEN};

use crate::bigint::U256;
use crate::compilation::PREAMBLE_MEMORY_LEN;
use crate::reference::{
    HintBuf, SqrtInputs, SwapInputs, emit_sqrt_hints, emit_swap_hints, raw_muldiv_inputs,
    sqrt_op_head_scalars, sqrt_op_result, sqrt_op_swap_slots, swap_public_input_for,
};

const UINT256_LIMBS: usize = 32;
const DIGEST_LEN: usize = 8;

/// Left-fold a slice (length a multiple of 8) into an 8-cell digest, matching
/// the guest's `commit_fold`:
///   acc_0   = chunk_0
///   acc_k+1 = poseidon16_compress_pair(acc_k, chunk_k+1)
fn commit_fold(data: &[F]) -> [F; DIGEST_LEN] {
    assert!(!data.is_empty() && data.len().is_multiple_of(DIGEST_LEN));
    let mut acc: [F; DIGEST_LEN] = data[..DIGEST_LEN].try_into().unwrap();
    let n_chunks = data.len() / DIGEST_LEN;
    for c in 1..n_chunks {
        let chunk: [F; DIGEST_LEN] = data[c * DIGEST_LEN..(c + 1) * DIGEST_LEN].try_into().unwrap();
        acc = poseidon16_compress_pair(&acc, &chunk);
    }
    acc
}

/// Public-input digest for a Phase-1 `(a, b, denom)` mulDiv, routed through the
/// op-7 (`RawMulDiv`) driver layout: the committed result is the floor quotient.
pub fn public_input_for(a: U256, b: U256, denom: U256) -> [F; PUBLIC_INPUT_LEN] {
    sqrt_public_input_for(&raw_muldiv_inputs(a, b, denom))
}

/// Turn an ordered list of `HintBuf`s into a `Hints` map, preserving per-label
/// FIFO order (the guest pops in the same order).
pub fn hints_from_bufs(bytecode: &Bytecode, bufs: &[HintBuf]) -> Hints {
    use std::collections::BTreeMap;
    let mut grouped: BTreeMap<&'static str, Vec<ArenaVec<F>>> = BTreeMap::new();
    for buf in bufs {
        grouped
            .entry(buf.label)
            .or_default()
            .push(ArenaVec::from_slice(&buf.cells));
    }
    let mut hints = Hints::default();
    for (label, entries) in grouped {
        let mut av: ArenaVec<ArenaVec<F>> = arena_vec![];
        for e in entries {
            av.push(e);
        }
        hints.insert(bytecode, label, av);
    }
    hints
}

/// Build the honest `(public_input, witness)` for a `(a, b, denom)` mulDiv.
pub fn build_muldiv_witness(
    bytecode: &Bytecode,
    a: U256,
    b: U256,
    denom: U256,
) -> ([F; PUBLIC_INPUT_LEN], ExecutionWitness) {
    let public_input = public_input_for(a, b, denom);
    let bufs = emit_sqrt_hints(&raw_muldiv_inputs(a, b, denom));
    let hints = hints_from_bufs(bytecode, &bufs);
    let witness = ExecutionWitness {
        preamble_memory_len: PREAMBLE_MEMORY_LEN,
        hints,
        ..ExecutionWitness::default()
    };
    (public_input, witness)
}

/// Build a witness from an EXPLICIT, possibly-corrupted hint list (for negative
/// soundness tests). `public_input` is computed from the honest `(a,b,denom)` so
/// the digest is well-formed; only the hint buffers are under test.
pub fn build_witness_with_bufs(
    bytecode: &Bytecode,
    a: U256,
    b: U256,
    denom: U256,
    bufs: &[HintBuf],
) -> ([F; PUBLIC_INPUT_LEN], ExecutionWitness) {
    let public_input = public_input_for(a, b, denom);
    let hints = hints_from_bufs(bytecode, bufs);
    let witness = ExecutionWitness {
        preamble_memory_len: PREAMBLE_MEMORY_LEN,
        hints,
        ..ExecutionWitness::default()
    };
    (public_input, witness)
}

// ===========================================================================
// Phase 2/3/4: SqrtPriceMath / LiquidityMath / TickMath / BitMath / TickBitmap
// driver witness + public input.
// ===========================================================================
//
// COMMIT LAYOUT (must match main.py exactly): 304 cells = 38 eight-cell chunks
//   head (16 cells) : [op, flag_a, y_sign, tick_sign, abs_tick, sign_p1, abs_p1,
//                      tick_spacing, bit_pos, next_val, initialized, fee_pips,
//                      amount_sign, 0,0,0]
//   sqrtA(32) | sqrtB(32) | liquidity(32) | amount(32) | result(32) | word(32)
//   | swap_in(32) | swap_out(32) | fee(32)
// folded by `commit_fold` into the 8-cell public input. The tick scalars bind
// the TickMath ops 8/9 answer; bit_pos binds BitMath ops 10/11; (next_val,
// initialized) bind TickBitmap op 12; `word` is the committed bitmap input;
// (fee_pips, amount_sign) + (result=sqrtRatioNext, swap_in, swap_out, fee) bind
// SwapMath op 13's inputs and all four outputs.

const COMMIT_HEAD: usize = 16;
const COMMIT_U256_SLOTS: usize = 9;

/// Public-input digest for a driver invocation, from the honest result + head.
pub fn sqrt_public_input_for(inp: &SqrtInputs) -> [F; PUBLIC_INPUT_LEN] {
    let result = sqrt_op_result(inp);
    let (next_val, initialized) = sqrt_op_head_scalars(inp);
    let (swap_in, swap_out, fee) = sqrt_op_swap_slots(inp);

    let mut data = Vec::with_capacity(COMMIT_HEAD + COMMIT_U256_SLOTS * UINT256_LIMBS);
    data.push(F::from_u32(inp.op.code()));
    data.push(F::from_u32(inp.flag_a as u32));
    data.push(F::from_u32(inp.y_sign as u32));
    data.push(F::from_u32(inp.tick_sign as u32));
    data.push(F::from_u32(inp.abs_tick));
    data.push(F::from_u32(inp.sign_p1 as u32));
    data.push(F::from_u32(inp.abs_p1));
    data.push(F::from_u32(inp.tick_spacing));
    data.push(F::from_u32(inp.bit_pos));
    data.push(next_val);
    data.push(F::from_u32(initialized));
    data.push(F::from_u32(inp.fee_pips));
    data.push(F::from_u32(inp.amount_sign as u32));
    for _ in 13..COMMIT_HEAD {
        data.push(F::ZERO);
    }
    data.extend(inp.sqrt_a.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(inp.sqrt_b.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(inp.liquidity.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(inp.amount.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(result.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(inp.word.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(swap_in.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(swap_out.to_le_base256_limbs(UINT256_LIMBS));
    data.extend(fee.to_le_base256_limbs(UINT256_LIMBS));
    commit_fold(&data)
}

/// Build the honest `(public_input, witness)` for a Phase-2 invocation.
pub fn build_sqrt_witness(bytecode: &Bytecode, inp: &SqrtInputs) -> ([F; PUBLIC_INPUT_LEN], ExecutionWitness) {
    let public_input = sqrt_public_input_for(inp);
    let bufs = emit_sqrt_hints(inp);
    let hints = hints_from_bufs(bytecode, &bufs);
    let witness = ExecutionWitness {
        preamble_memory_len: PREAMBLE_MEMORY_LEN,
        hints,
        ..ExecutionWitness::default()
    };
    (public_input, witness)
}

/// Build a Phase-2 witness from an EXPLICIT (possibly-corrupted) hint list, with
/// the public input pinned to the HONEST result of `inp` (so only the hints are
/// under test). Used by the negative soundness tests.
pub fn build_sqrt_witness_with_bufs(
    bytecode: &Bytecode,
    inp: &SqrtInputs,
    bufs: &[HintBuf],
) -> ([F; PUBLIC_INPUT_LEN], ExecutionWitness) {
    let public_input = sqrt_public_input_for(inp);
    let hints = hints_from_bufs(bytecode, bufs);
    let witness = ExecutionWitness {
        preamble_memory_len: PREAMBLE_MEMORY_LEN,
        hints,
        ..ExecutionWitness::default()
    };
    (public_input, witness)
}

// ===========================================================================
// Phase 5b: full Pool.swap (SWAP mode) witness + public input.
// ===========================================================================

/// Build the honest `(public_input, witness)` for a full Pool.swap.
pub fn build_swap_witness(bytecode: &Bytecode, inp: &SwapInputs) -> ([F; PUBLIC_INPUT_LEN], ExecutionWitness) {
    let public_input = swap_public_input_for(inp);
    let bufs = emit_swap_hints(inp);
    let hints = hints_from_bufs(bytecode, &bufs);
    let witness = ExecutionWitness {
        preamble_memory_len: PREAMBLE_MEMORY_LEN,
        hints,
        ..ExecutionWitness::default()
    };
    (public_input, witness)
}

/// Build a SWAP witness from an EXPLICIT (possibly-corrupted) hint list, with the
/// public input pinned to the HONEST result of `inp`. For negative soundness
/// tests (forged words / liquidityNets / num_steps / results).
pub fn build_swap_witness_with_bufs(
    bytecode: &Bytecode,
    inp: &SwapInputs,
    bufs: &[HintBuf],
) -> ([F; PUBLIC_INPUT_LEN], ExecutionWitness) {
    let public_input = swap_public_input_for(inp);
    let hints = hints_from_bufs(bytecode, bufs);
    let witness = ExecutionWitness {
        preamble_memory_len: PREAMBLE_MEMORY_LEN,
        hints,
        ..ExecutionWitness::default()
    };
    (public_input, witness)
}

/// Build a SWAP witness with the public input pinned to an EXPLICIT digest (for
/// forged-result negative tests: honest hints, tampered commitment).
pub fn build_swap_witness_with_pi(
    bytecode: &Bytecode,
    inp: &SwapInputs,
    public_input: [F; PUBLIC_INPUT_LEN],
) -> ([F; PUBLIC_INPUT_LEN], ExecutionWitness) {
    let bufs = emit_swap_hints(inp);
    let hints = hints_from_bufs(bytecode, &bufs);
    let witness = ExecutionWitness {
        preamble_memory_len: PREAMBLE_MEMORY_LEN,
        hints,
        ..ExecutionWitness::default()
    };
    (public_input, witness)
}
