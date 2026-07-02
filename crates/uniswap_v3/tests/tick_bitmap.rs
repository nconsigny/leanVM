//! Phase 4 — BitMath + TickBitmap: differential, edge, negative, prove+verify.
//!
//! 1. REFERENCE hand-vectors: ref_next_init / msb / lsb on known bit patterns
//!    (single set bit at each end, word==0, bitPos 0 and 255, negative ticks).
//! 2. DIFFERENTIAL via `try_execute` (witness from the trusted reference; Ok ==
//!    "guest agrees with the v3 BitMath/TickBitmap port"):
//!      - most/least_significant_bit over random + edge uint256 (op 10/11);
//!      - nextInitializedTickWithinOneWord over random (tick, ts in {1,10,60,200},
//!        lte both, random word) + edges (word==0, single-bit words, bitPos
//!        0/255, negative ticks crossing a word boundary) (op 12).
//! 3. NEGATIVE soundness: a forged msb/lsb position rejected (ops 10/11/12); a
//!    forged bitmap word rejected via the commitment; a non-canonical word limb
//!    rejected; a forged next_val commitment rejected — each with the honest
//!    counterpart passing.
//! 4. PROVE+VERIFY: representative MSB, LSB, and NextInit vectors (release only;
//!    debug trips a known prover-internal SIMD assert — see Phase 1).

use backend::PrimeCharacteristicRing;
use lean_vm::F;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::sync::{Mutex, MutexGuard};
use uniswap_v3::{
    SqrtInputs, U256, build_sqrt_witness, build_sqrt_witness_with_bufs, emit_sqrt_hints,
    get_bytecode, least_significant_bit_inputs, most_significant_bit_inputs, next_init_inputs,
    prove, ref_least_significant_bit, ref_most_significant_bit, ref_next_init, ref_position, setup,
    try_execute, verify,
};

static ARENA_LOCK: Mutex<()> = Mutex::new(());

fn serialize_arena() -> MutexGuard<'static, ()> {
    ARENA_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Random uint256 with a controllable number of nonzero words.
fn rand_u256(rng: &mut StdRng) -> U256 {
    U256 {
        words: [rng.random(), rng.random(), rng.random(), rng.random()],
    }
}

/// Run the guest on `inp` and assert it accepts. Witness from the trusted
/// reference, so Ok == "guest agrees with the v3 port".
fn assert_accepts(inp: &SqrtInputs) {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();
    let (public_input, witness) = build_sqrt_witness(bytecode, inp);
    try_execute(bytecode, &public_input, &witness).unwrap_or_else(|e| panic!("guest must accept {inp:?}: {e:?}"));
}

// ---------------------------------------------------------------------------
// 1. Reference hand-vectors (the magic-constant-free safety net for Phase 4).
// ---------------------------------------------------------------------------

#[test]
fn reference_bitmath_hand_vectors() {
    // mostSignificantBit / leastSignificantBit on known patterns.
    assert_eq!(ref_most_significant_bit(U256::from_u64(1)), 0);
    assert_eq!(ref_least_significant_bit(U256::from_u64(1)), 0);
    assert_eq!(ref_most_significant_bit(U256::from_u64(0b1011)), 3);
    assert_eq!(ref_least_significant_bit(U256::from_u64(0b1000)), 3);
    assert_eq!(ref_most_significant_bit(U256::ONE.shl(255)), 255);
    assert_eq!(ref_least_significant_bit(U256::ONE.shl(255)), 255);
    assert_eq!(ref_most_significant_bit(U256::MAX), 255);
    assert_eq!(ref_least_significant_bit(U256::MAX), 0);
    // 2^200 + 2^7: msb 200, lsb 7.
    let v = U256::ONE.shl(200).overflowing_add(U256::ONE.shl(7)).0;
    assert_eq!(ref_most_significant_bit(v), 200);
    assert_eq!(ref_least_significant_bit(v), 7);
}

#[test]
fn reference_position_signed() {
    // position(t): wordPos = floor(t/256), bitPos = t mod 256 (nonneg).
    assert_eq!(ref_position(0), (0, 0));
    assert_eq!(ref_position(255), (0, 255));
    assert_eq!(ref_position(256), (1, 0));
    assert_eq!(ref_position(-1), (-1, 255));
    assert_eq!(ref_position(-256), (-1, 0));
    assert_eq!(ref_position(-257), (-2, 255));
}

#[test]
fn reference_next_init_hand_vectors() {
    // word with a single set bit at position b; tick 0, ts 1.
    // lte from compressed=0, bitPos=0: mask = low 1 bit; only bit 0 visible.
    let only_bit0 = U256::ONE; // bit 0
    let (next, init, bp, _m, _wp) = ref_next_init(0, 1, true, only_bit0);
    assert!(init && bp == 0 && next == 0, "lte tick0 bit0: next={next} init={init} bp={bp}");

    // lte tick 0, word has only bit 5 set: bit 5 > bitPos(0) => not in low mask => uninit.
    let only_bit5 = U256::ONE.shl(5);
    let (next, init, _bp, _m, _wp) = ref_next_init(0, 1, true, only_bit5);
    assert!(!init && next == 0, "lte tick0 bit5 uninit: next={next} init={init}");

    // gt tick 0, ts 1: position(compressed+1=1) bitPos=1, mask = bits >= 1.
    // word bit 5 set: lsb(masked)=5 => next = (0+1 + (5-1))*1 = 5.
    let (next, init, bp, _m, _wp) = ref_next_init(0, 1, false, only_bit5);
    assert!(init && bp == 1 && next == 5, "gt tick0 bit5: next={next} init={init} bp={bp}");

    // gt tick 0, word == 0: uninit. next = (0+1 + (255 - bitPos)) * 1, bitPos=1 => 255.
    let (next, init, _bp, _m, _wp) = ref_next_init(0, 1, false, U256::ZERO);
    assert!(!init && next == 255, "gt tick0 word0: next={next} init={init}");

    // Negative tick crossing a word boundary. tick=-1, ts=1: compressed=-1.
    // lte: position(-1) bitPos=255. mask = all bits (low 256). word with bit 255:
    let bit255 = U256::ONE.shl(255);
    let (next, init, bp, _m, _wp) = ref_next_init(-1, 1, true, bit255);
    assert!(init && bp == 255 && next == -1, "lte tick-1 bit255: next={next} init={init} bp={bp}");
    // lte tick=-1, word bit 0 only: msb(masked)=0, next = (-1 - (255-0)) = -256.
    let (next, init, _bp, _m, _wp) = ref_next_init(-1, 1, true, U256::ONE);
    assert!(init && next == -256, "lte tick-1 bit0: next={next} init={init}");
}

// ---------------------------------------------------------------------------
// 2. DIFFERENTIAL.
// ---------------------------------------------------------------------------

#[test]
fn differential_msb_lsb_edges() {
    for v in [
        U256::from_u64(1),
        U256::from_u64(2),
        U256::from_u64(255),
        U256::from_u64(256),
        U256::ONE.shl(127),
        U256::ONE.shl(128),
        U256::ONE.shl(255),
        U256::MAX,
        U256::ONE.shl(200).overflowing_add(U256::ONE.shl(7)).0,
    ] {
        assert_accepts(&most_significant_bit_inputs(v));
        assert_accepts(&least_significant_bit_inputs(v));
    }
}

#[test]
fn differential_msb_lsb_random() {
    let mut rng = StdRng::seed_from_u64(0xB17_4A11);
    for _ in 0..30 {
        let v = rand_u256(&mut rng);
        if v.is_zero() {
            continue; // BitMath requires x != 0
        }
        assert_accepts(&most_significant_bit_inputs(v));
        assert_accepts(&least_significant_bit_inputs(v));
    }
}

#[test]
fn differential_next_init_random() {
    let mut rng = StdRng::seed_from_u64(0x7100_B17A);
    let spacings = [1u32, 10, 60, 200];
    for _ in 0..50 {
        let ts = spacings[(rng.random::<u64>() % 4) as usize];
        let lte = rng.random::<u64>() & 1 == 0;
        // tick in a wide range incl. negatives, multiple of small values sometimes.
        let tick = (rng.random::<u64>() % 200_001) as i32 - 100_000;
        let word = rand_u256(&mut rng);
        assert_accepts(&next_init_inputs(tick, ts, lte, word));
    }
}

#[test]
fn differential_next_init_edges() {
    // word == 0 both branches (the uninitialized fall-through).
    for lte in [true, false] {
        for tick in [0i32, 1, -1, 255, 256, -256, -257, 12345, -12345] {
            for ts in [1u32, 60] {
                assert_accepts(&next_init_inputs(tick, ts, lte, U256::ZERO));
            }
        }
    }
    // single-bit words at each end + a few interior bits.
    for b in [0u32, 1, 7, 8, 127, 128, 254, 255] {
        let word = U256::ONE.shl(b);
        for lte in [true, false] {
            for tick in [0i32, 5, -5, 256, -256, -1] {
                assert_accepts(&next_init_inputs(tick, 1, lte, word));
            }
        }
    }
    // bitPos == 0 (compressed multiple of 256, lte) and bitPos == 255 cases.
    // compressed = tick/ts; tick = 256, ts = 1 -> compressed 256 -> bitPos 0 (lte).
    assert_accepts(&next_init_inputs(256, 1, true, U256::MAX));
    // tick = 255 -> compressed 255 -> bitPos 255 (lte).
    assert_accepts(&next_init_inputs(255, 1, true, U256::MAX));
    // gt with compressed+1 hitting bitPos 0: tick = 255 -> compressed 255 -> +1=256 -> bitPos 0.
    assert_accepts(&next_init_inputs(255, 1, false, U256::MAX));
    // negative tick crossing a word boundary with a full word.
    assert_accepts(&next_init_inputs(-1, 1, true, U256::MAX));
    assert_accepts(&next_init_inputs(-1, 1, false, U256::MAX));
    assert_accepts(&next_init_inputs(-256, 1, true, U256::MAX));
    assert_accepts(&next_init_inputs(-257, 1, false, U256::MAX));
}

#[test]
fn differential_next_init_spacing_floor() {
    // Exercise the signed FLOOR division for negatives that are NOT multiples of
    // tickSpacing (compressed must round toward -inf, i.e. decrement).
    for (tick, ts) in [(-1i32, 60u32), (-59, 60), (-61, 60), (-199, 200), (-201, 200), (-7, 10)] {
        for lte in [true, false] {
            assert_accepts(&next_init_inputs(tick, ts, lte, U256::MAX));
            assert_accepts(&next_init_inputs(tick, ts, lte, U256::ONE.shl(130)));
        }
    }
}

// ---------------------------------------------------------------------------
// 3. NEGATIVE soundness tests (honest counterpart passes; forgery rejected).
// ---------------------------------------------------------------------------

#[test]
fn negative_forged_msb_position_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let x = U256::ONE.shl(200).overflowing_add(U256::ONE.shl(7)).0; // msb 200
    let honest = most_significant_bit_inputs(x);
    let h = emit_sqrt_hints(&honest);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &honest, &h);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest MSB must pass");

    // Forge p = 201 (a higher, non-set bit): 2^201 > x breaks `2^p <= x`. Pin the
    // public input to the FORGED commitment so the rejection comes from BitMath.
    let mut forged = honest;
    forged.bit_pos = 201;
    let fh = emit_sqrt_hints(&forged);
    let (pi_forged, w_forged) = build_sqrt_witness_with_bufs(bytecode, &forged, &fh);
    assert!(
        try_execute(bytecode, &pi_forged, &w_forged).is_err(),
        "guest must reject a too-high MSB position"
    );

    // Forge p = 199 (a lower bit, but bit 200 is set above it): x >= 2^200 breaks
    // `x < 2^(p+1)`.
    let mut forged_lo = honest;
    forged_lo.bit_pos = 199;
    let flh = emit_sqrt_hints(&forged_lo);
    let (pi_lo, w_lo) = build_sqrt_witness_with_bufs(bytecode, &forged_lo, &flh);
    assert!(
        try_execute(bytecode, &pi_lo, &w_lo).is_err(),
        "guest must reject a too-low MSB position"
    );
}

#[test]
fn negative_forged_lsb_position_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let x = U256::ONE.shl(200).overflowing_add(U256::ONE.shl(7)).0; // lsb 7
    let honest = least_significant_bit_inputs(x);
    let h = emit_sqrt_hints(&honest);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &honest, &h);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest LSB must pass");

    // Forge p = 8 (above the true lsb 7): keep_low(x, 9) has bit 7 set, != 2^8.
    let mut forged_hi = honest;
    forged_hi.bit_pos = 8;
    let fhh = emit_sqrt_hints(&forged_hi);
    let (pi_hi, w_hi) = build_sqrt_witness_with_bufs(bytecode, &forged_hi, &fhh);
    assert!(
        try_execute(bytecode, &pi_hi, &w_hi).is_err(),
        "guest must reject a too-high LSB position"
    );

    // Forge p = 6 (below the true lsb 7): keep_low(x, 7) == 0, != 2^6.
    let mut forged_lo = honest;
    forged_lo.bit_pos = 6;
    let flh = emit_sqrt_hints(&forged_lo);
    let (pi_lo, w_lo) = build_sqrt_witness_with_bufs(bytecode, &forged_lo, &flh);
    assert!(
        try_execute(bytecode, &pi_lo, &w_lo).is_err(),
        "guest must reject a too-low LSB position"
    );
}

#[test]
fn negative_forged_nextinit_bitpos_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // An INITIALIZED case so the msb/lsb hint is actually used. gt, tick 0, ts 1,
    // word bit 5 -> bitPos 1, masked bit 5, lsb 5.
    let word = U256::ONE.shl(5);
    let honest = next_init_inputs(0, 1, false, word);
    let (_n, init, _bp, _m, _wp) = ref_next_init(0, 1, false, word);
    assert!(init, "test setup: case must be initialized");
    let h = emit_sqrt_hints(&honest);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &honest, &h);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest NextInit must pass");

    // Forge the lsb hint to a wrong bit (4). Pin the PI to the forged commitment
    // (head bit_pos = 4); the rejection must come from BitMath's lsb verification.
    let mut forged = honest;
    forged.bit_pos = 4;
    let fh = emit_sqrt_hints(&forged);
    let (pi_forged, w_forged) = build_sqrt_witness_with_bufs(bytecode, &forged, &fh);
    assert!(
        try_execute(bytecode, &pi_forged, &w_forged).is_err(),
        "guest must reject a wrong msb/lsb hint in NextInit"
    );
}

#[test]
fn negative_forged_word_rejected_via_commitment() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Honest NextInit on a specific word.
    let word = U256::ONE.shl(100).overflowing_add(U256::ONE.shl(40)).0;
    let honest = next_init_inputs(0, 1, false, word);
    let h = emit_sqrt_hints(&honest);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &honest, &h);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest NextInit must pass");

    // Feed a DIFFERENT word in the witness but keep the public input pinned to the
    // honest word's commitment: the guest re-commits the (forged) word and the
    // digest no longer matches -> rejected by the commitment.
    let mut forged = honest;
    forged.word = U256::ONE.shl(101).overflowing_add(U256::ONE.shl(40)).0;
    // Recompute the honest answer for the forged word so its own internal math
    // would pass; only the commitment must catch the swap.
    let forged_full = next_init_inputs(0, 1, false, forged.word);
    let fh = emit_sqrt_hints(&forged_full);
    assert!(
        try_execute(bytecode, &pi_ok, &build_sqrt_witness_with_bufs(bytecode, &honest, &fh).1).is_err(),
        "guest must reject a bitmap word that differs from the committed one"
    );
}

#[test]
fn negative_noncanonical_word_limb_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let word = U256::ONE.shl(50);
    let honest = next_init_inputs(0, 1, false, word);
    let h = emit_sqrt_hints(&honest);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &honest, &h);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest must pass");

    // Corrupt a `word` limb to 256 (non-canonical): range_check_limbs(word) rejects.
    let mut corrupt = h.clone();
    let widx = corrupt.iter().position(|hb| hb.label == "word").unwrap();
    corrupt[widx].cells[3] = F::from_u32(256);
    assert!(
        try_execute(bytecode, &pi_ok, &build_sqrt_witness_with_bufs(bytecode, &honest, &corrupt).1).is_err(),
        "guest must reject a non-canonical (>= 256) word limb"
    );
}

#[test]
fn negative_forged_nextval_commitment_rejected() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // Honest NextInit (initialized) on word bit 5, gt, tick 0, ts 1 -> next 5.
    let word = U256::ONE.shl(5);
    let honest = next_init_inputs(0, 1, false, word);
    let h = emit_sqrt_hints(&honest);
    let (pi_ok, w_ok) = build_sqrt_witness_with_bufs(bytecode, &honest, &h);
    try_execute(bytecode, &pi_ok, &w_ok).expect("honest must pass");

    // Pin the public input to a DIFFERENT case's commitment (different next_val)
    // but feed the honest hints. The guest computes next=5 and commits it; the
    // digest won't match the pinned (wrong-next) public input.
    let other = next_init_inputs(0, 1, false, U256::ONE.shl(6)); // next would be 6
    let (pi_wrong, _) = build_sqrt_witness(bytecode, &other);
    assert!(
        try_execute(bytecode, &pi_wrong, &w_ok).is_err(),
        "guest must reject a forged next_val commitment"
    );
}

// ---------------------------------------------------------------------------
// 4. PROVE + VERIFY (release only).
// ---------------------------------------------------------------------------

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn prove_and_verify_bit_math() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // MSB of a multi-word value.
    let x = U256::ONE.shl(200).overflowing_add(U256::ONE.shl(7)).0;
    let inp_msb = most_significant_bit_inputs(x);
    let (pi, w) = build_sqrt_witness(bytecode, &inp_msb);
    try_execute(bytecode, &pi, &w).expect("witness gen (op10) must succeed");
    let p = prove(bytecode, &pi, &w, 1).expect("prove op10 must succeed");
    verify(bytecode, &pi, p).expect("verify op10 must succeed");

    // LSB of the same value.
    let inp_lsb = least_significant_bit_inputs(x);
    let (pi2, w2) = build_sqrt_witness(bytecode, &inp_lsb);
    try_execute(bytecode, &pi2, &w2).expect("witness gen (op11) must succeed");
    let p2 = prove(bytecode, &pi2, &w2, 1).expect("prove op11 must succeed");
    verify(bytecode, &pi2, p2).expect("verify op11 must succeed");
}

#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn prove_and_verify_next_init() {
    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    // An initialized gt case crossing into a negative compressed (exercises the
    // signed floor div + masking + lsb hint).
    let word = U256::ONE.shl(130).overflowing_add(U256::ONE.shl(3)).0;
    let inp = next_init_inputs(-61, 60, false, word);
    let (pi, w) = build_sqrt_witness(bytecode, &inp);
    try_execute(bytecode, &pi, &w).expect("witness gen (op12) must succeed");
    let p = prove(bytecode, &pi, &w, 1).expect("prove op12 must succeed");
    verify(bytecode, &pi, p).expect("verify op12 must succeed");

    // An uninitialized lte case (word == 0 fall-through).
    let inp0 = next_init_inputs(12345, 60, true, U256::ZERO);
    let (pi0, w0) = build_sqrt_witness(bytecode, &inp0);
    try_execute(bytecode, &pi0, &w0).expect("witness gen (op12 uninit) must succeed");
    let p0 = prove(bytecode, &pi0, &w0, 1).expect("prove op12 uninit must succeed");
    verify(bytecode, &pi0, p0).expect("verify op12 uninit must succeed");
}

/// Phase-4 memory/trace/timing budget probe (run with `--release --nocapture`).
#[cfg_attr(debug_assertions, ignore = "prover trips a debug-only SIMD assert; run with --release")]
#[test]
fn tick_bitmap_budget_report() {
    use lean_vm::try_execute_bytecode;
    use std::time::Instant;

    let _guard = serialize_arena();
    setup();
    let bytecode = get_bytecode();

    let x = U256::ONE.shl(200).overflowing_add(U256::ONE.shl(7)).0;
    let word = U256::ONE.shl(130).overflowing_add(U256::ONE.shl(3)).0;
    let cases: [(&str, SqrtInputs); 3] = [
        ("most_significant_bit(2^200+2^7)", most_significant_bit_inputs(x)),
        ("least_significant_bit(2^200+2^7)", least_significant_bit_inputs(x)),
        ("nextInit(tick=-61, ts=60, gt, word)", next_init_inputs(-61, 60, false, word)),
    ];

    for (label, inp) in cases {
        let (pi, w) = build_sqrt_witness(bytecode, &inp);
        let res = try_execute_bytecode(bytecode, &pi, &w, false).expect("exec must succeed");
        let total_cells = res.memory.0.len();
        let runtime_cells = res.runtime_memory_size;
        let cycles = res.n_cycles();
        let log2_total = (usize::BITS - total_cells.next_power_of_two().leading_zeros() - 1) as usize;

        let t0 = Instant::now();
        let proof = prove(bytecode, &pi, &w, 1).expect("prove must succeed");
        let t_prove = t0.elapsed();
        let t1 = Instant::now();
        verify(bytecode, &pi, proof).expect("verify must succeed");
        let t_verify = t1.elapsed();

        println!("\n=== {label} ===");
        println!(
            "  total memory cells = {total_cells} (~2^{log2_total}); runtime cells = {runtime_cells}; cycles = {cycles}",
        );
        println!("  prove = {t_prove:.3?}; verify = {t_verify:.3?}");
    }
}
