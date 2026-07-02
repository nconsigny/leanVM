# Uniswap V3 swap, proven in leanVM — implementation plan

> Shared spec for the multi-phase (“ultracode”) build. Every workflow agent should read this first.
> Status legend: ☐ not started · ◐ in progress · ☑ done.

## Goal & locked scope

Implement a **full Uniswap V3 single-pool swap** as a zkDSL guest program (`.py`, compiles to leanVM
bytecode) and a Rust harness that **WHIR-proves and verifies** it against a `U256` reference.

- **Fidelity:** full — multi-tick crossing + in-circuit tick discovery (tick bitmap).
- **Deliverable:** prove **and** verify, end-to-end, in a Rust integration test.
- **Parity:** mirror v3-core naming (`FullMath`, `SqrtPriceMath`, `TickMath`, `SwapMath`,
  `LiquidityMath`, `TickBitmap`, `BitMath`). The authoritative spec is the **v3-core Solidity
  libraries**; `developers.uniswap.org` (SDK/subgraph/AI docs) is reference only for input/state shapes.

This is **application-layer** work. The ISA, Rust VM/runner, compiler, and proving system are **fixed
external interfaces**. If a phase appears to need a change there, **stop and surface it** — do not work around it.

## Field & VM constraints that shape everything

- Field: KoalaBear `p = 2^31 − 2^24 + 1` (~31-bit). No native ints wider than the field.
- Comparisons: only `== != < <=` (no `>`/`>=`; flip operands). `assert a < b` requires `b ≤ 2^16`;
  `assert a <= b` requires `b < 2^16`. **Wide comparisons must be done limbwise.**
- Memory is **write-once**; total execution must fit `< 2^26` cells (`MAX_LOG_MEMORY_SIZE = 26`).
- `DIM = 5` (extension field), `PUBLIC_INPUT_LEN = DIGEST_LEN = 8` (public input is `memory[0..8]`).
- leanVM is a **real VM** (`ADD/MUL/DEREF/JUMP` + Poseidon/ext-field precompiles): `if`/`range`/`match`
  are genuine control flow, so untaken branches are *not* executed (unlike flat R1CS) — exploited in `TickMath`.

## Architecture — new crate `crates/uniswap_v3/` (mirrors `crates/rec_aggregation`)

```
crates/uniswap_v3/
├── Cargo.toml                         # workspace member; deps: lean_compiler, lean_vm, lean_prover, (backend as needed)
├── PLAN.md                            # this file
├── zkdsl_implem/                      # the guest (Python-syntax zkDSL)
│   ├── __init__.py
│   ├── preamble.py                    # memory offsets + build_preamble_memory()
│   ├── helpers.py                     # local copies of checked_less_than / checked_decompose_bits / euclidean_div / copy_ef
│   ├── bignum.py                      # ★ limbs, range_check, lt/eq, add/sub, mul_schoolbook, muldiv(_rounding_up), shl/shr_96
│   ├── full_math.py                   # mulDiv / mulDivRoundingUp (thin over bignum)
│   ├── sqrt_price_math.py             # getAmount0/1Delta, getNextSqrtPriceFromInput/Output (+RoundingUp/Down)
│   ├── liquidity_math.py              # addDelta (uint128 ± int128, checked)
│   ├── bit_math.py                    # most/least_significant_bit (hint+constrain)
│   ├── tick_math.py                   # getSqrtRatioAtTick (exact), getTickAtSqrtRatio (bucket-verify)
│   ├── tick_bitmap.py                 # nextInitializedTickWithinOneWord
│   ├── swap_math.py                   # computeSwapStep
│   ├── swap.py                        # Pool.swap multi-tick loop
│   └── main.py                        # build preamble → read hints → run swap → commit to 8-cell public input
├── src/
│   ├── lib.rs                         # OnceLock<Bytecode>, include_dir, compile, setup(), prove/verify wrappers
│   ├── compilation.rs                 # replacements + PREAMBLE_MEMORY_LEN
│   ├── bigint.rs                      # minimal U256/I256 for the reference + hint encoding
│   ├── reference.rs                   # faithful Rust port; ALSO emits every hint buffer in consumption order
│   └── witness.rs                     # build ExecutionWitness (hints, preamble_memory_len, public input)
└── tests / repo tests/test_uniswap_v3.rs
```

## Harness API quick-reference (verified against source — file:line)

- **Compile:** `lean_compiler::compile_program_with_flags(&ProgramSource::Embedded{entry, dir}, CompilationFlags{replacements})`
  → `Bytecode`. `ProgramSource` `lib.rs:89`; entrypoints `lib.rs:162,170`; `CompilationFlags` `lib.rs:142`
  (`<NAME>_PLACEHOLDER` whole-identifier substitution).
- **Embed dir:** `include_dir!("$CARGO_MANIFEST_DIR/zkdsl_implem")` (see `rec_aggregation/src/compilation.rs:43`); cache in `static OnceLock<Bytecode>`.
- **Witness:** `ExecutionWitness { preamble_memory_len: usize, hints: Hints, min_table_log_n_rows }` (`lean_vm/.../runner.rs:22`); `..Default::default()`.
- **Hints:** `Hints::insert(&bytecode, name: &'static str, entries: ArenaVec<ArenaVec<F>>)` (`runner.rs:42`); build with `arena_vec![...]` (`backend/zk-alloc/src/lib.rs:22`) and `ArenaVec::from_slice(&[F])`. One inner vec per `hint_witness(name, ptr)` call, FIFO per label. (Example: `lean_compiler/tests/test_compiler.rs:51-57`.)
- **Public input:** separate `&[F; 8]` arg (NOT in the witness); typically a Poseidon digest the guest re-derives and asserts.
- **Run (fast dev loop, assertion-only):** `try_execute_bytecode(&bytecode, &public_input, &witness, profiling)` (`runner.rs:59`); panic-variant `execute_bytecode` (`runner.rs:90`). Minimal template: `test_compiler.rs:9`.
- **Prove:** `lean_prover::prove_execution(&bytecode, &public_input, &witness, &whir_config, vm_profiler)` (`prove_execution.rs:19`); `whir_config = lean_prover::default_whir_config(log_inv_rate)` (`lean_prover/src/lib.rs:33`), `log_inv_rate ∈ [1,4]`.
- **Verify:** `lean_prover::verify_execution(&bytecode, &public_input, execution_proof.proof)` — `ExecutionProof { proof: Proof<F>, metadata }`, so pass the `.proof` field (NOT `.proof.proof`) (`verify_execution.rs:15`).
- **Setup (once, before proving):** `lean_multisig::setup_prover()` (`/projects/leanVM/src/lib.rs:21`) enables the arena allocator, `parallel::init()`, bytecode init, `precompute_dft_twiddles`. ⚠ It initializes the *aggregation* bytecode — a Uniswap crate needs a crate-local `setup()` doing the same generic init for **our** bytecode. **Never prove two things concurrently** (shared arena). Integration-test shape: `tests/test_multisignatures.rs:52`.
- **Preamble:** `build_preamble_memory()` pattern at `rec_aggregation/zkdsl_implem/utils.py:11`, offsets at `hashing.py:8`. The runner does **not** zero the preamble — the guest writes its own constants. `preamble_memory_len` must equal the cells the offsets occupy.

## Harness facts established by Phase 0 (✅ verified — reuse verbatim)

Phase 0 is **green**: a real WHIR `prove_execution` + `verify_execution` of a thin no-op guest passes (both
adversarial reviewers pass, 0 blocking). Files: `crates/uniswap_v3/{Cargo.toml, src/{lib,compilation}.rs,
zkdsl_implem/{__init__,preamble,main}.py, tests/harness.rs}`. The `crates/*` glob auto-registers the crate — **root
Cargo.toml stays untouched.** Established facts to reuse in every later phase:

- **Import sources** (getting these wrong costs a compile cycle): `F` (=KoalaBear) is `lean_vm::F`, *not* `backend::F`.
  `RawProof` / `Proof<F>` / `ProofError` live in `backend` (fiat_shamir). `RunnerError`, `ExecutionWitness`, `Hints`,
  `PUBLIC_INPUT_LEN`, `Bytecode`, `try_execute_bytecode` are in `lean_vm`. `ExecutionProof { proof: Proof<F>, metadata }`
  — `verify_execution` takes the `.proof` field by value. `enable_arena` / `arena_vec!` / `ArenaVec` / `parallel` come via `backend::*`.
- **Setup:** crate-local `setup()` (in `src/lib.rs`) replicates only setup_prover's *generic* init for **our** bytecode
  (`enable_arena` + `parallel::init` + `init_bytecode` + `precompute_dft_twiddles::<F>(1<<24)`); it does **not** call
  `lean_multisig::setup_prover`. Idempotent (`OnceLock`). The arena is one shared per-process region — **serialize every
  proving test behind a static `Mutex`; never prove two things concurrently** (`tests/harness.rs` does this).
- **Public-input commitment (the Phase-4 mechanism, proven here):** in-guest `poseidon16_compress_half(L, R, out)` writes the
  **full 8-cell** digest `Poseidon(L‖R)[0..8] + L` ("half" = split L/R *inputs*, not a 4-cell output — all 8 cells are
  AIR-constrained). It equals Rust `backend::poseidon16_compress_pair(&L, &R)`. Reuse this exact pairing: host hashes the
  canonical 16-cell-aligned encoding of all state/params/results, guest recomputes it and asserts `== memory[0..8]`.
- **Hint encoding:** one `ArenaVec<F>` per `hint_witness(name, ptr)` call, FIFO per label; the runner errors unless every
  queued entry is consumed. `hints.insert(&bytecode, name, arena_vec![ArenaVec::from_slice(&buf)])`. **Phase 1: encode each
  U256 as little-endian base-2⁸ limbs = 32 cells; queue `q`/`r`/operand buffers in exact guest consumption order.**
- **Proving floor / profile:** a no-op proves in ~2.7s debug / ~0.11s release at `log_inv_rate=1`; cost is dominated by the
  proving floor (memory padded to ≥ 2¹⁶ cells), not the guest — bignum has headroom. `.cargo/config.toml` already sets
  `RUST_MIN_STACK=512MiB` + `target-cpu=native`; debug `cargo test -p uniswap_v3` works. Use `--release` from Phase 3+.
- **PREAMBLE_MEMORY_LEN = 21** today (ZERO_VEC 16 + ONE_EF 5), duplicated in `preamble.py` and `src/compilation.rs` — keep
  them in sync; the witness's `preamble_memory_len` MUST equal the cells `build_preamble_memory()` writes.

**Carry-in cleanups for Phase 1** (minor, flagged by review — do while adding bignum): drop the redundant `zk-alloc` /
`parallel` / `tracing` deps from `Cargo.toml` (reach them via `backend`); switch `lib.rs` to
`#![cfg_attr(not(test), allow(unused_crate_dependencies))]` (mirrors rec_aggregation, surfaces dep drift); add a docstring on
`main.py`'s `compress_half` noting it emits the full 8-cell digest; have later tests assert on the committed result, not just `Ok`/`Err`.

## Bignum API established by Phase 1 (✅ green — reuse)

`zkdsl_implem/bignum.py` (base 2⁸, all widths are `: Const`): `range_check_limbs`, `bignum_eq`, `bignum_lt`(→0/1),
`bignum_add`(→carry), `bignum_sub` (requires `a ≥ b`), `bignum_mul_schoolbook(NX,NY)`→`Array(NX+NY)` (self-range-checks
inputs), `shl_96`/`shr_96` (12 limbs = 96 bits; shr = floor), `bignum_muldiv(NA,NB,ND,NOUT)`→`Array(NOUT)`,
`bignum_muldiv_rounding_up(...)`, `_is_bignum_nonzero(x,N)`. `full_math.py`: `mulDiv`/`mulDivRoundingUp` (uint256/32-limb).
Widths: uint128=16, uint160=20, uint256=32, product ≤64. `bignum_muldiv` = floor (FullMath.mulDiv);
`bignum_muldiv_rounding_up` = ceil. Pass each op its true width to save cycles.
Rust to reuse: `src/bigint.rs` (`U256`/`U512`, LE base-256; **Ord is hand-written MS-word-first**; divmod holds the remainder
in 512 bits), `src/reference.rs` (muldiv + ordered hint emission), `src/witness.rs` (`build_witness_with_bufs`,
perturbed-buffer negative tests, `commit_fold`/`public_input_for`).

**Hint-ordering contract (do not drift):** one (`muldiv_q` 64-limb, `muldiv_r` 32-limb) pair consumed per
`bignum_muldiv`/`_rounding_up` call, FIFO per label; the host (`reference.rs`) MUST push q/r buffers in the exact order the
guest calls muldiv. New `hint_witness` inputs (sqrtP, liquidity, amount, …) get new labels documented at the top of BOTH the
`.py` and `reference.rs`.

**⚠ Debug-mode prover caveat (fixed-interface — do NOT work around from guest/witness):** a non-trivial guest trips a
debug-only `debug_assert` (`log_packing_width ≤ eval_points.len()`) in `backend/poly/src/eq_mle.rs` during `prove_execution`
— a SIMD edge case, **compiled out in release**. Gate prove+verify tests with `#[cfg_attr(debug_assertions, ignore = "...")]`
and run them via `cargo test -p uniswap_v3 --release`; keep fast `try_execute` differential/negative tests ungated so debug
`cargo test -p uniswap_v3` stays green.

**Compiler gotchas:** `@inline` fns cannot take `: Const` params (use plain `def`); top-level constants must precede the first
`def`; duplicate constant names across `import *` is a compile error; use `div_floor(a,b)` not `//`; compile-time
`min(u,v) = u − saturating_sub(u,v)`.

**Phase-1 carry-in fixes to apply in Phase 2** (minor, from the panel): add `range_check_limbs` on `bignum_sub`'s inputs (it
gets its first caller in `SqrtPriceMath` — currently a latent non-canonical-limb forgery vector); add negative tests for the
`(q−1, r+denom)` forgery (identity holds but `r ≥ denom`) and for a quotient exceeding `NOUT`; use `NAB` (not `NQ`) when
copying the product in `bignum_muldiv_rounding_up` (equal today, latent).

## SqrtPriceMath / LiquidityMath + op-dispatcher established by Phase 2 (✅ green)

`zkdsl_implem/sqrt_price_math.py`: `get_amount0_delta` / `get_amount1_delta` (→ uint256/32), `get_next_sqrt_price_from_input` /
`_from_output` / `_from_amount0_rounding_up` / `_from_amount1_rounding_down` (→ uint160/20, asserted to fit). `liquidity_math.py`:
`add_delta(x16, y_sign∈{0,1}, y_mag16)` → uint128/16. All match v3-core exactly (every formula in its docstring); the
`amount ≤ uint160_max` gas-branch is collapsed (provably bit-identical — full 512-bit product, no truncation). Overflow guards
(`(amount·sqrtP)/amount == sqrtP`) are **derived in-circuit** via `bignum_high_is_zero(product,64,32)` — unforgeable, not hinted.
New bignum helpers: `bignum_cast`, `bignum_assert_fits`, `bignum_is_zero`, `bignum_high_is_zero`, `bignum_add_fits`.
Constants: `make_one_u256`, `make_q96_u256` (2⁹⁶ ⇒ limb[12]=1), `make_uint160_max` (in sqrt_price_math.py).

**Op-dispatcher driver (`main.py`):** reads an `op` selector hint, `assert op < NUM_OPS` (`NUM_OPS=8`), then `match op` over ops
0–7 (`SqrtOp` enum in `reference.rs`). Input block (FIFO, one buffer each): `op, flag_a, sqrtA, sqrtB, liquidity, amount, y_sign`;
then `muldiv_q`(64)/`muldiv_r`(32) pairs in guest call order. Public input commits
`[op|flag_a|y_sign|pad | sqrtA|sqrtB|liquidity|amount|result]` (168 cells) via the Poseidon fold. **Phase 3+ adds op 8+ on the
same input-block + commit pattern — bump `NUM_OPS`, keep the `assert op < NUM_OPS` guard, add a `match` arm.**

**Hint policy:** all muldiv runs at uint256 width (q=64, r=32) ⇒ one pair per call, FIFO — the Rust `MdRecorder` replays the
guest's in-circuit branch decisions (`amount_zero`/`fits256`/`no_overflow`, all derived) and appends pairs in the same order, so
guest/host pair counts provably match per branch.

**Gotchas:** WRITE-ONCE constant building — write each cell exactly once (zero-prefix, set the nonzero limb, zero-suffix); never
zero-all-then-overwrite. Width constants live in ONE module each (`UINT128_LIMBS` in liquidity_math, `UINT256_LIMBS` in
full_math, `UINT160_LIMBS`/`SQRT_RESULT_LIMBS` in sqrt_price_math) — never redefine across `import *` (compile error).

**Phase-2 carry-in fixes for Phase 3** (minor, from the panel): harden `bignum_add` with `range_check_limbs` on both inputs
(match `bignum_sub`/`mul_schoolbook`; fixes the false soundness claim in `liquidity_math.py`'s docstring — currently sound only
because callers pass canonical limbs); fix the stale `NUM_OPS=7 / {0..6}` comment in `main.py` (it's 8 / {0..7}).

**For Phase 3 (TickMath):** needs the 19 magic Q128.128 multipliers + `MIN_TICK=-887272` / `MAX_TICK=887272` as guest constants;
`MIN_SQRT_RATIO=4295128739`, `MAX_SQRT_RATIO=1461446703485210103287273052203988822378723970342` (currently only in `reference.rs`
tests). **Boundary validators:** `getSqrtRatioAtTick(0)=2⁹⁶`, `(MIN_TICK)=MIN_SQRT_RATIO`, `(MAX_TICK)=MAX_SQRT_RATIO`.

## TickMath established by Phase 3 (✅ green, prove+verified)

`zkdsl_implem/tick_math.py`: `get_sqrt_ratio_at_tick(tick_sign, abs_tick)` (op 8 → uint160) = the **exact** bit-selected product
of the 20 magic Q128.128 constants (bit0 picks initial MAGIC[0] vs 2¹²⁸; bits 1..19 conditionally `(ratio*MAGIC)>>128` via real
VM `if`), `uint256.max / ratio` inversion for tick>0, `(ratio>>32)+ceil` downcast asserted to uint160. `get_tick_at_sqrt_ratio`
(op 9 → tick) hints the tick and bucket-verifies `ratio(tick) <= P < ratio(tick+1)`, hinted tick range-checked to
[MIN_TICK, MAX_TICK], unique by monotonicity. `NUM_OPS=10`. Boundary vectors validated guest+reference:
`getSqrtRatioAtTick(0)=2⁹⁶`, `(MIN_TICK)=4295128739`, `(MAX_TICK)=1461446703485210103287273052203988822378723970342`.

**⚠ NEW compiler constraints discovered in Phase 3 (NOT in zkDSL.md — critical for Phase 4+):**
- The zkDSL compiler accepts **decimal integer literals only — NO `0x` hex literals.**
- It **rejects local (in-function) list literals.** Build wide constants by **per-cell decimal writes** (write-once), or hoist
  shared byte tables to **top-level constants** (keep the hex in a trailing comment for readability).

**Comparing against a bound > 2¹⁶ (the legal idiom):** you cannot `assert x <= BOUND` when `BOUND > 2¹⁶`. Instead decompose
`diff = BOUND − x` into N bits (checked) — the decomposition succeeds iff `x <= BOUND` (Phase 3 gates `abs_tick ≤ MAX_TICK` this
way). Likewise derive a "nonzero" flag from proven bits, NOT `checked_less_than(0, x)`, when `x` can exceed 2¹⁶.

**Memory (release, log_inv_rate=1, incl. full op-dispatch+commit wrapper):** `getSqrtRatioAtTick` ≈ 27.2k cells (~2¹⁵), prove
~81 ms; `getTickAtSqrtRatio` ≈ 57.8k cells (~2¹⁶, two internal getSqrtRatioAtTick calls), prove ~71 ms — both far under the 2²⁶
ceiling. `tests/tick_math.rs::tickmath_budget_report` is the reusable budget probe. **Phase 5 swap loop calls getSqrtRatioAtTick
per crossed tick → cache reused `sqrtRatio(tick)` and re-checkpoint memory after Phase 5.**

**Phase-3 audit carry-ins (minor, for Phase 4/5):** (1) **canonicalize the `tick==0` encoding** — constrain
`tick_sign·(1−abs_nonzero)==0` so zero has a single committed form (otherwise `(sign=0,abs=0)` and `(sign=1,abs=0)` are both
valid → commitment malleability; any consumer MUST decode to the signed value `abs − 2·sign·abs`, never compare raw `(sign,abs)`).
(2) add a fixed tick exercising magic bits {4,9,10,12,13,14,17} (uncovered by the boundary vectors; soften the "single wrong byte
fails a boundary" comment). (3) add committed negative tests for op9 band edges (`P<MIN_SQRT_RATIO`, `P==MAX_SQRT_RATIO`) and a
direct `abs_p1≤MAX_TICK` test.
**All three carry-ins APPLIED in Phase 4** (canonical-zero assert now lives inside `get_sqrt_ratio_at_tick`, covering ops 8 & 9).

## BitMath / TickBitmap established by Phase 4 (✅ green, prove+verified)

`zkdsl_implem/bit_math.py`: `most_significant_bit(x, p)` (op 10) / `least_significant_bit(x, p)` (op 11) — HINT-AND-VERIFY: the
position `p` is read by the caller under hint label `bit_pos` and PASSED IN (BitMath does NOT hint internally); both require
`x != 0`. MSB proves `2^p ≤ x` (`bignum_lt(x,2^p)==0`) AND `x < 2^(p+1)` (a **33-limb** `2^(p+1)` so `p==255 → 2^256` and the
bound holds vacuously). LSB proves `keep_low(x, p+1) == 2^p` (one equality = bit p set + lower bits zero). Helpers: `make_two_pow(p,
width)` builds `2^p` for runtime `p` (limb_idx=p//8, bit_in=p%8 via `hint_div_floor`; `2^bit_in` via `match_range(bit_in,
range(0,9), 2**i)` — **`match_range` must be the RHS of an assignment, NOT a bare `return`**, else compile error "Function used but
not defined: match_range"); `keep_low_split(word, k)` → `(kept=word mod 2^k, cleared=word−kept)` per-limb, every produced limb
range-checked < 256.

`zkdsl_implem/tick_bitmap.py`: `next_initialized_tick_within_one_word(tick_sign, abs_tick, ts, lte, word, bit_pos_hint)` (op 12) →
`(next_val signed-field, initialized∈{0,1})`. **EXACT v3 port.** Key sound constructions:
- **Signed floor div** `compressed = floor(tick/ts)`: `hint_div_floor(abs_tick, ts, q, r)`; bind via the **10+10 split** — decompose
  `q` to 20 bits, split into `q_hi,q_lo` (10 bits each), `p_hi=q_hi*ts`, `p_lo=q_lo*ts` (each `< 2^26 < p`), `assert p_hi < 1024`
  (honest max is **866**, brute-forced), `assert p_hi*1024 + p_lo + r == abs_tick` (`< 2^27 < p`, integer-exact). A single field
  `q*ts` would overflow (up to `2^36 > p`); the split is what keeps it sound. Then `compressed = (1−sign)*q − sign*(q + (r≠0))`.
- **position()**: `bitPos = base mod 256`, `wordPos = floor(base/256)` for signed `base` (= compressed for lte, **compressed+1 for
  gt** — v3 calls `position(compressed+1)` in the gt branch). Done by shifting `base` by `OFFSET = 2^20` (a multiple of 256, > max
  |base|) into the nonnegative range, then plain euclidean div by 256 (`bit_pos_of`); bitPos is OFFSET-invariant.
- **MASKING IS MODULAR (no general bitwise AND — v3 masks are contiguous):** lte `masked = keep_low(word, bitPos+1)` (low bitPos+1
  bits); gt `masked = clear_low(word, bitPos)` = `keep_low_split(word, bitPos).cleared` (bits ≥ bitPos). `initialized = masked != 0`.
- **next** = `compressed_adjusted * ts` (signed field): lte init `(base − bitPos + msb)*ts`, lte uninit `(base − bitPos)*ts`, gt
  init `(base + lsb − bitPos)*ts`, gt uninit `(base + 255 − bitPos)*ts`. `|next| < 2^25 < p`; inputs already pinned ⇒ product exact.
  msb/lsb of `masked` computed via BitMath **only on the `if initialized` arm** (masked≠0 there); `bit_pos_hint` is consumed
  unconditionally in the input block but used only when initialized.

**Driver (`main.py`):** `NUM_OPS=13` (ops 0..12). **New input-block hint labels** (FIFO, one buffer each, read AFTER `abs_p1`,
BEFORE muldiv pairs): `tick_spacing`(1), `bit_pos`(1), `word`(32). Ops 10/11/12 emit **ZERO** muldiv pairs. **Commit layout
changed** (208 cells = 26 chunks): head is now **16 cells** `[op, flag_a, y_sign, tick_sign, abs_tick, sign_p1, abs_p1,
tick_spacing, bit_pos, next_val, initialized, 0×5]`, then `sqrtA|sqrtB|liquidity|amount|result|word` (6×32). The **bitmap `word`
is a committed input**; MSB/LSB answers are committed via head `bit_pos`; NextInit answers via head `next_val`/`initialized`.
`next_val` is committed as a raw **signed field element** (negative = p−|next|); the Rust `signed_to_f` mirrors it.

Rust (`src/reference.rs`): `ref_most_significant_bit`/`ref_least_significant_bit` (leading/trailing zeros on `U256`),
`ref_compressed` (signed floor), `ref_position`, `ref_next_init(tick, ts, lte, word) → (next, initialized, bit_pos, masked,
word_pos)`. `SqrtOp` += 3 variants; `SqrtInputs` += `tick_spacing/bit_pos/word` (use `..SqrtInputs::default()`); `compute_op`
returns `OpOutput{result, next_val, initialized, pairs}`; `sqrt_op_head_scalars` exposes `(next_val, initialized)` to witness.rs.
Builders: `most_significant_bit_inputs`, `least_significant_bit_inputs`, `next_init_inputs` (the latter computes the msb/lsb hint).

**Memory (release, log_inv_rate=1, full driver+commit):** MSB ≈ 6.0k cells (~2¹³), LSB ≈ 5.0k, NextInit ≈ 6.8k — all dominated by
the 2¹⁶ proving floor; these ops are nearly free. Prove ~45–75 ms each.

**For Phase 5 (computeSwapStep + swap loop + unified pool-state commitment):**
- The `next_val` of op 12 can fall OUTSIDE `[MIN_TICK, MAX_TICK]` (v3 returns it raw; the swap loop clamps to the limit). Phase 5
  must clamp before feeding `next` back into `getSqrtRatioAtTick` (which requires `|tick| ≤ MAX_TICK`).
- **Hint-consumption is fixed & branch-INDEPENDENT** for the per-step scalars (op/flags/tick scalars/tick_spacing/bit_pos/word read
  unconditionally; muldiv pairs FIFO in call order). For the swap loop, replicate this discipline PER STEP and **do NOT use
  `parallel_range`** on the state-carried loop (checklist #8).
- **Unified pool-state commitment:** today each op commits one `word`; Phase 5 needs MANY touched bitmap words + crossed ticks'
  `liquidityNet` + current tick/√P/liquidity + swap params, all Poseidon-folded into the 8-cell public input. Extend the same
  `commit_fold` over a canonical 8-cell-aligned encoding; bind every per-step `word`/`tick`/`liquidityNet` the loop consumes (a
  free per-step `word` like Phase 4's is fine in isolation but MUST be bound to the pool-state commitment in the loop).
- Reusable signed-field idioms now exist: `signed_floor_compressed` (floor div), `bit_pos_of` (mod-256 of a signed value),
  `make_two_pow`/`keep_low_split` (runtime power/mask), and `next_val` committed as a raw signed `F`. Reuse, don't reinvent.
- The `next` multiply `comp_adj * ts` is sound only because comp_adj and ts are pinned; if Phase 5 multiplies larger pinned values,
  re-check the `< p` bound (decompose if a product can reach p).

## SwapMath.computeSwapStep established by Phase 5a (✅ green, prove+verified, audited)

`zkdsl_implem/swap_math.py`: `compute_swap_step(current, target, liquidity, amount_sign, magnitude, fee_pips)` →
`(sqrtRatioNext, amountIn, amountOut, feeAmount)` = op 13 (`NUM_OPS=14`). EXACT v3 port. All branch selectors are **derived
in-circuit** (none hinted): `zeroForOne=1−bignum_lt(current,target)`, `exactIn=1−amount_sign`, `reach=bignum_le_flag(stage1,budget)`,
`max=bignum_eq_flag(target,next)`, `cap=bignum_lt(mag,amountOut)`. New bignum helpers: `bignum_eq_flag` (1−lt−lt), `bignum_le_flag`
(1−lt). Fee: `FEE_DENOM=1e6`; `feePips<1e6` gated via **dual 20-bit decomposition** (of `fee_pips` and of `999999−fee_pips`,
since 1e6>2¹⁶); `feeAmount = (exactIn ∧ next≠target) ? mag−amountIn : mulDivRoundingUp(amountIn,feePips,1e6−feePips)`.
**Commit layout grew to 9 u256 slots / 304 cells** (`sqrtA|sqrtB|liq|amount|result|word|swap_in|swap_out|fee`, head=16;
`COMMIT_U256_SLOTS=9` kept in sync across main.py + witness.rs); ops 0–12 default the 3 swap slots to a reseated shared zero
buffer (no write-once conflict). All inputs + all 4 outputs committed.
Rust: `ref_/rec_compute_swap_step` in reference.rs (ONE MdRecorder threaded; per-branch muldiv-pair counts pinned 7/5/7);
`U256::from_dec_str`, `U512::low_u256` added to bigint.rs. **Memory:** op 13 ≈ 2¹⁷ cells (~7 muldivs heaviest), prove ~125 ms.

**Phase-5a carry-in for Phase 5b:** add a **canonical signed-zero** constraint for `amountRemaining` —
`amount_sign·(1−is_nonzero(magnitude))==0` — so `(sign=1,mag=0)` is rejected (mirrors the `tick==0` carry-in); the loop must
decode `amountRemaining=(1−2·sign)·magnitude` and never compare raw `(sign,magnitude)`. (Also a trivial stale comment at
swap_math.py ~line 274.)

## Pool.swap loop established by Phase 5b (✅ green, prove+verified, audited)

`zkdsl_implem/swap.py` + `main.py` **SWAP mode** (mode=1; op-dispatcher ops 0–13 kept as mode=0). Full v3 multi-tick loop:
carried state (sqrtP, tick sign+abs, liquidity, amountSpecifiedRemaining int256, amountCalculated int256, running commitment
digest) threads through write-once buffers over a runtime `range` loop (**NO parallel_range**), bounded by host `num_steps`.
**Verified termination** binds num_steps exactly: each executed step asserts the v3 guard held (remaining≠0 ∧ sqrtP≠limit); after
the loop asserts it's false (remaining==0 ∨ sqrtP==limit). Signed int256 via `signed_add_u256` (canonical zero); `next_val`
clamped to [MIN_TICK,MAX_TICK]. **Unified commitment** Poseidon-folds context (zeroForOne, amountSpecified, sqrtPriceLimit, fee,
tickSpacing, num_steps) + initial state (sqrtP₀, tick₀, liquidity₀) + EVERY per-step `word` & `liquidityNet` (as consumed) +
final results (sqrtP, tick, liquidity, remaining, calculated, amount0/amount1) into the 8-cell PI. Rust: `ref_pool_swap` (ONE
threaded MdRecorder, all hints in guest order), `PoolFixture` (bitmap + per-tick liquidityNet), `I256`. **Memory:** ~130k
cells/step; `SWAP_MAX_STEPS=64` ≈ 2²³ (max ~516 steps fit under 2²⁶). 104 tests pass.

**Phase-6 carry-ins (from the 5b panel + build):** (1) **expand differential coverage** — exactOut multi-tick crossing,
`!zeroForOne` crossing, tickSpacing ∈ {1,10,200}, a swap crossing an UNINITIALIZED-word boundary, a deeper 8–16-crossing
prove+verify, and a fixture exercising the `moved` branch (`getTickAtSqrtRatio` recompute) under prove. (2) document/widen the
**crossed-branch `MIN_TICK−1` edge** (guest rejects it; v3 permits, but unreachable for valid swaps). (3) deep-audit: per-step
muldiv **FIFO interleaving** across heterogeneous-branch steps (`emit_swap_hints` appends pairs at end), the `|next_val|<2²⁵`
clamp bound for large tickSpacing, `signed_add_u256` on adversarial sign+mag, `fold_step_data` canonical padding, and full
**unified-commitment completeness**. (4) the strongest validation: an **independent external-v3-oracle differential** of
`ref_pool_swap` over many randomized pools (as Phase 5a did for computeSwapStep). (5) fix the stale NUM_OPS comment.

## zkDSL idioms to reuse (copy locally; imports resolve within the embedded dir)

- `euclidean_div_runtime(a,b)` → hint `q,r`; `assert r<b; assert q<2^16; assert q*b+r==a` (`hashing.py:97`).
- `checked_decompose_bits_small_value_const(v, n_bits)` → boolean-constrained bits + recomposition (`utils.py:614`).
- `checked_less_than(a,b)` → hint + boolean + dual-branch re-derivation (`utils.py:238`) — operands must be `≤ 2^16`.
- Write-once subtraction: `add_ee(b, c, a)` ⇒ `c = a − b` (`utils.py:338`); multiply-by-one copy `dot_product_ee(a, ONE_EF_PTR, b)` (`utils.py:377`).
- Loop-carried state via read-only buffers (`zkDSL.md` “For loops”; `hashing.py:134`); runtime→const dispatch via `match_range`.

## Bignum design (Phase 1 foundation)

- **Limbs little-endian, base B = 2^8 to start** (overflow on the 31-bit field is then impossible by construction).
  Counts: uint128 = 16 limbs, uint160 = 20, uint256 = 32, product = up to 64. Operand widths parameterized so
  each call pays only for its true width (uint160×uint128 ≪ uint256×uint256). Base 2^16 / mixed-radix is a later optimization.
- `range_check_limbs`, `bignum_eq`, `bignum_lt` (MS-limb-first decided flag using `checked_less_than` on 8-bit limbs),
  `bignum_add`/`bignum_sub` (carry/borrow ∈ {0,1}, each limb range-checked; sub via write-once, assert top borrow 0),
  `bignum_mul_schoolbook` (per-column `Σ x[i]·y[j]` provably `< p`, carry-extract via `hint_div_floor(·,256)`),
  `shl_96`/`shr_96` (limb shift by 12 at base 2^8).
- **`mulDiv(a,b,denom)` = the crux:** host hints `q` (NA+NB limbs) and `r` (ND limbs) → constrain
  `a·b == q·denom + r` over full product width → `bignum_lt(r,denom)==1` → range-check **every** limb → assert `denom ≠ 0`.
  `mulDivRoundingUp` = `mulDiv` + (`r≠0 ? q+1 : q`).

## Uniswap math map: in-circuit vs hint-and-constrain

- **In-circuit (built on bignum):** `mulDiv`/`mulDivRoundingUp`, `getAmount0/1Delta`,
  `getNextSqrtPriceFrom{Input,Output}`, `addDelta`, `computeSwapStep`, the swap loop.
- **`getSqrtRatioAtTick` — exact, in-circuit:** bit-selected product of the 19 magic Q128.128 constants
  (compile-time arrays), gated by bits of `|tick|`; `if bit==1:` skips untaken multiplies (real VM control flow).
- **`getTickAtSqrtRatio` — hint+verify:** hint `tick`; assert `sqrtRatio(tick) ≤ √P < sqrtRatio(tick+1)` (two `bignum_lt`).
- **`BitMath` MSB/LSB — hint+verify:** hint bit position `p`; assert `2^p ≤ x` and `x < 2^(p+1)` (bignum).
- **`TickBitmap.nextInitializedTickWithinOneWord` — hint+verify:** word/bit indexing, mask, nearest-set-bit via BitMath;
  assert chosen bit set and no nearer bit set in the search direction.
- **No `sqrt`, no `log` in-circuit** — the swap path never needs them (√P is tracked state).

## Pool-state commitment

Host supplies all pool state + swap params as hints (current tick/√P/liquidity, touched bitmap words, crossed ticks’
`liquidityNet`, swap direction/amount/limit). The guest Poseidon-hashes the canonical encoding and **asserts it equals
the 8-cell public input**, plus commits the swap result. Binds everything the prover could otherwise vary freely.

## Phased build

- ☑ **Phase 0 — Harness first.** ✅ green; both adversarial reviewers pass, 0 blocking. Scaffold the crate + workspace entry, `preamble.py`, a thin-but-real `main.py`
  (build preamble · read public input · consume ≥1 `hint_witness` · ≥1 binding `assert` · return), and the full
  Rust path (compile → witness → `prove_execution` → `verify_execution`). **Prove+verify this no-op end-to-end before any math.**
- ☑ **Phase 1 — Bignum + `FullMath.mulDiv`** (+ Rust `U256` reference). ✅ green; 4-lens panel found 0 forgery paths. The crux; heaviest soundness review.
- ☑ **Phase 2 — `SqrtPriceMath` + `LiquidityMath`.** ✅ green; 4-lens panel (fidelity/soundness/differential/hint-ordering) 0 blocking.
- ☑ **Phase 3 — `TickMath`** — exact `getSqrtRatioAtTick` (bit-selected magic-constant product) + bucket-verify `getTickAtSqrtRatio`. ✅ green; prove+verified (ops 8/9), boundary vectors validated; **independent 4-lens audit clean (0 blocking)**. (`BitMath` moved to Phase 4 — its only consumer is `TickBitmap`.)
- ☑ **Phase 4 — `BitMath` + `TickBitmap.nextInitializedTickWithinOneWord` + bitmap-word commitment.** ✅ green; prove+verified (ops 10/11/12); full release suite 70 tests pass, debug green. See "BitMath/TickBitmap established by Phase 4" below.
- ☑ **Phase 5a — `SwapMath.computeSwapStep`** (op 13, all branches). ✅ green; prove+verified; 87 tests; **independent 4-lens audit clean (0 blocking)**. See "established by Phase 5a" below.
- ☑ **Phase 5b — `Pool.swap` multi-tick loop + unified pool-state commitment.** ✅ green; prove+verified (SWAP mode); 104 tests; **independent 4-lens audit clean (0 blocking)**. See "established by Phase 5b" below.
- ☑ **Phase 6 — End-to-end:** ✅ green (debug + release). Representative swaps (single-tick in-range, multi-tick-crossing,
  exactIn/exactOut, BOTH directions) + a DEEP 8-16-crossing swap proved+verified, committed PI == `swap_public_input_for`
  (== `ref_pool_swap`). Expanded differential coverage (exactOut multi-tick, !zfo crossing, tickSpacing ∈ {1,10,200},
  uninitialized-word-boundary crossing, `moved`/getTickAtSqrtRatio-recompute branch). INDEPENDENT external-v3-oracle
  differential (`tests/swap_oracle.rs`) cross-checks `ref_pool_swap`'s full result over thousands of randomized
  pools/swaps. Minor fixes applied: documented the crossed-branch MIN_TICK−1 decode edge (unreachable for valid swaps;
  v3 reverts SPL when limit ≤ MIN_SQRT_RATIO) and refreshed the stale driver header comment. See "established by Phase 6" below.

## End-to-end established by Phase 6 (✅ green, prove+verified, externally cross-checked)

- **E2E prove+verify matrix** (`tests/swap.rs`, release-only, gated by the documented debug-SIMD-assert): `e2e_prove_verify`
  proves AND verifies, asserting `pi == swap_public_input_for(inp)` (so the proof is about the CORRECT result). Covers
  single-tick in-range, multi-tick exactIn/exactOut × zfo/!zfo, and a deep 8-16-crossing swap. All pass.
- **Memory checkpoint** (release, `try_execute` metadata, log_inv_rate=1): single-step ≈ 187.9k cells (~2^17.5);
  per-step incremental ≈ 116k-130k; deep 13-step swap ≈ 1.51M cells (~2^20.5), ~116k/step. SWAP_MAX_STEPS=64 ⇒ ≈ 8.5M
  (~2^23). Max steps under 2^26 ≈ 516. All comfortably under the 2^26 ceiling. (`swap_budget_report` is the probe.)
- **External-oracle differential** (`tests/swap_oracle.rs`): a self-contained `oracle::oracle_swap` re-ports the v3
  Pool.swap core loop + FullMath/SqrtPriceMath/SwapMath/TickBitmap/LiquidityMath from the Solidity spec, sharing only the
  unit-tested `U256`/`U512` primitives and `ref_get_sqrt_ratio_at_tick` (used as a validated numeric table). It uses a
  DIFFERENT bitmap representation (an init-position set, brute-search) and host i128 net arithmetic. Cross-checks full
  result (amount0/1, final sqrtP/tick/liquidity, num_steps) over thousands of randomized configs (ts ∈ {1,10,200,60},
  both directions, exactIn/Out, varied nets/limits); both sides must agree on accept-vs-revert. 0 mismatches.

## Cross-cutting soundness checklist (adversarial review hammers these every phase)

1. Every hint (`q/r`, tick, √ratio, MSB/LSB, next-tick, bitmap word, `liquidityNet`) is **bound by an assert**.
2. Every limb range-checked `< base` (unconstrained limb = multiple-representation forgery).
3. `mulDiv`: `r < denom`, equality across **full** product width, `denom ≠ 0`.
4. **No `assert a < b` with `b > 2^16`** — wide compares limbwise only.
5. `match`/`match_range` selectors provably in range (hard `assert` when prover-controlled, not just `debug_assert`).
6. Public-input commitment covers all state + params + results.
7. Rounding direction matches v3 exactly (`mulDiv` vs `mulDivRoundingUp`) — economic soundness.
8. `parallel_range` never on the state-carried swap loop.  9. Memory `< 2^26` cells.

## Ultracode execution model (per module)

**build** (`zkdsl-engineer` writes `.py` + Rust reference) → **cross-check** (`try_execute_bytecode` on random vectors
vs reference — the fast dev loop) → **adversarial soundness vote** (independent agents try to forge a proof: missing
range-check? unbound hint? out-of-range `match`? rounding mismatch?) → **fix loop until clean**. Phase milestone = the
module’s prove+verify passes. Workflows run in sequence; the orchestrator reviews between phases.

## Definition of done

`cargo test -p uniswap_v3 --release` compiles the guest, **WHIR-proves and verifies** single-tick + multi-tick-crossing
swaps (exact-in/out, both directions); each committed result equals the `U256` reference; soundness checklist green; memory `< 2^26`.

## Risks

- Dominant cost = `TickMath` × steps + `mulDiv`. Memory-budget checkpoints after Phases 3 & 5; mitigations: operand-width
  narrowing, caching reused `sqrtRatio(tick)`, base 2^16.
- Repo TODO notes the compiler is rough (“rewrite the compiler”); complex guests may hit compiler edges → surface, don’t hack around.
