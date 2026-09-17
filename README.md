# vole-field

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/infinityabundance/vole-field/blob/main/colab/vole_field_demo.ipynb)

A minimal, rigorous, native-Rust proof of **one** proposition:

> Useful generative state earned by an open local model can survive process
> termination, be persisted through EntropyFS, be restored later, and reduce repeated
> model computation for related future observations while preserving the same
> model-output contract.

Everything in this repository exists to test that sentence.

Paper context: *VOLE-Field — Deterministic Multimodal Field-State Representation,
Inverse Compilation, Entropy-Native Persistence, and Late Materialization*,
DOI [10.5281/zenodo.22805773](https://zenodo.org/records/22805773).

This is deliberately **not** an implementation of that paper. It does not implement the
paper's mechanisms, it is not a runtime, and it is not a framework. It is one small
experiment with a fair baseline, a real persistence path, and an honest negative
control. See [What this does not prove](#what-this-does-not-prove).

---

## What is in the box

| | |
|---|---|
| One Cargo package | `src/`, 6 modules |
| One tiny recurrent model | a single-layer ConvLSTM with a residual head, 3,937 parameters |
| One frozen checkpoint | `assets/tiny_convlstm.safetensors`, 17,036 bytes |
| One scene | `moving-shape-01`: one streak moving over a 32×32 field |
| One persisted state | `H`, `C` and the carry, exact `f32`, 73,728 bytes |
| Three recovery paths | replay-from-scratch, raw tensor file, EntropyFS blob |
| One negative case | an unrelated scene's request, refused |
| One notebook | `colab/vole_field_demo.ipynb` |

The whole experiment takes about **8 seconds** of wall time on 16 CPU cores after
compilation, and it launches **159 child processes**, each of which is verified to be a
distinct OS process. A clean release build from scratch takes about **28 seconds** on the
same 16 cores. On a free Colab CPU that build is not seconds but **roughly 10–20 minutes**,
and it dominates everything else the notebook does — which is why the notebook states that
estimate on the page before it starts, rather than leaving the reader to wonder whether it
has hung.

### Size, and where it went

This repository was scoped to roughly 1,000–1,500 lines of meaningful Rust with a hard
ceiling of 2,000. **It exceeds that ceiling, and the deviation is stated here rather than
left for a reviewer to discover.** Counted over `src/` under one explicit rule — an
*implementation line* is a non-blank line that is neither comment-only (`//`, `///`, `//!`,
`/*`, `*`) nor inside a `#[cfg(test)]` module:

```text
src/ total                                                  6,985 lines
  blank                                                       424
  comment-only (including doc comments)                     1,453
  inside #[cfg(test)] modules                                 534
  implementation code                                       4,574

implementation, by module
  model + scene (the actual experiment)                       754
  state record + EntropyFS path + compatibility               496
  offline trainer and scorer (not exercised by the demo)      569
  CLI and child-process dispatch                              155
  process-cold experiment + evidence + killer table         2,595
  crate root                                                    5
```

Roughly a third of the overrun is the evidence machinery that the brief asks for in
detail: five measured phases, three recovery paths, an N-sweep, a per-child report type,
a ~60-field `results.json`, byte and MAC accounting from four angles, and a printed table.
That is mechanical code with a long surface, and it cannot be much shorter while still
reporting every quantity the brief enumerates. The remaining overrun is documentation:
every non-obvious decision in this repository is commented with the measurement that forced
it, because several of those decisions look arbitrary until you know what failed without
them.

A reader who wants only the experiment itself can read `scene.rs`, `model.rs` and
`state.rs` — 2,368 non-blank lines including their comments — and skip everything else.

---

## Quick start

### Google Colab (the canonical environment)

Free tier, CPU, **hardware accelerator: None**. No GPU, no API key, no account, no paid
model, no external inference service.

1. Click the **Open in Colab** badge at the top of this README, or open
   `colab/vole_field_demo.ipynb` in Colab. Cell 1 clones
   `github.com/infinityabundance/vole-field` and prints the exact commit under test; if you
   are running a fork, point `REPO_URL` in cell 1 there instead.
2. **Runtime → Run all.**

Four steps: install the pinned toolchain and print the exact commit →
`cargo build --release --locked` → `cargo run --release --locked` → display the montage
and the cost curve. Google changes free Colab resources without notice, so nothing here
depends on a particular CPU model, core count, RAM tier or runtime length.

### Local

```sh
cargo build --release --locked
cargo run  --release --locked      # runs the experiment into ./run/
```

The toolchain is pinned by `rust-toolchain.toml`; the dependency graph is pinned by
`Cargo.lock` and by exact-version constraints in `Cargo.toml`. `--locked` refuses to
move anything.

The frozen checkpoint is shipped in `assets/` **and** embedded in the binary, so the
installed binary works from any working directory rather than only from a checkout (see
[The frozen checkpoint](#the-frozen-checkpoint)):

```sh
cargo install vole-field
vole-field run                     # works from anywhere; writes ./run/
```

Offline tooling, none of which the demo needs:

```sh
cargo run --release --locked -- eval     # score the frozen checkpoint against truth
cargo run --release --locked -- train    # retrain it (~15 minutes of CPU)
cargo run --release --locked -- version
```

---

## The result of the reference run

Produced by `cargo run --release --locked` on 16 logical CPUs, all children pinned to
four threads. It is quoted verbatim from the run's own output; nothing below is typed by
hand and no number is hardcoded in the program.

```text
model:             tiny ConvLSTM (3_937 params, 8 hidden ch, 3x3 kernels)
device:            CPU (no hardware accelerator)
scene:             moving-shape-01
context frames:    256
future frames:     16

STATE
raw recurrent state:       73728 B (72.0 KiB)
  H:                       32768 B (32.0 KiB)   C: 32768 B (32.0 KiB)
  carry (last two frames): 8192 B (8.0 KiB)
VOLE logical blob:         73904 B (72.2 KiB) (176 header + payload)
raw checkpoint file:       73728 B (72.0 KiB)
EntropyFS mkfs floor:      6422 B (6.3 KiB)
EntropyFS physical delta:  65643 B (64.1 KiB)
EntropyFS engine figure:   67457 B (65.9 KiB)
model weights (shared):    17036 B (16.6 KiB)

PROCESS
producer pid:              3896368
consumer pids (23 distinct): 3896375 3896381 3896386 ... 3898676
fresh process:             PASS

IDENTITY
model weights hash:        20bc7c513833f20dcf9be3d97cc6a70bfb3259a2da12dc6cb6d10e1bb3fb07bb
checkpoint file hash:      6a35ad398cc066e41ed30edab757e9fc13aa75e3cc35b673300918bb8b04b180
original state hash:       0ae5c0b14360cb6c8a1137ac6c6c45d2fc6fd681eadbe68cb77a1d478822106e
restored state hash:       0ae5c0b14360cb6c8a1137ac6c6c45d2fc6fd681eadbe68cb77a1d478822106e
state exact:               PASS
state finite:              PASS

REQUESTS (same earned state, related futures)
  continue       baseline 46d00a93f060 vole 46d00a93f060 raw 46d00a93f060  identical PASS
  turn-left      baseline f5163074fd80 vole f5163074fd80 raw f5163074fd80  identical PASS
  turn-right     baseline 741475cb3836 vole 741475cb3836 raw 741475cb3836  identical PASS
  accelerate     baseline 8e9b09734d40 vole 8e9b09734d40 raw 8e9b09734d40  identical PASS

MODEL WORK (per request)
cell step arithmetic:      3_244_032 gate + 745_472 head = 3_989_504 MACs, 40_960 nonlinearities
baseline cell steps:       271  (16 head applications)
VOLE cell steps:           15  (16 head applications)
cell steps avoided:        256
baseline model MACs:       891_060_224
VOLE model MACs:           60_588_032
model MACs avoided:        830_472_192  (93.20% of baseline)

REUSE (cumulative cost of N related requests, one process each)
  N          baseline       raw ckpt           VOLE
  1          73.69 ms       78.77 ms       96.67 ms
  2          151.3 ms       87.47 ms       106.3 ms
  4          298.8 ms       105.0 ms       124.4 ms
  8          597.6 ms       138.6 ms       159.2 ms
  16        1209.0 ms       212.1 ms       235.4 ms
  one-time VOLE producer:    87.69 ms   one-time raw producer: 69.46 ms
  reuse break-even N*:       2

NEGATIVE CASE
requested scene:           moving-shape-02
offered state's scene:     moving-shape-01
reuse:                     rejected
failing check:             scene_identity
fallback:                  replay-from-scratch
fallback equals baseline:  PASS
cell steps executed:       271

CORRECTNESS GATES
  [PASS] producer_really_exits
  [PASS] consumers_are_distinct_processes
  [PASS] restored_state_bytes_match_original
  [PASS] restored_state_hash_matches_producer
  [PASS] baseline_and_restore_outputs_identical
  [PASS] raw_checkpoint_and_restore_outputs_identical
  [PASS] fewer_recurrent_steps
  [PASS] state_is_finite
  [PASS] negative_state_rejected
  [PASS] negative_fallback_matches_baseline
  [PASS] fresh_process_requests_are_bit_identical
  [PASS] montages_identical
  overall: PASS
```

Read as a sentence: **74 KB of persisted state avoided 830 million multiply-accumulates
of generative model work per related request, 93.2 % of the baseline, with byte-identical
output**, and the break-even point was the second request. That break-even figure is a
measured wall-clock result and is sensitive to the one-time persistence cost; the
deterministic work result above it is not. [Timing methodology](#timing-methodology)
quantifies that difference rather than glossing it.

Note one honest detail in the byte accounting: the EntropyFS store grew by **65,643
bytes** to hold a 73,904-byte record — about 11 % *less* than the unframed payload the
attribution control writes. EntropyFS's representation machinery compressed the state
rather than padding it. That is reported because it was measured; this proof does not
claim it will always happen (see [What this does not prove](#what-this-does-not-prove)).

---

## What the run does

The top-level binary is an orchestrator. It never executes model arithmetic itself; it
launches short-lived **child processes of itself** and reads their reports.

```text
vole-field run
  │
  ├─ 1. PRODUCER             one process
  │       load frozen model → consume 256 context frames → earn S = (H, C, carry)
  │       → put the VOLE record through EntropyFS → sync() → write the raw payload to
  │         an ordinary file → EXIT
  │
  ├─ 2. IDENTITY             twelve processes (4 requests × 3 paths)
  │       for each of continue / turn-left / turn-right / accelerate:
  │         baseline child   fresh process → replay context → generate
  │         raw child        fresh process → read raw tensors → generate
  │         vole child       fresh process → EntropyFS restore → generate
  │       → compare all three outputs BYTE FOR BYTE
  │
  ├─ 3. TIMING               parent-clocked
  │       phase medians inside one process (1 warm-up + 5 repetitions), and the
  │       end-to-end wall time of single-request processes (1 warm-up + 5 processes
  │       per path), and the fresh-process cost of each producer flavour
  │
  ├─ 4. REUSE                N ∈ {1, 2, 4, 8, 16}
  │       N related requests, each in its own fresh process, per path
  │       → cumulative cost → N*
  │
  ├─ 5. NEGATIVE             two processes
  │       a moving-shape-02 request is handed the moving-shape-01 state
  │       → compatibility check refuses it → fall back to replay → compare
  │
  └─ 6. EVIDENCE             run/results.json, run/reuse.csv, run/branches.ppm
```

### Where the process boundary really is

The producer is `wait`ed on before the first consumer is spawned; the run additionally
observes that its `/proc/<pid>` entry is **already gone** at that point. Every child's
self-reported PID is checked against the PID the operating system assigned, and a
mismatch aborts the run — 159 processes were verified this way in the reference run.

There is no shared memory, no daemon, no warm Rust object carrying the state, and no
inherited tensors. The only bridge between producer and consumer is persisted bytes: an
EntropyFS blob id on the VOLE path, an ordinary file on the raw path.

The OS page cache is **not** flushed. This is a **fresh-process restore** (a.k.a.
process-cold restore), not a cold-disk-cache experiment. That is stated in
`results.json` and repeated in the output.

---

## The three recovery paths, and why the third one exists

All three paths use the same frozen weights, the same 256-frame context, the same control
program, the same future length, and the same generation code. The only difference is
where the starting recurrent state comes from.

| path | starting state | cost paid |
|---|---|---|
| `baseline` | recomputed by replaying all 256 context frames through the ConvLSTM | 256 extra recurrent steps, **per request** |
| `raw` | read from an ordinary file holding the same `H ‖ C ‖ carry` bytes | one file read |
| `vole` | restored from an EntropyFS blob | one store open + `get_blob` + record decode |

The baseline replays the context **per request** because each request really is a fresh
process. That is the scenario under measurement — "process-cold reuse" — and it is stated
rather than assumed. A system that kept the model resident in one long-lived process
would not pay that cost; it also would not have a durable generative state, which is the
thing being demonstrated.

**Why the `raw` path exists.** It answers the obvious objection: *isn't the saving simply
caused by persisting recurrent state?* The honest answer is yes — durable recurrent state
is the mechanism being isolated. VOLE-Field adds the explicit persistent-state contract
and the EntropyFS path; this demo does not claim that saving an RNN state was previously
unknown. The raw control measures exactly how much of the result comes from *durable state
in general* and how much from *the VOLE record and the EntropyFS path*.

The raw payload also carries **no identity at all**, so it cannot be checked for
compatibility. That absence is itself part of the comparison: the record's header is what
makes the negative case below possible.

---

## Correctness gates

The run prints PASS/FAIL for each of these, and `results.json` records the detail.

```text
producer_really_exits                     producer PID differs from every consumer PID,
                                          and its /proc entry is gone before any consumer
                                          was spawned
consumers_are_distinct_processes          every child's self-reported PID matched the OS
                                          PID; no consumer PID repeats
restored_state_bytes_match_original       re-serialised state == persisted payload, byte
                                          for byte
restored_state_hash_matches_producer      the hash the consumer generated from == the
                                          producer's
baseline_and_restore_outputs_identical    exact f32 output bytes, every request
raw_checkpoint_and_restore_outputs_identical
fewer_recurrent_steps                     the restored path executes strictly fewer cell steps
state_is_finite                           no NaN, no infinity, before or after persistence
negative_state_rejected                   the unrelated-scene request refused the state
negative_fallback_matches_baseline        the refused request replayed to the same bytes
fresh_process_requests_are_bit_identical  repeats in different processes agree, and no two
                                          distinct request programs collide
montages_identical                        the two montages are byte-identical images
```

**Performance never determines PASS/FAIL.** If the VOLE path is slower than the raw
checkpoint, that is a passing run with a measured negative performance result. If no
crossover appears inside the tested range, that is a passing run reporting `N* > 16`. No
crossover is ever manufactured.

---

## Byte accounting

| quantity | reference value | meaning |
|---|---|---|
| raw recurrent state | 73,728 B | `H` (32,768) + `C` (32,768) + carry (8,192), exact `f32`, no quantisation |
| VOLE logical blob | 73,904 B | the 176-byte record header plus that payload, as handed to EntropyFS |
| raw checkpoint file | 73,728 B | the unframed payload, as written by the attribution control |
| EntropyFS mkfs floor | 6,422 B | the size of an empty store directory, so the store's fixed overhead is visible rather than hidden |
| EntropyFS physical delta | 65,643 B | bytes the store directory gained when the blob was put and made durable |
| EntropyFS engine figure | 67,457 B | the engine's own `physical_used_bytes`, reported as an independent cross-check |
| model weights | 17,036 B | **shared by every path and charged to none** |

The store size is measured from *outside* the engine (a recursive sum of file lengths) so
the number cannot be flattered by the engine's own accounting. The engine's figure is
reported beside it precisely because the two disagree slightly — one counts file lengths,
the other counts its reconciled live bytes — and hiding that would be the wrong instinct.

---

## Model-work accounting

Wall-clock alone is too noisy on a shared free Colab VM to be the primary evidence, so the
primary evidence is deterministic:

- the loops that execute model arithmetic increment a **runtime counter**, so the step
  count is an observation, not a restatement of the plan;
- the model declares its **per-step arithmetic** analytically, from its own configuration;
- the report multiplies the two, so both factors are visible.

Definitions, for the shipped configuration (`32×32`, `H = 8` hidden channels, `3×3`
kernels, 3 input channels, 2 carry channels, 8 head-hidden channels):

```text
cell step = one ConvLSTM update
  gate convolution MACs   (Hh·Ww) · (in + H) · 4H · k²       = 3,244,032
  gate bias adds          (Hh·Ww) · 4H                       =    32,768
  cell elementwise ops    (Hh·Ww) · 4H                       =    32,768
  cell nonlinearities     (Hh·Ww) · 5H  (3 sigmoid, 2 tanh)  =    40,960
head application = one emitted frame
  head convolution MACs   (Hh·Ww) · (8·(H+2)·k² + 8)         =   745,472
  head bias adds and the residual add, per pixel             =     2,048
total per cell step + emission                               = 3,989,504
```

One MAC is one multiply and one add. Because generating `T` frames needs `T` emissions
and therefore only `T − 1` successor cell steps:

```text
baseline  256 context + 15 generation = 271 cell steps = 891,060,224 MACs
vole        0 context + 15 generation =  15 cell steps =  60,588,032 MACs
avoided   256 cell steps                                 = 830,472,192 MACs (93.20%)
```

Persistence and restore costs are reported **separately** and are never counted as zero
compute.

---

## Timing methodology

Two different things are measured, and they are kept apart.

**Phase medians.** One child process runs a discarded warm-up repetition followed by five
measured repetitions, and reports the median of each phase plus every raw sample. This
isolates a phase from process startup:

```text
                                        baseline      raw       vole
model_load                                 0.11 ms   0.10 ms   0.09 ms
scene_generate                             0.32 ms      n/a      n/a
state_restore                                n/a     0.02 ms   0.39 ms
model_only_replay                         67.07 ms      n/a      n/a
model_only_generate                        6.12 ms   6.55 ms   6.22 ms
request_total                             73.47 ms   6.62 ms   6.66 ms
```

`scene_generate` is reported separately from `model_only_replay` so that rendering the
history cannot hide inside "model work", and vice versa. Note that the VOLE path does not
regenerate the scene at all.

**Fresh-process end-to-end.** Six separate processes run per path, each serving exactly one
request; the parent clocks each one from spawn to exit:

```text
baseline   75.30 ms   (per-process: 75.4 77.0 74.7 75.3 74.9)
raw         8.54 ms
vole        9.23 ms
```

The model weights are read by every process on every path, so they appear symmetrically in
this number and are neither hidden nor charged to one side.

**Producer cost**, measured as real processes on their own fresh stores rather than derived
by subtraction:

```text
producer (vole)   87.69 ms      store open 6.93  replay 68.97  persist 3.91
producer (raw)    69.46 ms      replay 68.19  state_serialize 0.02  raw write 0.04
```

Each timed producer is required to have earned exactly the canonical state hash, so the
raw-only producer is a genuine measurement of the raw mechanism rather than a
re-numbering of the VOLE one.

The repeated-request sweep sums the parent-measured wall times of the actual child
processes, so `C_·(N)` is a direct measurement of a batch of fresh processes, not a model
fitted to a per-request median.

**How stable is `N*`?** `N*` is not a property of the mechanism alone; it is where a
*measured* one-time cost gets repaid, so it inherits the noise of that cost. The structural
arithmetic is simple — the one-time producer pays one 256-frame replay plus the persistence
barrier, every restored request pays one 16-frame generation, and the baseline pays both per
request — which is why `N*` lands near
`producer / (baseline_per_request − restored_per_request) ≈ 88 / 65 ≈ 1.4`, i.e. the second
request. But the producer figure is the wall time of a whole child process that ends in an
`fsync` durability barrier, and an `fsync` stalls when the host is busy flushing other data.
Measured on the same 16-core machine, same code, same deterministic work:

```text
idle machine, three separate runs    producer 87.7 / 88.6 / 88.6 ms    N* = 2
started directly after a compile     producer median 449.5 ms          N* = 8
```

The second row is not a bad measurement; it is a correct measurement of an unlucky moment
(a page cache full of freshly written build output). The program computes `N*` from whatever
it actually measured and prints that — it never substitutes the structural estimate above for
the observation. This is precisely why the load-bearing evidence is the deterministic
step/MAC count and why performance never gates the run: the avoided work is a property of the
mechanism, whereas `N*` is a property of the machine on the day. On a shared free Colab VM,
expect the reported `N*` to move, and read it together with the recurrent-step column that
does not.

---

## Determinism: why byte-for-byte is a fair comparison

Byte-exact equality is asserted, not approximated. Four things make it possible:

1. **The scene uses no transcendental functions at runtime.** Steering is a rotation by a
   pair of source-literal constants; the body kernel is `max(0, 1 − r²)²` evaluated without
   a square root; the throttle is a memoryless multiply. Given a scene spec and a control
   schedule, the frame sequence is a pure function of those inputs and is bit-identical in
   every process. This matters because the baseline *regenerates* the history the producer
   consumed; if regeneration were not exact the whole comparison would be vacuous.
2. **The state is never quantised.** `H`, `C` and the carry are persisted as their exact
   `f32` bits, and recovery is checked by re-serialising and comparing bytes.
3. **No randomness on the inference path.** Sampling exists only in the offline trainer.
4. **Every child is pinned to the same thread count** by the parent, so thread count cannot
   become a source of divergence between producer and consumer. The pinned value is
   recorded in `results.json` (`pinned_threads`).

The test suite includes a check that the two source-literal rotation constants equal the
platform's own `cos`/`sin` to the last bit, so the constants' provenance stays honest as
the code changes.

---

## The frozen checkpoint

`assets/tiny_convlstm.safetensors` — **17,036 bytes**, well under 1 MB.

```text
input            32 × 32 grayscale frame + 2 constant control planes
hidden channels  8
layers           1 ConvLSTM + a residual output head with 8 hidden channels
kernel           3 × 3
dtype            f32
parameters       3,937
raw state        73,728 bytes (H 32,768 + C 32,768 + carry 8,192)
```

The same 17,036 bytes are also **embedded in the binary** (`include_bytes!`), and the
loader falls back to that copy *only* when both of these hold: the canonical path was the
one asked for, **and** it is absent. Two consequences are worth stating plainly.

- A `cargo install`ed `vole-field` works from any working directory. Without this, the
  binary would abort with `read "assets/tiny_convlstm.safetensors": No such file or
  directory` the moment it was run outside a checkout.
- An explicitly named checkpoint is **never** silently substituted. A missing
  `--checkpoint foo.safetensors` is an error, because quietly serving the canonical weights
  instead would make every hash this program reports describe a model the user did not ask
  for — and those hashes are the evidence that producer and consumer loaded the same thing.

When the file is present on disk it wins, so replacing `assets/` still changes what runs.
Two tests pin the behaviour: one asserts the embedded copy is byte-identical to the shipped
file, and one asserts the file and the embedded copy load to the same weights hash.

Networks are adapted from Shi et al., *Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting*, [arXiv:1506.04214](https://arxiv.org/abs/1506.04214).
Nothing about the architecture is claimed as novel.

The cell folds the two input-to-gate convolutions of the textbook formulation into one
convolution over the channel-concatenation of `[x; u]` and `H`. With same padding that is
an exact algebraic identity — a convolution is linear in its input channels and the bias is
added once either way — and it halves the number of kernel launches. The MAC count is
unchanged.

The decoder reads `[H ; carry]`, where the carry holds the **last two consumed frames**.
Keeping it inside the state rather than beside it means the persisted record remains a
complete description of everything the decoder needs: a restored state is self-sufficient
and no separate "resume frame" is bolted on beside it. The head is a *residual* branch,
`clamp01(carry₀ + 0.6·tanh(conv₁(tanh(convₖ([H ; carry])))))`. Each of those choices was
made for a measured reason — see the training notes below.

### Provenance

The checkpoint is trained **once, offline**, and frozen for every run. The demo never
trains. The trainer lives in this crate because "trust me, the weights are frozen" is
weaker than "here is the seed, the corpus generator, the objective and the command". The
checkpoint's safetensors metadata records the seed, iterations, batch size, window range,
loss, learning-rate schedule, scheduled-sampling ramp, head initialisation, corpus digest,
final loss and crate/compiler versions. `vole-field eval` scores it against ground truth
with no display involved.

```sh
cargo run --release --locked -- train     # reproduces the shipped checkpoint
cargo run --release --locked -- eval      # scores it
```

Reproducing the *exact* checkpoint needs the same pinned toolchain: the corpus is a pure
function of the seed and the code, but the arithmetic that consumes it is floating point.
The *demo* never needs the same bits twice; it only needs its own processes to agree, which
is what the gates check.

### Model quality, stated plainly

This is the weakest part of the artifact, so it is measured and reported rather than
described. `vole-field eval` scores the shipped checkpoint against the scene's own future:

```text
mean absolute error over the horizon            0.0243   (frame range 0..1)
centroid error, mean over the horizon           3.64 px
centroid error, final generated frame           6.55 px
generated peak brightness, mean                 0.56     (truth 0.97)
generated total ink, mean                       24.2     (truth 14.1)
```

The model holds a recognizable object in roughly the right place and fades gently over the
16-frame horizon — it neither fades to black nor diverges. But it is **not a good video
model**, and in particular:

- **its response to the control word is sub-pixel.** The scene's own branches diverge by
  2.5–12.6 px across the horizon; the model's four branches diverge by **0.08–0.29 px**.
  The branches are genuinely, bit-provably different — but they are not *visibly* different,
  and the montage shows that honestly rather than hiding it.
- it over-inks, roughly 1.7× the truth by the end of the horizon.

Nothing in the proposition, the gates, the byte accounting or the work accounting depends
on model quality. A better model would make the picture nicer and change none of the
numbers.

### Training notes (measured, not assumed)

Six failures were measured and fixed or worked around while fitting this checkpoint. They
are recorded because they are part of the artifact's provenance, and because the last one is
the reason the montage shows ground truth as well as model output.

- **MSE with a sigmoid head plateaus.** The gradient reaching the pre-activation is
  `2(σ−y)·σ·(1−σ)`, which vanishes exactly where the sparse bright object is. At 500
  iterations the model emitted a static, dim blur: peak 0.11 against a truth peak of 0.96.
- **Soft cross-entropy fixes the vanishing gradient but not the geometry.** With a sigmoid
  head, soft-BCE moved the same budget to peak 0.35 and 4.5 px centroid error, and 3,000
  iterations to peak 0.60. It then *diffused*: total ink stayed roughly constant while the
  peak collapsed from 0.85 to 0.08 within five autoregressive steps.
- **Teacher forcing alone is not enough.** Trained purely on ground-truth inputs, the model
  never learns to correct its own drift. Scheduled sampling — with probability `p` the next
  input is the model's own prediction — with `p` ramped linearly to 0.75–1.0 across the run
  is what stabilises the free-running rollout. Ramping `p` to 0.9 within the first 35 % of
  a short run is worse, because it starves the model of the teacher-forced signal it needs
  to learn the dynamics at all.
- **Training must cover the deployment horizon.** Training on a 9-step self-generated
  horizon produced a model whose *first* generated frame was exact and whose rollout then
  died. Supervising exactly the 16 steps that will be asked for is what fixed that.
- **The objective must not make the object expendable.** A frame is 1,024 pixels of which
  ~30 carry the object. Under an unweighted per-pixel loss, fading the object away costs
  less than a slightly misplaced one, and the optimiser takes that trade. Weighting each
  pixel by `1 + 6·target` makes losing the sparse bright region expensive, which is what
  stopped the fade.
- **Several architectural changes were needed for the motion to be representable at all**,
  each arrived at by measurement rather than taste:
  - decoding from `H` alone blurs, because `H` is a leaky accumulator and a single 3×3
    convolution cannot de-blur it; the decoder therefore reads `[H ; carry]`, the textbook
    ConvLSTM decode;
  - a *linear* head cannot express a heading-dependent translation at all — a fixed kernel
    applied to the carry can only translate it by a fixed vector — so the head carries a
    nonlinear residual branch over the carry;
  - an unbounded residual diverges, because the head's output becomes the next step's
    carry, so the correction is bounded by `tanh`;
  - the throttle is memoryless rather than integrated, because an integrated speed is a
    hidden scalar register that has to be inferred from hundreds of frames, which is the
    credit-assignment problem a 3,937-parameter convolutional recurrence does not solve,
    whereas a held control word is present in the input at every step;
  - the body is rendered as a streak along its heading, because a circular body's heading is
    invisible in every individual frame and the model then has to invent and maintain a
    hidden orientation register — which, measured, it never learns;
  - the carry holds two frames, not one, so that motion is present in the input as a
    difference and temporal extrapolation is a representable starting point.

Despite all of that, the control response remained sub-pixel, and the honest thing to do
with a limitation that will not move is to show it rather than to describe it away. The
montage's ground-truth block exists for exactly that reason.

---

## Artifacts

Everything is written under the run directory (`./run/` by default):

```text
results.json      the whole evidence file: hashes, PIDs, byte accounting, step counts,
                  MAC estimates, raw timing samples, medians, N*, the negative case, and
                  every correctness gate
reuse.csv         N,baseline_ms,raw_checkpoint_ms,vole_ms
branches.ppm      792 × 890, nine rows of eight frames
branches_baseline.ppm   the same montage from the from-scratch path (byte-identical)
futures/*.f32     the emitted frame payloads, exact f32 bytes
reports/*.json    every child's own report, for a human debugging a run
entropyfs-store/  the real EntropyFS store
raw_state.bin     the unframed payload written by the attribution control
```

The run directory is git-ignored: the artifacts are reproducible from the two commands in
[Quick start](#quick-start).

The montage is a PPM written directly from Rust — no image library:

```text
row 1        the observed context tail: the history the state was earned from
rows 2-5     the model's branches: continue / turn-left / turn-right / accelerate,
             generated after the producing process had already exited
row 6-9      the same four branches, generated by the scene itself (ground truth)
```

The left margin of each row is a distinct grey so rows are countable without a font. The
ground-truth block is included so a reader can see both what the controls do to the scene
and what the model makes of them, instead of having to guess which is which. The model
block is *not* an illustration: it is the exact `f32` output that the gates compare, and the
two montages are compared as byte strings.

The only lossy step anywhere is the display gamma applied when a frame is mapped to 8 bits
for the picture. Every measurement and every comparison uses the exact `f32` bytes.

---

## Repository layout

```text
Cargo.toml              one package, exact-version dependencies
Cargo.lock              pinned dependency graph (use --locked)
rust-toolchain.toml     pinned compiler
README.md
LICENSE-MIT
LICENSE-APACHE
assets/
    tiny_convlstm.safetensors      17,036 bytes, frozen, with provenance metadata; also
                                   embedded in the binary (see The frozen checkpoint)
src/
    main.rs             CLI: run | producer | request | train | eval | version
    lib.rs              crate documentation and module map
    scene.rs            the deterministic scene, controls, and named requests
    model.rs            the ConvLSTM and residual head, exact state bytes, declared work
    state.rs            the VOLE record, its compatibility predicate, EntropyFS
    experiment.rs       the process-cold experiment, evidence, and the killer table
    train.rs            offline trainer and checkpoint scorer (not part of the demo)
colab/
    vole_field_demo.ipynb
```

---

## Tests

```sh
cargo test --release
```

The suite covers: exact `f32` round-tripping of the whole state; the record's fixed header
layout field by field; tamper detection (payload, version, magic, truncation); every
compatibility check including the refusals; a genuine put/get through an EntropyFS store
including dedup idempotence and materialising the exact bytes; the declared parameter shapes
and MAC counts matching the built model; the persistence initialisation emitting the carry
bit for bit; the carry window shifting correctly; the scene's bit-level reproducibility;
the memoryless throttle; the bodies never leaving the field; distinct requests producing
distinct futures; the embedded checkpoint being byte-identical to the shipped file and
loading to the same weights hash; the fallback firing only for an absent canonical
checkpoint and never substituting for an explicitly named one; and the rotation constants
matching the platform's `cos`/`sin` to the last bit.

---

## What this proves

> A recurrent generative model can accumulate useful state from an observation history;
> that state can be durably persisted through EntropyFS across process lifetimes; a fresh
> process can restore it and generate related future observations without replaying the
> original history; and the storage/time/work trade can be measured across repeated
> requests.

## What this does not prove

```text
It does not prove VOLE-Field universally accelerates generative models.
It does not prove that ConvLSTM is a frontier video model.
It does not prove that persisting recurrent neural state is itself novel.
It does not prove that all models expose reusable state.
It does not prove that all related requests can reuse the same state.
It does not prove that VOLE is always faster than a raw checkpoint.
It does not prove that EntropyFS always compresses recurrent state.
It does not prove frontier-scale economics.
It does not exercise every mechanism disclosed by the VOLE-Field paper.
```

It also does not prove anything about the *quality* of the generated futures. The shipped
model's branches differ by 0.08–0.29 px and its control response is sub-pixel; the
mechanism claim is unaffected, but a reader who came for a video model should leave now.

The point is **mechanism feasibility**. Nothing more.

---

## Prior art and attribution

- ConvLSTM: Shi, Chen, Wang, Yeung, Wong, Woo, *Convolutional LSTM Network: A Machine
  Learning Approach for Precipitation Nowcasting*, NeurIPS 2015,
  [arXiv:1506.04214](https://arxiv.org/abs/1506.04214). Used as-is; not claimed.
- Recurrent-state compression, learned world models, action-conditioned video prediction,
  teacher forcing / scheduled sampling and residual heads are all established lines of work.
  This demo builds on them and claims none of them.
- [EntropyFS](https://crates.io/crates/entropyfs) 0.7.17 is the persistence path, used
  through its published embeddable `Engine` facade with `default-features = false` — no FUSE
  mount, no daemon, no privileged operation. The demo does not reimplement it, wrap it in
  `std::fs`, or fake it.
- [Candle](https://github.com/huggingface/candle) 0.11.0 provides the CPU tensors, autograd
  and training loops.

## License

MIT OR Apache-2.0.
