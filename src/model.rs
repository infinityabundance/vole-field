//! The tiny recurrent generative video model: one ConvLSTM layer plus a small
//! output head, in native Rust on Candle's CPU backend.
//!
//! # What is and is not claimed
//!
//! ConvLSTM is Shi et al. 2015, "Convolutional LSTM Network: A Machine Learning
//! Approach for Precipitation Nowcasting" (<https://arxiv.org/abs/1506.04214>).
//! Nothing about the *architecture* is novel here and nothing about it is
//! claimed. It is used because it has exactly the property this proof needs:
//! an explicit, compact, persistent recurrent state `(H, C)` that is *earned* by
//! consuming an observation history and that conditions all future predictions.
//!
//! # The cell
//!
//! ```text
//! gates = conv([x_t ; u_t ; H_{t-1}])          -> [4H, Hh, Ww]
//! i, f, o, g = split(gates)
//! C_t = sigmoid(f) * C_{t-1} + sigmoid(i) * tanh(g)
//! H_t = sigmoid(o) * tanh(C_t)
//! carry_t = x_t
//! ```
//!
//! The two input-to-gate convolutions of the textbook formulation are folded
//! into one convolution over the *channel-concatenation* of `[x; u]` and `H`.
//! With same padding this is an exact algebraic identity — a convolution is
//! linear in its input channels and the bias is added once either way — and it
//! halves the number of kernel launches. The MAC count is unchanged.
//!
//! # The decode, and the carry channel
//!
//! Shi et al. decode the next frame from the hidden state *and the current input*
//! jointly. Decoding from `H` alone is the tempting simplification, and it was
//! measured to fail here: the hidden state is a leaky accumulator, so its content is
//! a temporal smear of the trajectory, and a single 3x3 convolution cannot
//! de-blur it. The result was a model whose first generated frame was exact and
//! whose rollout then faded to black.
//!
//! So the decoder reads `[H_t ; carry_t]`, where `carry_t` holds the last two frames
//! the cell consumed, most recent first. Putting the carry *inside the state* rather
//! than beside it matters: the persisted record then remains a complete description of
//! everything the decoder needs, so a restored state is still self-sufficient and the
//! record needs no separate "resume frame" bolted on beside it.
//!
//! # The head
//!
//! `head(S_t) -> frame_{t+1}` is a 3x3 convolution over `[H_t ; carry_t]` with a
//! **linear** output and a clamp to the physical intensity range `[0, 1]`.
//!
//! The clamp is part of the declared output contract, applied identically on every
//! path, and not a presentation choice.
//!
//! The head is linear on purpose. Predicting the next frame of a slowly moving
//! object is a problem where "repeat the last frame, shifted a little" is nearly the
//! whole answer, and a linear head can represent that exactly. A sigmoid head cannot:
//! to emit a copy of the carry through `sigmoid`, the pre-activation would have to be
//! `logit(carry)`, which no convolution of the carry can produce, and the best a
//! sigmoid head can do is a squashed copy whose background sits near 0.5. Measured:
//! with a sigmoid head the rollout diffused — total ink held roughly constant while
//! the peak collapsed from 0.85 to 0.08 within five steps — because the model had no
//! representable way to be sharp. A linear head plus a squared-error objective makes
//! the persistence solution the *starting point* rather than an unreachable ideal.
//!
//! # The emission contract
//!
//! `cell(x_t, u_t, S_{t-1}) -> S_t` consumes the frame and the control word;
//! `head(S_t) -> frame_{t+1}` emits the next frame. A control word `u_t` is the
//! control held during the transition into `frame_{t+1}`, so it is always supplied
//! together with `frame_t`.

use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops as nn_ops;
use candle_nn::{Init, VarBuilder};

use crate::scene::{H, W};

/// Gates per cell: input, forget, output, cell.
pub const GATES: usize = 4;
/// Input channels consumed by the cell: one frame plus two control planes.
pub const IN_CHANNELS: usize = 3;
/// Channels of the carry: the last **two** consumed observations, held inside the
/// state so that the decoder alone can be handed the state.
///
/// Two, not one, and that is the point. A head that can only see the current frame has
/// no expressible way to know how the object is *moving*: the correct next frame is a
/// translation along the object's own heading, and no fixed map of a single frame can
/// produce a heading-dependent translation. With the previous frame present as well,
/// the motion is in the input as a difference, and a small linear map of the two
/// frames — temporal extrapolation — is a representable starting point rather than
/// something the optimiser has to invent.
pub const CARRY_CHANNELS: usize = 2;

// ---------------------------------------------------------------------------
// The frozen checkpoint artifact
// ---------------------------------------------------------------------------

/// Canonical checkpoint path, relative to the working directory.
///
/// The path is written exactly once, here.
/// [`crate::experiment::DEFAULT_CHECKPOINT`] points at this constant, so the CLI
/// default and the loader cannot drift apart into two literals.
pub const CHECKPOINT_PATH: &str = "assets/tiny_convlstm.safetensors";

/// The canonical checkpoint, embedded in the binary at compile time.
///
/// A `cargo install`ed binary is run from whatever directory the user happens to be
/// in, where `assets/` does not exist. Embedding the frozen bytes lets the demo and
/// `eval` work from any directory *without changing what is loaded*: these are
/// byte-for-byte the same weights the on-disk artifact holds, so every hash reported
/// for them is unchanged. The file on disk stays authoritative when it is present.
static CHECKPOINT_EMBEDDED: &[u8] = include_bytes!("../assets/tiny_convlstm.safetensors");

/// Whether `path` is the canonical checkpoint path.
pub fn is_canonical_checkpoint(path: &Path) -> bool {
    path == Path::new(CHECKPOINT_PATH)
}

/// The embedded canonical checkpoint's bytes.
///
/// Exposed so a caller — or a test — can obtain the frozen artifact without
/// depending on the working directory.
pub fn embedded_checkpoint() -> &'static [u8] {
    CHECKPOINT_EMBEDDED
}

/// Read a checkpoint's bytes, falling back to the embedded canonical checkpoint.
///
/// The fallback fires **only** when both of these hold:
///
/// 1. the caller asked for the canonical path, and
/// 2. that file is absent.
///
/// A checkpoint named explicitly is read from disk or is an error. It is never
/// silently swapped for the canonical weights, because that would make the hashes
/// this program prints describe a model the user did not ask for — and those hashes
/// are the evidence that the producer and the consumer agree on what they loaded.
pub fn read_checkpoint(path: &Path) -> std::result::Result<Vec<u8>, String> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(bytes),
        Err(e) => match fallback_for(path, &e) {
            Some(embedded) => Ok(embedded.to_vec()),
            None => Err(format!("read {path:?}: {e}")),
        },
    }
}

/// The fallback rule itself, as a pure function.
///
/// Split out from [`read_checkpoint`] so the rule can be tested directly: exercising
/// it through the reader would require deleting the real artifact from the working
/// directory, and a test that mutates the checkout to prove a point is a worse test
/// than one that does not.
///
/// Returns the embedded bytes only when the read failed with "not found" **and** the
/// caller asked for the canonical path. `None` means "propagate the error".
fn fallback_for(path: &Path, err: &std::io::Error) -> Option<&'static [u8]> {
    let not_found = err.kind() == std::io::ErrorKind::NotFound;
    (not_found && is_canonical_checkpoint(path)).then_some(CHECKPOINT_EMBEDDED)
}

/// Parameter name: folded gate convolution weight, `[4H, in+H, k, k]`.
pub const NAME_GATES_WEIGHT: &str = "convlstm.gates.weight";
/// Parameter name: folded gate convolution bias, `[4H]`.
pub const NAME_GATES_BIAS: &str = "convlstm.gates.bias";
/// Parameter name: residual head hidden convolution weight, `[M, H+carry, k, k]`.
pub const NAME_HEAD_MID_WEIGHT: &str = "convlstm.head.mid.weight";
/// Parameter name: residual head hidden convolution bias, `[M]`.
pub const NAME_HEAD_MID_BIAS: &str = "convlstm.head.mid.bias";
/// Parameter name: residual head output weight, `[1, M, 1, 1]`.
pub const NAME_HEAD_OUT_WEIGHT: &str = "convlstm.head.out.weight";
/// Parameter name: residual head output bias, `[1]`.
pub const NAME_HEAD_OUT_BIAS: &str = "convlstm.head.out.bias";

/// Every parameter name, in a fixed order used for the canonical weights hash.
pub const PARAM_NAMES: [&str; 6] = [
    NAME_GATES_WEIGHT,
    NAME_GATES_BIAS,
    NAME_HEAD_MID_WEIGHT,
    NAME_HEAD_MID_BIAS,
    NAME_HEAD_OUT_WEIGHT,
    NAME_HEAD_OUT_BIAS,
];

/// Hidden channels inside the residual head branch.
pub const HEAD_HIDDEN: usize = 8;

/// Bound on the residual correction the head may add to the carry in one step.
///
/// The correction is `RESIDUAL_SCALE * tanh(...)`, so no head can change any pixel by
/// more than this in a single step. That is a stability requirement, not an
/// aesthetic one: the head output becomes the carry of the next step, so an unbounded
/// residual is a feedback loop, and it diverges. Measured without the bound: the peak
/// stayed sharp while total ink grew 4.6x across the horizon, i.e. the object inflated
/// a little more every step.
///
/// The bound is generous for this scene — the object moves well under one pixel per
/// step, so the true frame-to-frame change at an edge is a small fraction of one.
pub const RESIDUAL_SCALE: f32 = 0.6;

/// Architecture configuration. Small enough to print in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Input channels: frame (1) + control planes (2).
    pub in_channels: usize,
    /// Recurrent hidden channels. `H` and `C` are each `[1, hidden, 32, 32]`.
    pub hidden_channels: usize,
    /// Square convolution kernel size.
    pub kernel: usize,
    /// Field height.
    pub height: usize,
    /// Field width.
    pub width: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            in_channels: IN_CHANNELS,
            hidden_channels: 8,
            kernel: 3,
            height: H,
            width: W,
        }
    }
}

impl ModelConfig {
    /// Canonical little-endian encoding, the preimage of `model_config_hash`.
    ///
    /// Explicit fields in a fixed order — not a serialiser dump — so the hash is
    /// stable across serde versions and cannot drift with field reordering.
    pub fn canonical_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..4].copy_from_slice(&1u32.to_le_bytes()); // config schema version
        out[4..8].copy_from_slice(&(self.in_channels as u32).to_le_bytes());
        out[8..12].copy_from_slice(&(self.hidden_channels as u32).to_le_bytes());
        out[12..16].copy_from_slice(&(CARRY_CHANNELS as u32).to_le_bytes());
        out[16..20].copy_from_slice(&(self.kernel as u32).to_le_bytes());
        out[20..24].copy_from_slice(&(self.height as u32).to_le_bytes());
        out[24..28].copy_from_slice(&(self.width as u32).to_le_bytes());
        out[28..32].copy_from_slice(&0u32.to_le_bytes()); // reserved, must be zero
        out
    }

    /// Pixels per frame.
    pub fn spatial(&self) -> u64 {
        (self.height * self.width) as u64
    }

    /// Floats in `H`, `C` and the carry — the *exact* raw recurrent state size.
    pub fn state_floats(&self) -> u64 {
        (2 * self.hidden_channels as u64 + CARRY_CHANNELS as u64) * self.spatial()
    }

    /// Parameters in the whole model.
    pub fn parameter_count(&self) -> u64 {
        let h = self.hidden_channels as u64;
        let k = (self.kernel * self.kernel) as u64;
        let ci = (self.in_channels as u64) + h;
        let m = HEAD_HIDDEN as u64;
        GATES as u64 * h * ci * k
            + GATES as u64 * h
            + m * (h + CARRY_CHANNELS as u64) * k
            + m
            + m
            + 1
    }

    /// Declared per-step work. See [`StepWork`].
    ///
    /// The head is a residual branch: a `k x k` convolution to `M` hidden channels, a
    /// `tanh`, then a `1 x 1` projection to one channel added to the carry. Its MACs
    /// are the two convolutions. The residual add and the output clamp are counted in
    /// `head_bias_adds`.
    pub fn step_work(&self) -> StepWork {
        let h = self.hidden_channels as u64;
        let s = self.spatial();
        let k = (self.kernel * self.kernel) as u64;
        let ci = (self.in_channels as u64) + h;
        let m = HEAD_HIDDEN as u64;
        StepWork {
            cell_macs: s * ci * GATES as u64 * h * k,
            cell_bias_adds: s * GATES as u64 * h,
            cell_mul_adds: 4 * h * s,
            cell_nonlinearities: 5 * h * s,
            head_macs: s * m * (h + CARRY_CHANNELS as u64) * k + s * m,
            head_bias_adds: 2 * s,
        }
    }
}

/// The declared arithmetic of one recurrent step and one head application.
///
/// These are *analytic* counts derived from the architecture, not measurements.
/// They are cross-checked against a runtime step counter, so the work claim is
/// `steps_actually_executed x arithmetic_per_step`, with both factors visible.
///
/// Definitions, stated so that nothing is hidden in a constant:
/// - `cell_macs`: multiply-accumulates in the folded gate convolution,
///   `Hh*Ww * (in+H) * 4H * k^2`. One MAC is one multiply and one add.
/// - `cell_bias_adds`: one add per gate channel per pixel.
/// - `cell_mul_adds`: the elementwise cell update, `4H` per pixel
///   (`f*C`, `i*g`, the sum, `o*tanh`).
/// - `cell_nonlinearities`: `5H` per pixel — three sigmoids and two tanhs.
/// - `head_macs`: multiply-accumulates in the output head, `Hh*Ww * H * k^2`.
/// - `head_bias_adds`: one add per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StepWork {
    /// Multiply-accumulates in the cell convolution, per recurrent step.
    pub cell_macs: u64,
    /// Bias additions in the cell, per recurrent step.
    pub cell_bias_adds: u64,
    /// Elementwise multiply/add in the cell update, per recurrent step.
    pub cell_mul_adds: u64,
    /// Sigmoid/tanh evaluations in the cell, per recurrent step.
    pub cell_nonlinearities: u64,
    /// Multiply-accumulates in the output head, per emission.
    pub head_macs: u64,
    /// Bias additions in the output head, per emission.
    pub head_bias_adds: u64,
}

/// Runtime counter of what actually executed.
///
/// The meter is incremented by the loops that execute the arithmetic, so the
/// reported step count is an observation, not a restatement of the plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkMeter {
    /// Recurrent cell steps executed.
    pub cell_steps: u64,
    /// Output-head applications executed.
    pub head_applications: u64,
}

/// Totals obtained by scaling [`StepWork`] by a [`WorkMeter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkTotals {
    /// Recurrent cell steps executed.
    pub cell_steps: u64,
    /// Output-head applications executed.
    pub head_applications: u64,
    /// Total convolution MACs.
    pub macs: u64,
    /// Total bias additions.
    pub bias_adds: u64,
    /// Total elementwise multiply/adds.
    pub mul_adds: u64,
    /// Total sigmoid/tanh evaluations.
    pub nonlinearities: u64,
}

impl WorkMeter {
    /// Increment the recurrent step count.
    pub fn add_cell_step(&mut self) {
        self.cell_steps += 1;
    }

    /// Increment the head application count.
    pub fn add_head_application(&mut self) {
        self.head_applications += 1;
    }

    /// Scale declared per-step work by what executed.
    pub fn totals(&self, w: &StepWork) -> WorkTotals {
        let c = self.cell_steps;
        let e = self.head_applications;
        WorkTotals {
            cell_steps: c,
            head_applications: e,
            macs: c * w.cell_macs + e * w.head_macs,
            bias_adds: c * w.cell_bias_adds + e * w.head_bias_adds,
            mul_adds: c * w.cell_mul_adds,
            nonlinearities: c * w.cell_nonlinearities,
        }
    }
}

impl WorkTotals {
    /// The empty account — no arithmetic executed.
    pub const ZERO: WorkTotals = WorkTotals {
        cell_steps: 0,
        head_applications: 0,
        macs: 0,
        bias_adds: 0,
        mul_adds: 0,
        nonlinearities: 0,
    };
}

impl std::ops::Add for WorkTotals {
    type Output = WorkTotals;

    fn add(self, rhs: WorkTotals) -> WorkTotals {
        WorkTotals {
            cell_steps: self.cell_steps + rhs.cell_steps,
            head_applications: self.head_applications + rhs.head_applications,
            macs: self.macs + rhs.macs,
            bias_adds: self.bias_adds + rhs.bias_adds,
            mul_adds: self.mul_adds + rhs.mul_adds,
            nonlinearities: self.nonlinearities + rhs.nonlinearities,
        }
    }
}

impl std::iter::Sum for WorkTotals {
    fn sum<I: Iterator<Item = WorkTotals>>(iter: I) -> WorkTotals {
        iter.fold(WorkTotals::ZERO, |a, b| a + b)
    }
}

/// Recurrent state: the pair `(H, C)` plus the carry channel.
///
/// This is the whole persisted payload, and it is deliberately everything the
/// decoder needs — including the most recent observation — so that restoring the
/// state is sufficient on its own.
#[derive(Debug, Clone)]
pub struct State {
    /// Hidden state, `[batch, H, height, width]`, f32.
    pub h: Tensor,
    /// Cell state, `[batch, H, height, width]`, f32.
    pub c: Tensor,
    /// The frame most recently consumed and the one before it,
    /// `[batch, carry, height, width]`, f32.
    pub carry: Tensor,
}

impl State {
    /// Zero state — a batch of `batch` fresh streams.
    pub fn zeros(cfg: &ModelConfig, batch: usize, dev: &Device) -> Result<State> {
        let shape = (batch, cfg.hidden_channels, cfg.height, cfg.width);
        Ok(State {
            h: Tensor::zeros(shape, DType::F32, dev)?,
            c: Tensor::zeros(shape, DType::F32, dev)?,
            carry: Tensor::zeros(
                (batch, CARRY_CHANNELS, cfg.height, cfg.width),
                DType::F32,
                dev,
            )?,
        })
    }

    /// A deep copy. Tensor clones share the underlying storage; the arithmetic
    /// downstream never mutates in place, so this is safe to treat as a value.
    pub fn copy(&self) -> State {
        State {
            h: self.h.clone(),
            c: self.c.clone(),
            carry: self.carry.clone(),
        }
    }

    /// Exact f32 little-endian bytes of `H`.
    pub fn h_bytes(&self) -> Result<Vec<u8>> {
        f32s_to_le_bytes(&self.h.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Exact f32 little-endian bytes of `C`.
    pub fn c_bytes(&self) -> Result<Vec<u8>> {
        f32s_to_le_bytes(&self.c.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Exact f32 little-endian bytes of the carry.
    pub fn carry_bytes(&self) -> Result<Vec<u8>> {
        f32s_to_le_bytes(&self.carry.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Rebuild a state from exact bytes. Every shape is taken from the config,
    /// never from the record, so a malformed record cannot allocate an arbitrary
    /// shape.
    pub fn from_bytes(
        cfg: &ModelConfig,
        batch: usize,
        h: &[u8],
        c: &[u8],
        carry: &[u8],
    ) -> Result<State> {
        let dev = Device::Cpu;
        Ok(State {
            h: Tensor::from_vec(
                le_bytes_to_f32s(h)?,
                (batch, cfg.hidden_channels, cfg.height, cfg.width),
                &dev,
            )?,
            c: Tensor::from_vec(
                le_bytes_to_f32s(c)?,
                (batch, cfg.hidden_channels, cfg.height, cfg.width),
                &dev,
            )?,
            carry: Tensor::from_vec(
                le_bytes_to_f32s(carry)?,
                (batch, CARRY_CHANNELS, cfg.height, cfg.width),
                &dev,
            )?,
        })
    }

    /// Whether every element is finite.
    ///
    /// A state containing NaN or infinity can still be persisted and restored
    /// byte-exactly, but it is not a *useful* earned state. Reporting this keeps a
    /// degenerate demo from passing on a technicality.
    pub fn is_finite(&self) -> Result<bool> {
        for t in [&self.h, &self.c, &self.carry] {
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            if v.iter().any(|x| !x.is_finite()) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// The ConvLSTM generator.
#[derive(Debug)]
pub struct ConvLstm {
    /// Architecture.
    pub cfg: ModelConfig,
    gates_w: Tensor,
    gates_b: Tensor,
    head_mid_w: Tensor,
    head_mid_b: Tensor,
    head_out_w: Tensor,
    head_out_b: Tensor,
}

impl ConvLstm {
    /// Build from a variable source (a `VarMap` while training, a tensor map for
    /// inference). Shapes are checked by the builder.
    pub fn new(cfg: ModelConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_channels;
        let k = cfg.kernel;
        let ci = cfg.in_channels + h;
        let w_scale = Init::Uniform {
            lo: -0.12,
            up: 0.12,
        };
        let gates_w = vb.get_with_hints((GATES * h, ci, k, k), NAME_GATES_WEIGHT, w_scale)?;
        let gates_b = vb.get_with_hints(GATES * h, NAME_GATES_BIAS, Init::Const(0.0))?;
        let head_mid_w = vb.get_with_hints(
            (HEAD_HIDDEN, h + CARRY_CHANNELS, k, k),
            NAME_HEAD_MID_WEIGHT,
            w_scale,
        )?;
        let head_mid_b = vb.get_with_hints(HEAD_HIDDEN, NAME_HEAD_MID_BIAS, Init::Const(0.0))?;
        // The residual branch starts at zero, so the head begins by emitting the carry
        // unchanged: the exactly-representable persistence solution.
        let head_out_w = vb.get_with_hints(
            (1, HEAD_HIDDEN, 1, 1),
            NAME_HEAD_OUT_WEIGHT,
            Init::Const(0.0),
        )?;
        let head_out_b = vb.get_with_hints(1, NAME_HEAD_OUT_BIAS, Init::Const(0.0))?;
        Ok(ConvLstm {
            cfg,
            gates_w,
            gates_b,
            head_mid_w,
            head_mid_b,
            head_out_w,
            head_out_b,
        })
    }

    /// The bias a gated recurrence wants at step zero: the forget gate open.
    ///
    /// This is the PyTorch LSTM default (`forget_gate_bias = 1`). It matters here
    /// because the first steps of a long context are otherwise dominated by a
    /// saturated cell, and a saturated `C` is a poor thing to demonstrate
    /// persistence of. Returns `4H` values in gate-major order.
    pub fn initial_gate_bias(cfg: &ModelConfig) -> Vec<f32> {
        let mut out = vec![0.0f32; GATES * cfg.hidden_channels];
        let h = cfg.hidden_channels;
        // Gate order in the folded weight is [input, forget, output, cell].
        for v in out[h..2 * h].iter_mut() {
            *v = 1.0;
        }
        out
    }

    /// The initialisation that makes the head a pure persistence predictor.
    ///
    /// The output projection starts at zero, so the residual branch contributes nothing
    /// and `head` returns the carry unchanged. Returns `(out_weight, out_bias)`. The
    /// hidden branch keeps its ordinary random initialisation, which is harmless while
    /// its output is zero and leaves a useful gradient from the very first step.
    pub fn persistence_head(_cfg: &ModelConfig) -> (Vec<f32>, Vec<f32>) {
        (vec![0.0f32; HEAD_HIDDEN], vec![0.0f32; 1])
    }

    /// The exact shape each parameter must have for `cfg`.
    pub fn parameter_shapes(cfg: &ModelConfig) -> Vec<(&'static str, Vec<usize>)> {
        let h = cfg.hidden_channels;
        let k = cfg.kernel;
        vec![
            (
                NAME_GATES_WEIGHT,
                vec![GATES * h, cfg.in_channels + h, k, k],
            ),
            (NAME_GATES_BIAS, vec![GATES * h]),
            (
                NAME_HEAD_MID_WEIGHT,
                vec![HEAD_HIDDEN, h + CARRY_CHANNELS, k, k],
            ),
            (NAME_HEAD_MID_BIAS, vec![HEAD_HIDDEN]),
            (NAME_HEAD_OUT_WEIGHT, vec![1, HEAD_HIDDEN, 1, 1]),
            (NAME_HEAD_OUT_BIAS, vec![1]),
        ]
    }

    /// Load a frozen checkpoint from a safetensors file.
    ///
    /// Every expected parameter must be present with exactly the expected shape;
    /// anything missing, extra or mis-shaped is an error rather than a silently
    /// initialised variable. A model that quietly invents weights would make
    /// every downstream hash comparison meaningless.
    ///
    /// When the *canonical* path is absent this falls back to the embedded copy; see
    /// [`read_checkpoint`] for exactly when that happens and why.
    pub fn load_safetensors(path: &Path, cfg: ModelConfig) -> Result<ConvLstm> {
        let bytes = read_checkpoint(path).map_err(candle_core::Error::Msg)?;
        ConvLstm::load_safetensors_bytes(&bytes, path, cfg)
    }

    /// Load a frozen checkpoint from safetensors bytes already in memory.
    ///
    /// The validation is identical to [`ConvLstm::load_safetensors`]; `origin` names
    /// the checkpoint in error messages. Taking the bytes rather than a path is what
    /// lets a caller hash *exactly* the bytes it loaded — reading the file twice would
    /// leave a window in which the hashed bytes and the loaded weights could differ.
    pub fn load_safetensors_bytes(
        bytes: &[u8],
        origin: &Path,
        cfg: ModelConfig,
    ) -> Result<ConvLstm> {
        let dev = Device::Cpu;
        let map = candle_core::safetensors::load_buffer(bytes, &dev)?;
        for (name, shape) in ConvLstm::parameter_shapes(&cfg) {
            let t = map.get(name).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "checkpoint {origin:?} is missing parameter {name}"
                ))
            })?;
            if t.dims() != shape.as_slice() {
                candle_core::bail!(
                    "checkpoint {origin:?}: parameter {name} has shape {:?}, expected {shape:?}",
                    t.dims()
                );
            }
            if t.dtype() != DType::F32 {
                candle_core::bail!(
                    "checkpoint {origin:?}: parameter {name} has dtype {:?}, expected f32",
                    t.dtype()
                );
            }
        }
        let extra: Vec<&String> = map
            .keys()
            .filter(|k| !PARAM_NAMES.contains(&k.as_str()))
            .collect();
        if !extra.is_empty() {
            candle_core::bail!("checkpoint {origin:?} has unexpected parameters {extra:?}");
        }
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        ConvLstm::new(cfg, vb)
    }

    /// The parameters as named tensors, for hashing and for saving.
    pub fn named_parameters(&self) -> Vec<(&'static str, Tensor)> {
        vec![
            (NAME_GATES_WEIGHT, self.gates_w.clone()),
            (NAME_GATES_BIAS, self.gates_b.clone()),
            (NAME_HEAD_MID_WEIGHT, self.head_mid_w.clone()),
            (NAME_HEAD_MID_BIAS, self.head_mid_b.clone()),
            (NAME_HEAD_OUT_WEIGHT, self.head_out_w.clone()),
            (NAME_HEAD_OUT_BIAS, self.head_out_b.clone()),
        ]
    }

    /// One recurrent step: consume `(frame_t, u_t)` and return `S_t`.
    ///
    /// The consumed frame becomes the new carry, so the returned state is a complete
    /// description of everything the decoder will read.
    pub fn cell(&self, frame: &Tensor, control: &Tensor, prev: &State) -> Result<State> {
        debug_assert_eq!(frame.dims().len(), 4);
        let x = Tensor::cat(&[frame, control, &prev.h], 1)?;
        let pad = self.cfg.kernel / 2;
        let gates = x.conv2d(&self.gates_w, pad, 1, 1, 1)?;
        let b = self
            .gates_b
            .reshape((1, GATES * self.cfg.hidden_channels, 1, 1))?;
        let gates = gates.broadcast_add(&b)?;

        let h = self.cfg.hidden_channels;
        let i = nn_ops::sigmoid(&gates.narrow(1, 0, h)?)?;
        let f = nn_ops::sigmoid(&gates.narrow(1, h, h)?)?;
        let o = nn_ops::sigmoid(&gates.narrow(1, 2 * h, h)?)?;
        let g = gates.narrow(1, 3 * h, h)?.tanh()?;

        let c = ((f * &prev.c)? + (i * g)?)?;
        let h_new = (o * c.tanh()?)?;
        // Shift the carry window: the frame just consumed becomes the newest entry and
        // the previous newest moves down one slot.
        let previous = prev.carry.narrow(1, 0, CARRY_CHANNELS - 1)?;
        let carry = Tensor::cat(&[frame, &previous], 1)?;
        Ok(State { h: h_new, c, carry })
    }

    /// Emit the prediction of `frame_{t+1}` from `S_t`.
    ///
    /// Reads `[H_t ; carry_t]` — the standard ConvLSTM decode — with a linear
    /// output clamped to the physical intensity range.
    /// `y = clamp01( carry_t + RESIDUAL_SCALE * tanh(conv_1(m)) )`
    ///
    /// Reading `[H_t ; carry_t]` is the textbook decode. The *residual* form, the
    /// hidden `tanh` layer and the bound on the correction are all load-bearing, and
    /// all three were arrived at by measurement:
    ///
    /// - The residual form makes "repeat the carry" exactly representable, so the
    ///   model starts at a rollout that cannot fade and only has to learn the motion.
    /// - The nonlinearity is what makes *heading-dependent* motion representable at
    ///   all. A fixed linear kernel applied to the carry can only translate it by a
    ///   fixed vector; the correct next frame is a translation *along the object's own
    ///   heading*, which a purely linear operator cannot express. Measured: a linear
    ///   head on this task expands the object and barely moves it.
    /// - The bound on the correction keeps the head from diverging, because the head's
    ///   output becomes the next step's carry.
    pub fn head(&self, st: &State) -> Result<Tensor> {
        let x = Tensor::cat(&[&st.h, &st.carry], 1)?;
        let pad = self.cfg.kernel / 2;
        let m = x.conv2d(&self.head_mid_w, pad, 1, 1, 1)?;
        let b = self.head_mid_b.reshape((1, HEAD_HIDDEN, 1, 1))?;
        let m = m.broadcast_add(&b)?.tanh()?;
        let g = m.conv2d(&self.head_out_w, 0, 1, 1, 1)?;
        let b = self.head_out_b.reshape((1, 1, 1, 1))?;
        let g = g.broadcast_add(&b)?.tanh()?;
        let newest = st.carry.narrow(1, 0, 1)?;
        (&newest + (g * RESIDUAL_SCALE as f64)?)?.clamp(0f32, 1f32)
    }
}

// ---------------------------------------------------------------------------
// Exact f32 <-> bytes
// ---------------------------------------------------------------------------

/// `f32` slice to little-endian bytes. Bit-preserving, including subnormals.
pub fn f32s_to_le_bytes(v: &[f32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    Ok(out)
}

/// Little-endian bytes to `f32`. Bit-preserving.
pub fn le_bytes_to_f32s(b: &[u8]) -> Result<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        candle_core::bail!("f32 payload length {} is not a multiple of 4", b.len());
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{hash_hex, weights_hash};

    /// A model with small random weights, which is what a fresh training run
    /// starts from. A zero-initialised model is degenerate on purpose (all gates
    /// saturate to their sigmoid midpoint), so it is not used for behaviour tests.
    fn model() -> (ConvLstm, candle_nn::VarMap) {
        let cfg = ModelConfig::default();
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let m = ConvLstm::new(cfg, vb).unwrap();
        (m, varmap)
    }

    #[test]
    fn parameter_shapes_match_the_built_model() {
        let (m, _vm) = model();
        let named = m.named_parameters();
        let shapes = ConvLstm::parameter_shapes(&m.cfg);
        assert_eq!(shapes.len(), named.len());
        for ((name, shape), (actual_name, t)) in shapes.iter().zip(named.iter()) {
            assert_eq!(name, actual_name);
            assert_eq!(shape.as_slice(), t.dims(), "shape of {name}");
        }
    }

    #[test]
    fn the_head_starts_as_a_pure_persistence_predictor() {
        let dev = Device::Cpu;
        let cfg = ModelConfig::default();
        let mut vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let m = ConvLstm::new(cfg, vb).unwrap();
        let (w, b) = ConvLstm::persistence_head(&cfg);
        vm.set_one(
            NAME_HEAD_OUT_WEIGHT,
            Tensor::from_vec(w, (1, HEAD_HIDDEN, 1, 1), &dev).unwrap(),
        )
        .unwrap();
        vm.set_one(NAME_HEAD_OUT_BIAS, Tensor::from_vec(b, 1, &dev).unwrap())
            .unwrap();

        let mut s = State::zeros(&cfg, 1, &dev).unwrap();
        let frame = Tensor::rand(0f32, 1f32, (1, 1, cfg.height, cfg.width), &dev).unwrap();
        let ctrl = Tensor::zeros((1, 2, cfg.height, cfg.width), DType::F32, &dev).unwrap();
        s = m.cell(&frame, &ctrl, &s).unwrap();
        let y = m.head(&s).unwrap();
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            frame.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the persistence initialisation must emit the carry bit for bit"
        );
    }

    #[test]
    fn declared_work_matches_the_architecture() {
        let cfg = ModelConfig::default();
        let w = cfg.step_work();
        let h = cfg.hidden_channels as u64;
        let s = cfg.spatial();
        assert_eq!(w.cell_macs, s * (IN_CHANNELS as u64 + h) * 4 * h * 9);
        assert_eq!(
            w.head_macs,
            s * HEAD_HIDDEN as u64 * (h + CARRY_CHANNELS as u64) * 9 + s * HEAD_HIDDEN as u64
        );
        assert_eq!(w.cell_nonlinearities, 5 * h * s);
        assert_eq!(cfg.state_floats(), (2 * h + CARRY_CHANNELS as u64) * s);
    }

    #[test]
    fn parameter_count_matches_the_named_tensors() {
        let (m, _vm) = model();
        let n: u64 = m
            .named_parameters()
            .iter()
            .map(|(_, t)| t.elem_count() as u64)
            .sum();
        assert_eq!(n, m.cfg.parameter_count());
    }

    #[test]
    fn state_round_trips_bit_exactly() {
        let cfg = ModelConfig::default();
        let mut st = State::zeros(&cfg, 1, &Device::Cpu).unwrap();
        st.h = Tensor::rand(-3.0f32, 3.0f32, st.h.shape(), &Device::Cpu).unwrap();
        st.c = Tensor::rand(-0.5f32, 0.5f32, st.c.shape(), &Device::Cpu).unwrap();
        st.carry = Tensor::rand(0f32, 1f32, st.carry.shape(), &Device::Cpu).unwrap();
        let hb = st.h_bytes().unwrap();
        let cb = st.c_bytes().unwrap();
        let kb = st.carry_bytes().unwrap();
        let back = State::from_bytes(&cfg, 1, &hb, &cb, &kb).unwrap();
        assert_eq!(back.h_bytes().unwrap(), hb);
        assert_eq!(back.c_bytes().unwrap(), cb);
        assert_eq!(back.carry_bytes().unwrap(), kb);
        assert_eq!(
            hash_hex(&hb),
            hash_hex(&back.h_bytes().unwrap()),
            "restored H must hash identically"
        );
        assert!(st.is_finite().unwrap());
    }

    #[test]
    fn cell_changes_state_and_head_stays_in_range() {
        let (m, _vm) = model();
        let cfg = m.cfg;
        let dev = Device::Cpu;
        let s0 = State::zeros(&cfg, 1, &dev).unwrap();
        let frame = Tensor::zeros((1, 1, cfg.height, cfg.width), DType::F32, &dev).unwrap();
        let ctrl = Tensor::ones((1, 2, cfg.height, cfg.width), DType::F32, &dev).unwrap();

        // From a zero state the head emits the clamped residual over a zero carry, so
        // the output is whatever the (randomly initialised) residual branch produces,
        // clamped to the physical range.
        let y0 = m.head(&s0).unwrap();
        let v0 = y0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(y0.dims(), &[1, 1, cfg.height, cfg.width]);
        assert!(v0.iter().all(|x| (0.0..=1.0).contains(x)));

        let s1 = m.cell(&frame, &ctrl, &s0).unwrap();
        let y = m.head(&s1).unwrap();
        assert_eq!(y.dims(), &[1, 1, cfg.height, cfg.width]);
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!(s1.h_bytes().unwrap() != s0.h_bytes().unwrap());
        assert!(s1.is_finite().unwrap());
        // The newest carry entry is exactly the frame the cell consumed.
        assert_eq!(s1.carry.dims(), &[1, CARRY_CHANNELS, cfg.height, cfg.width]);
        assert_eq!(
            s1.carry
                .narrow(1, 0, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            frame.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn repeated_stepping_is_deterministic() {
        let (m, _vm) = model();
        let cfg = m.cfg;
        let dev = Device::Cpu;
        let frame = Tensor::rand(0f32, 1f32, (1, 1, cfg.height, cfg.width), &dev).unwrap();
        let ctrl = Tensor::ones((1, 2, cfg.height, cfg.width), DType::F32, &dev).unwrap();
        let run = || {
            let mut s = State::zeros(&cfg, 1, &dev).unwrap();
            for _ in 0..16 {
                s = m.cell(&frame, &ctrl, &s).unwrap();
                s = m.cell(&frame, &ctrl, &s).unwrap();
            }
            hash_hex(&s.h_bytes().unwrap())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn canonical_bytes_encode_the_config() {
        let cfg = ModelConfig::default();
        let a = cfg.canonical_bytes();
        let mut other = cfg;
        other.hidden_channels = 6;
        assert_ne!(a, other.canonical_bytes());
        assert_eq!(a, ModelConfig::default().canonical_bytes());
    }

    #[test]
    fn initial_gate_bias_opens_the_forget_gate() {
        let cfg = ModelConfig::default();
        let b = ConvLstm::initial_gate_bias(&cfg);
        let h = cfg.hidden_channels;
        assert!(b[..h].iter().all(|v| *v == 0.0));
        assert!(b[h..2 * h].iter().all(|v| *v == 1.0));
        assert!(b[2 * h..].iter().all(|v| *v == 0.0));
    }

    // --- the embedded checkpoint, and the rule for falling back to it ---------

    #[test]
    fn the_embedded_checkpoint_is_byte_identical_to_the_shipped_file() {
        // The fallback is only honest if it serves exactly the frozen artifact. If the
        // checkpoint is ever retrained or replaced without rebuilding, this fails loudly
        // instead of letting the embedded copy and the file it is hashed as diverge.
        let on_disk = std::fs::read(CHECKPOINT_PATH).expect("shipped checkpoint on disk");
        assert_eq!(
            on_disk.as_slice(),
            embedded_checkpoint(),
            "the embedded checkpoint has drifted from {CHECKPOINT_PATH}"
        );
    }

    #[test]
    fn the_fallback_fires_only_for_an_absent_canonical_checkpoint() {
        let not_found = std::io::Error::from(std::io::ErrorKind::NotFound);
        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);

        assert!(is_canonical_checkpoint(Path::new(CHECKPOINT_PATH)));
        assert!(!is_canonical_checkpoint(Path::new(
            "elsewhere/other.safetensors"
        )));

        // The one case that may substitute.
        assert_eq!(
            fallback_for(Path::new(CHECKPOINT_PATH), &not_found),
            Some(embedded_checkpoint())
        );
        // A checkpoint named explicitly is never substituted, even when missing.
        assert!(fallback_for(Path::new("elsewhere/other.safetensors"), &not_found).is_none());
        // A canonical path that failed for any other reason is a real error.
        assert!(fallback_for(Path::new(CHECKPOINT_PATH), &denied).is_none());
    }

    #[test]
    fn a_checkpoint_that_was_named_explicitly_is_never_substituted() {
        // The canonical filename, but in a directory that does not exist. This must be an
        // error: quietly serving the embedded weights would make every hash the program
        // reports describe a model the caller did not ask for.
        let missing = Path::new("no-such-directory").join(CHECKPOINT_PATH);
        let err = read_checkpoint(&missing).unwrap_err();
        assert!(err.starts_with("read "), "unexpected error: {err}");
    }

    #[test]
    fn loading_the_embedded_bytes_yields_the_same_weights_as_the_shipped_file() {
        // The load-bearing property of the fallback: an installed binary, which has no
        // `assets/` beside it, loads exactly the model the repository ships. If these two
        // hashes ever differ, the fallback is not a fallback but a different experiment.
        let cfg = ModelConfig::default();
        let from_file = ConvLstm::load_safetensors(Path::new(CHECKPOINT_PATH), cfg).unwrap();
        let from_embedded = ConvLstm::load_safetensors_bytes(
            embedded_checkpoint(),
            Path::new(CHECKPOINT_PATH),
            cfg,
        )
        .unwrap();
        assert_eq!(
            weights_hash(&from_file).unwrap(),
            weights_hash(&from_embedded).unwrap()
        );
    }
}
