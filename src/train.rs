//! Offline training of the frozen checkpoint.
//!
//! # This is not part of the canonical demo
//!
//! The Colab notebook never runs this. It exists because the repository must be
//! able to produce the artifact it ships, and because "trust me, the weights are
//! frozen" is weaker than "here is the seed, the corpus generator, the objective
//! and the exact command".
//!
//! # The objective
//!
//! Teacher-forced next-frame prediction over truncated back-propagation through
//! time:
//!
//! ```text
//! burn b steps:        S_t = cell(x_t, u_t, S_{t-1})          (detached, no loss)
//! then for k in b..:   loss += MSE( head(S_k), x_{k+1} )
//!                      S_{k+1} = cell(x_{k+1}, u_{k+1}, S_k)   (gradients flow)
//! ```
//!
//! The burn-in is detached, so gradients do not flow through it: that is
//! truncated BPTT, and it costs a bounded amount of memory for an unbounded
//! amount of context. The burn-in length is resampled per iteration from
//! `[burn_min, burn_max]`, which matters because the demo earns its state from a
//! 256-frame context. A model trained only on short windows would be evaluated
//! far outside its training distribution; sampling long burn-ins keeps the
//! steady-state regime inside it.
//!
//! # Corpus
//!
//! Procedurally generated in-process by [`crate::scene`]: random single-blob
//! scenes with random initial position, direction and radius, driven by random
//! held-segment control schedules. No download, no dataset file. Reproducing the
//! checkpoint needs only the seed recorded in this file and the same toolchain.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::model::{
    ConvLstm, ModelConfig, State, NAME_GATES_BIAS, NAME_HEAD_OUT_BIAS, NAME_HEAD_OUT_WEIGHT,
};
use crate::scene::{self, Rng, SceneSpec, HW};
use crate::state::{hash_hex, weights_hash};

/// Default training hyper-parameters. Changing any of these changes the shipped
/// checkpoint, so they are recorded in its metadata.
pub const DEFAULT_SEED: u64 = 0x5601_E1D5_2C07_BA11;
/// Default optimisation steps.
pub const DEFAULT_ITERATIONS: usize = 12_000;
/// Default batch size.
pub const DEFAULT_BATCH: usize = 4;
/// Shortest burn-in, in frames.
pub const BURN_MIN: usize = 4;
/// Longest burn-in, in frames.
pub const BURN_MAX: usize = 12;
/// Number of gradient-carrying prediction steps per sample.
///
/// This is deliberately equal to the demo's generation horizon. Measured failure
/// mode: training on a shorter horizon produces a model whose *first* generated
/// frame is exact and whose rollout then fades away, because the only supervision it
/// ever received was for the first handful of self-generated steps. Supervising
/// exactly the horizon that will be asked for is what stops the fade.
pub const LOSS_STEPS: usize = 16;
/// Final probability of feeding the model its own prediction back as the next
/// input during training.
///
/// This is the single most important hyper-parameter here. With pure teacher
/// forcing the model is only ever asked to predict from ground-truth frames, so it
/// never learns to correct its own drift; at generation time its first blurry
/// prediction is fed back and the rollout settles into a soft, dim fixed point of
/// its own making. Scheduled sampling trains the model in the regime it will
/// actually be used in.
pub const SCHEDULED_SAMPLING_P_MAX: f32 = 1.0;
/// Fraction of the run over which the scheduled-sampling probability ramps to
/// [`SCHEDULED_SAMPLING_P_MAX`].
///
/// A linear ramp over the whole run works best here. Ramping early starves the
/// model of the teacher-forced signal it needs to learn the dynamics at all; that
/// was measured, not assumed (see the training notes in the README).
pub const SCHEDULED_SAMPLING_RAMP: f32 = 1.0;
/// Default AdamW learning rate (the schedule decays it to [`LR_FLOOR`]).
pub const DEFAULT_LR: f64 = 4e-3;
/// Final learning rate of the cosine decay.
pub const LR_FLOOR: f64 = 2e-4;
/// Where the frozen checkpoint is written.
pub const DEFAULT_OUT: &str = "assets/tiny_convlstm.safetensors";

/// Weight added to a pixel's squared error in proportion to its target intensity.
///
/// A frame here is 1,024 pixels of which roughly 40 carry the object. Under an
/// unweighted per-pixel loss, fading the object away is almost free: emitting black
/// scores a smaller error than emitting a slightly mispositioned object, so the
/// optimiser is rewarded for a rollout that quietly dies. Measured: rollouts trained
/// that way hold total ink for seven or eight steps and then collapse to almost
/// nothing by the end of the horizon.
///
/// Weighting each pixel by `1 + W * target` makes the sparse bright region expensive
/// to lose, which is what stops the fade. It is a property of the *objective*, not of
/// the model, the state, or any measurement.
pub const FOREGROUND_LOSS_WEIGHT: f32 = 6.0;

/// Objective used to fit the next frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Loss {
    /// Mean squared error, weighted by `1 + `[`FOREGROUND_LOSS_WEIGHT`]` * target`.
    ///
    /// The default. With the linear head of [`crate::model::ConvLstm`], squared error
    /// has a clean gradient and no saturation, and the weighting keeps the sparse
    /// object from being traded away for a quiet background.
    WeightedMse,
    /// Plain mean squared error, with every pixel weighted equally.
    ///
    /// Kept as a selectable alternative and as part of the checkpoint's provenance:
    /// it is the objective that produced the fading rollouts described in
    /// [`FOREGROUND_LOSS_WEIGHT`].
    Mse,
    /// Soft binary cross-entropy on the emitted intensity in `[0, 1]`.
    ///
    /// Also kept as a selectable alternative. Its gradient `sigma - y` was valuable
    /// while the head was a sigmoid; with a linear head it is the wrong objective, and
    /// on this task it diffuses — the rollout keeps total ink roughly constant while
    /// spreading it out.
    SoftBce,
}

impl Loss {
    /// CLI value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Loss::WeightedMse => "weighted-mse",
            Loss::Mse => "mse",
            Loss::SoftBce => "soft-bce",
        }
    }

    /// Parse a CLI value.
    pub fn parse(s: &str) -> Option<Loss> {
        match s {
            "weighted-mse" | "wmse" => Some(Loss::WeightedMse),
            "mse" => Some(Loss::Mse),
            "soft-bce" | "bce" => Some(Loss::SoftBce),
            _ => None,
        }
    }

    /// The scalar loss for one emitted frame, already reduced to a mean.
    fn per_frame(&self, pred: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
        let diff = (pred - target)?.sqr()?;
        match self {
            Loss::Mse => diff.mean_all(),
            Loss::WeightedMse => {
                let t = target.to_dtype(pred.dtype())?;
                let w = t.affine(FOREGROUND_LOSS_WEIGHT as f64, 1.0)?;
                (w * diff)?.mean_all()
            }
            Loss::SoftBce => {
                const EPS: f64 = 1e-6;
                let p = pred.clamp(EPS, 1.0 - EPS)?;
                let one_minus_p = p.affine(-1.0, 1.0)?;
                let t = target.to_dtype(pred.dtype())?;
                let t1 = t.affine(-1.0, 1.0)?;
                let a = (t * p.log()?)?;
                let b = (t1 * one_minus_p.log()?)?;
                (a + b)?.mean_all()?.affine(-1.0, 0.0)
            }
        }
    }
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// PRNG seed. The corpus is a pure function of this and the code.
    pub seed: u64,
    /// Optimisation steps.
    pub iterations: usize,
    /// Scenes per step.
    pub batch: usize,
    /// Shortest window, in frames.
    pub burn_min: usize,
    /// Longest window, in frames.
    pub burn_max: usize,
    /// Prediction steps that carry loss, counted back from the end of the window.
    pub loss_steps: usize,
    /// AdamW learning rate at step 0.
    pub lr: f64,
    /// Back-propagate through the whole window. When false the pre-loss steps are
    /// detached (truncated BPTT), so gradient does not reach state construction.
    pub full_bptt: bool,
    /// Objective to optimise.
    pub loss: Loss,
    /// Final probability of replacing a ground-truth input with the model's own
    /// prediction. Ramped linearly from zero over the run.
    pub scheduled_sampling_p_max: f32,
    /// Checkpoint output path.
    pub out: PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            seed: DEFAULT_SEED,
            iterations: DEFAULT_ITERATIONS,
            batch: DEFAULT_BATCH,
            burn_min: BURN_MIN,
            burn_max: BURN_MAX,
            loss_steps: LOSS_STEPS,
            lr: DEFAULT_LR,
            full_bptt: true,
            loss: Loss::WeightedMse,
            scheduled_sampling_p_max: SCHEDULED_SAMPLING_P_MAX,
            out: PathBuf::from(DEFAULT_OUT),
        }
    }
}

/// What the trainer reports when it finishes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainReport {
    /// Seed used.
    pub seed: u64,
    /// Iterations run.
    pub iterations: usize,
    /// Batch size.
    pub batch: usize,
    /// Burn-in range used.
    pub burn_range: [usize; 2],
    /// Gradient-carrying prediction steps.
    pub loss_steps: usize,
    /// Learning rate at step 0.
    pub lr: f64,
    /// Learning rate at the final step.
    pub lr_final: f64,
    /// Whether gradients flowed through the whole window.
    pub full_bptt: bool,
    /// Objective.
    pub loss: Loss,
    /// Final teacher-forcing replacement probability.
    pub scheduled_sampling_p_max: f32,
    /// Loss from the first step.
    pub first_loss: f64,
    /// Lowest loss seen at any step.
    pub best_loss: f64,
    /// Loss from the last step.
    pub last_loss: f64,
    /// Mean loss over the final tenth of the run.
    pub final_mean_loss: f64,
    /// Wall-clock seconds.
    pub elapsed_s: f64,
    /// Fingerprint of the first sampled batch: the sampling pipeline's identity.
    pub corpus_digest: String,
    /// BLAKE3 of the written checkpoint file.
    pub checkpoint_file_hash: String,
    /// Behaviour-determining hash of the parameters.
    pub weights_hash: String,
    /// Parameters in the model.
    pub parameter_count: u64,
    /// Checkpoint file size in bytes.
    pub checkpoint_bytes: u64,
    /// Whether the checkpoint reloads with an identical weights hash.
    pub reload_verified: bool,
    /// Objective quality of the checkpoint against the canonical scene.
    pub eval: EvalReport,
}

/// Sample `steps + 1` batched frames and `steps` batched control planes.
///
/// Returns `(frames, controls, first_batch_digest)`. `frames[k]` has shape
/// `[batch, 1, 32, 32]`; `controls[k]` has shape `[batch, 2, 32, 32]`.
fn sample_batch(
    rng: &mut Rng,
    batch: usize,
    steps: usize,
    digest: &mut blake3::Hasher,
    digest_frames: bool,
) -> candle_core::Result<(Vec<Tensor>, Vec<Tensor>)> {
    let dev = Device::Cpu;
    let mut frames: Vec<Vec<f32>> = (0..=steps)
        .map(|_| Vec::with_capacity(batch * HW))
        .collect();
    let mut turns: Vec<Vec<f32>> = (0..steps).map(|_| Vec::with_capacity(batch * HW)).collect();
    let mut accels: Vec<Vec<f32>> = (0..steps).map(|_| Vec::with_capacity(batch * HW)).collect();

    for _ in 0..batch {
        let spec = SceneSpec::random(rng);
        let sched = rng.schedule(steps);
        let mut bodies = spec.working();
        let f0 = scene::render(&bodies);
        if digest_frames {
            for x in &f0 {
                digest.update(&x.to_le_bytes());
            }
        }
        frames[0].extend_from_slice(&f0);
        for k in 0..steps {
            let u = sched.at(k);
            for b in bodies.iter_mut() {
                b.advance(u);
            }
            let f = scene::render(&bodies);
            if digest_frames {
                for x in &f {
                    digest.update(&x.to_le_bytes());
                }
            }
            frames[k + 1].extend_from_slice(&f);
            let [t, a] = u.planes();
            turns[k].extend(std::iter::repeat_n(t, HW));
            accels[k].extend(std::iter::repeat_n(a, HW));
        }
    }

    let fshape = (batch, 1, scene::H, scene::W);
    let cshape = (batch, 2, scene::H, scene::W);
    let mut ft = Vec::with_capacity(steps + 1);
    for f in frames {
        ft.push(Tensor::from_vec(f, fshape, &dev)?);
    }
    let mut ct = Vec::with_capacity(steps);
    for (mut t, a) in turns.into_iter().zip(accels) {
        t.extend_from_slice(&a);
        ct.push(Tensor::from_vec(t, cshape, &dev)?);
    }
    Ok((ft, ct))
}

/// Run the trainer.
pub fn run(tc: &TrainConfig, quiet: bool) -> candle_core::Result<TrainReport> {
    let dev = Device::Cpu;
    let cfg = ModelConfig::default();
    let start = std::time::Instant::now();

    // --- parameters ---------------------------------------------------------
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvLstm::new(cfg, vb)?;
    // PyTorch's LSTM default: the forget gate starts open, so a long context is
    // remembered rather than saturated away at step zero.
    let bias = Tensor::from_vec(
        ConvLstm::initial_gate_bias(&cfg),
        cfg.hidden_channels * crate::model::GATES,
        &dev,
    )?;
    varmap.set_one(NAME_GATES_BIAS, &bias)?;
    // Start the head as a pure persistence predictor: emit the carry unchanged. See
    // `ConvLstm::persistence_head`.
    let (hw, hb) = ConvLstm::persistence_head(&cfg);
    varmap.set_one(
        NAME_HEAD_OUT_WEIGHT,
        Tensor::from_vec(hw, (1, crate::model::HEAD_HIDDEN, 1, 1), &dev)?,
    )?;
    varmap.set_one(NAME_HEAD_OUT_BIAS, Tensor::from_vec(hb, 1, &dev)?)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: tc.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;
    let lr_at = |iter: usize| -> f64 {
        // Cosine decay with a floor: a constant rate leaves the last mile on the
        // table, and a decay to zero wastes the final iterations.
        let p = (iter as f64 / tc.iterations.max(1) as f64).min(1.0);
        LR_FLOOR + (tc.lr - LR_FLOOR) * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
    };

    // --- optimisation loop --------------------------------------------------
    let mut rng = Rng::new(tc.seed);
    let mut digest = blake3::Hasher::new();
    let mut losses: Vec<f64> = Vec::with_capacity(tc.iterations);
    let mut last_loss = f64::NAN;
    let tail_from = tc.iterations.saturating_sub(tc.iterations / 10 + 1);

    for iter in 0..tc.iterations {
        opt.set_learning_rate(lr_at(iter));
        let burn = tc.burn_min + rng.below((tc.burn_max - tc.burn_min + 1) as u64) as usize;
        let steps = burn + tc.loss_steps;
        let (frames, controls) = sample_batch(&mut rng, tc.batch, steps, &mut digest, iter == 0)?;

        let mut s = State::zeros(&cfg, tc.batch, &dev)?;
        let mut total = Tensor::zeros((), DType::F32, &dev)?;
        let counted = steps - (burn - 1);
        let p = tc.scheduled_sampling_p_max
            * (iter as f32 / (SCHEDULED_SAMPLING_RAMP * tc.iterations as f32).max(1.0)).min(1.0);

        if tc.full_bptt {
            // One pass, no detach: the graph spans the whole window, so gradient
            // reaches the construction of the recurrent state itself.
            let mut input = frames[0].clone();
            for k in 0..steps {
                s = model.cell(&input, &controls[k], &s)?;
                let pred = model.head(&s)?;
                let target = &frames[k + 1];
                if k + 1 >= burn {
                    total = (total + tc.loss.per_frame(&pred, target)?)?;
                }
                if k + 1 == steps {
                    break;
                }
                // Scheduled sampling: with probability `p` the next input is the
                // model's own prediction, otherwise the ground truth.
                input = if p > 0.0 && rng.f32_unit() < p {
                    pred
                } else {
                    target.clone()
                };
            }
        } else {
            // Truncated BPTT: execute the burn-in, but detach it.
            for k in 0..burn {
                s = model.cell(&frames[k], &controls[k], &s)?;
                s = State {
                    h: s.h.detach(),
                    c: s.c.detach(),
                    carry: s.carry.detach(),
                };
            }
            for k in (burn - 1..).take(tc.loss_steps) {
                let pred = model.head(&s)?;
                total = (total + tc.loss.per_frame(&pred, &frames[k + 1])?)?;
                s = model.cell(&frames[k + 1], &controls[k + 1], &s)?;
            }
        }

        let loss = (total / counted as f64)?;
        opt.backward_step(&loss)?;

        last_loss = loss.to_scalar::<f32>()? as f64;
        losses.push(last_loss);
        if !quiet && (iter % 100 == 0 || iter + 1 == tc.iterations) {
            let tail: f64 = losses[losses.len().saturating_sub(50)..]
                .iter()
                .sum::<f64>()
                / losses[losses.len().saturating_sub(50)..].len().max(1) as f64;
            let secs = start.elapsed().as_secs_f64();
            eprintln!(
                "iter {iter:>5}/{} loss {last_loss:.6} (mean last50 {tail:.6}) window {steps:>3} lr {:.5} p {p:.2} {:.1}s",
                tc.iterations,
                lr_at(iter),
                secs
            );
        }
    }

    let final_mean_loss = {
        let tail = &losses[tail_from..];
        tail.iter().sum::<f64>() / tail.len().max(1) as f64
    };

    // --- write the checkpoint with provenance -------------------------------
    let params = model.named_parameters();
    let mut bytes_per_param: Vec<(&'static str, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, t) in &params {
        let dims = t.dims().to_vec();
        let raw = crate::model::f32s_to_le_bytes(&t.flatten_all()?.to_vec1::<f32>()?)?;
        bytes_per_param.push((name, dims, raw));
    }
    let mut views = std::collections::BTreeMap::new();
    for (name, dims, raw) in &bytes_per_param {
        views.insert(
            (*name).to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, dims.clone(), raw)
                .map_err(|e| candle_core::Error::Msg(format!("safetensors view: {e}")))?,
        );
    }
    let mut meta = std::collections::HashMap::new();
    meta.insert("producer".to_string(), "vole-field train".to_string());
    meta.insert("crate".to_string(), env!("CARGO_PKG_NAME").to_string());
    meta.insert(
        "crate_version".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    );
    meta.insert("rustc".to_string(), rustc_version_string());
    meta.insert("seed".to_string(), format!("0x{:016x}", tc.seed));
    meta.insert("iterations".to_string(), tc.iterations.to_string());
    meta.insert("batch".to_string(), tc.batch.to_string());
    meta.insert(
        "burn_range".to_string(),
        format!("{}..={}", tc.burn_min, tc.burn_max),
    );
    meta.insert("loss_steps".to_string(), tc.loss_steps.to_string());
    meta.insert("lr_initial".to_string(), format!("{}", tc.lr));
    meta.insert("lr_schedule".to_string(), "cosine to 2e-4".to_string());
    meta.insert("full_bptt".to_string(), tc.full_bptt.to_string());
    meta.insert("loss".to_string(), tc.loss.as_str().to_string());
    meta.insert(
        "foreground_loss_weight".to_string(),
        FOREGROUND_LOSS_WEIGHT.to_string(),
    );
    meta.insert(
        "scheduled_sampling_p_max".to_string(),
        tc.scheduled_sampling_p_max.to_string(),
    );
    meta.insert(
        "scheduled_sampling_ramp".to_string(),
        SCHEDULED_SAMPLING_RAMP.to_string(),
    );
    meta.insert(
        "objective".to_string(),
        "next-frame prediction; scheduled sampling ramp; full BPTT over the window".to_string(),
    );
    meta.insert(
        "head_initialisation".to_string(),
        "persistence (emit the carry unchanged)".to_string(),
    );
    meta.insert(
        "corpus_digest".to_string(),
        digest.clone().finalize().to_hex().to_string(),
    );
    meta.insert(
        "final_mean_loss".to_string(),
        format!("{final_mean_loss:.8}"),
    );
    meta.insert(
        "note".to_string(),
        "Tiny ConvLSTM trained on procedurally generated moving-shape scenes; not part of the canonical demo."
            .to_string(),
    );
    if let Some(parent) = tc.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    safetensors::tensor::serialize_to_file(views, Some(meta), &tc.out)
        .map_err(|e| candle_core::Error::Msg(format!("safetensors write: {e}")))?;

    // --- verify what we just wrote ------------------------------------------
    let file_bytes = std::fs::read(&tc.out)?;
    let reloaded = ConvLstm::load_safetensors(&tc.out, cfg)?;
    let reload_verified = weights_hash(&reloaded)? == weights_hash(&model)?;
    let eval = evaluate(&tc.out, CANONICAL_EVAL_CONTEXT, CANONICAL_EVAL_FUTURE)?;

    Ok(TrainReport {
        seed: tc.seed,
        iterations: tc.iterations,
        batch: tc.batch,
        burn_range: [tc.burn_min, tc.burn_max],
        loss_steps: tc.loss_steps,
        lr: tc.lr,
        lr_final: lr_at(tc.iterations.saturating_sub(1)),
        full_bptt: tc.full_bptt,
        loss: tc.loss,
        scheduled_sampling_p_max: tc.scheduled_sampling_p_max,
        first_loss: losses.first().copied().unwrap_or(f64::NAN),
        best_loss: losses.iter().copied().fold(f64::INFINITY, f64::min),
        last_loss,
        final_mean_loss,
        elapsed_s: start.elapsed().as_secs_f64(),
        corpus_digest: digest.finalize().to_hex().to_string(),
        checkpoint_file_hash: hash_hex(&file_bytes),
        weights_hash: crate::state::hex32(&weights_hash(&model)?),
        parameter_count: cfg.parameter_count(),
        checkpoint_bytes: file_bytes.len() as u64,
        reload_verified,
        eval,
    })
}

/// Context length used for the trainer's own quality check. Matches the demo.
pub const CANONICAL_EVAL_CONTEXT: usize = 256;
/// Future length used for the trainer's own quality check. Matches the demo.
pub const CANONICAL_EVAL_FUTURE: usize = 16;

/// The compiler version, for checkpoint provenance.
fn rustc_version_string() -> String {
    option_env!("RUSTC_VERSION")
        .unwrap_or("unknown")
        .to_string()
}

/// Per-request quality of a checkpoint against the ground-truth scene.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RequestEval {
    /// Request name.
    pub request: String,
    /// Mean absolute error over all generated frames.
    pub mae: f64,
    /// Mean absolute error of the final generated frame.
    pub final_mae: f64,
    /// Mean centroid error in pixels over the horizon.
    pub centroid_err_px_mean: f64,
    /// Centroid error in pixels of the final generated frame.
    pub centroid_err_px_final: f64,
    /// Mean peak brightness of the generated frames.
    pub generated_peak_mean: f64,
    /// Peak brightness of the final generated frame.
    pub generated_peak_final: f64,
}

/// How well a checkpoint reproduces the canonical scene.
///
/// This exists so that "the frozen model is good enough to demonstrate the point"
/// is a *measured* statement rather than an assumption, and so that a training run
/// can be judged without looking at pictures.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvalReport {
    /// Frames consumed before the branches start.
    pub context_len: usize,
    /// Frames generated per branch.
    pub future_len: usize,
    /// Mean absolute error over every generated frame of every branch.
    pub mae: f64,
    /// Mean absolute error of the final generated frame of every branch.
    pub final_mae: f64,
    /// Mean centroid error in pixels.
    pub centroid_err_px_mean: f64,
    /// Mean centroid error of the final frames.
    pub centroid_err_px_final: f64,
    /// Mean peak brightness of generated frames (ground truth peaks near 1.0).
    pub generated_peak_mean: f64,
    /// Mean peak brightness of the ground-truth frames.
    pub truth_peak_mean: f64,
    /// Mean total ink of generated frames.
    pub generated_mass_mean: f64,
    /// Mean total ink of ground-truth frames.
    pub truth_mass_mean: f64,
    /// Whether every generated number is finite.
    pub finite: bool,
    /// Per-branch detail.
    pub per_request: Vec<RequestEval>,
    /// Mean absolute error by generation step, averaged over all branches. The first
    /// entry is the one-step prediction from the earned state; the last is the end
    /// of the rollout. A rising profile is autoregressive drift.
    pub mae_by_horizon: Vec<f64>,
    /// Mean peak brightness by generation step, averaged over all branches.
    pub peak_by_horizon: Vec<f64>,
    /// Mean total ink by generation step, averaged over all branches.
    pub mass_by_horizon: Vec<f64>,
    /// Ground-truth mean total ink by generation step.
    pub truth_mass_by_horizon: Vec<f64>,
}

/// Total ink and its centroid, for comparing generated frames with the truth.
fn mass_centroid(f: &[f32]) -> (f64, f64, f64, f64) {
    let mut m = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut peak = 0.0f64;
    for (i, v) in f.iter().enumerate() {
        let v = *v as f64;
        m += v;
        sx += (i % scene::W) as f64 * v;
        sy += (i / scene::W) as f64 * v;
        peak = peak.max(v);
    }
    if m <= 1e-9 {
        (m, 0.0, 0.0, peak)
    } else {
        (m, sx / m, sy / m, peak)
    }
}

/// Evaluate a checkpoint against the canonical scene and the request set.
///
/// Uses exactly the generator the demo uses, so a good number here means the demo
/// will look right. Any error from loading or generating is propagated rather than
/// swallowed: a checkpoint that cannot generate must not score well by accident.
pub fn evaluate(
    checkpoint: &Path,
    context_len: usize,
    future_len: usize,
) -> candle_core::Result<EvalReport> {
    let scene = crate::scene::SceneId::MovingShape01;
    let gen = crate::experiment::Generator::load(
        checkpoint,
        scene,
        context_len,
        future_len,
        ModelConfig::default(),
    )?;
    let ctx = gen.context();
    let mut meter = crate::model::WorkMeter::default();
    let state = gen.earn(&ctx, &mut meter)?;
    let mut finite = state.is_finite()?;

    let spec = SceneSpec::canonical(scene);
    let mut per_request = Vec::new();
    let mut mae_sum = 0.0;
    let mut final_mae_sum = 0.0;
    let mut cent_sum = 0.0;
    let mut cent_final_sum = 0.0;
    let mut gpeak_sum = 0.0;
    let mut tpeak_sum = 0.0;
    let mut gmass_sum = 0.0;
    let mut tmass_sum = 0.0;
    let mut mae_by_horizon = vec![0.0f64; future_len];
    let mut peak_by_horizon = vec![0.0f64; future_len];
    let mut mass_by_horizon = vec![0.0f64; future_len];
    let mut truth_mass_by_horizon = vec![0.0f64; future_len];

    for req in crate::scene::Request::ALL {
        let mut m2 = crate::model::WorkMeter::default();
        let generated = gen.generate(&state, req, &mut m2)?;
        // Ground truth: the same scene, rolled forward from the end of the
        // context under the same request program.
        let mut bodies = spec.working();
        let ctxprog = crate::scene::context_program();
        for t in 0..context_len {
            for b in bodies.iter_mut() {
                b.advance(ctxprog.at(t));
            }
        }
        let sched = req.schedule();
        let mut truth = Vec::with_capacity(future_len);
        for j in 0..future_len {
            for b in bodies.iter_mut() {
                b.advance(sched.at(j));
            }
            truth.push(scene::render(&bodies));
        }

        let mut mae = 0.0;
        let mut final_mae = 0.0;
        let mut cent = 0.0;
        let mut n_cent = 0usize;
        let mut gpeak = 0.0;
        let mut tpeak = 0.0;
        let mut gmass = 0.0;
        let mut tmass = 0.0;
        for j in 0..future_len {
            let g = &generated[j];
            let t = &truth[j];
            finite &= g.iter().all(|v| v.is_finite());
            let e: f64 = g
                .iter()
                .zip(t.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .sum::<f64>()
                / g.len() as f64;
            mae += e;
            if j + 1 == future_len {
                final_mae = e;
            }
            let (gm, gx, gy, gp) = mass_centroid(g);
            let (tm, tx, ty, tp) = mass_centroid(t);
            mae_by_horizon[j] += e / 6.0;
            peak_by_horizon[j] += gp / 6.0;
            mass_by_horizon[j] += gm / 6.0;
            truth_mass_by_horizon[j] += tm / 6.0;
            if gm > 1e-9 && tm > 1e-9 {
                cent += ((gx - tx).powi(2) + (gy - ty).powi(2)).sqrt();
                n_cent += 1;
            }
            gpeak += gp;
            tpeak += tp;
            gmass += gm;
            tmass += tm;
        }
        let f = future_len as f64;
        let cent_mean = cent / n_cent.max(1) as f64;
        mae_sum += mae / f;
        final_mae_sum += final_mae;
        cent_sum += cent_mean;
        cent_final_sum += per_request_centroid_final(&generated, &truth);
        gpeak_sum += gpeak / f;
        tpeak_sum += tpeak / f;
        gmass_sum += gmass / f;
        tmass_sum += tmass / f;
        per_request.push(RequestEval {
            request: req.name().to_string(),
            mae: mae / f,
            final_mae,
            centroid_err_px_mean: cent_mean,
            centroid_err_px_final: per_request_centroid_final(&generated, &truth),
            generated_peak_mean: gpeak / f,
            generated_peak_final: per_request_peak(&generated),
        });
    }

    let n = crate::scene::Request::ALL.len() as f64;
    Ok(EvalReport {
        context_len,
        future_len,
        mae: mae_sum / n,
        final_mae: final_mae_sum / n,
        centroid_err_px_mean: cent_sum / n,
        centroid_err_px_final: cent_final_sum / n,
        generated_peak_mean: gpeak_sum / n,
        truth_peak_mean: tpeak_sum / n,
        generated_mass_mean: gmass_sum / n,
        truth_mass_mean: tmass_sum / n,
        finite,
        per_request,
        mae_by_horizon,
        peak_by_horizon,
        mass_by_horizon,
        truth_mass_by_horizon,
    })
}

fn per_request_peak(frames: &[Vec<f32>]) -> f64 {
    frames.last().map(|f| mass_centroid(f).3).unwrap_or(0.0)
}

fn per_request_centroid_final(generated: &[Vec<f32>], truth: &[Vec<f32>]) -> f64 {
    let (gm, gx, gy, _) = mass_centroid(generated.last().map(|v| v.as_slice()).unwrap_or(&[]));
    let (tm, tx, ty, _) = mass_centroid(truth.last().map(|v| v.as_slice()).unwrap_or(&[]));
    if gm > 1e-9 && tm > 1e-9 {
        ((gx - tx).powi(2) + (gy - ty).powi(2)).sqrt()
    } else {
        f64::NAN
    }
}

/// Path the trainer writes to, resolved relative to the process working dir.
pub fn default_out() -> &'static Path {
    Path::new(DEFAULT_OUT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_produces_the_declared_shapes_and_is_reproducible() {
        let steps = 6;
        let mut a = Rng::new(7);
        let mut h1 = blake3::Hasher::new();
        let (fa, ca) = sample_batch(&mut a, 2, steps, &mut h1, true).unwrap();
        let mut b = Rng::new(7);
        let mut h2 = blake3::Hasher::new();
        let (fb, _cb) = sample_batch(&mut b, 2, steps, &mut h2, true).unwrap();
        assert_eq!(h1.finalize(), h2.finalize());
        assert_eq!(fa.len(), steps + 1);
        assert_eq!(ca.len(), steps);
        for (x, y) in fa.iter().zip(fb.iter()) {
            assert_eq!(
                x.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
        assert_eq!(fa[0].dims(), &[2, 1, scene::H, scene::W]);
        assert_eq!(ca[0].dims(), &[2, 2, scene::H, scene::W]);
    }

    #[test]
    fn a_short_run_produces_a_loadable_checkpoint_that_improves_the_loss() {
        let tc = TrainConfig {
            iterations: 250,
            batch: 2,
            burn_min: 2,
            burn_max: 6,
            loss_steps: 6,
            out: std::env::temp_dir().join("vole-field-train-smoke.safetensors"),
            ..Default::default()
        };
        let r = run(&tc, true).unwrap();
        assert!(r.reload_verified);
        assert!(r.checkpoint_bytes > 0);
        // The head starts at the persistence solution, so the first loss is already
        // low; what must be true is that optimisation improved on it.
        assert!(
            r.best_loss < r.first_loss,
            "loss never improved on the initialisation: best {} vs first {}",
            r.best_loss,
            r.first_loss
        );
        assert!(r.eval.finite);
        assert!(
            r.eval.generated_mass_mean > 0.0,
            "the model emitted nothing"
        );
        assert_eq!(r.eval.mae_by_horizon.len(), r.eval.future_len);
        let _ = std::fs::remove_file(&tc.out);
    }
}
