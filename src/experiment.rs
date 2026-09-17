//! The process-cold durable-generative-state-reuse experiment.
//!
//! # Shape of the experiment
//!
//! The top-level `vole-field run` is an orchestrator. It never executes model
//! arithmetic itself. It launches short-lived child processes of itself:
//!
//! ```text
//! vole-field run
//!   ├── producer  <scene>            earn state, persist it, EXIT
//!   ├── baseline  <request>          fresh process, replay context, generate, EXIT
//!   ├── raw       <request>          fresh process, read raw tensors, generate, EXIT
//!   └── vole      <request>          fresh process, restore via EntropyFS, generate, EXIT
//! ```
//!
//! Every child is a distinct OS process with a distinct PID. No shared memory, no
//! daemon, no warm Rust object carrying the state, no inherited tensors. The only
//! bridge from the producer to a consumer is persisted bytes:
//!
//! - the VOLE path: bytes inside an EntropyFS store, addressed by a blob id;
//! - the raw control: the same payload bytes in an ordinary file.
//!
//! The OS page cache is *not* flushed, so this is a **fresh-process restore**, not
//! a cold-disk-cache experiment. That is stated plainly wherever a result appears.
//!
//! # Fairness rules obeyed here
//!
//! - the same frozen weights, the same context, the same control program, the
//!   same future length, and the same generation code on every path;
//! - the baseline runs the *same* generation code; the only difference is whether
//!   the earned state is recomputed by replaying history or recovered from
//!   persisted bytes;
//! - the baseline replays the context *per request*, because each request really
//!   is a fresh process. That is the scenario under measurement ("process-cold
//!   reuse") and it is stated rather than assumed;
//! - the model weights are shared by both paths and are reported as such: neither
//!   hidden, nor charged to one side;
//! - performance never decides PASS/FAIL. A slower VOLE path is a passing run with
//!   an honest negative performance result, and no crossover is ever manufactured.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_core::{Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::model::{ConvLstm, ModelConfig, State, StepWork, WorkMeter, WorkTotals};
use crate::scene::{self, Control, Request, SceneId, SceneSpec, Schedule, H, HW, W};
use crate::state::{
    config_hash, hash_hex, hex32, state_hash, weights_hash, Check, Expectation, StateRecord,
    VoleStore,
};

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/// Default run directory, relative to the working directory.
pub const DEFAULT_RUN_DIR: &str = "run";
/// Default frozen checkpoint.
pub const DEFAULT_CHECKPOINT: &str = "assets/tiny_convlstm.safetensors";
/// Frames the producer consumes to earn the state.
pub const DEFAULT_CONTEXT: usize = 256;
/// Frames each branch request generates.
pub const DEFAULT_FUTURE: usize = 16;
/// Measured repetitions per timing phase, after one discarded warm-up.
pub const TIMING_REPS: usize = 5;
/// The N values of the repeated-request sweep.
pub const N_POINTS: [usize; 5] = [1, 2, 4, 8, 16];
/// Threads pinned for every child, so thread count can never be a source of
/// divergence between the producer and a consumer.
pub const MAX_PINNED_THREADS: usize = 4;
/// Marker prefix on a child's report line.
pub const CHILD_MARKER: &str = "@@VOLE_FIELD_CHILD@@";
/// Magic prefix of an emitted frame file.
pub const FRAME_MAGIC: [u8; 8] = *b"VOLEFRM1";
/// Frames drawn per row in the montage.
pub const MONTAGE_COLS: usize = 8;
/// Integer upscale factor for the montage.
pub const MONTAGE_SCALE: usize = 3;
/// Gap between montage tiles, in output pixels.
pub const MONTAGE_GAP: usize = 2;
/// Width of the montage's per-row marker margin, in output pixels.
pub const MONTAGE_MARGIN: usize = 6;
/// Extra gap separating the model block from the ground-truth block, in pixels.
pub const MONTAGE_GROUP_GAP: usize = 6;
/// Display gamma applied when rendering a frame to 8-bit. Display only: every
/// measurement and comparison in this crate uses the exact `f32` bytes.
pub const MONTAGE_GAMMA: f32 = 0.7;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Median of a sample set. `None` when there is nothing to take a median of, so
/// an absent phase can never masquerade as a measured zero.
pub fn median(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn now_unix_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Evenly spaced indices over `0..len`, exactly `k` of them.
pub fn sample_indices(len: usize, k: usize) -> Vec<usize> {
    if len == 0 || k == 0 {
        return Vec::new();
    }
    if len <= k {
        return (0..len).collect();
    }
    if k == 1 {
        return vec![0];
    }
    (0..k).map(|i| i * (len - 1) / (k - 1)).collect()
}

/// One measured phase within one process.
///
/// `median_ms` is `None` when the phase did not run in this mode at all (for
/// example `model_only_replay` on the VOLE path, which never replays anything).
/// An absent phase is reported as absent, never as a zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTiming {
    /// Stable phase name.
    pub phase: String,
    /// What the number means.
    pub definition: String,
    /// The discarded warm-up repetition, if one ran.
    pub warmup_ms: Option<f64>,
    /// Every measured repetition, in order, in milliseconds.
    pub samples_ms: Vec<f64>,
    /// Median of `samples_ms`, or `None` if the phase never ran.
    pub median_ms: Option<f64>,
}

impl PhaseTiming {
    fn new(phase: &str, definition: &str, warmup: Option<f64>, samples: Vec<f64>) -> PhaseTiming {
        let median_ms = median(&samples);
        PhaseTiming {
            phase: phase.to_string(),
            definition: definition.to_string(),
            warmup_ms: warmup,
            samples_ms: samples,
            median_ms,
        }
    }
}

/// One phase aggregated across independent child processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseAcross {
    /// Phase name.
    pub phase: String,
    /// What the number means.
    pub definition: String,
    /// One value per child process, in order.
    pub per_process_ms: Vec<f64>,
    /// Median across processes, or `None` if no process reported the phase.
    pub median_ms: Option<f64>,
}

impl PhaseAcross {
    fn from_reports(reports: &[serde_json::Value], phase: &str) -> Option<PhaseAcross> {
        let mut per = Vec::new();
        let mut def = String::new();
        for r in reports {
            let timings: Vec<PhaseTiming> =
                serde_json::from_value(r.get("timings")?.clone()).ok()?;
            let hit = timings.iter().find(|t| t.phase == phase)?;
            per.push(hit.median_ms?);
            def = hit.definition.clone();
        }
        Some(PhaseAcross {
            phase: phase.to_string(),
            definition: def,
            median_ms: median(&per),
            per_process_ms: per,
        })
    }
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Every path the run uses, derived from one root directory.
#[derive(Debug, Clone)]
pub struct Paths {
    /// Run root.
    pub root: PathBuf,
}

impl Paths {
    /// Derive all paths from a run root.
    pub fn new(root: impl Into<PathBuf>) -> Paths {
        Paths { root: root.into() }
    }

    /// The EntropyFS store directory.
    pub fn store(&self) -> PathBuf {
        self.root.join("entropyfs-store")
    }

    /// The raw-checkpoint file (unframed payload).
    pub fn raw_state(&self) -> PathBuf {
        self.root.join("raw_state.bin")
    }

    /// Emitted branch frame files.
    pub fn futures(&self) -> PathBuf {
        self.root.join("futures")
    }

    /// Child report JSON files, for a human debugging a run.
    pub fn reports(&self) -> PathBuf {
        self.root.join("reports")
    }

    /// The evidence file.
    pub fn results_json(&self) -> PathBuf {
        self.root.join("results.json")
    }

    /// The cumulative-cost table.
    pub fn reuse_csv(&self) -> PathBuf {
        self.root.join("reuse.csv")
    }

    /// The branch montage produced from the restore path.
    pub fn montage(&self) -> PathBuf {
        self.root.join("branches.ppm")
    }

    /// The branch montage produced from the from-scratch replay path.
    pub fn montage_baseline(&self) -> PathBuf {
        self.root.join("branches_baseline.ppm")
    }

    /// A scratch subdirectory for the timing producers.
    pub fn timing_dir(&self, name: &str) -> PathBuf {
        self.root.join("timing").join(name)
    }
}

// ---------------------------------------------------------------------------
// Child argument contracts
//
// Both ends of the parent/child boundary are built and parsed from the same
// struct, so the two can never drift apart.
// ---------------------------------------------------------------------------

/// Which state-recovery mechanism a request child uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Fresh process, replay the whole context from scratch.
    Baseline,
    /// Fresh process, read the exact `(H, C)` bytes from an ordinary file.
    Raw,
    /// Fresh process, restore the state blob from EntropyFS.
    Vole,
}

impl Mode {
    /// CLI value.
    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Baseline => "baseline",
            Mode::Raw => "raw",
            Mode::Vole => "vole",
        }
    }

    /// Parse a CLI value.
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "baseline" => Some(Mode::Baseline),
            "raw" => Some(Mode::Raw),
            "vole" => Some(Mode::Vole),
            _ => None,
        }
    }
}

/// Arguments of a `producer` child.
#[derive(Debug, Clone)]
pub struct ProducerArgs {
    /// Run root.
    pub run_dir: PathBuf,
    /// Frozen checkpoint.
    pub checkpoint: PathBuf,
    /// Scene to watch.
    pub scene: SceneId,
    /// Frames consumed.
    pub context_len: usize,
    /// Skip the EntropyFS path entirely and write only the raw payload file.
    ///
    /// This exists so the attribution control's one-time cost is *measured*
    /// rather than derived: a raw-only producer is a real, comparable process,
    /// not the VOLE producer with numbers subtracted by hand.
    pub skip_entropyfs: bool,
    /// Where to write the child's report.
    pub report: Option<PathBuf>,
}

impl ProducerArgs {
    /// The exact argv tail for this child.
    pub fn argv(&self) -> Vec<String> {
        let mut a = vec![
            "producer".to_string(),
            "--run-dir".into(),
            self.run_dir.display().to_string(),
            "--checkpoint".into(),
            self.checkpoint.display().to_string(),
            "--scene".into(),
            self.scene.as_str().to_string(),
            "--context".into(),
            self.context_len.to_string(),
        ];
        if self.skip_entropyfs {
            a.push("--skip-entropyfs".into());
        }
        if let Some(r) = &self.report {
            a.push("--report".into());
            a.push(r.display().to_string());
        }
        a
    }

    /// Parse the argv tail. Errors name the offending flag.
    pub fn parse(args: &[String]) -> std::result::Result<ProducerArgs, String> {
        let scene_s = req_str(args, "--scene")?;
        let scene =
            SceneId::parse(scene_s).ok_or_else(|| format!("unknown --scene {scene_s:?}"))?;
        Ok(ProducerArgs {
            run_dir: req_path(args, "--run-dir")?,
            checkpoint: req_path(args, "--checkpoint")?,
            scene,
            context_len: req_usize(args, "--context")?,
            skip_entropyfs: args.iter().any(|a| a == "--skip-entropyfs"),
            report: opt_path(args, "--report"),
        })
    }
}

/// Arguments of a `request` child.
#[derive(Debug, Clone)]
pub struct RequestArgs {
    /// Run root.
    pub run_dir: PathBuf,
    /// Frozen checkpoint.
    pub checkpoint: PathBuf,
    /// Recovery mechanism.
    pub mode: Mode,
    /// Scene the request belongs to.
    pub scene: SceneId,
    /// Scene the request *expects* the offered state to belong to. Differs from
    /// `scene` only in the negative case.
    pub expect_scene: SceneId,
    /// Frames the request believes precede the future.
    pub context_len: usize,
    /// Frames to generate.
    pub future_len: usize,
    /// Which related request.
    pub request: Request,
    /// Measured repetitions.
    pub reps: usize,
    /// Whether to run one discarded warm-up repetition first.
    pub warmup: bool,
    /// Where to write the generated frames.
    pub emit: Option<PathBuf>,
    /// EntropyFS blob id, on the VOLE path.
    pub vole_blob: Option<String>,
    /// Raw payload file, on the raw path.
    pub raw_file: Option<PathBuf>,
    /// Where to write the child's report.
    pub report: Option<PathBuf>,
}

impl RequestArgs {
    /// The exact argv tail for this child.
    pub fn argv(&self) -> Vec<String> {
        let mut a = vec![
            "request".to_string(),
            "--run-dir".into(),
            self.run_dir.display().to_string(),
            "--checkpoint".into(),
            self.checkpoint.display().to_string(),
            "--mode".into(),
            self.mode.as_str().to_string(),
            "--scene".into(),
            self.scene.as_str().to_string(),
            "--expect-scene".into(),
            self.expect_scene.as_str().to_string(),
            "--context".into(),
            self.context_len.to_string(),
            "--future".into(),
            self.future_len.to_string(),
            "--request".into(),
            self.request.name().to_string(),
            "--reps".into(),
            self.reps.to_string(),
        ];
        if self.warmup {
            a.push("--warmup".into());
        }
        if let Some(p) = &self.emit {
            a.push("--emit".into());
            a.push(p.display().to_string());
        }
        if let Some(b) = &self.vole_blob {
            a.push("--vole-blob".into());
            a.push(b.clone());
        }
        if let Some(f) = &self.raw_file {
            a.push("--raw-file".into());
            a.push(f.display().to_string());
        }
        if let Some(r) = &self.report {
            a.push("--report".into());
            a.push(r.display().to_string());
        }
        a
    }

    /// Parse the argv tail. Errors name the offending flag.
    pub fn parse(args: &[String]) -> std::result::Result<RequestArgs, String> {
        let mode_s = req_str(args, "--mode")?;
        let mode = Mode::parse(mode_s).ok_or_else(|| format!("unknown --mode {mode_s:?}"))?;
        let scene_s = req_str(args, "--scene")?;
        let scene =
            SceneId::parse(scene_s).ok_or_else(|| format!("unknown --scene {scene_s:?}"))?;
        let expect_scene = match opt_str(args, "--expect-scene") {
            Some(s) => SceneId::parse(s).ok_or_else(|| format!("unknown --expect-scene {s:?}"))?,
            None => scene,
        };
        let request_s = req_str(args, "--request")?;
        let request =
            Request::parse(request_s).ok_or_else(|| format!("unknown --request {request_s:?}"))?;
        Ok(RequestArgs {
            run_dir: req_path(args, "--run-dir")?,
            checkpoint: req_path(args, "--checkpoint")?,
            mode,
            scene,
            expect_scene,
            context_len: req_usize(args, "--context")?,
            future_len: req_usize(args, "--future")?,
            request,
            reps: opt_usize(args, "--reps").unwrap_or(1).max(1),
            warmup: args.iter().any(|a| a == "--warmup"),
            emit: opt_path(args, "--emit"),
            vole_blob: opt_str(args, "--vole-blob").map(|s| s.to_string()),
            raw_file: opt_path(args, "--raw-file"),
            report: opt_path(args, "--report"),
        })
    }
}

fn opt_str<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn req_str<'a>(args: &'a [String], name: &str) -> std::result::Result<&'a str, String> {
    opt_str(args, name).ok_or_else(|| format!("missing {name}"))
}

fn opt_usize(args: &[String], name: &str) -> Option<usize> {
    opt_str(args, name).and_then(|v| v.parse().ok())
}

fn req_usize(args: &[String], name: &str) -> std::result::Result<usize, String> {
    opt_usize(args, name).ok_or_else(|| format!("missing or malformed {name}"))
}

fn opt_path(args: &[String], name: &str) -> Option<PathBuf> {
    opt_str(args, name).map(PathBuf::from)
}

fn req_path(args: &[String], name: &str) -> std::result::Result<PathBuf, String> {
    opt_path(args, name).ok_or_else(|| format!("missing {name}"))
}

// ---------------------------------------------------------------------------
// Frame files (the visual-evidence carrier)
// ---------------------------------------------------------------------------

/// `magic || frame_count || height || width || f32 LE payload`.
///
/// The hash the experiment compares is over the *payload only*, so a header
/// difference can never be mistaken for a frame difference.
pub fn encode_frames(frames: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(20 + frames.len() * HW * 4);
    out.extend_from_slice(&FRAME_MAGIC);
    out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    out.extend_from_slice(&(H as u32).to_le_bytes());
    out.extend_from_slice(&(W as u32).to_le_bytes());
    for f in frames {
        debug_assert_eq!(f.len(), HW);
        for x in f {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

/// Inverse of [`encode_frames`]. Validates magic, dimensions and length.
pub fn decode_frames(bytes: &[u8]) -> std::result::Result<Vec<Vec<f32>>, String> {
    if bytes.len() < 20 {
        return Err(format!("frame file is {} bytes, need >= 20", bytes.len()));
    }
    if bytes[0..8] != FRAME_MAGIC {
        return Err("frame file magic mismatch".into());
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let hh = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let ww = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    if hh != H || ww != W {
        return Err(format!("frame file is {hh}x{ww}, expected {H}x{W}"));
    }
    let payload = &bytes[20..];
    if payload.len() != count * HW * 4 {
        return Err(format!(
            "frame file payload is {} bytes, expected {}",
            payload.len(),
            count * HW * 4
        ));
    }
    Ok(payload
        .chunks_exact(HW * 4)
        .map(|c| {
            c.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect())
}

/// The bytes that identify a frame sequence: the payload, without the header.
pub fn frame_payload(bytes: &[u8]) -> &[u8] {
    if bytes.len() < 20 {
        &[]
    } else {
        &bytes[20..]
    }
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

/// A loaded frozen model plus the scene/context/request geometry it is used with.
pub struct Generator {
    /// The frozen model.
    pub model: ConvLstm,
    /// Architecture.
    pub cfg: ModelConfig,
    /// Behaviour-determining hash of the loaded weights.
    pub weights_hash: [u8; 32],
    /// Architecture hash.
    pub config_hash: [u8; 32],
    /// BLAKE3 of the checkpoint file as shipped.
    pub checkpoint_file_hash: String,
    /// Checkpoint file size in bytes.
    pub checkpoint_bytes: u64,
    /// Scene identity this generator is watching.
    pub scene: SceneId,
    /// Frames the producer consumes.
    pub context_len: usize,
    /// Frames a request generates.
    pub future_len: usize,
}

/// The observed history: frames plus the controls held during them.
pub struct Context {
    /// Exactly `context_len` frames.
    pub frames: Vec<Vec<f32>>,
    /// `at(k)` is the control held during the transition out of frame `k`.
    pub schedule: Schedule,
}

impl Generator {
    /// Load the frozen checkpoint and derive its identity hashes.
    pub fn load(
        checkpoint: &Path,
        scene: SceneId,
        context_len: usize,
        future_len: usize,
        cfg: ModelConfig,
    ) -> Result<Generator> {
        let bytes = std::fs::read(checkpoint)
            .map_err(|e| candle_core::Error::Msg(format!("read {checkpoint:?}: {e}")))?;
        let model = ConvLstm::load_safetensors(checkpoint, cfg)?;
        Ok(Generator {
            weights_hash: weights_hash(&model)?,
            config_hash: config_hash(&cfg),
            checkpoint_file_hash: hash_hex(&bytes),
            checkpoint_bytes: bytes.len() as u64,
            model,
            cfg,
            scene,
            context_len,
            future_len,
        })
    }

    /// The deterministic context for this generator's scene.
    ///
    /// Recomputed on every call. That is deliberate: the baseline's claim is that
    /// it rebuilds the state "by replaying the original history", and rebuilding
    /// the history is part of that. The cost of rendering is reported separately
    /// from the model arithmetic so neither can hide inside the other.
    pub fn context(&self) -> Context {
        let spec = SceneSpec::canonical(self.scene);
        let schedule = scene::context_program();
        let frames = scene::rollout(&spec, self.context_len - 1, |t| schedule.at(t));
        debug_assert_eq!(frames.len(), self.context_len);
        Context { frames, schedule }
    }

    /// Consume the context and return the earned state, counting the arithmetic.
    pub fn earn(&self, ctx: &Context, meter: &mut WorkMeter) -> Result<State> {
        let dev = Device::Cpu;
        let mut s = State::zeros(&self.cfg, 1, &dev)?;
        for k in 0..self.context_len {
            let x = frame_tensor(&self.cfg, &ctx.frames[k])?;
            let u = control_tensor(&self.cfg, ctx.schedule.at(k))?;
            s = self.model.cell(&x, &u, &s)?;
            meter.add_cell_step();
        }
        Ok(s)
    }

    /// Generate one branch future from a state, counting the arithmetic.
    ///
    /// The loop is `emit, then advance`: the recovered state emits the first frame
    /// directly, and the last emitted frame needs no successor, so `T` frames cost
    /// `T` head applications and `T - 1` cell steps. No step is performed that
    /// nothing consumes, in either path.
    pub fn generate(
        &self,
        state: &State,
        request: Request,
        meter: &mut WorkMeter,
    ) -> Result<Vec<Vec<f32>>> {
        let sched = request.schedule();
        let mut s = state.copy();
        let mut out = Vec::with_capacity(self.future_len);
        for j in 0..self.future_len {
            let pred = self.model.head(&s)?;
            meter.add_head_application();
            out.push(pred.flatten_all()?.to_vec1::<f32>()?);
            if j + 1 < self.future_len {
                let u = control_tensor(&self.cfg, sched.at(j))?;
                s = self.model.cell(&pred, &u, &s)?;
                meter.add_cell_step();
            }
        }
        Ok(out)
    }

    /// The model's declared per-step arithmetic.
    pub fn step_work(&self) -> StepWork {
        self.cfg.step_work()
    }

    /// The expectation a request places on an offered state record.
    pub fn expectation(&self, expect_scene: SceneId) -> Expectation {
        Expectation {
            model_weights_hash: self.weights_hash,
            config_hash: self.config_hash,
            scene: expect_scene,
            context_len: self.context_len as u32,
            cfg: self.cfg,
        }
    }
}

fn frame_tensor(cfg: &ModelConfig, f: &[f32]) -> Result<Tensor> {
    Tensor::from_vec(f.to_vec(), (1, 1, cfg.height, cfg.width), &Device::Cpu)
}

fn control_tensor(cfg: &ModelConfig, u: Control) -> Result<Tensor> {
    let [t, a] = u.planes();
    let mut v = Vec::with_capacity(2 * HW);
    v.extend(std::iter::repeat_n(t, HW));
    v.extend(std::iter::repeat_n(a, HW));
    Tensor::from_vec(v, (1, 2, cfg.height, cfg.width), &Device::Cpu)
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

/// The producer's account of one earning and persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerReport {
    /// Process id of the producer.
    pub pid: u32,
    /// Scene watched.
    pub scene: String,
    /// Frames consumed.
    pub context_len: usize,
    /// Behaviour-determining weights hash in use.
    pub model_weights_hash: String,
    /// Architecture hash in use.
    pub model_config_hash: String,
    /// Shipped checkpoint file hash.
    pub checkpoint_file_hash: String,
    /// Checkpoint file size.
    pub checkpoint_bytes: u64,
    /// Hash of the earned `(H, C)`.
    pub state_hash: String,
    /// Whether every element of the earned state is finite.
    pub state_finite: bool,
    /// `H` payload bytes.
    pub h_bytes: usize,
    /// `C` payload bytes.
    pub c_bytes: usize,
    /// Carry payload bytes.
    pub carry_bytes: usize,
    /// `h_bytes + c_bytes + carry_bytes`: the raw recurrent state size.
    pub raw_state_bytes: usize,
    /// Serialized VOLE record size (header plus payload).
    pub vole_record_bytes: usize,
    /// Whether the EntropyFS path was exercised at all.
    pub entropyfs_used: bool,
    /// Whether the store had to be created for this producer.
    pub store_created: bool,
    /// Store directory bytes for a freshly created (empty) store: the mkfs floor.
    pub store_mkfs_bytes: u64,
    /// Store directory bytes immediately before the put.
    pub store_bytes_before: u64,
    /// Store directory bytes after the put and the durability barrier.
    pub store_bytes_after: u64,
    /// `store_bytes_after - store_bytes_before`: this blob's physical footprint.
    pub store_physical_delta: u64,
    /// The engine's own physical-used figure after the put, as a cross-check.
    pub engine_physical_used_after: u64,
    /// EntropyFS blob id (64 hex characters), empty if EntropyFS was skipped.
    pub blob_id: String,
    /// Whether the blob was already present before this put (dedup hit).
    pub blob_already_present: bool,
    /// Path of the unframed payload written for the attribution control.
    pub raw_file: String,
    /// Raw payload bytes written.
    pub raw_file_bytes: u64,
    /// Phase timings, in execution order.
    pub timings: Vec<PhaseTiming>,
    /// Model arithmetic executed by the producer.
    pub work: WorkTotals,
    /// Failure, if the child failed.
    pub error: Option<String>,
}

/// How a request child obtained (or refused) its starting state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RestoreReport {
    /// Whether a persisted state was offered to this request at all.
    pub offered: bool,
    /// The scene identity the offered record actually carried, as decoded from the
    /// record itself rather than from the caller's request.
    pub offered_scene: Option<String>,
    /// The state hash the offered record carried.
    pub offered_state_hash: Option<String>,
    /// Whether the offered bytes carried an identity header that could be checked.
    ///
    /// True on the VOLE path, false on the raw path: the raw control is an
    /// unframed byte blob with no compatibility predicate at all, which is one of
    /// the things the VOLE record adds on top of bare tensors.
    pub identity_available: bool,
    /// Whether the offered state was accepted for reuse.
    pub accepted: bool,
    /// `entropyfs`, `raw-file` or `replay-from-scratch`.
    pub mechanism: String,
    /// Every compatibility check that was evaluated, passed or not.
    pub checks: Vec<Check>,
    /// Name of the first failing check, if reuse was rejected.
    pub rejection: Option<String>,
    /// What the request did instead, if reuse was rejected.
    pub fallback: Option<String>,
}

/// A request child's account of one recovery and generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestReport {
    /// Process id of this request.
    pub pid: u32,
    /// `baseline`, `raw` or `vole`.
    pub mode: String,
    /// Scene the request belongs to.
    pub scene: String,
    /// Scene the request expected the offered state to belong to.
    pub expect_scene: String,
    /// Request name.
    pub request: String,
    /// Request human label.
    pub request_label: String,
    /// Request control codes, one per step.
    pub request_controls: Vec<u8>,
    /// Frames the request believes precede the future.
    pub context_len: usize,
    /// Frames generated.
    pub future_len: usize,
    /// Measured repetitions.
    pub reps: usize,
    /// Whether a warm-up repetition ran.
    pub warmup: bool,
    /// Weights hash in use.
    pub model_weights_hash: String,
    /// Shipped checkpoint file hash.
    pub checkpoint_file_hash: String,
    /// How the starting state was obtained.
    pub restore: RestoreReport,
    /// Hash of the state this request actually generated from.
    pub start_state_hash: Option<String>,
    /// Hash of the state the producer earned, when one was offered.
    pub offered_state_hash: Option<String>,
    /// Whether re-serialising the recovered state reproduces the offered bytes
    /// exactly. `None` when no bytes were offered.
    pub restore_exact: Option<bool>,
    /// Whether the starting state was finite.
    pub start_state_finite: bool,
    /// BLAKE3 of the generated frame payload.
    pub output_hash: String,
    /// Frames generated.
    pub output_frames: usize,
    /// Payload bytes of the generated frames.
    pub output_bytes: usize,
    /// Where the frames were written.
    pub emitted: Option<String>,
    /// Phase timings, in execution order.
    pub timings: Vec<PhaseTiming>,
    /// Arithmetic executed while recovering the state by replay.
    pub work_context: WorkTotals,
    /// Arithmetic executed while generating.
    pub work_generate: WorkTotals,
    /// `work_context + work_generate`, for one request.
    pub work_total: WorkTotals,
    /// Failure, if the child failed.
    pub error: Option<String>,
}

fn write_report<T: Serialize>(path: &Option<PathBuf>, report: &T) {
    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(report) {
            let _ = std::fs::write(p, json);
        }
    }
}

/// The single line a child prints for its parent to parse.
pub fn emit_child_report<T: Serialize>(report: &T) -> std::result::Result<(), String> {
    let json = serde_json::to_string(report).map_err(|e| e.to_string())?;
    println!("{CHILD_MARKER}{json}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Child: producer
// ---------------------------------------------------------------------------

/// Earn a state, persist it through EntropyFS, write the raw payload, exit.
pub fn run_producer(args: &ProducerArgs) -> std::result::Result<ProducerReport, String> {
    let paths = Paths::new(&args.run_dir);
    standard_dir(&paths.root)?;
    standard_dir(&paths.futures())?;
    standard_dir(&paths.reports())?;
    let mut timings = Vec::new();
    let mut work = WorkMeter::default();

    // --- load -----------------------------------------------------------------
    let t = Instant::now();
    let gen = Generator::load(
        &args.checkpoint,
        args.scene,
        args.context_len,
        DEFAULT_FUTURE,
        ModelConfig::default(),
    )
    .map_err(|e| format!("load checkpoint: {e}"))?;
    timings.push(PhaseTiming::new(
        "model_load",
        "read the frozen safetensors checkpoint and derive its identity hashes",
        None,
        vec![ms(t.elapsed())],
    ));

    // --- store, if this producer uses the EntropyFS path ----------------------
    let mut store = None;
    let mut created = false;
    let mut mkfs_bytes = 0u64;
    if !args.skip_entropyfs {
        let t = Instant::now();
        let (s, was_created) = VoleStore::open_or_create(&paths.store())
            .map_err(|e| format!("open/create entropyfs store: {e}"))?;
        let mkfs_ms = ms(t.elapsed());
        mkfs_bytes = if was_created {
            // Measure the mkfs floor on a throwaway sibling so the reported number
            // is "the store before any blob", not "the store as it happens to be".
            let probe = paths.root.join("mkfs-probe");
            let _ = std::fs::remove_dir_all(&probe);
            let p = VoleStore::create(&probe).map_err(|e| format!("mkfs probe: {e}"))?;
            let b = p.physical_bytes().map_err(|e| format!("{e}"))?;
            p.close().map_err(|e| format!("{e}"))?;
            let _ = std::fs::remove_dir_all(&probe);
            b
        } else {
            0
        };
        timings.push(PhaseTiming::new(
            "store_open_or_create",
            "open the EntropyFS store, or mkfs it when absent",
            None,
            vec![mkfs_ms],
        ));
        store = Some(s);
        created = was_created;
    }

    // --- regenerate the scene -------------------------------------------------
    let t = Instant::now();
    let ctx = gen.context();
    timings.push(PhaseTiming::new(
        "scene_generate",
        "deterministically render the context frames (no model arithmetic)",
        None,
        vec![ms(t.elapsed())],
    ));

    // --- earn -----------------------------------------------------------------
    let t = Instant::now();
    let state = gen
        .earn(&ctx, &mut work)
        .map_err(|e| format!("earn state: {e}"))?;
    timings.push(PhaseTiming::new(
        "model_only_replay",
        "consume the context through the ConvLSTM: model arithmetic only",
        None,
        vec![ms(t.elapsed())],
    ));

    // --- exact bytes ----------------------------------------------------------
    let t = Instant::now();
    let h = state.h_bytes().map_err(|e| format!("H bytes: {e}"))?;
    let c = state.c_bytes().map_err(|e| format!("C bytes: {e}"))?;
    let carry = state
        .carry_bytes()
        .map_err(|e| format!("carry bytes: {e}"))?;
    let finite = state.is_finite().map_err(|e| format!("finiteness: {e}"))?;
    let record = StateRecord::new(
        gen.weights_hash,
        gen.config_hash,
        args.scene,
        args.context_len as u32,
        &gen.cfg,
        h,
        c,
        carry,
    );
    let encoded = record.encode();
    timings.push(PhaseTiming::new(
        "state_serialize",
        "encode the VOLE state record: 176-byte header plus exact f32 payload",
        None,
        vec![ms(t.elapsed())],
    ));

    // --- persist ---------------------------------------------------------------
    let mut store_bytes_before = 0u64;
    let mut store_bytes_after = 0u64;
    let mut engine_physical_used_after = 0u64;
    let mut blob_id = String::new();
    let mut already = false;

    if let Some(store) = &store {
        store_bytes_before = store.physical_bytes().map_err(|e| format!("{e}"))?;
        already = {
            let id = VoleStore::parse_id(&hash_hex(&encoded)).map_err(|e| format!("{e}"))?;
            let t = Instant::now();
            let present = store.contains(id).map_err(|e| format!("{e}"))?;
            timings.push(PhaseTiming::new(
                "entropyfs_contains",
                "namespace lookup of the blob id before the put",
                None,
                vec![ms(t.elapsed())],
            ));
            present
        };
        let t = Instant::now();
        let id = store
            .put(&encoded)
            .map_err(|e| format!("entropyfs put_blob: {e}"))?;
        let put_ms = ms(t.elapsed());
        let t = Instant::now();
        store
            .sync()
            .map_err(|e| format!("entropyfs durability barrier: {e}"))?;
        let sync_ms = ms(t.elapsed());
        timings.push(PhaseTiming::new(
            "state_persist_entropyfs",
            "put_blob then sync(): mutation-log append and the power-durability barrier",
            None,
            vec![put_ms + sync_ms],
        ));
        store_bytes_after = store.physical_bytes().map_err(|e| format!("{e}"))?;
        engine_physical_used_after = store
            .engine_physical_used()
            .map_err(|e| format!("engine metrics: {e}"))?;
        blob_id = id.to_string();
    }

    // --- the raw attribution control ------------------------------------------
    let raw_payload = record.raw_payload();
    let raw_file = paths.raw_state();
    let t = Instant::now();
    if let Some(parent) = raw_file.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {parent:?}: {e}"))?;
    }
    std::fs::write(&raw_file, &raw_payload).map_err(|e| format!("write raw state: {e}"))?;
    let raw_write_ms = ms(t.elapsed());
    timings.push(PhaseTiming::new(
        "state_persist_raw_file",
        "write the same payload bytes to an ordinary file (the attribution control)",
        None,
        vec![raw_write_ms],
    ));

    if let Some(store) = store {
        store.close().map_err(|e| format!("close store: {e}"))?;
    }

    let report = ProducerReport {
        pid: std::process::id(),
        scene: args.scene.as_str().to_string(),
        context_len: args.context_len,
        model_weights_hash: hex32(&gen.weights_hash),
        model_config_hash: hex32(&gen.config_hash),
        checkpoint_file_hash: gen.checkpoint_file_hash.clone(),
        checkpoint_bytes: gen.checkpoint_bytes,
        state_hash: hex32(&record.state_hash),
        state_finite: finite,
        h_bytes: record.h.len(),
        c_bytes: record.c.len(),
        carry_bytes: record.carry.len(),
        raw_state_bytes: record.h.len() + record.c.len() + record.carry.len(),
        vole_record_bytes: encoded.len(),
        entropyfs_used: !args.skip_entropyfs,
        store_created: created,
        store_mkfs_bytes: mkfs_bytes,
        store_bytes_before,
        store_bytes_after,
        store_physical_delta: store_bytes_after.saturating_sub(store_bytes_before),
        engine_physical_used_after,
        blob_id,
        blob_already_present: already,
        raw_file: raw_file.display().to_string(),
        raw_file_bytes: raw_payload.len() as u64,
        timings,
        work: work.totals(&gen.step_work()),
        error: None,
    };
    write_report(&args.report, &report);
    Ok(report)
}

fn standard_dir(p: &Path) -> std::result::Result<(), String> {
    std::fs::create_dir_all(p).map_err(|e| format!("create {p:?}: {e}"))
}

/// Record one phase sample, either as the discarded warm-up or as a measurement.
fn push_sample(
    is_warmup: bool,
    warmups: &mut BTreeMap<&'static str, f64>,
    samples: &mut Vec<f64>,
    name: &'static str,
    v: f64,
) {
    if is_warmup {
        warmups.insert(name, v);
    } else {
        samples.push(v);
    }
}

// ---------------------------------------------------------------------------
// Child: request
// ---------------------------------------------------------------------------

/// Recover a state (or refuse to), generate one related future, exit.
pub fn run_request(args: &RequestArgs) -> std::result::Result<RequestReport, String> {
    let paths = Paths::new(&args.run_dir);

    let t = Instant::now();
    let gen = Generator::load(
        &args.checkpoint,
        args.scene,
        args.context_len,
        args.future_len,
        ModelConfig::default(),
    )
    .map_err(|e| format!("load checkpoint: {e}"))?;
    let model_load_ms = ms(t.elapsed());

    let expectation = gen.expectation(args.expect_scene);
    let step_work = gen.step_work();

    let mut scene_samples = Vec::new();
    let mut replay_samples = Vec::new();
    let mut restore_samples = Vec::new();
    let mut generate_samples = Vec::new();
    let mut total_samples = Vec::new();
    let mut warmups: BTreeMap<&str, f64> = BTreeMap::new();

    let mut restore = RestoreReport::default();
    let mut offered_state_hash: Option<String> = None;
    let mut start_state_hash: Option<String> = None;
    let mut restore_exact: Option<bool> = None;
    let mut start_finite = true;
    let mut frames: Vec<Vec<f32>> = Vec::new();
    let mut m_ctx = WorkMeter::default();
    let mut m_gen = WorkMeter::default();

    let reps = if args.warmup {
        args.reps + 1
    } else {
        args.reps
    };
    for rep in 0..reps {
        let is_warmup = args.warmup && rep == 0;
        let rep_start = Instant::now();
        // Fresh meters each repetition: the reported work is ONE request's work,
        // not `reps` requests' work.
        m_ctx = WorkMeter::default();
        m_gen = WorkMeter::default();

        // --- offer a persisted state, if this mode has one -------------------
        let offered: Option<StateRecord> = match args.mode {
            Mode::Baseline => None,
            Mode::Raw => {
                let path = args
                    .raw_file
                    .clone()
                    .ok_or_else(|| "--raw-file is required for --mode raw".to_string())?;
                let t = Instant::now();
                let raw = std::fs::read(&path).map_err(|e| format!("read {path:?}: {e}"))?;
                let rec = StateRecord::from_raw_payload(
                    &raw,
                    &gen.cfg,
                    gen.weights_hash,
                    gen.config_hash,
                    args.expect_scene,
                    args.context_len as u32,
                )
                .map_err(|e| format!("raw payload: {e}"))?;
                push_sample(
                    is_warmup,
                    &mut warmups,
                    &mut restore_samples,
                    "state_restore",
                    ms(t.elapsed()),
                );
                Some(rec)
            }
            Mode::Vole => {
                let blob = args
                    .vole_blob
                    .clone()
                    .ok_or_else(|| "--vole-blob is required for --mode vole".to_string())?;
                let id = VoleStore::parse_id(&blob).map_err(|e| format!("{e}"))?;
                let t = Instant::now();
                let store = VoleStore::open(&paths.store())
                    .map_err(|e| format!("open entropyfs store: {e}"))?;
                let bytes = store
                    .get(id)
                    .map_err(|e| format!("entropyfs get_blob: {e}"))?;
                store.close().map_err(|e| format!("close store: {e}"))?;
                let rec = StateRecord::decode(&bytes).map_err(|e| format!("decode: {e}"))?;
                push_sample(
                    is_warmup,
                    &mut warmups,
                    &mut restore_samples,
                    "state_restore",
                    ms(t.elapsed()),
                );
                Some(rec)
            }
        };

        // --- compatibility gate, then recovery -------------------------------
        let start: State = match &offered {
            None => {
                restore.offered = false;
                restore.mechanism = "replay-from-scratch".into();
                let t = Instant::now();
                let ctx = gen.context();
                push_sample(
                    is_warmup,
                    &mut warmups,
                    &mut scene_samples,
                    "scene_generate",
                    ms(t.elapsed()),
                );
                let t = Instant::now();
                let s = gen
                    .earn(&ctx, &mut m_ctx)
                    .map_err(|e| format!("replay context: {e}"))?;
                push_sample(
                    is_warmup,
                    &mut warmups,
                    &mut replay_samples,
                    "model_only_replay",
                    ms(t.elapsed()),
                );
                s
            }
            Some(rec) => {
                let checks = rec.compatibility(&expectation);
                let rejection = rec.first_rejection(&checks).map(|c| c.name.clone());
                offered_state_hash = Some(hex32(&rec.state_hash));
                restore.offered = true;
                restore.offered_scene = Some(rec.scene.as_str().to_string());
                restore.offered_state_hash = Some(hex32(&rec.state_hash));
                restore.identity_available = args.mode == Mode::Vole;
                restore.checks = checks;

                if let Some(name) = rejection {
                    // Fail closed: refuse the state and pay for a full replay. This
                    // is the negative case, and it is not a branch that exists only
                    // for the demo — it is the branch a wrong-key lookup really
                    // takes.
                    restore.accepted = false;
                    restore.rejection = Some(name);
                    restore.mechanism = "replay-from-scratch".into();
                    restore.fallback = Some("replay-from-scratch".into());
                    let t = Instant::now();
                    let ctx = gen.context();
                    push_sample(
                        is_warmup,
                        &mut warmups,
                        &mut scene_samples,
                        "scene_generate",
                        ms(t.elapsed()),
                    );
                    let t = Instant::now();
                    let s = gen
                        .earn(&ctx, &mut m_ctx)
                        .map_err(|e| format!("fallback replay: {e}"))?;
                    push_sample(
                        is_warmup,
                        &mut warmups,
                        &mut replay_samples,
                        "model_only_replay",
                        ms(t.elapsed()),
                    );
                    s
                } else {
                    restore.accepted = true;
                    restore.mechanism = if args.mode == Mode::Vole {
                        "entropyfs".into()
                    } else {
                        "raw-file".into()
                    };
                    let rebuilt = State::from_bytes(&gen.cfg, 1, &rec.h, &rec.c, &rec.carry)
                        .map_err(|e| format!("rebuild state: {e}"))?;
                    // Exactness is checked by re-serialising the rebuilt state and
                    // comparing bytes with what was offered. Byte comparison, not a
                    // tolerance: that is the whole point of the proof.
                    let rh = rebuilt.h_bytes().map_err(|e| format!("{e}"))?;
                    let rc = rebuilt.c_bytes().map_err(|e| format!("{e}"))?;
                    let rk = rebuilt.carry_bytes().map_err(|e| format!("{e}"))?;
                    restore_exact = Some(rh == rec.h && rc == rec.c && rk == rec.carry);
                    rebuilt
                }
            }
        };

        start_finite = start.is_finite().map_err(|e| format!("{e}"))?;
        let sh = start.h_bytes().map_err(|e| format!("{e}"))?;
        let sc = start.c_bytes().map_err(|e| format!("{e}"))?;
        let sk = start.carry_bytes().map_err(|e| format!("{e}"))?;
        start_state_hash = Some(hex32(&state_hash(&sh, &sc, &sk)));

        // --- generate --------------------------------------------------------
        let t = Instant::now();
        let out = gen
            .generate(&start, args.request, &mut m_gen)
            .map_err(|e| format!("generate: {e}"))?;
        push_sample(
            is_warmup,
            &mut warmups,
            &mut generate_samples,
            "model_only_generate",
            ms(t.elapsed()),
        );
        push_sample(
            is_warmup,
            &mut warmups,
            &mut total_samples,
            "__total",
            ms(rep_start.elapsed()),
        );
        frames = out;
    }

    let timings = vec![
        PhaseTiming::new(
            "model_load",
            "read the frozen safetensors checkpoint and derive its identity hashes",
            None,
            vec![model_load_ms],
        ),
        PhaseTiming::new(
            "scene_generate",
            "deterministically render the context frames (no model arithmetic)",
            warmups.get("scene_generate").copied(),
            scene_samples,
        ),
        PhaseTiming::new(
            "model_only_replay",
            "consume the context through the ConvLSTM: model arithmetic only",
            warmups.get("model_only_replay").copied(),
            replay_samples,
        ),
        PhaseTiming::new(
            "state_restore",
            "obtain the starting state from persisted bytes (open, get, decode, rebuild)",
            warmups.get("state_restore").copied(),
            restore_samples,
        ),
        PhaseTiming::new(
            "model_only_generate",
            "emit the branch future: model arithmetic only",
            warmups.get("model_only_generate").copied(),
            generate_samples,
        ),
        PhaseTiming::new(
            "request_total",
            "one whole request inside this process: state recovery plus generation",
            warmups.get("__total").copied(),
            total_samples,
        ),
    ];

    let encoded = encode_frames(&frames);
    let payload = frame_payload(&encoded);
    let mut emitted = None;
    if let Some(p) = &args.emit {
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create {parent:?}: {e}"))?;
        }
        std::fs::write(p, &encoded).map_err(|e| format!("write {p:?}: {e}"))?;
        emitted = Some(p.display().to_string());
    }

    let report = RequestReport {
        pid: std::process::id(),
        mode: args.mode.as_str().to_string(),
        scene: args.scene.as_str().to_string(),
        expect_scene: args.expect_scene.as_str().to_string(),
        request: args.request.name().to_string(),
        request_label: args.request.label().to_string(),
        request_controls: args.request.schedule().codes(),
        context_len: args.context_len,
        future_len: args.future_len,
        reps: args.reps,
        warmup: args.warmup,
        model_weights_hash: hex32(&gen.weights_hash),
        checkpoint_file_hash: gen.checkpoint_file_hash.clone(),
        restore,
        start_state_hash,
        offered_state_hash,
        restore_exact,
        start_state_finite: start_finite,
        output_hash: hash_hex(payload),
        output_frames: frames.len(),
        output_bytes: payload.len(),
        emitted,
        timings,
        work_context: m_ctx.totals(&step_work),
        work_generate: m_gen.totals(&step_work),
        work_total: m_ctx.totals(&step_work) + m_gen.totals(&step_work),
        error: None,
    };
    write_report(&args.report, &report);
    Ok(report)
}

// ---------------------------------------------------------------------------
// Parent mechanics
// ---------------------------------------------------------------------------

/// The outcome of one child process.
#[derive(Debug, Clone)]
pub struct ChildOutcome {
    /// Wall-clock milliseconds measured by the parent around spawn and wait.
    pub wall_ms: f64,
    /// PID the child reported for itself.
    pub pid: u32,
    /// PID the operating system assigned, as seen by the parent.
    pub os_pid: u32,
    /// The child's parsed report.
    pub report: serde_json::Value,
    /// The child's stderr, kept so a failure is diagnosable.
    pub stderr: String,
}

/// Spawn one child of this same binary, with the experiment's pinned environment.
pub fn spawn_child(
    args: &[String],
    extra_env: &[(String, String)],
) -> std::result::Result<ChildOutcome, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let t = Instant::now();
    let mut cmd = Command::new(&exe);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let child = cmd.spawn().map_err(|e| format!("spawn {exe:?}: {e}"))?;
    let os_pid = child.id();
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait for child: {e}"))?;
    let wall_ms = ms(t.elapsed());
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    if !out.status.success() {
        return Err(format!(
            "child {args:?} exited with {}\nstderr:\n{stderr}",
            out.status
        ));
    }
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.starts_with(CHILD_MARKER))
        .ok_or_else(|| {
            format!("child {args:?} printed no report line\nstdout:\n{stdout}\nstderr:\n{stderr}")
        })?;
    let report: serde_json::Value = serde_json::from_str(&line[CHILD_MARKER.len()..])
        .map_err(|e| format!("child report is not JSON: {e}"))?;
    let pid = report.get("pid").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    Ok(ChildOutcome {
        wall_ms,
        pid,
        os_pid,
        report,
        stderr,
    })
}

/// Parse a child's report into a concrete type.
pub fn parse_report<T: for<'de> Deserialize<'de>>(
    v: &serde_json::Value,
) -> std::result::Result<T, String> {
    serde_json::from_value(v.clone()).map_err(|e| format!("report deserialisation: {e}"))
}

/// Read one phase's median out of a report's timing table.
pub fn phase_median(timings: &[PhaseTiming], phase: &str) -> Option<f64> {
    timings
        .iter()
        .find(|t| t.phase == phase)
        .and_then(|t| t.median_ms)
}

// ---------------------------------------------------------------------------
// Montage
// ---------------------------------------------------------------------------

/// The scene's own future for a request: the same trajectory the model is asked to
/// predict, rolled forward from the end of the observed context.
///
/// Used only to draw the ground-truth block of the montage. It is never handed to a
/// request child and never enters a measurement.
pub fn truth_future(
    scene: SceneId,
    context_len: usize,
    future_len: usize,
    request: Request,
) -> Vec<Vec<f32>> {
    let spec = SceneSpec::canonical(scene);
    let mut bodies = spec.working();
    let ctx = scene::context_program();
    for t in 0..context_len {
        for b in bodies.iter_mut() {
            b.advance(ctx.at(t));
        }
    }
    let sched = request.schedule();
    let mut out = Vec::with_capacity(future_len);
    for j in 0..future_len {
        for b in bodies.iter_mut() {
            b.advance(sched.at(j));
        }
        out.push(scene::render(&bodies));
    }
    out
}

/// Render the branch montage.
///
/// ```text
/// row 1          the observed context tail: the history the state was earned from
/// rows 2-5       the model's branches, generated after the producer had exited
/// (group gap)
/// rows 6-9       the same four branches, generated by the scene itself
/// ```
///
/// The ground-truth block is included deliberately. The model that ships here is small
/// and its response to the controls is sub-pixel (see `eval`); showing only its
/// branches would make the artifact look like the controls had no effect, when in fact
/// the scene's futures diverge strongly and the model reproduces only part of that.
/// Displaying both blocks puts the truth next to the model instead of leaving the
/// reader to guess what was supposed to happen. It is also not an illustration of the
/// result: the model block is the exact `f32` output that the gates compare.
///
/// The gamma is a display transform only. Every measurement and every comparison in
/// this crate uses the exact `f32` bytes.
pub fn build_montage(
    context_tail: &[Vec<f32>],
    model_branches: &[(Request, Vec<Vec<f32>>)],
    truth_branches: &[(Request, Vec<Vec<f32>>)],
) -> Vec<u8> {
    let rows = 1 + model_branches.len() + truth_branches.len();
    let cols = MONTAGE_COLS;
    let tile = H * MONTAGE_SCALE;
    let gap = MONTAGE_GAP;
    let margin = MONTAGE_MARGIN;
    let group_gap = MONTAGE_GROUP_GAP;

    // Row starts: one gap between rows, plus an extra group gap before the truth block.
    let group_at = 1 + model_branches.len();
    let mut row_y = Vec::with_capacity(rows);
    let mut y = gap;
    for row in 0..rows {
        if row == group_at {
            y += group_gap;
        }
        row_y.push(y);
        y += tile + gap;
    }
    let height = y;
    let grid_x0 = margin + gap;
    let grid_w = cols * tile + cols.saturating_sub(1) * gap;
    let width = grid_x0 + grid_w + gap;

    let mut img = vec![0u8; width * height * 3];

    // Separators: a flat mid-grey lattice bounding the grid and the tiles.
    for yy in 0..height {
        for xx in 0..width {
            let inside_x = xx >= grid_x0 && xx < grid_x0 + grid_w;
            let on_tile_x = inside_x && (xx - grid_x0) % (tile + gap) < tile;
            let on_tile_y = row_y.iter().any(|y0| yy >= *y0 && yy < *y0 + tile);
            if !(on_tile_x && on_tile_y) {
                let i = (yy * width + xx) * 3;
                img[i] = 70;
                img[i + 1] = 70;
                img[i + 2] = 70;
            }
        }
    }

    // Row markers: a distinct grey per row, so rows are countable without a font.
    for (row, y0) in row_y.iter().enumerate() {
        let grey = (24 + 26 * row).min(255) as u8;
        for yy in *y0..(y0 + tile).min(height) {
            for xx in 0..margin {
                let i = (yy * width + xx) * 3;
                img[i] = grey;
                img[i + 1] = grey;
                img[i + 2] = grey;
            }
        }
    }

    // Tiles.
    let rows_frames: Vec<Vec<Vec<f32>>> = std::iter::once(context_tail.to_vec())
        .chain(model_branches.iter().map(|(_, f)| f.clone()))
        .chain(truth_branches.iter().map(|(_, f)| f.clone()))
        .collect();
    for (row, frames) in rows_frames.iter().enumerate() {
        let picks = sample_indices(frames.len(), cols);
        let y0 = row_y[row];
        for (col, pick) in picks.iter().enumerate() {
            let x0 = grid_x0 + col * (tile + gap);
            let f = &frames[*pick];
            for ty in 0..tile {
                let sy = ty / MONTAGE_SCALE;
                for tx in 0..tile {
                    let sx = tx / MONTAGE_SCALE;
                    let v = f[sy * W + sx].clamp(0.0, 1.0).powf(MONTAGE_GAMMA);
                    let g = (v * 255.0).round() as u8;
                    let (px, py) = (x0 + tx, y0 + ty);
                    if px < width && py < height {
                        let i = (py * width + px) * 3;
                        img[i] = g;
                        img[i + 1] = g;
                        img[i + 2] = g;
                    }
                }
            }
        }
    }

    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(&img);
    out
}

// ---------------------------------------------------------------------------
// Evidence types
// ---------------------------------------------------------------------------

/// Host and environment facts that a reader needs to interpret the timings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// `std::env::consts::OS`.
    pub os: String,
    /// `std::env::consts::ARCH`.
    pub arch: String,
    /// Logical CPUs the process can see.
    pub available_parallelism: usize,
    /// `RAYON_NUM_THREADS` pinned for every child, or 0 if not pinned.
    pub pinned_threads: usize,
    /// Crate version.
    pub crate_version: String,
    /// `git rev-parse HEAD`, when the run happened inside a checkout.
    pub git_commit: Option<String>,
}

/// The frozen model's identity and declared arithmetic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Checkpoint path.
    pub checkpoint: String,
    /// Checkpoint bytes as shipped.
    pub checkpoint_bytes: u64,
    /// BLAKE3 of the checkpoint file.
    pub checkpoint_file_hash: String,
    /// Behaviour-determining parameters hash.
    pub weights_hash: String,
    /// Architecture hash.
    pub config_hash: String,
    /// Architecture.
    pub config: ModelConfig,
    /// Parameters.
    pub parameter_count: u64,
    /// Declared arithmetic per recurrent step and per emission.
    pub step_work: StepWork,
    /// Model weights are read by *both* paths; they are not a VOLE saving.
    pub weights_are_shared_by_both_paths: bool,
}

/// The scene and request geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneInfo {
    /// Canonical scene identity.
    pub id: String,
    /// Human label.
    pub label: String,
    /// Unrelated scene identity used only by the negative case.
    pub unrelated_id: String,
    /// Unrelated scene label.
    pub unrelated_label: String,
    /// Frame height.
    pub height: usize,
    /// Frame width.
    pub width: usize,
    /// Frames consumed to earn a state.
    pub context_len: usize,
    /// Frames generated per request.
    pub future_len: usize,
    /// The context control program, as one code per step, run-length collapsed.
    pub context_controls_rle: Vec<(u8, usize)>,
    /// Every related request, with its control codes.
    pub requests: Vec<RequestIdentity>,
}

/// One related request and how the three paths compared on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestIdentity {
    /// Request name.
    pub request: String,
    /// Human label.
    pub label: String,
    /// Control codes, one per generation step.
    pub controls: Vec<u8>,
    /// Baseline output hash.
    pub baseline_hash: String,
    /// Restore-path output hash.
    pub vole_hash: String,
    /// Raw-checkpoint output hash.
    pub raw_hash: String,
    /// Bytes compared, per path.
    pub compared_bytes: usize,
    /// `baseline_hash == vole_hash`.
    pub baseline_eq_vole: bool,
    /// `baseline_hash == raw_hash`.
    pub baseline_eq_raw: bool,
    /// Byte-for-byte equality of the emitted frame payloads.
    pub byte_identical: bool,
}

/// Byte accounting for the persisted state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    /// `H` payload bytes.
    pub h_bytes: usize,
    /// `C` payload bytes.
    pub c_bytes: usize,
    /// Carry payload bytes.
    pub carry_bytes: usize,
    /// `h_bytes + c_bytes + carry_bytes`: the raw recurrent state.
    pub raw_state_bytes: usize,
    /// Serialized VOLE record bytes (header plus payload).
    pub vole_record_bytes: usize,
    /// Record header bytes.
    pub vole_header_bytes: usize,
    /// Raw-checkpoint file bytes.
    pub raw_file_bytes: u64,
    /// BLAKE3 of the earned `(H, C)`.
    pub state_hash: String,
    /// BLAKE3 of the recovered `(H, C)`.
    pub restored_state_hash: Option<String>,
    /// Recovered bytes equal offered bytes.
    pub state_exact: bool,
    /// Every element of the state is finite.
    pub state_finite: bool,
    /// EntropyFS accounting.
    pub entropyfs: EntropyFsInfo,
    /// Model weight bytes, shared by both paths.
    pub model_weight_bytes: u64,
}

/// EntropyFS physical accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyFsInfo {
    /// Blob id.
    pub blob_id: String,
    /// Bytes of a freshly created (empty, mkfs'd) store: the fixed floor.
    pub store_mkfs_bytes: u64,
    /// Store bytes before the put.
    pub store_bytes_before: u64,
    /// Store bytes after the put and the durability barrier.
    pub store_bytes_after: u64,
    /// `store_bytes_after - store_bytes_before`.
    pub physical_delta_bytes: u64,
    /// The engine's own physical-used figure after the put.
    pub engine_physical_used_bytes: u64,
    /// Whether the blob was already present before the put.
    pub blob_already_present: bool,
    /// Bytes the record added relative to the raw payload: header plus store overhead.
    pub overhead_vs_raw_bytes: i64,
    /// Whether the store was measured from outside the engine.
    pub measured_from_outside_the_engine: bool,
}

/// Process-isolation facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// PID of the producer that earned and persisted the canonical state.
    pub producer_pid: u32,
    /// Every consumer PID, keyed by role tag. Long-running roles are overwritten
    /// by the most recent child, so this is a sample rather than a full census.
    pub consumer_pids: BTreeMap<String, u32>,
    /// How many child processes reported a PID that matched the one the operating
    /// system assigned. A mismatch aborts the run, so this is a count of verified
    /// processes, not a hopeful flag.
    pub children_pid_verified: usize,
    /// Producer PID differs from every recorded consumer PID.
    pub producer_differs_from_all_consumers: bool,
    /// `/proc/<producer pid>` was already gone when checked immediately after the
    /// producer was waited on. `None` off Linux.
    pub producer_proc_entry_gone: Option<bool>,
    /// The producer was waited on before any consumer was spawned.
    pub producer_exited_before_consumers: bool,
    /// The only bridge is persisted bytes.
    pub bridge: String,
    /// This is not a cold-disk-cache experiment.
    pub cold_disk_cache: bool,
    /// What the isolation claim is, in words.
    pub claim: String,
}

/// Model-work accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkInfo {
    /// Baseline per request.
    pub baseline_per_request: WorkTotals,
    /// VOLE per request.
    pub vole_per_request: WorkTotals,
    /// Raw-checkpoint per request.
    pub raw_per_request: WorkTotals,
    /// Baseline cell steps minus VOLE cell steps, per request.
    pub cell_steps_avoided_per_request: u64,
    /// Baseline MACs minus VOLE MACs, per request.
    pub macs_avoided_per_request: u64,
    /// `macs_avoided / baseline_macs`.
    pub macs_avoided_fraction: f64,
    /// The producer's own arithmetic.
    pub producer: WorkTotals,
}

/// One N point of the repeated-request sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReusePoint {
    /// Number of fresh related requests.
    pub n: usize,
    /// Cumulative wall time of `n` fresh baseline processes.
    pub c_baseline_ms: f64,
    /// Cumulative wall time of `n` fresh VOLE processes, plus the one-time
    /// producer that earned and persisted the state.
    pub c_vole_ms: f64,
    /// Cumulative wall time of `n` fresh raw-checkpoint processes, plus the
    /// one-time raw-only producer.
    pub c_raw_ms: f64,
    /// One-time producer cost, VOLE mechanism (median over fresh processes).
    pub vole_producer_ms: f64,
    /// One-time producer cost, raw mechanism (median over fresh processes).
    pub raw_producer_ms: f64,
    /// Request programs used, in order.
    pub requests: Vec<String>,
    /// Distinct request programs asked for at this N.
    pub distinct_request_programs: usize,
    /// Distinct output hashes seen among the baseline children at this N. Equal to
    /// [`ReusePoint::distinct_request_programs`] unless two programs collide.
    pub distinct_baseline_outputs: usize,
    /// Per-request wall times of the baseline children, in order.
    pub baseline_per_request_ms: Vec<f64>,
    /// Per-request wall times of the VOLE children, in order.
    pub vole_per_request_ms: Vec<f64>,
    /// Per-request wall times of the raw children, in order.
    pub raw_per_request_ms: Vec<f64>,
}

/// The repeated-request result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseInfo {
    /// Tested N values.
    pub n_points: Vec<usize>,
    /// The sweep.
    pub points: Vec<ReusePoint>,
    /// Smallest tested N with `C_vole(N) < C_baseline(N)`.
    pub n_star: Option<usize>,
    /// A sentence stating the N* result, including the honest no-crossover case.
    pub n_star_note: String,
    /// How many fresh processes each N point ran, per mode.
    pub fresh_processes_per_mode: usize,
    /// Whether every request really ran in its own fresh process.
    pub every_request_in_a_fresh_process: bool,
}

/// The negative case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeInfo {
    /// Scene the request wanted.
    pub requested_scene: String,
    /// Scene the offered record belonged to.
    pub offered_scene: String,
    /// Whether reuse was refused.
    pub rejected: bool,
    /// The failing check.
    pub rejection: String,
    /// Every check that was evaluated.
    pub checks: Vec<Check>,
    /// What the request did instead.
    pub fallback: String,
    /// Hash of the fallback output.
    pub fallback_output_hash: String,
    /// Hash of a genuine from-scratch baseline output for the same request.
    pub reference_baseline_hash: String,
    /// Fallback output equals the genuine baseline output.
    pub fallback_matches_baseline: bool,
    /// Cell steps the refused request still had to execute.
    pub cell_steps_executed: u64,
}

/// One correctness gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gate {
    /// Gate name.
    pub name: String,
    /// Whether it passed.
    pub ok: bool,
    /// What was compared.
    pub detail: String,
}

/// Artifact locations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactInfo {
    /// Montage from the restore path.
    pub montage: String,
    /// Montage from the from-scratch path.
    pub montage_baseline: String,
    /// The two montages are byte-identical.
    pub montages_identical: bool,
    /// Cumulative-cost CSV.
    pub reuse_csv: String,
    /// This file.
    pub results_json: String,
    /// Directory holding every child's own report.
    pub child_reports: String,
}

/// The whole evidence file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results {
    /// Schema id.
    pub schema: String,
    /// Generation time, unix seconds.
    pub generated_unix_s: u64,
    /// Host facts.
    pub host: HostInfo,
    /// Model facts.
    pub model: ModelInfo,
    /// Scene and requests.
    pub scene: SceneInfo,
    /// Persisted-state accounting.
    pub state: StateInfo,
    /// Process isolation.
    pub process: ProcessInfo,
    /// Model-work accounting.
    pub work: WorkInfo,
    /// Phase timings across fresh processes.
    pub timing: Vec<PhaseAcross>,
    /// Fresh-process end-to-end wall times, one entry per mode.
    pub fresh_process_end_to_end_ms: Vec<PhaseAcross>,
    /// Producer phase timings across fresh processes, EntropyFS path.
    pub producer_timing_vole: Vec<PhaseAcross>,
    /// Producer phase timings across fresh processes, raw-file-only path.
    pub producer_timing_raw: Vec<PhaseAcross>,
    /// The repeated-request result.
    pub reuse: ReuseInfo,
    /// The negative case.
    pub negative: NegativeInfo,
    /// The correctness gates; performance is deliberately absent from them.
    pub gates: Vec<Gate>,
    /// All gates passed.
    pub pass: bool,
    /// Artifacts.
    pub artifacts: ArtifactInfo,
    /// Honest notes about what the numbers do and do not mean.
    pub notes: Vec<String>,
}

/// Options for a whole demonstration run.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Run root.
    pub run_dir: PathBuf,
    /// Frozen checkpoint.
    pub checkpoint: PathBuf,
    /// Frames consumed.
    pub context_len: usize,
    /// Frames generated per request.
    pub future_len: usize,
    /// Threads pinned for children; 0 means "leave the default".
    pub threads: usize,
    /// Delete the run directory before starting.
    pub clean: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            run_dir: PathBuf::from(DEFAULT_RUN_DIR),
            checkpoint: PathBuf::from(DEFAULT_CHECKPOINT),
            context_len: DEFAULT_CONTEXT,
            future_len: DEFAULT_FUTURE,
            threads: default_threads(),
            clean: true,
        }
    }
}

/// Threads to pin for children: the machine's logical CPUs, capped.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, MAX_PINNED_THREADS)
}

/// Control codes as run-length pairs, for compact evidence.
pub fn rle(codes: &[u8]) -> Vec<(u8, usize)> {
    let mut out: Vec<(u8, usize)> = Vec::new();
    for c in codes {
        match out.last_mut() {
            Some((last, n)) if *last == *c => *n += 1,
            _ => out.push((*c, 1)),
        }
    }
    out
}

fn git_commit() -> Option<String> {
    let out = Command::new("git")
        .args(["--no-optional-locks", "rev-parse", "HEAD"])
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

// ---------------------------------------------------------------------------
// The demonstration run
// ---------------------------------------------------------------------------

/// Phase names a request child reports.
const REQUEST_PHASES: [&str; 6] = [
    "model_load",
    "scene_generate",
    "model_only_replay",
    "state_restore",
    "model_only_generate",
    "request_total",
];

/// Phase names a producer child reports.
const PRODUCER_PHASES: [&str; 8] = [
    "model_load",
    "store_open_or_create",
    "scene_generate",
    "model_only_replay",
    "state_serialize",
    "entropyfs_contains",
    "state_persist_entropyfs",
    "state_persist_raw_file",
];

fn aggregate(reports: &[serde_json::Value], phases: &[&str], prefix: &str) -> Vec<PhaseAcross> {
    let mut out = Vec::new();
    for p in phases {
        if let Some(mut a) = PhaseAcross::from_reports(reports, p) {
            a.phase = format!("{prefix}.{p}");
            out.push(a);
        }
    }
    out
}

/// Whether the process's entry in `/proc` is gone. `None` off Linux.
fn gate(name: &str, ok: bool, detail: impl Into<String>) -> Gate {
    Gate {
        name: name.to_string(),
        ok,
        detail: detail.into(),
    }
}

/// Count distinct values, for the "did this batch really ask for different things"
/// checks.
fn distinct<T: std::hash::Hash + Eq + Clone>(v: &[T]) -> usize {
    v.iter()
        .cloned()
        .collect::<std::collections::HashSet<T>>()
        .len()
}

/// Whether the process's entry in `/proc` is gone. `None` off Linux.
///
/// This turns "the producer exited" from a claim about the parent's control flow
/// into an observation: after `wait()` returns, the kernel must no longer be listing
/// that pid.
fn process_entry_gone(pid: u32) -> Option<bool> {
    if cfg!(target_os = "linux") {
        Some(!Path::new(&format!("/proc/{pid}")).exists())
    } else {
        None
    }
}

fn fmt_ms(v: f64) -> String {
    if v >= 100.0 {
        format!("{v:.1} ms")
    } else {
        format!("{v:.2} ms")
    }
}

fn fmt_ms_opt(v: Option<f64>) -> String {
    v.map(fmt_ms).unwrap_or_else(|| "n/a".to_string())
}

fn fmt_bytes(v: u64) -> String {
    if v >= 1024 * 1024 {
        format!("{v} B ({:.2} MiB)", v as f64 / (1024.0 * 1024.0))
    } else if v >= 1024 {
        format!("{v} B ({:.1} KiB)", v as f64 / 1024.0)
    } else {
        format!("{v} B")
    }
}

fn fmt_count(v: u64) -> String {
    let s = v.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push('_');
        }
        out.push(c);
    }
    out
}

/// A tiny stdout section header.
fn banner(text: &str) {
    println!("\n-- {text}");
}

/// Count of child processes whose self-reported PID matched the OS-assigned one.
/// Incremented by [`child`], so it is an exact census rather than a hand count.
static CHILDREN_PID_VERIFIED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Run one child and require it to succeed, with a self-reported PID that matches
/// the PID the operating system assigned.
///
/// The PID check is not decoration: it is the evidence that the report in hand came
/// from the process the parent actually spawned, rather than from a re-exec, a
/// wrapper, or a mis-parse.
fn child(args: &[String], env: &[(String, String)]) -> std::result::Result<ChildOutcome, String> {
    let out = spawn_child(args, env)?;
    if out.pid == 0 {
        return Err(format!("child {args:?} reported no pid"));
    }
    if out.pid != out.os_pid {
        return Err(format!(
            "child {args:?} reported pid {} but the OS assigned {}",
            out.pid, out.os_pid
        ));
    }
    CHILDREN_PID_VERIFIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(out)
}

/// Run `reps` measured children, optionally preceded by one discarded warm-up.
fn child_series(
    make_args: impl Fn(usize) -> Vec<String>,
    reps: usize,
    warmup: bool,
    env: &[(String, String)],
) -> std::result::Result<(Option<ChildOutcome>, Vec<ChildOutcome>), String> {
    let mut warm = None;
    if warmup {
        warm = Some(child(&make_args(0), env)?);
    }
    let mut out = Vec::with_capacity(reps);
    for i in 0..reps {
        out.push(child(&make_args(i + 1), env)?);
    }
    Ok((warm, out))
}

/// Execute the whole process-cold experiment and write the evidence.
pub fn run_demo(opts: &RunOptions) -> std::result::Result<Results, String> {
    let paths = Paths::new(&opts.run_dir);
    if opts.clean {
        let _ = std::fs::remove_dir_all(&paths.root);
    }
    standard_dir(&paths.root)?;
    standard_dir(&paths.futures())?;
    standard_dir(&paths.reports())?;

    let env: Vec<(String, String)> = if opts.threads > 0 {
        vec![("RAYON_NUM_THREADS".to_string(), opts.threads.to_string())]
    } else {
        Vec::new()
    };
    let canonical_scene = SceneId::MovingShape01;
    let unrelated_scene = SceneId::MovingShape02;
    let run_start = Instant::now();

    // ======================================================================
    // 1. Producer: earn, persist through EntropyFS, exit.
    // ======================================================================
    banner("PRODUCER - earn generative state, persist through EntropyFS, exit");
    let pargs = ProducerArgs {
        run_dir: paths.root.clone(),
        checkpoint: opts.checkpoint.clone(),
        scene: canonical_scene,
        context_len: opts.context_len,
        skip_entropyfs: false,
        report: Some(paths.reports().join("producer-canonical.json")),
    };
    let p_out = child(&pargs.argv(), &env)?;
    let producer: ProducerReport = parse_report(&p_out.report)?;
    if producer.pid != p_out.os_pid {
        return Err(format!(
            "producer reported pid {} but the OS assigned {}",
            producer.pid, p_out.os_pid
        ));
    }
    if !producer.state_finite {
        return Err("the earned state contains a non-finite value".into());
    }
    println!(
        "   producer pid {} exited after {} (replay {})",
        producer.pid,
        fmt_ms(p_out.wall_ms),
        fmt_ms_opt(phase_median(&producer.timings, "model_only_replay"))
    );
    println!(
        "   state hash {} ({} H + {} C + {} carry bytes)",
        &producer.state_hash[..16],
        producer.h_bytes,
        producer.c_bytes,
        producer.carry_bytes
    );
    // Observed, not asserted: the producer is waited on here, and its /proc entry
    // must already be gone before any consumer exists.
    let producer_proc_gone = process_entry_gone(producer.pid);
    match producer_proc_gone {
        Some(true) => println!("   producer /proc entry already gone: PASS"),
        Some(false) => {
            return Err(format!(
                "producer pid {} still has a /proc entry after wait()",
                producer.pid
            ))
        }
        None => println!("   /proc check unavailable on this platform"),
    }
    println!(
        "   entropyfs blob {} ; store delta {}",
        &producer.blob_id[..16.min(producer.blob_id.len())],
        fmt_bytes(producer.store_physical_delta)
    );

    // ======================================================================
    // 2. Identity: same earned state, four related futures, three paths.
    // ======================================================================
    banner("IDENTITY - one earned state, four related futures, three recovery paths");
    let mut consumer_pids: BTreeMap<String, u32> = BTreeMap::new();
    let mut requests_evidence: Vec<RequestIdentity> = Vec::new();
    let mut montage_ok = true;
    let mut restore_exact_all = true;
    let mut start_hash_all = true;
    let mut finite_all = true;
    let mut baseline_frames: Vec<(Request, Vec<Vec<f32>>)> = Vec::new();
    let mut vole_frames: Vec<(Request, Vec<Vec<f32>>)> = Vec::new();

    for req in Request::MONTAGE {
        let mut files: Vec<(&str, PathBuf)> = Vec::new();
        for mode in [Mode::Baseline, Mode::Raw, Mode::Vole] {
            let emit = paths
                .futures()
                .join(format!("{}_{}.f32", mode.as_str(), req.name()));
            let args = RequestArgs {
                run_dir: paths.root.clone(),
                checkpoint: opts.checkpoint.clone(),
                mode,
                scene: canonical_scene,
                expect_scene: canonical_scene,
                context_len: opts.context_len,
                future_len: opts.future_len,
                request: req,
                reps: 1,
                warmup: false,
                emit: Some(emit.clone()),
                vole_blob: (mode == Mode::Vole).then(|| producer.blob_id.clone()),
                raw_file: (mode == Mode::Raw).then(|| paths.raw_state()),
                report: Some(paths.reports().join(format!(
                    "identity-{}-{}.json",
                    req.name(),
                    mode.as_str()
                ))),
            };
            let out = child(&args.argv(), &env)?;
            consumer_pids.insert(format!("montage-{}-{}", req.name(), mode.as_str()), out.pid);
            let r: RequestReport = parse_report(&out.report)?;
            if mode == Mode::Vole {
                restore_exact_all &= r.restore_exact == Some(true);
                start_hash_all &=
                    r.start_state_hash.as_deref() == Some(producer.state_hash.as_str());
            }
            finite_all &= r.start_state_finite;
            files.push((mode.as_str(), emit));
        }

        let read = |p: &Path| -> std::result::Result<Vec<u8>, String> {
            std::fs::read(p).map_err(|e| format!("read {p:?}: {e}"))
        };
        let b_bytes = read(&files[0].1)?;
        let r_bytes = read(&files[1].1)?;
        let v_bytes = read(&files[2].1)?;
        let identical = frame_payload(&b_bytes) == frame_payload(&v_bytes)
            && frame_payload(&b_bytes) == frame_payload(&r_bytes);
        montage_ok &= identical;
        baseline_frames.push((req, decode_frames(&b_bytes)?));
        vole_frames.push((req, decode_frames(&v_bytes)?));
        requests_evidence.push(RequestIdentity {
            request: req.name().to_string(),
            label: req.label().to_string(),
            controls: req.schedule().codes(),
            baseline_hash: hash_hex(frame_payload(&b_bytes)),
            vole_hash: hash_hex(frame_payload(&v_bytes)),
            raw_hash: hash_hex(frame_payload(&r_bytes)),
            compared_bytes: frame_payload(&b_bytes).len(),
            baseline_eq_vole: frame_payload(&b_bytes) == frame_payload(&v_bytes),
            baseline_eq_raw: frame_payload(&b_bytes) == frame_payload(&r_bytes),
            byte_identical: identical,
        });
        println!(
            "   {:<14} baseline {} vole {} raw {} -> {}",
            req.name(),
            &hash_hex(frame_payload(&b_bytes))[..12],
            &hash_hex(frame_payload(&v_bytes))[..12],
            &hash_hex(frame_payload(&r_bytes))[..12],
            if identical {
                "byte-identical PASS"
            } else {
                "DIFFER FAIL"
            }
        );
    }

    // ======================================================================
    // 3. Timing across fresh processes.
    // ======================================================================
    banner("TIMING - phase medians across fresh processes, plus fresh-process end-to-end");
    let mut request_reports: BTreeMap<&'static str, Vec<serde_json::Value>> = BTreeMap::new();
    let mut end_to_end: Vec<PhaseAcross> = Vec::new();
    for mode in [Mode::Baseline, Mode::Raw, Mode::Vole] {
        // (a) phase medians: one child that runs `1 + TIMING_REPS` repetitions
        //     internally, so each phase's steady state is what is recorded.
        let make = |i: usize| -> Vec<String> {
            RequestArgs {
                run_dir: paths.root.clone(),
                checkpoint: opts.checkpoint.clone(),
                mode,
                scene: canonical_scene,
                expect_scene: canonical_scene,
                context_len: opts.context_len,
                future_len: opts.future_len,
                request: Request::TurnLeft,
                reps: TIMING_REPS,
                warmup: true,
                emit: None,
                vole_blob: (mode == Mode::Vole).then(|| producer.blob_id.clone()),
                raw_file: (mode == Mode::Raw).then(|| paths.raw_state()),
                report: Some(
                    paths
                        .reports()
                        .join(format!("timing-{}-{i}.json", mode.as_str())),
                ),
            }
            .argv()
        };
        let (_warm, outs) = child_series(make, TIMING_REPS, true, &env)?;
        for o in &outs {
            request_reports
                .entry(mode.as_str())
                .or_default()
                .push(o.report.clone());
            consumer_pids.insert(format!("timing-{}", mode.as_str()), o.pid);
        }

        // (b) fresh-process end-to-end: `1 + TIMING_REPS` separate processes, each
        //     serving exactly ONE request, clocked from outside by the parent. This
        //     is the number that includes process spawn, dynamic loading, model
        //     load, recovery, generation and exit. It is deliberately not derived
        //     from the phase table above.
        let make_one = |i: usize| -> Vec<String> {
            RequestArgs {
                run_dir: paths.root.clone(),
                checkpoint: opts.checkpoint.clone(),
                mode,
                scene: canonical_scene,
                expect_scene: canonical_scene,
                context_len: opts.context_len,
                future_len: opts.future_len,
                request: Request::TurnLeft,
                reps: 1,
                warmup: false,
                emit: None,
                vole_blob: (mode == Mode::Vole).then(|| producer.blob_id.clone()),
                raw_file: (mode == Mode::Raw).then(|| paths.raw_state()),
                report: Some(
                    paths
                        .reports()
                        .join(format!("e2e-{}-{i}.json", mode.as_str())),
                ),
            }
            .argv()
        };
        let (_warm, ones) = child_series(make_one, TIMING_REPS, true, &env)?;
        let walls: Vec<f64> = ones.iter().map(|o| o.wall_ms).collect();
        for o in &ones {
            consumer_pids.insert(format!("e2e-{}", mode.as_str()), o.pid);
        }
        end_to_end.push(PhaseAcross {
            phase: format!("{}.fresh_process_end_to_end", mode.as_str()),
            definition: "parent-measured wall time of one fresh process serving exactly one \
                         request: spawn, load, recover, generate, exit"
                .into(),
            median_ms: median(&walls),
            per_process_ms: walls,
        });
        println!(
            "   {:<9} fresh-process end-to-end median {} (6 processes)",
            mode.as_str(),
            fmt_ms_opt(end_to_end.last().unwrap().median_ms)
        );
    }

    // Producer timing: a real process of each flavour, on its own fresh store.
    let mut producer_timing_vole: Vec<PhaseAcross> = Vec::new();
    let mut producer_timing_raw: Vec<PhaseAcross> = Vec::new();
    let mut vole_producer_ms = 0.0f64;
    let mut raw_producer_ms = 0.0f64;
    for skip in [false, true] {
        let label = if skip { "raw" } else { "vole" };
        let make = |i: usize| -> Vec<String> {
            ProducerArgs {
                run_dir: paths.timing_dir(&format!("producer-{label}-{i}")),
                checkpoint: opts.checkpoint.clone(),
                scene: canonical_scene,
                context_len: opts.context_len,
                skip_entropyfs: skip,
                report: None,
            }
            .argv()
        };
        let (_warm, outs) = child_series(make, TIMING_REPS, true, &env)?;
        let walls: Vec<f64> = outs.iter().map(|o| o.wall_ms).collect();
        let reports: Vec<serde_json::Value> = outs.iter().map(|o| o.report.clone()).collect();
        // Every timed producer must have earned exactly the canonical state.
        for o in &outs {
            let r: ProducerReport = parse_report(&o.report)?;
            if r.state_hash != producer.state_hash {
                return Err(format!(
                    "{label} timing producer earned state {} != canonical {}",
                    r.state_hash, producer.state_hash
                ));
            }
        }
        let agg = aggregate(&reports, &PRODUCER_PHASES, &format!("producer-{label}"));
        let wall_across = PhaseAcross {
            phase: format!("producer-{label}.fresh_process_end_to_end"),
            definition: "parent-measured wall time of one whole producer child, on a fresh store"
                .into(),
            median_ms: median(&walls),
            per_process_ms: walls,
        };
        if skip {
            raw_producer_ms = wall_across
                .median_ms
                .ok_or("raw producer timing produced no median")?;
            producer_timing_raw = std::iter::once(wall_across).chain(agg).collect();
        } else {
            vole_producer_ms = wall_across
                .median_ms
                .ok_or("vole producer timing produced no median")?;
            producer_timing_vole = std::iter::once(wall_across).chain(agg).collect();
        }
        println!(
            "   producer ({label:<4}) fresh-process median {}",
            fmt_ms(if skip {
                raw_producer_ms
            } else {
                vole_producer_ms
            })
        );
    }

    // ======================================================================
    // 4. Repeated related requests: cumulative cost and N*.
    // ======================================================================
    banner("REUSE - N fresh related requests per point, every request in its own process");
    // One discarded warm-up child per mode, so the sweep is not measuring the
    // first-touch of a cold binary image.
    for mode in [Mode::Baseline, Mode::Raw, Mode::Vole] {
        let a = RequestArgs {
            run_dir: paths.root.clone(),
            checkpoint: opts.checkpoint.clone(),
            mode,
            scene: canonical_scene,
            expect_scene: canonical_scene,
            context_len: opts.context_len,
            future_len: opts.future_len,
            request: Request::Continue,
            reps: 1,
            warmup: false,
            emit: None,
            vole_blob: (mode == Mode::Vole).then(|| producer.blob_id.clone()),
            raw_file: (mode == Mode::Raw).then(|| paths.raw_state()),
            report: None,
        };
        child(&a.argv(), &env)?;
    }

    let mut points: Vec<ReusePoint> = Vec::new();
    // request programme -> output hash, per path, to prove determinism across fresh
    // processes and non-identity across programmes.
    let mut seen: BTreeMap<(&'static str, String), String> = BTreeMap::new();
    let mut repeated_identical = true;
    let mut distinct_programs: BTreeMap<String, String> = BTreeMap::new();
    let mut programs_collide = false;
    let mut fresh_processes_per_mode = 0usize;

    for &n in N_POINTS.iter() {
        let requests: Vec<Request> = (0..n)
            .map(|i| Request::ALL[i % Request::ALL.len()])
            .collect();
        fresh_processes_per_mode += n;
        let mut per: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        let mut last_pid: BTreeMap<&'static str, u32> = BTreeMap::new();
        let mut baseline_hashes: Vec<String> = Vec::with_capacity(n);
        for req in requests.iter() {
            for mode in [Mode::Baseline, Mode::Raw, Mode::Vole] {
                let a = RequestArgs {
                    run_dir: paths.root.clone(),
                    checkpoint: opts.checkpoint.clone(),
                    mode,
                    scene: canonical_scene,
                    expect_scene: canonical_scene,
                    context_len: opts.context_len,
                    future_len: opts.future_len,
                    request: *req,
                    reps: 1,
                    warmup: false,
                    emit: None,
                    vole_blob: (mode == Mode::Vole).then(|| producer.blob_id.clone()),
                    raw_file: (mode == Mode::Raw).then(|| paths.raw_state()),
                    report: None,
                };
                let out = child(&a.argv(), &env)?;
                let r: RequestReport = parse_report(&out.report)?;
                per.entry(mode.as_str()).or_default().push(out.wall_ms);
                last_pid.insert(mode.as_str(), out.pid);
                let key = (mode.as_str(), req.name().to_string());
                match seen.get(&key) {
                    Some(prev) if prev != &r.output_hash => repeated_identical = false,
                    None => {
                        seen.insert(key, r.output_hash.clone());
                    }
                    _ => {}
                }
                if mode == Mode::Baseline {
                    baseline_hashes.push(r.output_hash.clone());
                    match distinct_programs.get(req.name()) {
                        Some(prev) if prev != &r.output_hash => programs_collide = true,
                        None => {
                            distinct_programs.insert(req.name().to_string(), r.output_hash.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
        for (mode, pid) in &last_pid {
            consumer_pids.insert(format!("sweep-{mode}-last"), *pid);
        }
        let sum = |v: &Vec<f64>| v.iter().sum::<f64>();
        let bl = per.get("baseline").cloned().unwrap_or_default();
        let rw = per.get("raw").cloned().unwrap_or_default();
        let vo = per.get("vole").cloned().unwrap_or_default();
        let c_baseline = sum(&bl);
        let c_vole = vole_producer_ms + sum(&vo);
        let c_raw = raw_producer_ms + sum(&rw);
        println!(
            "   N={:<2} baseline {:>10}  raw {:>10}  vole {:>10}   VOLE cheaper: {}",
            n,
            fmt_ms(c_baseline),
            fmt_ms(c_raw),
            fmt_ms(c_vole),
            if c_vole < c_baseline { "yes" } else { "no" }
        );
        points.push(ReusePoint {
            n,
            c_baseline_ms: c_baseline,
            c_vole_ms: c_vole,
            c_raw_ms: c_raw,
            vole_producer_ms,
            raw_producer_ms,
            requests: requests.iter().map(|r| r.name().to_string()).collect(),
            distinct_request_programs: distinct(
                &requests.iter().map(|r| r.name()).collect::<Vec<_>>(),
            ),
            distinct_baseline_outputs: distinct(&baseline_hashes),
            baseline_per_request_ms: bl,
            vole_per_request_ms: vo,
            raw_per_request_ms: rw,
        });
    }

    let n_star = points
        .iter()
        .find(|p| p.c_vole_ms < p.c_baseline_ms)
        .map(|p| p.n);
    let n_star_note = match n_star {
        Some(n) => format!(
            "smallest tested N with C_vole(N) < C_baseline(N): {n} of {}",
            N_POINTS.last().copied().unwrap_or(0)
        ),
        None => format!(
            "no crossover within the tested range: N* > {}",
            N_POINTS.last().copied().unwrap_or(0)
        ),
    };

    // ======================================================================
    // 5. The negative case.
    // ======================================================================
    banner("NEGATIVE - a request for the unrelated scene is offered the canonical state");
    let neg_req = Request::Continue;
    let neg_vole = child(
        &RequestArgs {
            run_dir: paths.root.clone(),
            checkpoint: opts.checkpoint.clone(),
            mode: Mode::Vole,
            scene: unrelated_scene,
            expect_scene: unrelated_scene,
            context_len: opts.context_len,
            future_len: opts.future_len,
            request: neg_req,
            reps: 1,
            warmup: false,
            emit: None,
            vole_blob: Some(producer.blob_id.clone()),
            raw_file: None,
            report: Some(paths.reports().join("negative-offered.json")),
        }
        .argv(),
        &env,
    )?;
    let neg_offered: RequestReport = parse_report(&neg_vole.report)?;
    consumer_pids.insert("negative-offered".into(), neg_vole.pid);

    let neg_ref = child(
        &RequestArgs {
            run_dir: paths.root.clone(),
            checkpoint: opts.checkpoint.clone(),
            mode: Mode::Baseline,
            scene: unrelated_scene,
            expect_scene: unrelated_scene,
            context_len: opts.context_len,
            future_len: opts.future_len,
            request: neg_req,
            reps: 1,
            warmup: false,
            emit: None,
            vole_blob: None,
            raw_file: None,
            report: Some(paths.reports().join("negative-baseline.json")),
        }
        .argv(),
        &env,
    )?;
    let neg_baseline: RequestReport = parse_report(&neg_ref.report)?;
    consumer_pids.insert("negative-baseline".into(), neg_ref.pid);

    let negative = NegativeInfo {
        requested_scene: unrelated_scene.as_str().to_string(),
        offered_scene: neg_offered
            .restore
            .offered_scene
            .clone()
            .unwrap_or_else(|| "<none>".into()),
        rejected: !neg_offered.restore.accepted,
        rejection: neg_offered
            .restore
            .rejection
            .clone()
            .unwrap_or_else(|| "<none>".into()),
        checks: neg_offered.restore.checks.clone(),
        fallback: neg_offered
            .restore
            .fallback
            .clone()
            .unwrap_or_else(|| "<none>".into()),
        fallback_output_hash: neg_offered.output_hash.clone(),
        reference_baseline_hash: neg_baseline.output_hash.clone(),
        fallback_matches_baseline: neg_offered.output_hash == neg_baseline.output_hash,
        cell_steps_executed: neg_offered.work_total.cell_steps,
    };
    println!(
        "   reuse rejected on check {:?}; fallback {:?}; fallback output {} reference",
        negative.rejection,
        negative.fallback,
        if negative.fallback_matches_baseline {
            "MATCHES"
        } else {
            "DIFFERS FROM"
        }
    );

    // ======================================================================
    // 6. Visual evidence.
    // ======================================================================
    banner("VISUALS - context tail and four related branches");
    let gen_for_scene = SceneSpec::canonical(canonical_scene);
    let sched = scene::context_program();
    let all_ctx = scene::rollout(&gen_for_scene, opts.context_len - 1, |t| sched.at(t));
    let ctx_tail: Vec<Vec<f32>> = all_ctx[all_ctx.len() - MONTAGE_COLS..].to_vec();
    let truth_branches: Vec<(Request, Vec<Vec<f32>>)> = Request::MONTAGE
        .iter()
        .map(|r| {
            (
                *r,
                truth_future(canonical_scene, opts.context_len, opts.future_len, *r),
            )
        })
        .collect();
    let montage_v = build_montage(&ctx_tail, &vole_frames, &truth_branches);
    let montage_b = build_montage(&ctx_tail, &baseline_frames, &truth_branches);
    std::fs::write(paths.montage(), &montage_v).map_err(|e| format!("{e}"))?;
    std::fs::write(paths.montage_baseline(), &montage_b).map_err(|e| format!("{e}"))?;
    let montages_identical = montage_v == montage_b;
    println!(
        "   wrote {} ({} bytes) and {} ; identical: {}",
        paths.montage().display(),
        montage_v.len(),
        paths.montage_baseline().display(),
        montages_identical
    );

    // ======================================================================
    // 7. Work accounting and gates.
    // ======================================================================
    let baseline_ref: RequestReport = parse_report(
        &request_reports
            .get("baseline")
            .and_then(|v| v.first())
            .ok_or("no baseline timing report")?
            .clone(),
    )?;
    let vole_ref: RequestReport = parse_report(
        &request_reports
            .get("vole")
            .and_then(|v| v.first())
            .ok_or("no vole timing report")?
            .clone(),
    )?;
    let raw_ref: RequestReport = parse_report(
        &request_reports
            .get("raw")
            .and_then(|v| v.first())
            .ok_or("no raw timing report")?
            .clone(),
    )?;

    let avoided_steps = baseline_ref
        .work_total
        .cell_steps
        .saturating_sub(vole_ref.work_total.cell_steps);
    let avoided_macs = baseline_ref
        .work_total
        .macs
        .saturating_sub(vole_ref.work_total.macs);
    let work = WorkInfo {
        baseline_per_request: baseline_ref.work_total,
        vole_per_request: vole_ref.work_total,
        raw_per_request: raw_ref.work_total,
        cell_steps_avoided_per_request: avoided_steps,
        macs_avoided_per_request: avoided_macs,
        macs_avoided_fraction: if baseline_ref.work_total.macs == 0 {
            0.0
        } else {
            avoided_macs as f64 / baseline_ref.work_total.macs as f64
        },
        producer: producer.work,
    };

    let all_consumer_pids: Vec<u32> = consumer_pids.values().copied().collect();
    let producer_differs = all_consumer_pids.iter().all(|p| *p != producer.pid);
    let no_duplicate_consumer_pids = all_consumer_pids.len()
        == all_consumer_pids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
    let children_pid_verified_count =
        CHILDREN_PID_VERIFIED.load(std::sync::atomic::Ordering::Relaxed);

    let mut gates: Vec<Gate> = Vec::new();
    gates.push(gate(
        "producer_really_exits",
        producer_differs && producer_proc_gone.unwrap_or(true),
        format!(
            "producer pid {}; every recorded consumer pid differs from it; the producer was \
             waited on before any consumer was spawned and its /proc entry was already gone \
             ({:?})",
            producer.pid, producer_proc_gone
        ),
    ));
    gates.push(gate(
        "consumers_are_distinct_processes",
        no_duplicate_consumer_pids && children_pid_verified_count > 0,
        format!(
            "{} recorded consumer roles spanning {} distinct pids, none equal to the producer's; \
             {} child processes reported a pid matching the one the OS assigned",
            all_consumer_pids.len(),
            all_consumer_pids
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            children_pid_verified_count
        ),
    ));
    gates.push(gate(
        "restored_state_bytes_match_original",
        restore_exact_all,
        "re-serialised (H, C) compared byte-for-byte with the persisted payload on every \
         VOLE request"
            .to_string(),
    ));
    gates.push(gate(
        "restored_state_hash_matches_producer",
        start_hash_all,
        format!(
            "producer state hash {} equals the hash of the state every VOLE request generated from",
            &producer.state_hash[..16]
        ),
    ));
    gates.push(gate(
        "baseline_and_restore_outputs_identical",
        montage_ok,
        format!(
            "{} frames of exact f32 bytes compared per request, on {} requests",
            requests_evidence
                .first()
                .map(|r| r.compared_bytes)
                .unwrap_or(0),
            requests_evidence.len()
        ),
    ));
    gates.push(gate(
        "raw_checkpoint_and_restore_outputs_identical",
        requests_evidence.iter().all(|r| r.baseline_eq_raw),
        "the raw-checkpoint path produced the same bytes as the baseline on every request"
            .to_string(),
    ));
    gates.push(gate(
        "fewer_recurrent_steps",
        avoided_steps > 0,
        format!(
            "baseline {} cell steps vs VOLE {} cell steps per request; {} avoided",
            baseline_ref.work_total.cell_steps, vole_ref.work_total.cell_steps, avoided_steps
        ),
    ));
    gates.push(gate(
        "state_is_finite",
        finite_all && producer.state_finite,
        "no NaN or infinity in H or C, before or after persistence".to_string(),
    ));
    gates.push(gate(
        "negative_state_rejected",
        negative.rejected && negative.rejection != "<none>",
        format!(
            "a {} request offered a {} record failed the {} check",
            negative.requested_scene, negative.offered_scene, negative.rejection
        ),
    ));
    gates.push(gate(
        "negative_fallback_matches_baseline",
        negative.fallback_matches_baseline,
        "after refusing reuse the request replayed from scratch and produced the same bytes as \
         an independent from-scratch baseline for the unrelated scene"
            .to_string(),
    ));
    gates.push(gate(
        "fresh_process_requests_are_bit_identical",
        repeated_identical && !programs_collide,
        format!(
            "{} request programs; repeats of the same program in different fresh processes \
             produced identical bytes, and no two programs collided",
            distinct_programs.len()
        ),
    ));
    gates.push(gate(
        "montages_identical",
        montages_identical,
        "the from-scratch montage and the restored montage are byte-identical images".to_string(),
    ));
    let pass = gates.iter().all(|g| g.ok);

    // ======================================================================
    // 8. Assemble the evidence file.
    // ======================================================================
    let mut timing: Vec<PhaseAcross> = Vec::new();
    for (mode, reports) in &request_reports {
        timing.extend(aggregate(reports, &REQUEST_PHASES, mode));
    }
    timing.extend(end_to_end.iter().cloned());

    let context_controls: Vec<u8> = (0..opts.context_len)
        .map(|t| scene::context_program().at(t).code())
        .collect();

    let results = Results {
        schema: "vole-field/results/1".into(),
        generated_unix_s: now_unix_s(),
        host: HostInfo {
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            available_parallelism: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0),
            pinned_threads: opts.threads,
            crate_version: env!("CARGO_PKG_VERSION").into(),
            git_commit: git_commit(),
        },
        model: ModelInfo {
            checkpoint: opts.checkpoint.display().to_string(),
            checkpoint_bytes: producer.checkpoint_bytes,
            checkpoint_file_hash: producer.checkpoint_file_hash.clone(),
            weights_hash: producer.model_weights_hash.clone(),
            config_hash: producer.model_config_hash.clone(),
            config: ModelConfig::default(),
            parameter_count: ModelConfig::default().parameter_count(),
            step_work: ModelConfig::default().step_work(),
            weights_are_shared_by_both_paths: true,
        },
        scene: SceneInfo {
            id: canonical_scene.as_str().into(),
            label: canonical_scene.label().into(),
            unrelated_id: unrelated_scene.as_str().into(),
            unrelated_label: unrelated_scene.label().into(),
            height: H,
            width: W,
            context_len: opts.context_len,
            future_len: opts.future_len,
            context_controls_rle: rle(&context_controls),
            requests: requests_evidence.clone(),
        },
        state: StateInfo {
            h_bytes: producer.h_bytes,
            c_bytes: producer.c_bytes,
            carry_bytes: producer.carry_bytes,
            raw_state_bytes: producer.raw_state_bytes,
            vole_record_bytes: producer.vole_record_bytes,
            vole_header_bytes: producer.vole_record_bytes - producer.raw_state_bytes,
            raw_file_bytes: producer.raw_file_bytes,
            state_hash: producer.state_hash.clone(),
            restored_state_hash: vole_ref.start_state_hash.clone(),
            state_exact: restore_exact_all,
            state_finite: producer.state_finite,
            entropyfs: EntropyFsInfo {
                blob_id: producer.blob_id.clone(),
                store_mkfs_bytes: producer.store_mkfs_bytes,
                store_bytes_before: producer.store_bytes_before,
                store_bytes_after: producer.store_bytes_after,
                physical_delta_bytes: producer.store_physical_delta,
                engine_physical_used_bytes: producer.engine_physical_used_after,
                blob_already_present: producer.blob_already_present,
                overhead_vs_raw_bytes: producer.store_physical_delta as i64
                    - producer.raw_file_bytes as i64,
                measured_from_outside_the_engine: true,
            },
            model_weight_bytes: producer.checkpoint_bytes,
        },
        process: ProcessInfo {
            producer_pid: producer.pid,
            consumer_pids: consumer_pids.clone(),
            children_pid_verified: children_pid_verified_count,
            producer_differs_from_all_consumers: producer_differs,
            producer_proc_entry_gone: producer_proc_gone,
            producer_exited_before_consumers: true,
            bridge: "persisted bytes only: an EntropyFS blob id on the VOLE path, an ordinary \
                     file on the raw path"
                .into(),
            cold_disk_cache: false,
            claim: "fresh-process restore (a.k.a. process-cold restore). The producer was waited \
                    on and its /proc entry was already gone before the first consumer was \
                    spawned. The OS page cache was not flushed; this is not a cold-disk-cache \
                    experiment."
                .into(),
        },
        work,
        timing,
        fresh_process_end_to_end_ms: end_to_end,
        producer_timing_vole,
        producer_timing_raw,
        reuse: ReuseInfo {
            n_points: N_POINTS.to_vec(),
            points,
            n_star,
            n_star_note,
            fresh_processes_per_mode,
            every_request_in_a_fresh_process: true,
        },
        negative,
        gates: gates.clone(),
        pass,
        artifacts: ArtifactInfo {
            montage: paths.montage().display().to_string(),
            montage_baseline: paths.montage_baseline().display().to_string(),
            montages_identical,
            reuse_csv: paths.reuse_csv().display().to_string(),
            results_json: paths.results_json().display().to_string(),
            child_reports: paths.reports().display().to_string(),
        },
        notes: vec![
            "Model weights are read by every process on every path. They are not a saving of \
             this technique and they are not charged to one side."
                .into(),
            "Persistence and restore costs are reported separately from model arithmetic; \
             persistence I/O is never counted as zero compute."
                .into(),
            "The recurrent-step and MAC counts are the load-bearing evidence that less model \
             work happened. Wall-clock confirms whether that translated into runtime on this \
             machine."
                .into(),
            "The raw-checkpoint control exists to keep the result honest: it measures how much \
             of the saving comes from durable state in general, and how much from the VOLE \
             record and the EntropyFS path."
                .into(),
            "The raw-checkpoint payload carries no identity, so it cannot be checked for \
             compatibility. That absence is itself part of the comparison."
                .into(),
            "Performance does not determine PASS/FAIL. A slower VOLE path or an N* beyond the \
             tested range is a passing run with an honest result."
                .into(),
        ],
    };

    write_results(&paths, &results)?;
    println!(
        "\n   total experiment wall time {}",
        fmt_ms(ms(run_start.elapsed()))
    );
    print_table(&results, &paths);
    Ok(results)
}

fn write_results(paths: &Paths, results: &Results) -> std::result::Result<(), String> {
    let json = serde_json::to_string_pretty(results).map_err(|e| format!("serialise: {e}"))?;
    std::fs::write(paths.results_json(), json).map_err(|e| format!("write results: {e}"))?;

    let mut csv = String::from("N,baseline_ms,raw_checkpoint_ms,vole_ms\n");
    for p in &results.reuse.points {
        csv.push_str(&format!(
            "{},{:.3},{:.3},{:.3}\n",
            p.n, p.c_baseline_ms, p.c_raw_ms, p.c_vole_ms
        ));
    }
    std::fs::write(paths.reuse_csv(), csv).map_err(|e| format!("write reuse.csv: {e}"))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The killer table
// ---------------------------------------------------------------------------

fn rule(width: usize) -> String {
    "=".repeat(width)
}

/// Print the human-readable summary.
pub fn print_table(r: &Results, paths: &Paths) {
    let w = 78;
    let step = &r.model.step_work;
    let b = &r.work.baseline_per_request;
    let v = &r.work.vole_per_request;

    println!("\n{}", rule(w));
    println!("VOLE-FIELD - DURABLE GENERATIVE-STATE REUSE");
    println!("{}", rule(w));
    println!(
        "model:             tiny ConvLSTM ({} params, {} hidden ch, {}x{} kernels)",
        fmt_count(r.model.parameter_count),
        r.model.config.hidden_channels,
        r.model.config.kernel,
        r.model.config.kernel
    );
    println!("device:            CPU (no hardware accelerator)");
    println!("scene:             {}", r.scene.id);
    println!("context frames:    {}", r.scene.context_len);
    println!("future frames:     {}", r.scene.future_len);
    println!();
    println!("STATE");
    println!(
        "raw recurrent state:       {}",
        fmt_bytes(r.state.raw_state_bytes as u64)
    );
    println!(
        "  H:                       {}   C: {}",
        fmt_bytes(r.state.h_bytes as u64),
        fmt_bytes(r.state.c_bytes as u64)
    );
    println!(
        "  carry (last two frames): {}",
        fmt_bytes(r.state.carry_bytes as u64)
    );
    println!(
        "VOLE logical blob:         {} ({} header + payload)",
        fmt_bytes(r.state.vole_record_bytes as u64),
        r.state.vole_header_bytes
    );
    println!(
        "raw checkpoint file:       {}",
        fmt_bytes(r.state.raw_file_bytes)
    );
    println!(
        "EntropyFS mkfs floor:      {}",
        fmt_bytes(r.state.entropyfs.store_mkfs_bytes)
    );
    println!(
        "EntropyFS physical delta:  {}",
        fmt_bytes(r.state.entropyfs.physical_delta_bytes)
    );
    println!(
        "EntropyFS engine figure:   {}",
        fmt_bytes(r.state.entropyfs.engine_physical_used_bytes)
    );
    println!(
        "model weights (shared):    {}",
        fmt_bytes(r.state.model_weight_bytes)
    );
    println!();
    println!("PROCESS");
    println!("producer pid:              {}", r.process.producer_pid);
    let mut pids: Vec<String> = r
        .process
        .consumer_pids
        .values()
        .map(|p| p.to_string())
        .collect();
    pids.sort();
    pids.dedup();
    println!(
        "consumer pids ({} distinct): {}",
        pids.len(),
        pids.join(" ")
    );
    println!(
        "fresh process:             {}",
        pass_str(r.process.producer_differs_from_all_consumers)
    );
    println!();
    println!("IDENTITY");
    println!("model weights hash:        {}", r.model.weights_hash);
    println!(
        "checkpoint file hash:      {}",
        r.model.checkpoint_file_hash
    );
    println!("original state hash:       {}", r.state.state_hash);
    println!(
        "restored state hash:       {}",
        r.state
            .restored_state_hash
            .clone()
            .unwrap_or_else(|| "<none>".into())
    );
    println!(
        "state exact:               {}",
        pass_str(r.state.state_exact)
    );
    println!(
        "state finite:              {}",
        pass_str(r.state.state_finite)
    );
    println!();
    println!("REQUESTS (same earned state, related futures)");
    for q in &r.scene.requests {
        println!(
            "  {:<14} baseline {} vole {} raw {}  identical {}",
            q.request,
            &q.baseline_hash[..12],
            &q.vole_hash[..12],
            &q.raw_hash[..12],
            pass_str(q.byte_identical)
        );
    }
    println!();
    println!("MODEL WORK (per request)");
    println!(
        "cell step arithmetic:      {} gate + {} head = {} MACs, {} nonlinearities",
        fmt_count(step.cell_macs),
        fmt_count(step.head_macs),
        fmt_count(step.cell_macs + step.head_macs),
        fmt_count(step.cell_nonlinearities)
    );
    println!(
        "baseline cell steps:       {}  ({} head applications)",
        fmt_count(b.cell_steps),
        fmt_count(b.head_applications)
    );
    println!(
        "VOLE cell steps:           {}  ({} head applications)",
        fmt_count(v.cell_steps),
        fmt_count(v.head_applications)
    );
    println!(
        "cell steps avoided:        {}",
        fmt_count(r.work.cell_steps_avoided_per_request)
    );
    println!("baseline model MACs:       {}", fmt_count(b.macs));
    println!("VOLE model MACs:           {}", fmt_count(v.macs));
    println!(
        "model MACs avoided:        {}  ({:.2}% of baseline)",
        fmt_count(r.work.macs_avoided_per_request),
        r.work.macs_avoided_fraction * 100.0
    );
    println!();
    println!("TIMING");
    for t in &r.timing {
        println!("  {:<40} {}", t.phase, fmt_ms_opt(t.median_ms));
    }
    for t in &r.producer_timing_vole {
        println!("  {:<40} {}", t.phase, fmt_ms_opt(t.median_ms));
    }
    for t in &r.producer_timing_raw {
        println!("  {:<40} {}", t.phase, fmt_ms_opt(t.median_ms));
    }
    println!();
    println!("REUSE (cumulative cost of N related requests, one process each)");
    println!(
        "  {:<4} {:>14} {:>14} {:>14}",
        "N", "baseline", "raw ckpt", "VOLE"
    );
    for p in &r.reuse.points {
        println!(
            "  {:<4} {:>14} {:>14} {:>14}",
            p.n,
            fmt_ms(p.c_baseline_ms),
            fmt_ms(p.c_raw_ms),
            fmt_ms(p.c_vole_ms)
        );
    }
    println!(
        "  one-time VOLE producer:    {}   one-time raw producer: {}",
        fmt_ms_opt(producer_once_ms(r, "vole")),
        fmt_ms_opt(producer_once_ms(r, "raw"))
    );
    println!(
        "  reuse break-even N*:       {}",
        match r.reuse.n_star {
            Some(n) => n.to_string(),
            None => format!("> {}", r.reuse.n_points.last().copied().unwrap_or(0)),
        }
    );
    println!("  {}", r.reuse.n_star_note);
    println!();
    println!("NEGATIVE CASE");
    println!("requested scene:           {}", r.negative.requested_scene);
    println!("offered state's scene:     {}", r.negative.offered_scene);
    println!(
        "reuse:                     {}",
        if r.negative.rejected {
            "rejected"
        } else {
            "accepted (FAIL)"
        }
    );
    println!("failing check:             {}", r.negative.rejection);
    println!("fallback:                  {}", r.negative.fallback);
    println!(
        "fallback equals baseline:  {}",
        pass_str(r.negative.fallback_matches_baseline)
    );
    println!(
        "cell steps executed:       {}",
        fmt_count(r.negative.cell_steps_executed)
    );
    println!();
    println!("CORRECTNESS GATES");
    for g in &r.gates {
        println!("  [{}] {}", if g.ok { "PASS" } else { "FAIL" }, g.name);
    }
    println!("  overall: {}", if r.pass { "PASS" } else { "FAIL" });
    println!("{}", rule(w));
    println!("artifacts: {}", paths.root.display());
    println!("  results.json  {}", r.artifacts.results_json);
    println!("  reuse.csv     {}", r.artifacts.reuse_csv);
    println!("  montage       {}", r.artifacts.montage);
    println!("  child reports {}", r.artifacts.child_reports);
    println!("{}", rule(w));
}

fn pass_str(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn producer_once_ms(r: &Results, which: &str) -> Option<f64> {
    let tag = format!("producer-{which}.fresh_process_end_to_end");
    r.producer_timing_vole
        .iter()
        .chain(r.producer_timing_raw.iter())
        .find(|p| p.phase == tag)
        .and_then(|p| p.median_ms)
}
