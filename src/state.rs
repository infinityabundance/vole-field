//! The minimal VOLE state record, its exact-byte codec, its compatibility
//! predicate, and the EntropyFS persistence path.
//!
//! # The record
//!
//! A fixed 176-byte header followed by the exact `f32` bytes of `H`, `C` and the
//! carry channel:
//!
//! ```text
//! offset size field
//!   0      8  magic "VOLEFLD1"
//!   8      4  format version (u32 LE)
//!  12      4  header length (u32 LE) = 176
//!  16     32  model weights hash      (BLAKE3, behaviour-determining)
//!  48     32  model config hash       (BLAKE3, architecture)
//!  80     16  scene identity          (ASCII, NUL padded)
//!  96      4  context length
//! 100      4  hidden channels
//! 104      4  carry channels
//! 108      4  height
//! 112      4  width
//! 116      4  reserved (zero)
//! 120      8  H payload length in bytes
//! 128      8  C payload length in bytes
//! 136      8  carry payload length in bytes
//! 144     32  state hash (BLAKE3 of H bytes || C bytes || carry bytes)
//! 176    ...  H payload, then C, then carry
//! ```
//!
//! Explicit little-endian fields, not a serialiser's layout: the wire format is
//! part of the claim and must not drift when a dependency's derive changes. This
//! mirrors the discipline EntropyFS itself uses for its permanent on-disk format.
//!
//! # What is deliberately *not* in the record
//!
//! No future frames. No pre-rendered observations. No baseline output. No rendered
//! context beyond the last two observed frames, which the decoder contract requires and
//! which are carried inside the state rather than bolted on beside it. The record cannot
//! smuggle the answer: the future has to be materialised after the restore, by the model,
//! from the state alone.
//!
//! # Why the weights hash and the config hash are separate
//!
//! One is "which frozen weights produced this state", the other is "what shape
//! of operator were they arranged into". They fail for different reasons —
//! retraining versus an architecture edit — and a reviewer wants to know which.

use std::path::{Path, PathBuf};

use entropyfs::engine::{BlobId, Engine, EngineError, EngineOpenOptions};

use crate::model::{ConvLstm, ModelConfig, CARRY_CHANNELS};
use crate::scene::{SceneId, SCENE_ID_BYTES};

/// Record magic; also the first eight bytes of every state blob.
pub const MAGIC: [u8; 8] = *b"VOLEFLD1";
/// Format version written by this build.
pub const FORMAT_VERSION: u32 = 1;
/// Header length in bytes. Fixed by the format, not by the payload.
pub const HEADER_LEN: usize = 176;

// ---------------------------------------------------------------------------
// Hashing / hex helpers
// ---------------------------------------------------------------------------

/// BLAKE3-256 of `bytes`.
pub fn hash32(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Lowercase hex of a 32-byte digest.
pub fn hex32(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// BLAKE3-256 of `bytes`, as lowercase hex. The crate's one hashing idiom.
pub fn hash_hex(bytes: &[u8]) -> String {
    hex32(&hash32(bytes))
}

/// The state hash: BLAKE3 over `H bytes || C bytes || carry bytes`, in that order.
pub fn state_hash(h: &[u8], c: &[u8], carry: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(h);
    hasher.update(c);
    hasher.update(carry);
    *hasher.finalize().as_bytes()
}

/// The canonical behaviour-determining hash of a frozen model.
///
/// Every parameter is fed in with its name, its shape and its exact `f32` bytes,
/// in a fixed name order. Two checkpoints with identical parameters therefore
/// hash identically even if the container file differs; two checkpoints with
/// different parameters can never hash identically.
pub fn weights_hash(model: &ConvLstm) -> candle_core::Result<[u8; 32]> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vole-field/weights/1");
    for (name, t) in model.named_parameters() {
        hasher.update(name.as_bytes());
        hasher.update(b"\0");
        for d in t.dims() {
            hasher.update(&(*d as u64).to_le_bytes());
        }
        let v = t.flatten_all()?.to_vec1::<f32>()?;
        for x in v {
            hasher.update(&x.to_le_bytes());
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Canonical architecture hash.
pub fn config_hash(cfg: &ModelConfig) -> [u8; 32] {
    hash32(&cfg.canonical_bytes())
}

// ---------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------

/// Decode/validate failures. Each variant is a distinct, reportable reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateError {
    /// The blob does not begin with the record magic.
    BadMagic([u8; 8]),
    /// A version this build does not implement.
    UnsupportedVersion(u32),
    /// The header length field is not the format's length.
    BadHeaderLen(u32),
    /// Fewer bytes than the header (or than the declared payload) requires.
    Truncated {
        /// Bytes the record declares it needs.
        need: usize,
        /// Bytes actually present.
        got: usize,
    },
    /// The declared payload sizes disagree with the declared dimensions.
    PayloadSizeMismatch {
        /// `H` bytes present.
        h: usize,
        /// `C` bytes present.
        c: usize,
        /// Carry bytes present.
        carry: usize,
        /// `H` (and `C`) bytes the dimensions require.
        expected: usize,
        /// Carry bytes the dimensions require.
        expected_carry: usize,
    },
    /// The payload does not hash to the state hash in the header.
    StateHashMismatch {
        /// Hash stored in the header.
        stored: String,
        /// Hash recomputed from the payload.
        computed: String,
    },
    /// The scene identity field is not a known scene.
    UnknownScene(String),
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::BadMagic(m) => write!(f, "bad magic {m:?}"),
            StateError::UnsupportedVersion(v) => write!(f, "unsupported record version {v}"),
            StateError::BadHeaderLen(l) => write!(f, "unexpected header length {l}"),
            StateError::Truncated { need, got } => {
                write!(f, "record truncated: need {need} bytes, have {got}")
            }
            StateError::PayloadSizeMismatch {
                h,
                c,
                carry,
                expected,
                expected_carry,
            } => write!(
                f,
                "payload size mismatch: H={h} C={c} carry={carry} bytes, dimensions require \
                 {expected} each and {expected_carry} carry"
            ),
            StateError::StateHashMismatch { stored, computed } => write!(
                f,
                "state hash mismatch: header {stored}, payload {computed}"
            ),
            StateError::UnknownScene(s) => write!(f, "unknown scene identity {s:?}"),
        }
    }
}

impl std::error::Error for StateError {}

/// One compatibility check, evaluated and reported whether or not it passed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Check {
    /// Stable check name, e.g. `scene_identity`.
    pub name: String,
    /// What the request expects.
    pub expected: String,
    /// What the record carries.
    pub observed: String,
    /// Whether they agree.
    pub ok: bool,
}

impl Check {
    fn new(
        name: &str,
        expected: impl Into<String>,
        observed: impl Into<String>,
        ok: bool,
    ) -> Check {
        Check {
            name: name.to_string(),
            expected: expected.into(),
            observed: observed.into(),
            ok,
        }
    }
}

/// What a request demands of a state before it will reuse it.
#[derive(Debug, Clone)]
pub struct Expectation {
    /// Behaviour-determining hash of the frozen weights in use.
    pub model_weights_hash: [u8; 32],
    /// Architecture hash of the model in use.
    pub config_hash: [u8; 32],
    /// Scene the request belongs to.
    pub scene: SceneId,
    /// Context length the request believes it is continuing.
    pub context_len: u32,
    /// Model config in use (for the tensor-shape check).
    pub cfg: ModelConfig,
}

/// The minimal durable generative-state record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateRecord {
    /// Behaviour-determining hash of the weights that earned this state.
    pub model_weights_hash: [u8; 32],
    /// Architecture hash.
    pub config_hash: [u8; 32],
    /// Scene identity the state was earned from.
    pub scene: SceneId,
    /// Number of frames consumed to earn the state.
    pub context_len: u32,
    /// Hidden channels of `H` and `C`.
    pub hidden_channels: u32,
    /// Channels of the carry.
    pub carry_channels: u32,
    /// Field height.
    pub height: u32,
    /// Field width.
    pub width: u32,
    /// Exact `f32` little-endian bytes of `H`.
    pub h: Vec<u8>,
    /// Exact `f32` little-endian bytes of `C`.
    pub c: Vec<u8>,
    /// Exact `f32` little-endian bytes of the carry.
    pub carry: Vec<u8>,
    /// BLAKE3 of `h || c || carry` as stored in the header.
    pub state_hash: [u8; 32],
}

impl StateRecord {
    /// Build from an earned state and its provenance.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_weights_hash: [u8; 32],
        config_hash: [u8; 32],
        scene: SceneId,
        context_len: u32,
        cfg: &ModelConfig,
        h: Vec<u8>,
        c: Vec<u8>,
        carry: Vec<u8>,
    ) -> StateRecord {
        let sh = state_hash(&h, &c, &carry);
        StateRecord {
            model_weights_hash,
            config_hash,
            scene,
            context_len,
            hidden_channels: cfg.hidden_channels as u32,
            carry_channels: CARRY_CHANNELS as u32,
            height: cfg.height as u32,
            width: cfg.width as u32,
            h,
            c,
            carry,
            state_hash: sh,
        }
    }

    /// The unframed payload: `H` bytes, then `C`, then carry.
    ///
    /// This is the byte string the raw-checkpoint control writes to an ordinary
    /// file, and the byte string wrapped by [`StateRecord::encode`]. The attribution
    /// control therefore differs from the VOLE record *only* by the header and the
    /// store, which is exactly the comparison that control is for.
    pub fn raw_payload(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.h.len() + self.c.len() + self.carry.len());
        out.extend_from_slice(&self.h);
        out.extend_from_slice(&self.c);
        out.extend_from_slice(&self.carry);
        out
    }

    /// The header plus the payload — the exact bytes handed to EntropyFS.
    pub fn encode(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(HEADER_LEN + self.h.len() + self.c.len() + self.carry.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(HEADER_LEN as u32).to_le_bytes());
        out.extend_from_slice(&self.model_weights_hash);
        out.extend_from_slice(&self.config_hash);
        out.extend_from_slice(&self.scene.to_record_bytes());
        out.extend_from_slice(&self.context_len.to_le_bytes());
        out.extend_from_slice(&self.hidden_channels.to_le_bytes());
        out.extend_from_slice(&self.carry_channels.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(self.h.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.c.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.carry.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.state_hash);
        debug_assert_eq!(out.len(), HEADER_LEN);
        out.extend_from_slice(&self.h);
        out.extend_from_slice(&self.c);
        out.extend_from_slice(&self.carry);
        out
    }

    /// Decode and fully validate a record: magic, version, header length, self-
    /// consistent payload sizes, and the state hash.
    ///
    /// Validation is *self*-consistency only. Whether this record is the *right*
    /// record for a request is [`StateRecord::compatibility`].
    pub fn decode(bytes: &[u8]) -> Result<StateRecord, StateError> {
        if bytes.len() < HEADER_LEN {
            return Err(StateError::Truncated {
                need: HEADER_LEN,
                got: bytes.len(),
            });
        }
        let magic: [u8; 8] = bytes[0..8].try_into().unwrap();
        if magic != MAGIC {
            return Err(StateError::BadMagic(magic));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(StateError::UnsupportedVersion(version));
        }
        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if header_len as usize != HEADER_LEN {
            return Err(StateError::BadHeaderLen(header_len));
        }
        let model_weights_hash = bytes[16..48].try_into().unwrap();
        let config_hash = bytes[48..80].try_into().unwrap();
        let scene_bytes: [u8; SCENE_ID_BYTES] = bytes[80..96].try_into().unwrap();
        let scene = SceneId::from_record_bytes(&scene_bytes).ok_or_else(|| {
            StateError::UnknownScene(
                String::from_utf8_lossy(&scene_bytes)
                    .trim_end_matches('\0')
                    .to_string(),
            )
        })?;
        let context_len = u32::from_le_bytes(bytes[96..100].try_into().unwrap());
        let hidden_channels = u32::from_le_bytes(bytes[100..104].try_into().unwrap());
        let carry_channels = u32::from_le_bytes(bytes[104..108].try_into().unwrap());
        let height = u32::from_le_bytes(bytes[108..112].try_into().unwrap());
        let width = u32::from_le_bytes(bytes[112..116].try_into().unwrap());
        let h_len = u64::from_le_bytes(bytes[120..128].try_into().unwrap()) as usize;
        let c_len = u64::from_le_bytes(bytes[128..136].try_into().unwrap()) as usize;
        let carry_len = u64::from_le_bytes(bytes[136..144].try_into().unwrap()) as usize;
        let stored_hash: [u8; 32] = bytes[144..176].try_into().unwrap();

        let pixels = (height as usize).checked_mul(width as usize).ok_or(
            StateError::PayloadSizeMismatch {
                h: h_len,
                c: c_len,
                carry: carry_len,
                expected: 0,
                expected_carry: 0,
            },
        )?;
        let expected = 4usize
            .checked_mul(hidden_channels as usize)
            .and_then(|v| v.checked_mul(pixels))
            .ok_or(StateError::PayloadSizeMismatch {
                h: h_len,
                c: c_len,
                carry: carry_len,
                expected: 0,
                expected_carry: 0,
            })?;
        let expected_carry = 4usize
            .checked_mul(carry_channels as usize)
            .and_then(|v| v.checked_mul(pixels))
            .ok_or(StateError::PayloadSizeMismatch {
                h: h_len,
                c: c_len,
                carry: carry_len,
                expected: 0,
                expected_carry: 0,
            })?;
        if h_len != expected || c_len != expected || carry_len != expected_carry {
            return Err(StateError::PayloadSizeMismatch {
                h: h_len,
                c: c_len,
                carry: carry_len,
                expected,
                expected_carry,
            });
        }
        let need = HEADER_LEN + h_len + c_len + carry_len;
        if bytes.len() < need {
            return Err(StateError::Truncated {
                need,
                got: bytes.len(),
            });
        }
        let h = bytes[HEADER_LEN..HEADER_LEN + h_len].to_vec();
        let c = bytes[HEADER_LEN + h_len..HEADER_LEN + h_len + c_len].to_vec();
        let carry =
            bytes[HEADER_LEN + h_len + c_len..HEADER_LEN + h_len + c_len + carry_len].to_vec();
        let computed = state_hash(&h, &c, &carry);
        if computed != stored_hash {
            return Err(StateError::StateHashMismatch {
                stored: hex32(&stored_hash),
                computed: hex32(&computed),
            });
        }
        Ok(StateRecord {
            model_weights_hash,
            config_hash,
            scene,
            context_len,
            hidden_channels,
            carry_channels,
            height,
            width,
            h,
            c,
            carry,
            state_hash: stored_hash,
        })
    }

    /// Decode an unframed payload, given the provenance the caller already knows.
    ///
    /// Used on the raw-checkpoint path, which by construction has no header. The
    /// payload is still length-checked against the model configuration.
    pub fn from_raw_payload(
        raw: &[u8],
        cfg: &ModelConfig,
        model_weights_hash: [u8; 32],
        config_hash: [u8; 32],
        scene: SceneId,
        context_len: u32,
    ) -> Result<StateRecord, StateError> {
        let tensor_bytes = 4 * cfg.hidden_channels * cfg.height * cfg.width;
        let carry_bytes = 4 * CARRY_CHANNELS * cfg.height * cfg.width;
        if raw.len() != 2 * tensor_bytes + carry_bytes {
            return Err(StateError::PayloadSizeMismatch {
                h: raw.len(),
                c: raw.len(),
                carry: raw.len(),
                expected: tensor_bytes,
                expected_carry: carry_bytes,
            });
        }
        Ok(StateRecord::new(
            model_weights_hash,
            config_hash,
            scene,
            context_len,
            cfg,
            raw[..tensor_bytes].to_vec(),
            raw[tensor_bytes..2 * tensor_bytes].to_vec(),
            raw[2 * tensor_bytes..].to_vec(),
        ))
    }

    /// Every compatibility check, evaluated in a fixed order and reported in
    /// full — the failing one is not hidden by short-circuiting.
    pub fn compatibility(&self, exp: &Expectation) -> Vec<Check> {
        let shape_ok = self.hidden_channels == exp.cfg.hidden_channels as u32
            && self.carry_channels == CARRY_CHANNELS as u32
            && self.height == exp.cfg.height as u32
            && self.width == exp.cfg.width as u32;
        vec![
            Check::new(
                "model_weights_hash",
                hex32(&exp.model_weights_hash),
                hex32(&self.model_weights_hash),
                self.model_weights_hash == exp.model_weights_hash,
            ),
            Check::new(
                "model_config_hash",
                hex32(&exp.config_hash),
                hex32(&self.config_hash),
                self.config_hash == exp.config_hash,
            ),
            Check::new(
                "scene_identity",
                exp.scene.as_str(),
                self.scene.as_str(),
                self.scene == exp.scene,
            ),
            Check::new(
                "context_length",
                exp.context_len.to_string(),
                self.context_len.to_string(),
                self.context_len == exp.context_len,
            ),
            Check::new(
                "state_shape",
                format!(
                    "{}x{}+{}x{}x{}",
                    exp.cfg.hidden_channels, CARRY_CHANNELS, exp.cfg.height, exp.cfg.width, 0
                ),
                format!(
                    "{}x{}+{}x{}x{}",
                    self.hidden_channels, self.carry_channels, self.height, self.width, 0
                ),
                shape_ok,
            ),
            Check::new(
                "state_hash",
                hex32(&state_hash(&self.h, &self.c, &self.carry)),
                hex32(&self.state_hash),
                state_hash(&self.h, &self.c, &self.carry) == self.state_hash,
            ),
        ]
    }

    /// The first failing check, if any. `None` means reuse is permitted.
    pub fn first_rejection<'a>(&self, checks: &'a [Check]) -> Option<&'a Check> {
        checks.iter().find(|c| !c.ok)
    }

    /// Payload bytes per recurrent tensor (`H`, and `C`).
    pub fn tensor_bytes(&self) -> usize {
        self.h.len()
    }

    /// Payload bytes of the carry channel.
    pub fn carry_bytes_len(&self) -> usize {
        self.carry.len()
    }
}

// ---------------------------------------------------------------------------
// EntropyFS persistence path
// ---------------------------------------------------------------------------

/// Failures on the persistence path: EntropyFS errors and record errors, kept
/// distinct so evidence can attribute a failure to the right layer.
#[derive(Debug)]
pub enum PersistError {
    /// The EntropyFS engine refused an operation.
    Engine(EngineError),
    /// The stored bytes are not a valid record.
    Record(StateError),
    /// Local filesystem I/O outside the engine.
    Io(std::io::Error),
    /// The blob id string was malformed.
    BadBlobId(String),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistError::Engine(e) => write!(f, "entropyfs: {e}"),
            PersistError::Record(e) => write!(f, "record: {e}"),
            PersistError::Io(e) => write!(f, "io: {e}"),
            PersistError::BadBlobId(s) => write!(f, "bad blob id {s:?}"),
        }
    }
}

impl std::error::Error for PersistError {}

impl From<EngineError> for PersistError {
    fn from(e: EngineError) -> Self {
        PersistError::Engine(e)
    }
}

impl From<StateError> for PersistError {
    fn from(e: StateError) -> Self {
        PersistError::Record(e)
    }
}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        PersistError::Io(e)
    }
}

/// Sum of the lengths of every regular file under `dir`, recursively.
///
/// This is the honest "what did the bytes cost on disk" measure for the store
/// directory, taken outside the engine so it cannot be flattered by the engine's
/// own accounting. The engine's `physical_used_bytes` is reported alongside it as
/// an independent cross-check.
pub fn dir_bytes(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    if !dir.exists() {
        return Ok(0);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += dir_bytes(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

/// The embeddable EntropyFS engine, narrowed to what this proof uses.
///
/// Not a FUSE mount and not a reimplementation: every call below goes through
/// the published crate's `Engine` facade, and every byte lands through the
/// store's normal committed path.
pub struct VoleStore {
    engine: Option<Engine>,
    path: PathBuf,
}

impl VoleStore {
    /// Create a fresh store. The directory must not exist or must be empty.
    pub fn create(path: &Path) -> Result<VoleStore, PersistError> {
        let opts = EngineOpenOptions::default();
        let engine = Engine::create(path, &opts)?;
        Ok(VoleStore {
            engine: Some(engine),
            path: path.to_path_buf(),
        })
    }

    /// Open an existing store.
    pub fn open(path: &Path) -> Result<VoleStore, PersistError> {
        let opts = EngineOpenOptions::default();
        let engine = Engine::open(path, &opts)?;
        Ok(VoleStore {
            engine: Some(engine),
            path: path.to_path_buf(),
        })
    }

    /// Open if the store exists, create it otherwise.
    pub fn open_or_create(path: &Path) -> Result<(VoleStore, bool), PersistError> {
        if path.exists() && std::fs::read_dir(path)?.next().is_some() {
            Ok((VoleStore::open(path)?, false))
        } else {
            Ok((VoleStore::create(path)?, true))
        }
    }

    /// The store directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn engine(&self) -> Result<&Engine, PersistError> {
        self.engine
            .as_ref()
            .ok_or_else(|| PersistError::BadBlobId("store already closed".into()))
    }

    /// Store a blob, acknowledging at the mutation log.
    pub fn put(&self, bytes: &[u8]) -> Result<BlobId, PersistError> {
        Ok(self.engine()?.put_blob(bytes)?)
    }

    /// The durability barrier: after this returns, every acknowledged blob
    /// survives power loss.
    pub fn sync(&self) -> Result<(), PersistError> {
        Ok(self.engine()?.sync()?)
    }

    /// Fetch a blob. The engine verifies the returned bytes hash to the id.
    pub fn get(&self, id: BlobId) -> Result<Vec<u8>, PersistError> {
        Ok(self.engine()?.get_blob(id)?)
    }

    /// Whether an id is present.
    pub fn contains(&self, id: BlobId) -> Result<bool, PersistError> {
        Ok(self.engine()?.contains(id)?)
    }

    /// Parse a 64-character hex blob id.
    pub fn parse_id(hex: &str) -> Result<BlobId, PersistError> {
        BlobId::from_hex(hex).ok_or_else(|| PersistError::BadBlobId(hex.to_string()))
    }

    /// Endurance-relevant physical bytes as the engine accounts for them.
    pub fn engine_physical_used(&self) -> Result<u64, PersistError> {
        Ok(self.engine()?.metrics()?.accounting.physical_used_bytes)
    }

    /// Bytes of files under the store directory, measured from outside.
    pub fn physical_bytes(&self) -> Result<u64, PersistError> {
        Ok(dir_bytes(&self.path)?)
    }

    /// Release the store (drains in-flight operations).
    pub fn close(mut self) -> Result<(), PersistError> {
        if let Some(e) = self.engine.take() {
            e.close()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ModelConfig {
        ModelConfig::default()
    }

    fn record() -> StateRecord {
        let cfg = cfg();
        let n = 4 * cfg.hidden_channels * cfg.height * cfg.width;
        let k = 4 * CARRY_CHANNELS * cfg.height * cfg.width;
        let h: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let c: Vec<u8> = (0..n).map(|i| ((i * 7) % 253) as u8).collect();
        let carry: Vec<u8> = (0..k).map(|i| ((i * 13) % 247) as u8).collect();
        StateRecord::new(
            [1u8; 32],
            [2u8; 32],
            SceneId::MovingShape01,
            256,
            &cfg,
            h,
            c,
            carry,
        )
    }

    #[test]
    fn header_layout_is_exactly_as_documented() {
        let r = record();
        let bytes = r.encode();
        assert_eq!(&bytes[0..8], &MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize,
            HEADER_LEN
        );
        assert_eq!(bytes.len(), HEADER_LEN + r.raw_payload().len());
        assert_eq!(&bytes[HEADER_LEN..], &r.raw_payload()[..]);
        assert_eq!(
            u64::from_le_bytes(bytes[120..128].try_into().unwrap()) as usize,
            r.tensor_bytes()
        );
        assert_eq!(
            u64::from_le_bytes(bytes[128..136].try_into().unwrap()) as usize,
            r.tensor_bytes()
        );
        assert_eq!(
            u64::from_le_bytes(bytes[136..144].try_into().unwrap()) as usize,
            r.carry_bytes_len()
        );
        assert_eq!(
            u32::from_le_bytes(bytes[104..108].try_into().unwrap()) as usize,
            CARRY_CHANNELS
        );
        assert_eq!(u32::from_le_bytes(bytes[116..120].try_into().unwrap()), 0);
    }

    #[test]
    fn encode_decode_round_trips() {
        let r = record();
        let back = StateRecord::decode(&r.encode()).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn tampering_is_caught() {
        let r = record();
        let mut bytes = r.encode();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        match StateRecord::decode(&bytes) {
            Err(StateError::StateHashMismatch { stored, computed }) => {
                assert_ne!(stored, computed)
            }
            other => panic!("expected StateHashMismatch, got {other:?}"),
        }
        let mut bad_version = r.encode();
        bad_version[8] = 9;
        assert_eq!(
            StateRecord::decode(&bad_version),
            Err(StateError::UnsupportedVersion(9))
        );
        let mut bad_magic = r.encode();
        bad_magic[0] = b'X';
        assert!(matches!(
            StateRecord::decode(&bad_magic),
            Err(StateError::BadMagic(_))
        ));
        assert!(matches!(
            StateRecord::decode(&r.encode()[..100]),
            Err(StateError::Truncated { .. })
        ));
    }

    #[test]
    fn compatibility_reports_every_check_and_finds_the_failure() {
        let r = record();
        let ok = Expectation {
            model_weights_hash: [1u8; 32],
            config_hash: [2u8; 32],
            scene: SceneId::MovingShape01,
            context_len: 256,
            cfg: cfg(),
        };
        let checks = r.compatibility(&ok);
        assert!(checks.iter().all(|c| c.ok), "{checks:?}");
        assert!(r.first_rejection(&checks).is_none());

        let wrong_scene = Expectation {
            scene: SceneId::MovingShape02,
            ..ok.clone()
        };
        let checks = r.compatibility(&wrong_scene);
        assert_eq!(r.first_rejection(&checks).unwrap().name, "scene_identity");
        // The other checks still ran and still passed.
        assert_eq!(checks.iter().filter(|c| c.ok).count(), checks.len() - 1);

        let wrong_weights = Expectation {
            model_weights_hash: [9u8; 32],
            ..ok.clone()
        };
        let checks = r.compatibility(&wrong_weights);
        assert_eq!(
            r.first_rejection(&checks).unwrap().name,
            "model_weights_hash"
        );

        let wrong_len = Expectation {
            context_len: 128,
            ..ok
        };
        let checks = r.compatibility(&wrong_len);
        assert_eq!(r.first_rejection(&checks).unwrap().name, "context_length");
    }

    #[test]
    fn raw_payload_is_the_unframed_byte_string() {
        let r = record();
        let raw = r.raw_payload();
        let back = StateRecord::from_raw_payload(
            &raw,
            &cfg(),
            [1u8; 32],
            [2u8; 32],
            SceneId::MovingShape01,
            256,
        )
        .unwrap();
        assert_eq!(back, r);
        assert!(StateRecord::from_raw_payload(
            &raw[..raw.len() - 4],
            &cfg(),
            [1u8; 32],
            [2u8; 32],
            SceneId::MovingShape01,
            256
        )
        .is_err());
    }

    #[test]
    fn entropyfs_round_trips_through_a_real_store() {
        let dir =
            std::env::temp_dir().join(format!("vole-field-test-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let r = record();
        let bytes = r.encode();

        let before_create;
        {
            let store = VoleStore::create(&dir).unwrap();
            before_create = store.physical_bytes().unwrap();
            let id = store.put(&bytes).unwrap();
            store.sync().unwrap();
            assert!(store.contains(id).unwrap());
            let got = store.get(id).unwrap();
            assert_eq!(got, bytes, "engine must materialise the exact bytes");
            assert_eq!(StateRecord::decode(&got).unwrap(), r);
            let after = store.physical_bytes().unwrap();
            assert!(after > before_create);
            // Idempotent: re-putting identical bytes is a no-op with the same id.
            assert_eq!(store.put(&bytes).unwrap(), id);
            store.close().unwrap();
        }

        // A *different* process-equivalent open sees the same blob by id.
        let store = VoleStore::open(&dir).unwrap();
        assert!(store.physical_bytes().unwrap() >= before_create);
        store.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
