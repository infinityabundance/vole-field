//! `vole-field` — a minimal, rigorous proof that *useful generative state earned
//! by an open local model can survive process termination, be persisted through
//! EntropyFS, be restored later, and reduce repeated model computation for
//! related future observations while preserving the same model-output contract.*
//!
//! Paper context: VOLE-Field — Deterministic Multimodal Field-State
//! Representation, Inverse Compilation, Entropy-Native Persistence, and Late
//! Materialization (DOI 10.5281/zenodo.22805773).
//!
//! # What this crate is
//!
//! One package, one tiny recurrent generative video model, one boring scene, one
//! persisted state, a few related future requests, one from-scratch baseline, one
//! raw-state attribution control, one honest negative case, one Colab notebook.
//!
//! # What this crate is not
//!
//! Not a framework, not a runtime, not a full implementation of the paper, not a
//! benchmark suite. It demonstrates *mechanism feasibility* and nothing more; the
//! README lists precisely what it does not show.
//!
//! # Reading order
//!
//! 1. [`scene`] — the deterministic scene and the related future requests.
//! 2. [`model`] — the tiny ConvLSTM and the exact `(H, C)` recurrent state.
//! 3. [`state`] — the minimal VOLE state record, its compatibility predicate, and
//!    the EntropyFS persistence path.
//! 4. [`experiment`] — the process-cold experiment and its measurements.
//! 5. [`train`] — the offline trainer that produced the frozen checkpoint.

pub mod experiment;
pub mod model;
pub mod scene;
pub mod state;
pub mod train;
