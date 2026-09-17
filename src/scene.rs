//! The synthetic scene: one bright shape moving over a dark 32x32 field, plus a
//! small discrete control input.
//!
//! # Determinism contract
//!
//! The scene generator uses only IEEE-754 binary32 `+`, `-`, `*`, `/`, `min`
//! and `max`, plus f32 constants written into the source. It calls **no**
//! transcendental function at run time, allocates no randomness at run time, and
//! reads no clock. Given a [`SceneSpec`] and a control schedule, the frame
//! sequence is therefore a pure function of those inputs, bit-identical in every
//! process on every correctly-rounded binary32 platform.
//!
//! This is a load-bearing property, not a nicety: the *baseline* path
//! regenerates from scratch the same context the *producer* consumed. If
//! regeneration were not bit-exact the two paths would not be comparing like
//! with like, and the whole experiment would be vacuous.
//!
//! # The scene
//!
//! One compactly-supported quartic blob kernel,
//!
//! ```text
//! k(d) = max(0, 1 - d^2 / r^2)^2
//! ```
//!
//! rendered at a sub-pixel position over a zero background. The body carries a
//! unit direction vector and a scalar speed, so per-step motion costs one
//! rotation (by a precomputed cos/sin pair) and one multiply-add. The control
//! input is discrete: `turn` and `accel` each in `{-1, 0, +1}`.
//!
//! # Why this shape of task
//!
//! A single rendered frame does *not* reveal the body's velocity: the velocity
//! lives in the difference between consecutive frames. Any predictor that gets
//! the next frame right must therefore carry an inferred motion estimate across
//! time. That is exactly the state this proof persists, so the experiment is
//! about earned recurrent state rather than about a stateless frame function.

/// Field height in pixels. Deliberately tiny.
pub const H: usize = 32;
/// Field width in pixels.
pub const W: usize = 32;
/// Pixels per frame.
pub const HW: usize = H * W;
/// Fixed width, in bytes, of a scene identity inside the VOLE state record.
pub const SCENE_ID_BYTES: usize = 16;

/// Rotation applied per unit of `turn`, in radians.
///
/// Deliberately large. The control has to remain *visible* through a small,
/// imperfect model: the divergence a branch shows is the divergence the objective
/// taught it, and a control whose ground-truth effect is a few tenths of a pixel
/// over the horizon cannot teach a visible one. At 0.18 rad per step a held turn
/// sweeps more than a right angle across the generation horizon.
pub const TURN_ANGLE: f32 = 0.18;
/// `cos(TURN_ANGLE)` as a source literal.
///
/// A source literal, not a runtime call, so the scene cannot depend on the
/// platform's libm. `turn_constants_match_libm` checks that the literal equals the
/// rounded `f32` cosine on the build platform, which keeps the provenance of the
/// constant honest and catches typos.
pub const TURN_COS: f32 = 0.983_843_7;
/// `sin(TURN_ANGLE)` as a source literal. See [`TURN_COS`].
pub const TURN_SIN: f32 = 0.179_029_58;

/// Base speed in pixels per frame, held when the throttle word is zero.
pub const SPEED_BASE: f32 = 0.55;
/// Fractional speed change per unit of `accel`. Toppings: `+1` gives
/// [`SPEED_BASE`]`* 1.28`, `-1` gives `[`SPEED_BASE`]`* 0.72`.
pub const ACCEL_STEP: f32 = 0.28;
/// Slowest permitted speed, in pixels per frame.
///
/// Kept sub-pixel on purpose. A single 3x3 convolution can move a feature by at most
/// one pixel, so a body moving faster than that cannot be represented by a shift of
/// its own pattern and the tiny model would be asked to do something its architecture
/// forbids. Keeping the motion sub-pixel makes the task *representable*, which is the
/// only thing this proof needs from the model.
pub const SPEED_MIN: f32 = 0.32;
/// Fastest permitted speed, in pixels per frame. See [`SPEED_MIN`].
pub const SPEED_MAX: f32 = 0.78;

/// Semi-axis along the heading, in pixels. The body is drawn as a streak.
///
/// Sized so that the object occupies a meaningful fraction of the field. A per-pixel
/// objective on a mostly-empty frame barely notices whether a small object moved, so
/// an object that is too small makes motion effectively free to get wrong.
pub const RADIUS_ALONG: f32 = 5.6;
/// Semi-axis across the heading, in pixels. The aspect ratio is what makes the
/// heading visible in a single frame.
pub const RADIUS_ACROSS: f32 = 2.4;

/// Distance from a wall at which a body reflects, in pixels.
pub const WALL_MARGIN: f32 = 7.0;

/// `x` positions are reflected into `[WALL_MARGIN, X_HI]`.
const X_HI: f32 = (W as f32 - 1.0) - WALL_MARGIN;
/// `y` positions are reflected into `[WALL_MARGIN, Y_HI]`.
const Y_HI: f32 = (H as f32 - 1.0) - WALL_MARGIN;

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

/// One discrete control word. Both fields are in `{-1, 0, +1}`.
///
/// `turn` rotates the body's direction by `turn * TURN_ANGLE` in screen coordinates
/// (`+x` right, `+y` down), so `turn = +1` curves the trajectory toward screen-right
/// and `turn = -1` toward screen-left.
///
/// `accel` sets the body's speed *for that step only*: `speed = SPEED_BASE *
/// (1 + accel * ACCEL_STEP)`, clamped. It is deliberately memoryless. Integrating
/// the throttle would make the current speed a hidden scalar that a predictor has to
/// carry as a register inferred from hundreds of frames, and that is precisely the
/// credit-assignment problem a 3,273-parameter convolutional recurrence does not
/// solve. A held control word, by contrast, is present in the input at *every* step,
/// so a memoryless speed is directly readable by the model and the task stays
/// representable. The scene remains deterministic, and a held `accel` still produces
/// a visibly different trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Control {
    /// Steering: `-1`, `0` or `+1`.
    pub turn: i8,
    /// Throttle: `-1`, `0` or `+1`.
    pub accel: i8,
}

impl Control {
    /// No steering, no throttle change.
    pub const NONE: Control = Control { turn: 0, accel: 0 };

    /// Construct from raw values. The discrete domain is an invariant.
    pub fn new(turn: i8, accel: i8) -> Control {
        assert!(
            (-1..=1).contains(&turn) && (-1..=1).contains(&accel),
            "control words are discrete: turn/accel in {{-1,0,+1}}"
        );
        Control { turn, accel }
    }

    /// The two constant input planes appended to the frame, in the exact order
    /// the model expects: `[turn, accel]`.
    pub fn planes(&self) -> [f32; 2] {
        [self.turn as f32, self.accel as f32]
    }

    /// Dense code in `0..9`, useful for prose and evidence.
    pub fn code(&self) -> u8 {
        ((self.turn + 1) * 3 + (self.accel + 1)) as u8
    }
}

// ---------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------

/// One rendered body: sub-pixel position, unit direction, scalar speed.
///
/// The body is drawn as an ellipse elongated along its direction of travel — a motion
/// blur. That is a deliberate choice with a measured justification: the heading is a
/// *global* property of the body, and a 3x3 convolution can only move or deform a
/// pattern locally. With a circular body the heading is invisible in every individual
/// frame, so the network has to invent and maintain a hidden orientation register, and
/// it never learns the control's effect on it. Rendering the heading into the frame
/// turns a change of heading into a bounded, local deformation of a visible pattern —
/// exactly what a convolution can represent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Body {
    /// Sub-pixel centre, x.
    pub x: f32,
    /// Sub-pixel centre, y.
    pub y: f32,
    /// Unit direction, x (the heading, also the ellipse's major axis).
    pub dir_x: f32,
    /// Unit direction, y.
    pub dir_y: f32,
    /// Speed in pixels per frame.
    pub speed: f32,
    /// Peak brightness in `[0, 1]`.
    pub amplitude: f32,
}

/// The speed a control word commands. Memoryless by construction.
pub fn commanded_speed(accel: i8) -> f32 {
    let v = SPEED_BASE * (1.0 + accel as f32 * ACCEL_STEP);
    v.clamp(SPEED_MIN, SPEED_MAX)
}

impl Body {
    /// Advance one frame: apply the control, move, reflect off the walls.
    ///
    /// `u` is the control held *during* the transition, so it is the input that
    /// the model is given alongside the current frame to predict the next one.
    pub fn advance(&mut self, u: Control) {
        // Steering: rotate the unit direction by (turn * TURN_ANGLE).
        if u.turn != 0 {
            let (c, s) = if u.turn > 0 {
                (TURN_COS, TURN_SIN)
            } else {
                (TURN_COS, -TURN_SIN)
            };
            let (dx, dy) = (self.dir_x, self.dir_y);
            self.dir_x = c * dx - s * dy;
            self.dir_y = s * dx + c * dy;
        }
        // Throttle: a memoryless set-point, readable from the control word itself.
        self.speed = commanded_speed(u.accel);

        // Move.
        self.x += self.speed * self.dir_x;
        self.y += self.speed * self.dir_y;

        // Reflect off the walls so a long context never loses the body.
        if self.x < WALL_MARGIN {
            self.x = WALL_MARGIN + (WALL_MARGIN - self.x);
            self.dir_x = -self.dir_x;
        } else if self.x > X_HI {
            self.x = X_HI - (self.x - X_HI);
            self.dir_x = -self.dir_x;
        }
        if self.y < WALL_MARGIN {
            self.y = WALL_MARGIN + (WALL_MARGIN - self.y);
            self.dir_y = -self.dir_y;
        } else if self.y > Y_HI {
            self.y = Y_HI - (self.y - Y_HI);
            self.dir_y = -self.dir_y;
        }
    }

    /// Accumulate this body's contribution into `out` (length [`HW`]).
    ///
    /// The kernel is a compactly supported quartic evaluated in the body's own
    /// frame: `(1 - r²)²` with
    ///
    /// ```text
    /// r² = (along² / radius_along²) + (across² / radius_across²)
    /// ```
    ///
    /// where `along` and `across` are the pixel offsets projected onto the heading
    /// and its perpendicular. No square root, no transcendental function.
    pub fn render_into(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), HW);
        let inv_a2 = 1.0 / (RADIUS_ALONG * RADIUS_ALONG);
        let inv_b2 = 1.0 / (RADIUS_ACROSS * RADIUS_ACROSS);
        for yy in 0..H {
            let dy = yy as f32 - self.y;
            let row = yy * W;
            for xx in 0..W {
                let dx = xx as f32 - self.x;
                let along = dx * self.dir_x + dy * self.dir_y;
                let across = dy * self.dir_x - dx * self.dir_y;
                let r2 = along * along * inv_a2 + across * across * inv_b2;
                if r2 < 1.0 {
                    let t = 1.0 - r2;
                    let v = t * t * self.amplitude;
                    let slot = &mut out[row + xx];
                    *slot = (*slot + v).min(1.0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scene identity and specs
// ---------------------------------------------------------------------------

/// The closed set of scene identities this proof knows about.
///
/// Scene identity is a *first-class part of the VOLE state record*, not a
/// comment: it is what makes the negative case in this experiment possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneId {
    /// `moving-shape-01`: the canonical scene — one blob, gentle S-curve context.
    MovingShape01,
    /// `moving-shape-02`: an unrelated scene — two blobs on crossing paths.
    /// Exists solely to be the honest negative case.
    MovingShape02,
}

impl SceneId {
    /// The 15-character identity string carried in the state record.
    pub fn as_str(&self) -> &'static str {
        match self {
            SceneId::MovingShape01 => "moving-shape-01",
            SceneId::MovingShape02 => "moving-shape-02",
        }
    }

    /// Parse an identity string. Exact match only.
    pub fn parse(s: &str) -> Option<SceneId> {
        match s {
            "moving-shape-01" => Some(SceneId::MovingShape01),
            "moving-shape-02" => Some(SceneId::MovingShape02),
            _ => None,
        }
    }

    /// The identity as fixed-width NUL-padded bytes for the state record.
    pub fn to_record_bytes(&self) -> [u8; SCENE_ID_BYTES] {
        let mut out = [0u8; SCENE_ID_BYTES];
        let s = self.as_str().as_bytes();
        assert!(
            s.len() < SCENE_ID_BYTES,
            "scene id must fit the record field"
        );
        out[..s.len()].copy_from_slice(s);
        out
    }

    /// Decode fixed-width NUL-padded bytes back to an identity.
    pub fn from_record_bytes(bytes: &[u8]) -> Option<SceneId> {
        if bytes.len() != SCENE_ID_BYTES {
            return None;
        }
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(SCENE_ID_BYTES);
        std::str::from_utf8(&bytes[..end])
            .ok()
            .and_then(SceneId::parse)
    }

    /// A one-line human description, for evidence and the killer table.
    pub fn label(&self) -> &'static str {
        match self {
            SceneId::MovingShape01 => "one blob, gentle S-curve, canonical",
            SceneId::MovingShape02 => "two blobs, crossing paths, unrelated",
        }
    }

    /// The maximum body count across the closed set.
    pub const MAX_BODIES: usize = 2;
}

/// A fully specified, deterministic scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneSpec {
    /// Which scene this is.
    pub id: SceneId,
    /// Initial bodies. At most [`SceneId::MAX_BODIES`].
    pub bodies: Vec<Body>,
}

impl SceneSpec {
    /// The canonical scene and the unrelated scene, by identity.
    ///
    /// Initial conditions are source literals so that every process starts from
    /// exactly the same bits.
    pub fn canonical(id: SceneId) -> SceneSpec {
        match id {
            SceneId::MovingShape01 => SceneSpec {
                id,
                bodies: vec![Body {
                    x: 9.0,
                    y: 16.0,
                    dir_x: 1.0,
                    dir_y: 0.0,
                    speed: SPEED_BASE,
                    amplitude: 1.0,
                }],
            },
            SceneId::MovingShape02 => SceneSpec {
                id,
                bodies: vec![
                    Body {
                        x: 22.0,
                        y: 10.0,
                        dir_x: -1.0,
                        dir_y: 0.0,
                        speed: SPEED_BASE,
                        amplitude: 1.0,
                    },
                    Body {
                        x: 10.0,
                        y: 22.0,
                        dir_x: 0.0,
                        dir_y: -1.0,
                        speed: SPEED_BASE,
                        amplitude: 0.8,
                    },
                ],
            },
        }
    }

    /// A randomised single-body scene, used **only** by the offline trainer.
    ///
    /// This is the sole place in the crate that calls transcendental functions
    /// on scene data, and it is never on the inference path: the canonical scenes
    /// above are built from literals, and once the checkpoint is frozen the
    /// trainer is not part of the demo.
    pub fn random(rng: &mut Rng) -> SceneSpec {
        let theta = rng.f32_unit() * std::f32::consts::TAU;
        SceneSpec {
            id: SceneId::MovingShape01,
            bodies: vec![Body {
                x: 10.0 + rng.f32_unit() * 12.0,
                y: 10.0 + rng.f32_unit() * 12.0,
                dir_x: theta.cos(),
                dir_y: theta.sin(),
                speed: SPEED_BASE,
                amplitude: 1.0,
            }],
        }
    }

    /// A mutable working copy of the bodies.
    pub fn working(&self) -> Vec<Body> {
        self.bodies.clone()
    }
}

// ---------------------------------------------------------------------------
// Rendering and rollout
// ---------------------------------------------------------------------------

/// Render the current body configuration into a fresh frame.
pub fn render(bodies: &[Body]) -> Vec<f32> {
    let mut out = vec![0.0f32; HW];
    for b in bodies {
        b.render_into(&mut out);
    }
    out
}

/// Append the control planes of `u` to frame `f`, yielding the model's input
/// layout `[1 + 2, H, W]` as a flat `Vec<f32>`.
pub fn model_input(f: &[f32], u: Control) -> Vec<f32> {
    debug_assert_eq!(f.len(), HW);
    let mut out = Vec::with_capacity(3 * HW);
    out.extend_from_slice(f);
    let [t, a] = u.planes();
    out.extend(std::iter::repeat_n(t, HW));
    out.extend(std::iter::repeat_n(a, HW));
    out
}

/// Roll a scene forward, returning every frame including the initial one.
///
/// `frames.len() == steps + 1`. `control(t)` is the control held during the
/// transition from frame `t` to frame `t + 1`.
pub fn rollout(
    spec: &SceneSpec,
    steps: usize,
    control: impl Fn(usize) -> Control,
) -> Vec<Vec<f32>> {
    let mut bodies = spec.working();
    let mut frames = Vec::with_capacity(steps + 1);
    frames.push(render(&bodies));
    for t in 0..steps {
        let u = control(t);
        for b in bodies.iter_mut() {
            b.advance(u);
        }
        frames.push(render(&bodies));
    }
    frames
}

// ---------------------------------------------------------------------------
// Control schedules (the "related future requests")
// ---------------------------------------------------------------------------

/// A finite list of control words, held on the last word past the end.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    /// Control held at each step, in order.
    pub steps: Vec<Control>,
}

impl Schedule {
    /// The control held at step `t`, clamped to the final word.
    pub fn at(&self, t: usize) -> Control {
        if self.steps.is_empty() {
            Control::NONE
        } else if t < self.steps.len() {
            self.steps[t]
        } else {
            self.steps[self.steps.len() - 1]
        }
    }

    /// Hold one control word forever.
    pub fn constant(u: Control) -> Schedule {
        Schedule { steps: vec![u] }
    }

    /// Build from `(step_from, control)` segments, covering `0..len`.
    pub fn segments(len: usize, segs: &[(usize, Control)]) -> Schedule {
        let mut steps = Vec::with_capacity(len);
        for t in 0..len {
            let mut chosen = Control::NONE;
            for (from, u) in segs {
                if t >= *from {
                    chosen = *u;
                }
            }
            steps.push(chosen);
        }
        Schedule { steps }
    }

    /// The control words as raw evidence, one `code` per step.
    pub fn codes(&self) -> Vec<u8> {
        self.steps.iter().map(|u| u.code()).collect()
    }
}

/// The canonical context program: "the scene the model watches".
///
/// A gentle S-curve that ends travelling straight, so the first generated frame
/// is an honest straight-line extrapolation and a branch control takes effect
/// from the second generated frame onward.
pub fn context_program() -> Schedule {
    Schedule::segments(
        512,
        &[
            (0, Control::NONE),
            (64, Control::new(-1, 0)),
            (112, Control::NONE),
            (160, Control::new(1, 0)),
            (208, Control::NONE),
        ],
    )
}

/// A named related future request. These are the branches of one earned state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Request {
    /// Hold course.
    Continue,
    /// Steady left turn.
    TurnLeft,
    /// Steady right turn.
    TurnRight,
    /// Steady acceleration.
    Accelerate,
    /// Brake for a while, then turn left.
    BrakeThenLeft,
    /// Turn right, then turn left — an S of the future.
    SCurve,
}

impl Request {
    /// All requests, in evidence order.
    pub const ALL: [Request; 6] = [
        Request::Continue,
        Request::TurnLeft,
        Request::TurnRight,
        Request::Accelerate,
        Request::BrakeThenLeft,
        Request::SCurve,
    ];

    /// The four requests drawn in the branch montage.
    pub const MONTAGE: [Request; 4] = [
        Request::Continue,
        Request::TurnLeft,
        Request::TurnRight,
        Request::Accelerate,
    ];

    /// Stable short name (also the CLI value).
    pub fn name(&self) -> &'static str {
        match self {
            Request::Continue => "continue",
            Request::TurnLeft => "turn-left",
            Request::TurnRight => "turn-right",
            Request::Accelerate => "accelerate",
            Request::BrakeThenLeft => "brake-then-left",
            Request::SCurve => "s-curve",
        }
    }

    /// Parse a CLI value.
    pub fn parse(s: &str) -> Option<Request> {
        Request::ALL.iter().copied().find(|r| r.name() == s)
    }

    /// The request's control program, indexed from the first generated step.
    pub fn schedule(&self) -> Schedule {
        match self {
            Request::Continue => Schedule::constant(Control::NONE),
            Request::TurnLeft => Schedule::constant(Control::new(-1, 0)),
            Request::TurnRight => Schedule::constant(Control::new(1, 0)),
            Request::Accelerate => Schedule::constant(Control::new(0, 1)),
            Request::BrakeThenLeft => {
                Schedule::segments(32, &[(0, Control::new(0, -1)), (6, Control::new(-1, 0))])
            }
            Request::SCurve => {
                Schedule::segments(32, &[(0, Control::new(1, 0)), (8, Control::new(-1, 0))])
            }
        }
    }

    /// One-line description for the killer table.
    pub fn label(&self) -> &'static str {
        match self {
            Request::Continue => "hold course",
            Request::TurnLeft => "steady left turn",
            Request::TurnRight => "steady right turn",
            Request::Accelerate => "steady acceleration",
            Request::BrakeThenLeft => "brake 6 steps, then turn left",
            Request::SCurve => "right for 8 steps, then left",
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64) — offline training only
// ---------------------------------------------------------------------------

/// The smallest thing that is still a real PRNG. Used by the trainer's scene
/// sampler; never used on the inference or persistence path.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    /// Next 64 bits (splitmix64).
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform `f32` in `[0, 1)` with 24 bits of resolution.
    pub fn f32_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// A random control word.
    pub fn control(&mut self) -> Control {
        Control {
            turn: (self.below(3) as i8) - 1,
            accel: (self.below(3) as i8) - 1,
        }
    }

    /// A random control schedule.
    ///
    /// Two shapes, because deployment uses both: a single control held for the
    /// whole horizon (the constant requests — hold course, steady turn, steady
    /// throttle) and a run of held segments (brake-then-turn, S-curve). Training
    /// only on the segmented shape measurably weakens the response to the constant
    /// ones, so a third of the samples are pure single-control windows.
    pub fn schedule(&mut self, len: usize) -> Schedule {
        if self.below(3) == 0 {
            let u = self.control();
            return Schedule {
                steps: vec![u; len],
            };
        }
        let mut steps = Vec::with_capacity(len);
        while steps.len() < len {
            let u = self.control();
            let hold = if self.below(4) == 0 {
                20 + self.below(29) as usize
            } else {
                1 + self.below(12) as usize
            };
            for _ in 0..hold.min(len - steps.len()) {
                steps.push(u);
            }
        }
        Schedule { steps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_constants_match_libm() {
        assert_eq!(TURN_COS, TURN_ANGLE.cos());
        assert_eq!(TURN_SIN, TURN_ANGLE.sin());
    }

    #[test]
    fn speed_is_a_memoryless_function_of_the_control_word() {
        assert_eq!(commanded_speed(1), SPEED_BASE * (1.0 + ACCEL_STEP));
        assert_eq!(commanded_speed(0), SPEED_BASE);
        assert_eq!(commanded_speed(-1), SPEED_BASE * (1.0 - ACCEL_STEP));
        assert!(commanded_speed(1) <= SPEED_MAX && commanded_speed(-1) >= SPEED_MIN);
        // A held throttle gives the same speed at every step, which is what makes it
        // readable from the input rather than a hidden register.
        let mut s = SPEED_BASE;
        for _ in 0..10 {
            s = commanded_speed(1);
        }
        assert_eq!(s, commanded_speed(1));
        // And the three throttle settings really are distinguishable.
        assert!(commanded_speed(-1) < commanded_speed(0));
        assert!(commanded_speed(0) < commanded_speed(1));
    }

    #[test]
    fn scene_is_bit_reproducible() {
        let spec = SceneSpec::canonical(SceneId::MovingShape01);
        let a = rollout(&spec, 64, |t| context_program().at(t));
        let b = rollout(&spec, 64, |t| context_program().at(t));
        assert_eq!(a, b);
    }

    #[test]
    fn scene_ids_round_trip_through_record_bytes() {
        for id in [SceneId::MovingShape01, SceneId::MovingShape02] {
            let bytes = id.to_record_bytes();
            assert_eq!(bytes.len(), SCENE_ID_BYTES);
            assert_eq!(SceneId::from_record_bytes(&bytes), Some(id));
        }
        assert_eq!(SceneId::from_record_bytes(&[0u8; SCENE_ID_BYTES]), None);
    }

    #[test]
    fn bodies_stay_inside_the_field() {
        let mut spec = SceneSpec::canonical(SceneId::MovingShape01);
        let sched = Request::SCurve.schedule();
        for t in 0..2000 {
            for b in spec.bodies.iter_mut() {
                b.advance(sched.at(t));
                assert!((0.0..W as f32).contains(&b.x), "x escaped at {t}: {}", b.x);
                assert!((0.0..H as f32).contains(&b.y), "y escaped at {t}: {}", b.y);
            }
        }
    }

    #[test]
    fn frames_are_bright_somewhere() {
        let spec = SceneSpec::canonical(SceneId::MovingShape01);
        for f in rollout(&spec, 32, |t| context_program().at(t)) {
            let peak = f.iter().cloned().fold(0.0f32, f32::max);
            assert!(peak > 0.9, "frame went dark, peak {peak}");
        }
    }

    #[test]
    fn distinct_requests_give_distinct_futures() {
        // The future is generated by the *scene* here, independent of any model:
        // this pins the target behaviour the model is trained toward.
        let spec = SceneSpec::canonical(SceneId::MovingShape01);
        let ctx = context_program();
        let mut ends = Vec::new();
        for r in Request::ALL {
            let sched = r.schedule();
            let mut bodies = spec.working();
            for t in 0..256 {
                for b in bodies.iter_mut() {
                    b.advance(ctx.at(t));
                }
            }
            let mut frames = vec![render(&bodies)];
            for t in 0..16 {
                for b in bodies.iter_mut() {
                    b.advance(sched.at(t));
                }
                frames.push(render(&bodies));
            }
            ends.push(frames[16].clone());
        }
        for i in 0..ends.len() {
            for j in (i + 1)..ends.len() {
                let d: f32 = ends[i]
                    .iter()
                    .zip(ends[j].iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                assert!(d > 1.0, "requests {i} and {j} ended identically ({d})");
            }
        }
    }
}
