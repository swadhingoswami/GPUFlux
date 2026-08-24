use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::object::ObjectSpec;

/// Progress report emitted at executor checkpoints. `fraction_done` is in
/// [0, 1] over the total work of the path (for move: write+read bytes; for
/// recompute: all passes).
#[derive(Debug, Clone)]
pub struct Progress {
    pub phase: &'static str,
    pub elapsed: Duration,
    pub fraction_done: f64,
    pub bytes_done: u64,
}

/// Mid-execution control passed to executors so the runtime can observe
/// progress and abort a path that is going to miss its deadline (Phase 6
/// replanning). `on_checkpoint` is invoked at coarse checkpoints; `abort` is
/// polled by the executor, which stops and reports `aborted = true` when set.
pub struct ExecutionControl<'a> {
    pub on_checkpoint: Option<&'a mut dyn FnMut(&Progress)>,
    pub abort: &'a AtomicBool,
}

static NOOP_ABORT: AtomicBool = AtomicBool::new(false);

impl<'a> ExecutionControl<'a> {
    /// A control that never aborts and reports no checkpoints.
    pub fn none() -> ExecutionControl<'static> {
        ExecutionControl {
            on_checkpoint: None,
            abort: &NOOP_ABORT,
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.abort.load(Ordering::Relaxed)
    }
}

pub struct MoveReport {
    pub total: Duration,
    pub bytes: u64,
    pub write_elapsed: Duration,
    pub read_elapsed: Duration,
    pub deadline_met: bool,
    /// True if the executor stopped early due to `ExecutionControl.abort`.
    pub aborted: bool,
}

impl MoveReport {
    /// Effective throughput in megabytes (10^6 bytes) per second.
    pub fn throughput_mb_s(&self) -> f64 {
        let s = self.total.as_secs_f64();
        if s <= 0.0 {
            0.0
        } else {
            self.bytes as f64 / s / 1e6
        }
    }
}

pub struct RecomputeReport {
    pub total: Duration,
    pub bytes: u64,
    pub passes: u32,
    pub deadline_met: bool,
    pub aborted: bool,
}

impl RecomputeReport {
    /// Effective throughput in megabytes (10^6 bytes) per second.
    pub fn throughput_mb_s(&self) -> f64 {
        let s = self.total.as_secs_f64();
        if s <= 0.0 {
            0.0
        } else {
            self.bytes as f64 / s / 1e6
        }
    }
}

/// A path that produces X by moving existing bytes into GPU memory.
///
/// `deadline` models "stage B needs X no later than this instant". The
/// executor reports whether it finished in time; the decision engine (later
/// phases) is responsible for *predicting* that, this trait only reports
/// ground truth.
pub trait MoveExecutor {
    fn name(&self) -> &str;
    fn move_to_gpu(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<MoveReport>;
}

/// A path that produces X by recomputing it from scratch.
pub trait RecomputeExecutor {
    fn name(&self) -> &str;
    fn recompute(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<RecomputeReport>;
}
