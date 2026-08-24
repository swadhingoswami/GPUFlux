use std::time::{Duration, Instant};

use crate::error::Result;
use crate::executor::traits::{ExecutionControl, Progress};
use crate::object::ObjectSpec;

pub struct RemoteRecomputeReport {
    pub total: Duration,
    pub remote_compute_elapsed: Duration,
    pub transfer_elapsed: Duration,
    pub bytes: u64,
    pub deadline_met: bool,
    pub aborted: bool,
}

/// A path that produces X on a remote CPU and fetches it over the network.
pub trait RemoteExecutor {
    fn name(&self) -> &str;
    fn remote_recompute(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<RemoteRecomputeReport>;
}

/// Simulated remote node (Phase 8). There is no second physical machine in the
/// dev environment, so the remote path is modeled explicitly and tunably:
///
///   T_remote = T_remote_compute + T_network_transfer
///   T_remote_compute = passes * size / (remote_rate * (1 - remote_load))
///   T_network_transfer = network_latency_ms + size / network_bandwidth
///
/// The compute and transfer phases are "executed" by sleeping their modeled
/// durations (the remote node and network are simulated), while the *decision
/// logic* around them is real. This isolates the network/remote-queue cost
/// terms the Phase 8 decision study targets.
pub struct SimRemoteExecutor {
    /// Simulated remote CPU utilization in [0,1).
    pub remote_load: f64,
    /// Remote per-pass compute rate, bytes/s.
    pub remote_rate_bps: f64,
    /// Recompute passes (matches the local workload).
    pub passes: f64,
    /// One-way network latency, ms.
    pub network_latency_ms: f64,
    /// Network bandwidth, bytes/s.
    pub network_bandwidth_bps: f64,
}

impl Default for SimRemoteExecutor {
    fn default() -> Self {
        Self {
            remote_load: 0.1,
            remote_rate_bps: 4.2e9,
            passes: 1.0,
            network_latency_ms: 5.0,
            network_bandwidth_bps: 10e9,
        }
    }
}

impl SimRemoteExecutor {
    fn modeled(&self, spec: &ObjectSpec) -> (f64, f64) {
        let rate = self.remote_rate_bps * (1.0 - self.remote_load).max(0.05);
        let compute_ms = self.passes * spec.size_bytes as f64 / rate * 1000.0;
        let transfer_ms =
            self.network_latency_ms + spec.size_bytes as f64 / self.network_bandwidth_bps * 1000.0;
        (compute_ms, transfer_ms)
    }
}

impl RemoteExecutor for SimRemoteExecutor {
    fn name(&self) -> &str {
        "sim/remote"
    }

    fn remote_recompute(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<RemoteRecomputeReport> {
        let t0 = Instant::now();
        let (compute_ms, transfer_ms) = self.modeled(spec);

        if let Some(cb) = control.on_checkpoint.as_deref_mut() {
            cb(&Progress {
                phase: "remote:compute",
                elapsed: t0.elapsed(),
                fraction_done: 0.0,
                bytes_done: 0,
            });
        }
        if control.is_aborted() {
            return Ok(RemoteRecomputeReport {
                total: t0.elapsed(),
                remote_compute_elapsed: Duration::ZERO,
                transfer_elapsed: Duration::ZERO,
                bytes: 0,
                deadline_met: false,
                aborted: true,
            });
        }
        std::thread::sleep(Duration::from_secs_f64(compute_ms / 1000.0));
        let compute_done = t0.elapsed();
        let compute_frac = compute_ms / (compute_ms + transfer_ms).max(1.0);
        if let Some(cb) = control.on_checkpoint.as_deref_mut() {
            cb(&Progress {
                phase: "remote:network",
                elapsed: compute_done,
                fraction_done: compute_frac,
                bytes_done: spec.size_bytes,
            });
        }
        if control.is_aborted() {
            return Ok(RemoteRecomputeReport {
                total: t0.elapsed(),
                remote_compute_elapsed: compute_done,
                transfer_elapsed: Duration::ZERO,
                bytes: spec.size_bytes,
                deadline_met: false,
                aborted: true,
            });
        }
        std::thread::sleep(Duration::from_secs_f64(transfer_ms / 1000.0));

        let total = t0.elapsed();
        let deadline_met = deadline.is_none_or(|d| Instant::now() <= d);
        Ok(RemoteRecomputeReport {
            total,
            remote_compute_elapsed: compute_done,
            transfer_elapsed: total - compute_done,
            bytes: spec.size_bytes,
            deadline_met,
            aborted: false,
        })
    }
}
