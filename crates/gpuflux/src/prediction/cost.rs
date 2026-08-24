use crate::object::ObjectSpec;
use crate::prediction::deadline::deadline_exceed_probability;
use crate::resource::ResourceState;

/// Predicted completion-time distribution for one action. Level 1 (current
/// state) fills only `expected_ms`; the historical engine (Phase 3) fills the
/// quantiles; the deadline model (Phase 5) fills `deadline_probability`.
#[derive(Debug, Clone, Copy)]
pub struct CostEstimate {
    pub expected_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub deadline_probability: Option<f64>,
}

impl CostEstimate {
    /// A degenerate estimate with no distribution.
    pub fn point(expected_ms: f64) -> Self {
        Self {
            expected_ms,
            p50_ms: expected_ms,
            p90_ms: expected_ms,
            p95_ms: expected_ms,
            deadline_probability: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ActionPredictions {
    pub move_est: CostEstimate,
    pub recompute_est: CostEstimate,
    /// Remote recompute (Phase 8): remote CPU + network transfer.
    pub remote_est: CostEstimate,
}

pub trait CostModel {
    fn move_cost(&self, state: &ResourceState, object: &ObjectSpec) -> CostEstimate;
    fn recompute_cost(&self, state: &ResourceState, object: &ObjectSpec) -> CostEstimate;
    fn remote_recompute_cost(&self, state: &ResourceState, object: &ObjectSpec) -> CostEstimate;
}

/// Produces per-action estimates. `deadline_remaining_ms` lets the predictor
/// fill `deadline_probability` (Phase 5); Level 0-2 predictors ignore it.
/// `Any` lets benches/tools downcast to inspect concrete predictors.
pub trait Predictor: std::any::Any {
    fn predict(
        &self,
        object: &ObjectSpec,
        state: &ResourceState,
        deadline_remaining_ms: Option<f64>,
    ) -> ActionPredictions;

    /// Feed one completed action's (state, actual) back into the model.
    /// Default no-op; online learners override this. Called by the engine after
    /// execution for every action that ran to completion.
    fn update(
        &mut self,
        _action: &str,
        _object: &ObjectSpec,
        _state: &ResourceState,
        _actual_ms: f64,
    ) {
    }
}

/// Level 1 (current-state) cost model: derives completion time from the live
/// resource snapshot, ignoring history.
///
/// Coefficients are calibrated against Phase 0 measurements on the dev box
/// (move ~1.5 GB/s round-trip, recompute ~4.2 GB/s per fill pass, passes=2).
/// They are placeholder values to be re-derived on the CUDA box; the *shape*
/// of the model (contention scalings) is what matters here.
#[derive(Debug, Clone)]
pub struct CurrentStateCostModel {
    /// Effective move bytes/s over a full write+read+sync round trip (idle).
    pub move_rate_bps: f64,
    /// Recompute bytes/s per fill pass (idle).
    pub recompute_rate_bps: f64,
    /// Number of recompute passes in the target workload (couples to the
    /// recompute executor configuration). May be fractional.
    pub recompute_passes: f64,
    pub move_overhead_ms: f64,
    pub recompute_overhead_ms: f64,
    /// Reference NVMe read latency (us); higher measured latency scales move
    /// throughput down.
    pub nvme_latency_ref_us: f64,
    /// Per-unit-of-queue-depth throughput penalty for move.
    pub queue_penalty: f64,
    /// Fraction of recompute throughput lost per unit of cpu_util.
    pub cpu_contention: f64,
    /// Fraction of recompute throughput lost per unit of gpu_util (a busy GPU
    /// delays kernels on the real box).
    pub gpu_contention: f64,
    /// Whether a remote recompute path exists (Phase 8). When false, the remote
    /// estimate is +inf and no policy will ever choose it.
    pub remote_enabled: bool,
    /// Remote per-pass compute rate, bytes/s.
    pub remote_rate_bps: f64,
    /// Fraction of remote compute throughput lost per unit of remote_cpu_util.
    pub remote_contention: f64,
    /// One-way network latency, ms.
    pub network_latency_ms: f64,
    /// Network bandwidth, bytes/s.
    pub network_bandwidth_bps: f64,
}

impl Default for CurrentStateCostModel {
    fn default() -> Self {
        Self {
            move_rate_bps: 1.5e9,
            recompute_rate_bps: 4.2e9,
            recompute_passes: 2.0,
            move_overhead_ms: 2.0,
            recompute_overhead_ms: 1.0,
            nvme_latency_ref_us: 150.0,
            queue_penalty: 0.5,
            cpu_contention: 0.7,
            gpu_contention: 0.7,
            remote_enabled: false,
            remote_rate_bps: 4.2e9,
            remote_contention: 0.7,
            network_latency_ms: 5.0,
            network_bandwidth_bps: 10e9,
        }
    }
}

impl CurrentStateCostModel {
    fn move_rate(&self, state: &ResourceState) -> f64 {
        let mut rate = self.move_rate_bps;
        if let Some(lat) = state.nvme_latency_us {
            rate *= (self.nvme_latency_ref_us / lat.max(1.0)).clamp(0.2, 1.0);
        }
        if let Some(q) = state.nvme_queue_depth {
            rate /= 1.0 + self.queue_penalty * q as f64;
        }
        if let Some(pcie) = state.pcie_bandwidth {
            rate = rate.min(pcie);
        }
        rate
    }

    fn recompute_rate(&self, state: &ResourceState) -> f64 {
        let mut rate = self.recompute_rate_bps;
        if let Some(u) = state.cpu_util {
            rate *= (1.0 - self.cpu_contention * u).max(0.05);
        }
        if let Some(g) = state.gpu_util {
            rate *= (1.0 - self.gpu_contention * g).max(0.05);
        }
        rate
    }
}

impl CostModel for CurrentStateCostModel {
    fn move_cost(&self, state: &ResourceState, object: &ObjectSpec) -> CostEstimate {
        let est = self.move_overhead_ms + object.size_bytes as f64 / self.move_rate(state) * 1000.0;
        CostEstimate::point(est)
    }

    fn recompute_cost(&self, state: &ResourceState, object: &ObjectSpec) -> CostEstimate {
        let est = self.recompute_overhead_ms
            + (self.recompute_passes * object.size_bytes as f64) / self.recompute_rate(state)
                * 1000.0;
        CostEstimate::point(est)
    }

    fn remote_recompute_cost(&self, state: &ResourceState, object: &ObjectSpec) -> CostEstimate {
        if !self.remote_enabled {
            return CostEstimate::point(f64::INFINITY);
        }
        let load = state.remote_cpu_util.unwrap_or(0.0);
        let rate = self.remote_rate_bps * (1.0 - self.remote_contention * load).max(0.05);
        let compute_ms = self.recompute_passes * object.size_bytes as f64 / rate * 1000.0;
        let transfer_ms = self.network_latency_ms
            + object.size_bytes as f64 / self.network_bandwidth_bps * 1000.0;
        CostEstimate::point(self.recompute_overhead_ms + compute_ms + transfer_ms)
    }
}

pub struct CurrentStatePredictor {
    pub model: CurrentStateCostModel,
}

impl Predictor for CurrentStatePredictor {
    fn predict(
        &self,
        object: &ObjectSpec,
        state: &ResourceState,
        deadline_remaining_ms: Option<f64>,
    ) -> ActionPredictions {
        let mut move_est = self.model.move_cost(state, object);
        let mut recompute_est = self.model.recompute_cost(state, object);
        let mut remote_est = self.model.remote_recompute_cost(state, object);
        if let Some(d) = deadline_remaining_ms {
            move_est.deadline_probability = Some(deadline_exceed_probability(&move_est, d));
            recompute_est.deadline_probability =
                Some(deadline_exceed_probability(&recompute_est, d));
            remote_est.deadline_probability = Some(deadline_exceed_probability(&remote_est, d));
        }
        ActionPredictions {
            move_est,
            recompute_est,
            remote_est,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{DataLoc, ObjectSpec};

    fn spec(size: u64) -> ObjectSpec {
        ObjectSpec::new(1, size, DataLoc::GpuMemory)
    }

    #[test]
    fn move_cost_scales_with_size() {
        let m = CurrentStateCostModel::default();
        let small = m
            .move_cost(&ResourceState::default(), &spec(1_000_000))
            .expected_ms;
        let big = m
            .move_cost(&ResourceState::default(), &spec(10_000_000))
            .expected_ms;
        assert!(big > small);
    }

    #[test]
    fn nvme_latency_raises_move_cost() {
        let m = CurrentStateCostModel::default();
        let idle = ResourceState::default();
        let contended = ResourceState {
            nvme_latency_us: Some(2000.0),
            ..Default::default()
        };
        let s = spec(1_000_000);
        let c_idle = m.move_cost(&idle, &s).expected_ms;
        let c_busy = m.move_cost(&contended, &s).expected_ms;
        assert!(c_busy > c_idle);
    }

    #[test]
    fn recompute_cost_scales_with_passes() {
        let m1 = CurrentStateCostModel {
            recompute_passes: 1.0,
            ..Default::default()
        };
        let m2 = CurrentStateCostModel {
            recompute_passes: 3.0,
            ..Default::default()
        };
        let s = spec(1_000_000);
        let c1 = m1.recompute_cost(&ResourceState::default(), &s).expected_ms;
        let c2 = m2.recompute_cost(&ResourceState::default(), &s).expected_ms;
        assert!(c2 > c1);
    }
}
