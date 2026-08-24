use crate::object::ObjectSpec;
use crate::observation::ObservationStore;
use crate::prediction::bucket::action_bucket;
use crate::prediction::cost::{
    ActionPredictions, CostEstimate, CostModel, CurrentStateCostModel, Predictor,
};
use crate::prediction::deadline::deadline_exceed_probability;
use crate::resource::ResourceState;

/// Level 3 predictor: estimates each action from historical observations in the
/// store, using the analytic current-state model only as a cold-start prior.
///
/// Rationale (observed in Phase 2): the analytic model's contention
/// assumptions are poorly calibrated (it assumed CPU pressure would slow the
/// memory-bound recompute ~40%, but it barely did). History captures the
/// workload's *actual* response, so once enough samples exist it should
/// dominate; the analytic model is only used until history matures.
///
/// History weight ramps with sample count: `w = min(n/k, max_weight)`.
/// Cold start (n=0) falls back to the current-state estimate.
pub struct HistoricalPredictor {
    store: ObservationStore,
    model: CurrentStateCostModel,
    /// Samples before history reaches full trust.
    pub history_k: f64,
    pub history_weight_max: f64,
}

impl HistoricalPredictor {
    pub fn new(store: ObservationStore, model: CurrentStateCostModel) -> Self {
        Self {
            store,
            model,
            history_k: 20.0,
            history_weight_max: 0.95,
        }
    }

    fn weight(&self, n: u64) -> f64 {
        if n == 0 {
            return 0.0;
        }
        (n as f64 / self.history_k).min(self.history_weight_max)
    }

    fn estimate(&self, bucket: &str, current: CostEstimate) -> CostEstimate {
        let row = match self.store.aggregate(bucket) {
            Ok(Some(r)) if r.sample_count > 0 => r,
            _ => return current,
        };
        let w = self.weight(row.sample_count);
        let blend = |hist: f64, cur: f64| w * hist + (1.0 - w) * cur;
        let quantile = |q: f64| {
            if row.samples.is_empty() {
                current.expected_ms
            } else {
                blend(row.p(q), current.expected_ms)
            }
        };
        CostEstimate {
            expected_ms: blend(row.ewma_mean, current.expected_ms),
            p50_ms: quantile(0.50),
            p90_ms: quantile(0.90),
            p95_ms: quantile(0.95),
            deadline_probability: None,
        }
    }
}

impl Predictor for HistoricalPredictor {
    fn predict(
        &self,
        object: &ObjectSpec,
        state: &ResourceState,
        deadline_remaining_ms: Option<f64>,
    ) -> ActionPredictions {
        let mut move_est = self.estimate(
            &action_bucket(object, "move", state),
            self.model.move_cost(state, object),
        );
        let mut recompute_est = self.estimate(
            &action_bucket(object, "recompute", state),
            self.model.recompute_cost(state, object),
        );
        let mut remote_est = self.estimate(
            &action_bucket(object, "remote", state),
            self.model.remote_recompute_cost(state, object),
        );
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
