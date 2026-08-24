//! Context-aware online prediction (Phase 7).
//!
//! The Phase 4 experiments showed the fixed-coefficient current-state model
//! misreads the state→cost relationship (it assumed CPU pressure slows the
//! memory-bound recompute ~40%; it barely does). Instead of hand-tuning
//! coefficients, learn the mapping from live resource features to observed
//! completion time with an online linear regressor (recursive least squares
//! with exponential forgetting, so it adapts to drift). Residuals give the
//! completion-time distribution for deadline risk.

use crate::object::ObjectSpec;
use crate::prediction::cost::{
    ActionPredictions, CostEstimate, CostModel, CurrentStateCostModel, Predictor,
};
use crate::prediction::deadline::deadline_exceed_probability;
use crate::resource::ResourceState;

pub const N_FEATURES: usize = 4;

/// Normalized feature vector: bias, cpu_util, nvme_latency (scaled), queue depth.
/// Missing values become 0 (neutral); features are scaled so coefficients are
/// comparable.
pub fn features(state: &ResourceState) -> [f64; N_FEATURES] {
    [
        1.0,
        state.cpu_util.unwrap_or(0.0),
        state.nvme_latency_us.map(|l| l / 500.0).unwrap_or(0.0),
        state.nvme_queue_depth.map(|q| q as f64).unwrap_or(0.0),
    ]
}

/// Recursive-least-squares linear regressor with exponential forgetting.
#[derive(Debug, Clone)]
pub struct OnlineLinearRegression {
    w: [f64; N_FEATURES],
    p: [[f64; N_FEATURES]; N_FEATURES],
    n: u64,
    residuals: Vec<f64>,
    forget: f64,
    l2: f64,
}

impl Default for OnlineLinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineLinearRegression {
    pub fn new() -> Self {
        let mut p = [[0.0; N_FEATURES]; N_FEATURES];
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = 100.0;
        }
        Self {
            w: [0.0; N_FEATURES],
            p,
            n: 0,
            residuals: Vec::new(),
            forget: 0.99,
            l2: 1e-3,
        }
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn weights(&self) -> [f64; N_FEATURES] {
        self.w
    }

    pub fn predict(&self, x: &[f64; N_FEATURES]) -> f64 {
        self.w.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum()
    }

    /// One online update (y - w·x), standard RLS with forgetting + L2 damping.
    pub fn update(&mut self, x: &[f64; N_FEATURES], y: f64) {
        self.n += 1;
        let pred = self.predict(x);
        let resid = y - pred;
        self.residuals.push(resid);
        if self.residuals.len() > 512 {
            self.residuals = self.residuals.iter().step_by(2).copied().collect();
        }

        let mut px = [0.0; N_FEATURES];
        for (i, row) in self.p.iter().enumerate() {
            px[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
        let xpx: f64 = x.iter().zip(px.iter()).map(|(a, b)| a * b).sum();
        let denom = self.forget + xpx + self.l2;
        let k = px.map(|v| v / denom);
        for (i, ki) in k.iter().enumerate() {
            self.w[i] += ki * resid;
        }
        for (i, row) in self.p.iter_mut().enumerate() {
            for (j, pij) in row.iter_mut().enumerate() {
                *pij = (*pij - k[i] * px[j]) / self.forget;
            }
        }
    }

    /// Residual quantile (e.g. p90) around the mean prediction, ms.
    pub fn residual_quantile(&self, q: f64) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        let mut r = self.residuals.clone();
        r.sort_by(f64::total_cmp);
        let idx = ((r.len() - 1) as f64 * q).round() as usize;
        r[idx.min(r.len() - 1)]
    }
}

/// Predictor that learns a per-action linear model from (state, actual) pairs,
/// falling back to the analytic current-state model until enough samples exist.
pub struct OnlineRegressionPredictor {
    move_reg: OnlineLinearRegression,
    recompute_reg: OnlineLinearRegression,
    remote_reg: OnlineLinearRegression,
    model: CurrentStateCostModel,
    pub min_samples: u64,
}

impl OnlineRegressionPredictor {
    pub fn new(model: CurrentStateCostModel) -> Self {
        Self {
            move_reg: OnlineLinearRegression::new(),
            recompute_reg: OnlineLinearRegression::new(),
            remote_reg: OnlineLinearRegression::new(),
            model,
            min_samples: 5,
        }
    }

    /// Learned weights of the regressors (for inspection).
    pub fn weights(&self) -> ([f64; N_FEATURES], [f64; N_FEATURES], [f64; N_FEATURES]) {
        (
            self.move_reg.weights(),
            self.recompute_reg.weights(),
            self.remote_reg.weights(),
        )
    }
}

impl Predictor for OnlineRegressionPredictor {
    fn predict(
        &self,
        object: &ObjectSpec,
        state: &ResourceState,
        deadline_remaining_ms: Option<f64>,
    ) -> ActionPredictions {
        let x = features(state);
        let move_est = if self.move_reg.n() >= self.min_samples {
            let base = self.move_reg.predict(&x);
            CostEstimate {
                expected_ms: base + self.move_reg.residual_quantile(0.50),
                p50_ms: base + self.move_reg.residual_quantile(0.50),
                p90_ms: base + self.move_reg.residual_quantile(0.90),
                p95_ms: base + self.move_reg.residual_quantile(0.95),
                deadline_probability: None,
            }
        } else {
            self.model.move_cost(state, object)
        };
        let recompute_est = if self.recompute_reg.n() >= self.min_samples {
            let base = self.recompute_reg.predict(&x);
            CostEstimate {
                expected_ms: base + self.recompute_reg.residual_quantile(0.50),
                p50_ms: base + self.recompute_reg.residual_quantile(0.50),
                p90_ms: base + self.recompute_reg.residual_quantile(0.90),
                p95_ms: base + self.recompute_reg.residual_quantile(0.95),
                deadline_probability: None,
            }
        } else {
            self.model.recompute_cost(state, object)
        };
        let remote_est = if self.remote_reg.n() >= self.min_samples {
            let base = self.remote_reg.predict(&x);
            CostEstimate {
                expected_ms: base + self.remote_reg.residual_quantile(0.50),
                p50_ms: base + self.remote_reg.residual_quantile(0.50),
                p90_ms: base + self.remote_reg.residual_quantile(0.90),
                p95_ms: base + self.remote_reg.residual_quantile(0.95),
                deadline_probability: None,
            }
        } else {
            self.model.remote_recompute_cost(state, object)
        };

        let mut move_est = move_est;
        let mut recompute_est = recompute_est;
        let mut remote_est = remote_est;
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

    fn update(
        &mut self,
        action: &str,
        _object: &ObjectSpec,
        state: &ResourceState,
        actual_ms: f64,
    ) {
        let x = features(state);
        match action {
            "move" => self.move_reg.update(&x, actual_ms),
            "recompute" => self.recompute_reg.update(&x, actual_ms),
            "remote" => self.remote_reg.update(&x, actual_ms),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_linear_mapping() {
        let mut reg = OnlineLinearRegression::new();
        let mut rng = 12_345u64;
        for _ in 0..4000 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let cpu = (rng >> 32) as f64 / u32::MAX as f64;
            let x = [1.0, cpu, 0.0, 0.0];
            let y = 10.0 + 5.0 * cpu;
            reg.update(&x, y);
        }
        let w = reg.weights();
        assert!((w[0] - 10.0).abs() < 1.0, "bias={}", w[0]);
        assert!((w[1] - 5.0).abs() < 1.0, "cpu={}", w[1]);
        let yhat = reg.predict(&[1.0, 0.5, 0.0, 0.0]);
        assert!((yhat - 12.5).abs() < 1.0, "yhat={yhat}");
    }

    #[test]
    fn residual_quantiles_reflect_noise() {
        let mut reg = OnlineLinearRegression::new();
        // Zero-noise, long enough to evict pre-convergence residuals from the
        // bounded reservoir: residuals collapse to ~0.
        for _ in 0..2000 {
            reg.update(&[1.0, 0.0, 0.0, 0.0], 100.0);
        }
        assert!(reg.residual_quantile(0.9).abs() < 0.5);
        // Inject uniform noise in [0, 100): model tracks the mean (~150), so
        // residuals are symmetric around 0 and p90 is positive but bounded.
        for i in 0..1000u64 {
            reg.update(&[1.0, 0.0, 0.0, 0.0], 100.0 + (i % 100) as f64);
        }
        let r90 = reg.residual_quantile(0.9);
        let r50 = reg.residual_quantile(0.5);
        assert!(r90 > 20.0 && r90 < 60.0, "r90={r90}");
        assert!(r50.abs() < 10.0, "r50={r50}");
    }
}
