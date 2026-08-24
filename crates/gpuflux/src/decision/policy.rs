use std::time::Instant;

use crate::decision::engine::Action;
use crate::object::ObjectSpec;
use crate::prediction::cost::{ActionPredictions, CostEstimate};
use crate::prediction::deadline::uncertainty_ms;
use crate::resource::ResourceState;

/// Everything a policy is allowed to see when making a choice. Policies are
/// deliberately not given the observation store: reading history is the
/// prediction engine's job (Phase 3+). The `pred` argument carries the
/// per-action estimates produced by the predictor, so policies stay pure
/// decision functions.
#[derive(Debug, Clone)]
pub struct DecisionContext {
    pub object: ObjectSpec,
    pub resource: ResourceState,
    pub deadline: Option<Instant>,
}

pub trait Policy {
    fn name(&self) -> &'static str;
    fn choose(&self, ctx: &DecisionContext, pred: &ActionPredictions) -> Action;
}

/// Level 0 baseline: always move.
pub struct AlwaysMove;

impl Policy for AlwaysMove {
    fn name(&self) -> &'static str {
        "always_move"
    }

    fn choose(&self, _ctx: &DecisionContext, _pred: &ActionPredictions) -> Action {
        Action::Move
    }
}

/// Level 0 baseline: always recompute.
pub struct AlwaysRecompute;

impl Policy for AlwaysRecompute {
    fn name(&self) -> &'static str {
        "always_recompute"
    }

    fn choose(&self, _ctx: &DecisionContext, _pred: &ActionPredictions) -> Action {
        Action::Recompute
    }
}

/// Level 1: pick whichever action the current-state cost model expects to
/// finish sooner. This is the "dumb intelligent" baseline that Phase 3+ must
/// improve upon.
pub struct ExpectedCost;

impl Policy for ExpectedCost {
    fn name(&self) -> &'static str {
        "expected_cost"
    }

    fn choose(&self, _ctx: &DecisionContext, pred: &ActionPredictions) -> Action {
        let mut best = Action::Move;
        let mut best_cost = pred.move_est.expected_ms;
        if pred.recompute_est.expected_ms < best_cost {
            best = Action::Recompute;
            best_cost = pred.recompute_est.expected_ms;
        }
        if pred.remote_est.expected_ms < best_cost {
            best = Action::RemoteRecompute;
        }
        best
    }
}

/// Level 4: score each action with a risk penalty for estimate uncertainty,
/// `J(a) = E[T_a] + mu * U_a` where `U_a` is the p90-p50 spread. No deadline
/// awareness; selects the action with the lowest risk-adjusted expected cost.
pub struct RiskAware {
    pub mu: f64,
}

impl Policy for RiskAware {
    fn name(&self) -> &'static str {
        "risk_aware"
    }

    fn choose(&self, _ctx: &DecisionContext, pred: &ActionPredictions) -> Action {
        let score = |est: &CostEstimate| est.expected_ms + self.mu * uncertainty_ms(est);
        let mut best = Action::Move;
        let mut best_score = score(&pred.move_est);
        let rc = score(&pred.recompute_est);
        if rc < best_score {
            best = Action::Recompute;
            best_score = rc;
        }
        let rm = score(&pred.remote_est);
        if rm < best_score {
            best = Action::RemoteRecompute;
        }
        best
    }
}

/// Level 5: the GPUFlux core decision score,
///   J(a) = E[T_a] + lambda * P(T_a > D) + mu * U_a
/// with `P(T_a > D)` estimated from the historical completion-time distribution
/// and `U_a` the estimate uncertainty. This is what lets a riskier-but-faster
/// path lose to a slower-but-safer one when a deadline is near.
pub struct DeadlineAware {
    pub lambda: f64,
    pub mu: f64,
}

impl Policy for DeadlineAware {
    fn name(&self) -> &'static str {
        "deadline_aware"
    }

    fn choose(&self, ctx: &DecisionContext, pred: &ActionPredictions) -> Action {
        let deadline_remaining = ctx
            .deadline
            .and_then(|d| d.checked_duration_since(Instant::now()))
            .map(|r| r.as_secs_f64() * 1000.0);

        let score = |est: &CostEstimate| {
            let miss = match (deadline_remaining, est.deadline_probability) {
                (Some(_d), Some(p)) => p,
                (Some(d), None) => crate::prediction::deadline::deadline_exceed_probability(est, d),
                _ => 0.0,
            };
            est.expected_ms + self.lambda * miss + self.mu * uncertainty_ms(est)
        };

        let mut best = Action::Move;
        let mut best_score = score(&pred.move_est);
        let rc = score(&pred.recompute_est);
        if rc < best_score {
            best = Action::Recompute;
            best_score = rc;
        }
        let rm = score(&pred.remote_est);
        if rm < best_score {
            best = Action::RemoteRecompute;
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{DataLoc, ObjectSpec};
    use crate::prediction::cost::ActionPredictions;
    use crate::resource::ResourceState;
    use std::time::{Duration, Instant};

    fn est(expected: f64, p50: f64, p90: f64, miss: Option<f64>) -> CostEstimate {
        CostEstimate {
            expected_ms: expected,
            p50_ms: p50,
            p90_ms: p90,
            p95_ms: p90 + (p90 - p50),
            deadline_probability: miss,
        }
    }

    fn ctx(deadline: bool) -> DecisionContext {
        DecisionContext {
            object: ObjectSpec::new(1, 1024, DataLoc::GpuMemory),
            resource: ResourceState::now(),
            deadline: if deadline {
                Some(Instant::now() + Duration::from_millis(1000))
            } else {
                None
            },
        }
    }

    #[test]
    fn expected_cost_picks_min() {
        let pred = ActionPredictions {
            move_est: est(100.0, 100.0, 100.0, None),
            recompute_est: est(80.0, 80.0, 80.0, None),
            remote_est: est(1e9, 1e9, 1e9, None),
        };
        assert_eq!(ExpectedCost.choose(&ctx(false), &pred), Action::Recompute);
    }

    #[test]
    fn deadline_aware_inverts_on_risk() {
        // move: faster expected but risky tail near deadline
        let pred = ActionPredictions {
            move_est: est(120.0, 120.0, 300.0, Some(0.9)),
            recompute_est: est(150.0, 150.0, 160.0, Some(0.0)),
            remote_est: est(1e9, 1e9, 1e9, None),
        };
        let policy = DeadlineAware {
            lambda: 200.0,
            mu: 0.0,
        };
        assert_eq!(policy.choose(&ctx(true), &pred), Action::Recompute);
    }

    #[test]
    fn deadline_aware_prefers_move_when_safe() {
        let pred = ActionPredictions {
            move_est: est(120.0, 120.0, 125.0, Some(0.0)),
            recompute_est: est(150.0, 150.0, 160.0, Some(0.0)),
            remote_est: est(1e9, 1e9, 1e9, None),
        };
        let policy = DeadlineAware {
            lambda: 200.0,
            mu: 0.0,
        };
        assert_eq!(policy.choose(&ctx(true), &pred), Action::Move);
    }

    #[test]
    fn risk_aware_prefers_low_spread() {
        let pred = ActionPredictions {
            move_est: est(130.0, 130.0, 300.0, None),
            recompute_est: est(140.0, 140.0, 150.0, None),
            remote_est: est(1e9, 1e9, 1e9, None),
        };
        let policy = RiskAware { mu: 0.5 };
        assert_eq!(policy.choose(&ctx(false), &pred), Action::Recompute);
    }
}
