use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::decision::policy::{DecisionContext, Policy};
use crate::error::Result;
use crate::executor::{
    ExecutionControl, MoveExecutor, Progress, RecomputeExecutor, RemoteExecutor,
};
use crate::object::ObjectSpec;
use crate::observation::{DecisionEvent, ObservationStore};
use crate::prediction::bucket::action_bucket;
use crate::prediction::cost::{ActionPredictions, Predictor};
use crate::resource::ResourceState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Move,
    Recompute,
    /// Recompute on a remote CPU and fetch over the network (Phase 8).
    RemoteRecompute,
}

impl Action {
    pub fn as_str(&self) -> &'static str {
        match self {
            Action::Move => "move",
            Action::Recompute => "recompute",
            Action::RemoteRecompute => "remote",
        }
    }

    pub const ALL: [Action; 3] = [Action::Move, Action::Recompute, Action::RemoteRecompute];

    /// The other candidate actions.
    pub fn alternatives(&self) -> [Action; 2] {
        match self {
            Action::Move => [Action::Recompute, Action::RemoteRecompute],
            Action::Recompute => [Action::Move, Action::RemoteRecompute],
            Action::RemoteRecompute => [Action::Move, Action::Recompute],
        }
    }

    fn expected(&self, pred: &ActionPredictions) -> f64 {
        match self {
            Action::Move => pred.move_est.expected_ms,
            Action::Recompute => pred.recompute_est.expected_ms,
            Action::RemoteRecompute => pred.remote_est.expected_ms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecisionOutcome {
    pub action: Action,
    pub actual_ms: f64,
    /// Regret = chosen - min over all measured actions; None when not measured.
    pub regret_ms: Option<f64>,
    pub deadline_met: bool,
    /// Prediction error (actual - predicted expected) for the chosen action.
    pub prediction_error_ms: Option<f64>,
    /// True when the chosen path was aborted and an alternative executed.
    pub fallback_used: bool,
    /// Time wasted on the aborted path before fallback (ms).
    pub wasted_ms: f64,
}

/// The runtime decision loop: predict -> choose -> execute -> record -> log.
pub struct DecisionEngine {
    store: ObservationStore,
    policy: Box<dyn Policy>,
    predictor: Box<dyn Predictor>,
    move_exec: Box<dyn MoveExecutor>,
    recompute_exec: Box<dyn RecomputeExecutor>,
    remote_exec: Option<Box<dyn RemoteExecutor>>,
    next_decision_id: u64,
    decisions: u64,
}

impl DecisionEngine {
    pub fn new(
        store: ObservationStore,
        policy: Box<dyn Policy>,
        predictor: Box<dyn Predictor>,
        move_exec: Box<dyn MoveExecutor>,
        recompute_exec: Box<dyn RecomputeExecutor>,
    ) -> Self {
        Self {
            store,
            policy,
            predictor,
            move_exec,
            recompute_exec,
            remote_exec: None,
            next_decision_id: 1,
            decisions: 0,
        }
    }

    /// Enable the remote recompute path (Phase 8).
    pub fn with_remote_exec(mut self, exec: Box<dyn RemoteExecutor>) -> Self {
        self.remote_exec = Some(exec);
        self
    }

    pub fn policy_name(&self) -> &'static str {
        self.policy.name()
    }

    /// Access the active predictor (e.g. for inspection/downcasting in tools).
    pub fn predictor(&self) -> &dyn Predictor {
        &*self.predictor
    }

    /// Whether an action has an executor configured. The remote path is
    /// optional (Phase 8); move/recompute always exist.
    pub fn configured(&self, action: Action) -> bool {
        match action {
            Action::Move | Action::Recompute => true,
            Action::RemoteRecompute => self.remote_exec.is_some(),
        }
    }

    fn measure_controlled(
        &mut self,
        action: Action,
        object: &ObjectSpec,
        deadline_remaining_ms: Option<f64>,
        control: &mut ExecutionControl,
    ) -> Result<(f64, bool, bool)> {
        let deadline =
            deadline_remaining_ms.map(|r| Instant::now() + Duration::from_secs_f64(r / 1000.0));
        match action {
            Action::Move => {
                let r = self.move_exec.move_to_gpu(object, deadline, control)?;
                Ok((r.total.as_secs_f64() * 1000.0, r.deadline_met, r.aborted))
            }
            Action::Recompute => {
                let r = self.recompute_exec.recompute(object, deadline, control)?;
                Ok((r.total.as_secs_f64() * 1000.0, r.deadline_met, r.aborted))
            }
            Action::RemoteRecompute => {
                let exec = self.remote_exec.as_mut().ok_or_else(|| {
                    crate::error::Error::Invalid("remote executor not configured".into())
                })?;
                let r = exec.remote_recompute(object, deadline, control)?;
                Ok((r.total.as_secs_f64() * 1000.0, r.deadline_met, r.aborted))
            }
        }
    }

    fn measure(
        &mut self,
        action: Action,
        object: &ObjectSpec,
        deadline_remaining_ms: Option<f64>,
    ) -> Result<(f64, bool)> {
        let (ms, met, _) = self.measure_controlled(
            action,
            object,
            deadline_remaining_ms,
            &mut ExecutionControl::none(),
        )?;
        Ok((ms, met))
    }

    fn log_event(
        &mut self,
        object: &ObjectSpec,
        resource: &ResourceState,
        pred: &ActionPredictions,
        deadline_remaining_ms: Option<f64>,
        outcome: &DecisionOutcome,
    ) -> Result<()> {
        let mut ev = DecisionEvent::new(self.next_decision_id, object.id);
        self.next_decision_id += 1;
        ev.resource_snapshot = format!("{:?}", resource);
        ev.predicted_costs_ms = vec![
            pred.move_est.expected_ms,
            pred.recompute_est.expected_ms,
            pred.remote_est.expected_ms,
        ];
        ev.chosen_action = outcome.action.as_str().to_string();
        ev.actual_cost_ms = outcome.actual_ms;
        ev.deadline_remaining_ms = deadline_remaining_ms;
        ev.deadline_met = Some(outcome.deadline_met);
        ev.prediction_error_ms = outcome.prediction_error_ms;
        ev.fallback_used = outcome.fallback_used;
        ev.wasted_ms = outcome.wasted_ms;
        ev.regret_ms = outcome.regret_ms;
        self.store.log_event(ev)
    }

    /// Runtime mode: predict, choose, execute, record, log. Regret is not
    /// measured because the alternatives are not executed.
    pub fn decide(
        &mut self,
        object: &ObjectSpec,
        resource: ResourceState,
        deadline_remaining_ms: Option<f64>,
    ) -> Result<DecisionOutcome> {
        let deadline =
            deadline_remaining_ms.map(|r| Instant::now() + Duration::from_secs_f64(r / 1000.0));
        let ctx = DecisionContext {
            object: object.clone(),
            resource: resource.clone(),
            deadline,
        };
        let pred = self
            .predictor
            .predict(object, &resource, deadline_remaining_ms);
        let action = self.policy.choose(&ctx, &pred);
        let (actual_ms, met) = self.measure(action, object, deadline_remaining_ms)?;

        self.store.record(
            &action_bucket(object, action.as_str(), &resource),
            actual_ms,
        )?;
        self.store
            .record_deadline(&action_bucket(object, action.as_str(), &resource), met)?;
        self.predictor
            .update(action.as_str(), object, &resource, actual_ms);

        let outcome = DecisionOutcome {
            action,
            actual_ms,
            regret_ms: None,
            deadline_met: met,
            prediction_error_ms: Some(actual_ms - action.expected(&pred)),
            fallback_used: false,
            wasted_ms: 0.0,
        };
        self.log_event(object, &resource, &pred, deadline_remaining_ms, &outcome)?;
        self.decisions += 1;
        Ok(outcome)
    }

    /// Benchmark mode: also executes both alternatives so regret against the
    /// oracle (min over all actions) can be computed. The chosen action is
    /// measured first so its timing is unaffected by the alternatives.
    pub fn decide_and_measure(
        &mut self,
        object: &ObjectSpec,
        resource: ResourceState,
        deadline_remaining_ms: Option<f64>,
    ) -> Result<DecisionOutcome> {
        let deadline =
            deadline_remaining_ms.map(|r| Instant::now() + Duration::from_secs_f64(r / 1000.0));
        let ctx = DecisionContext {
            object: object.clone(),
            resource: resource.clone(),
            deadline,
        };
        let pred = self
            .predictor
            .predict(object, &resource, deadline_remaining_ms);
        let action = self.policy.choose(&ctx, &pred);

        let (chosen_ms, chosen_met) = self.measure(action, object, deadline_remaining_ms)?;

        // Measure both alternatives (no deadline) for the oracle. Skip any
        // alternative with no configured executor (e.g. remote when disabled).
        let mut min_actual = chosen_ms;
        for alt in action.alternatives() {
            if !self.configured(alt) {
                continue;
            }
            let (a, _) = self.measure(alt, object, None)?;
            min_actual = min_actual.min(a);
            self.store
                .record(&action_bucket(object, alt.as_str(), &resource), a)?;
            self.predictor.update(alt.as_str(), object, &resource, a);
        }

        let regret_ms = Some(chosen_ms - min_actual);

        self.store.record(
            &action_bucket(object, action.as_str(), &resource),
            chosen_ms,
        )?;
        self.store.record_deadline(
            &action_bucket(object, action.as_str(), &resource),
            chosen_met,
        )?;
        self.predictor
            .update(action.as_str(), object, &resource, chosen_ms);

        let outcome = DecisionOutcome {
            action,
            actual_ms: chosen_ms,
            regret_ms,
            deadline_met: chosen_met,
            prediction_error_ms: Some(chosen_ms - action.expected(&pred)),
            fallback_used: false,
            wasted_ms: 0.0,
        };
        self.log_event(object, &resource, &pred, deadline_remaining_ms, &outcome)?;
        self.decisions += 1;
        Ok(outcome)
    }

    /// Runtime mode with replanning (Phase 6): execute the chosen path with
    /// progress checkpoints; if it is on track to miss the deadline AND the best
    /// alternative is expected to still finish in time, abort and fall back.
    pub fn decide_with_fallback(
        &mut self,
        object: &ObjectSpec,
        resource: ResourceState,
        deadline_remaining_ms: Option<f64>,
    ) -> Result<DecisionOutcome> {
        let deadline =
            deadline_remaining_ms.map(|r| Instant::now() + Duration::from_secs_f64(r / 1000.0));
        let ctx = DecisionContext {
            object: object.clone(),
            resource: resource.clone(),
            deadline,
        };
        let pred = self
            .predictor
            .predict(object, &resource, deadline_remaining_ms);
        let action = self.policy.choose(&ctx, &pred);

        // Best alternative by predicted expected cost among configured actions.
        let alt = action
            .alternatives()
            .iter()
            .filter(|a| self.configured(**a))
            .min_by(|a, b| a.expected(&pred).total_cmp(&b.expected(&pred)))
            .copied()
            .ok_or_else(|| {
                crate::error::Error::Invalid("no configured alternative for fallback".into())
            })?;
        let alt_expected = alt.expected(&pred);

        let abort = Arc::new(AtomicBool::new(false));

        let (chosen_ms, chosen_met, aborted) = {
            let abort_flag = abort.clone();
            let deadline_rem = deadline_remaining_ms;
            let mut cb = |p: &Progress| {
                if p.fraction_done > 0.0 {
                    let elapsed_ms = p.elapsed.as_secs_f64() * 1000.0;
                    let est_finish_ms = elapsed_ms / p.fraction_done;
                    if let Some(d) = deadline_rem {
                        let remaining_now = d - elapsed_ms;
                        if est_finish_ms > remaining_now && alt_expected < remaining_now {
                            abort_flag.store(true, Ordering::Relaxed);
                        }
                    }
                }
            };
            let mut control = ExecutionControl {
                on_checkpoint: Some(&mut cb),
                abort: &abort,
            };
            self.measure_controlled(action, object, deadline_remaining_ms, &mut control)?
        };

        let outcome = if aborted {
            let wasted_ms = chosen_ms;
            let alt_remaining = deadline_remaining_ms.map(|d| (d - wasted_ms).max(0.0));
            let (alt_ms, alt_met) = self.measure(alt, object, alt_remaining)?;
            self.store
                .record(&action_bucket(object, alt.as_str(), &resource), alt_ms)?;
            self.store
                .record_deadline(&action_bucket(object, alt.as_str(), &resource), alt_met)?;
            self.predictor
                .update(alt.as_str(), object, &resource, alt_ms);
            DecisionOutcome {
                action,
                actual_ms: wasted_ms + alt_ms,
                regret_ms: None,
                deadline_met: alt_met,
                prediction_error_ms: None,
                fallback_used: true,
                wasted_ms,
            }
        } else {
            self.store.record(
                &action_bucket(object, action.as_str(), &resource),
                chosen_ms,
            )?;
            self.store.record_deadline(
                &action_bucket(object, action.as_str(), &resource),
                chosen_met,
            )?;
            self.predictor
                .update(action.as_str(), object, &resource, chosen_ms);
            DecisionOutcome {
                action,
                actual_ms: chosen_ms,
                regret_ms: None,
                deadline_met: chosen_met,
                prediction_error_ms: Some(chosen_ms - action.expected(&pred)),
                fallback_used: false,
                wasted_ms: 0.0,
            }
        };

        self.log_event(object, &resource, &pred, deadline_remaining_ms, &outcome)?;
        self.decisions += 1;
        Ok(outcome)
    }
}
