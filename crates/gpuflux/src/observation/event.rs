use crate::now_unix_ms;
use crate::observation::codec::{
    push_bool, push_f64, push_opt_bool, push_opt_f64, push_str, push_u64, push_vec_f64, take_bool,
    take_f64, take_opt_bool, take_opt_f64, take_str, take_u64, take_vec_f64,
};

/// One raw decision event, appended to the event log. This is the detailed
/// record used for offline analysis and model training; the AggregateRow tables
/// are the fast hot-path statistics.
#[derive(Debug, Clone)]
pub struct DecisionEvent {
    pub decision_id: u64,
    pub object_id: u64,
    pub timestamp_unix_ms: u64,
    /// Coarse serialization of the ResourceState snapshot (Phase 2+).
    pub resource_snapshot: String,
    /// Predicted completion cost (ms) per candidate action, in stable order.
    pub predicted_costs_ms: Vec<f64>,
    pub chosen_action: String,
    pub actual_cost_ms: f64,
    /// Remaining time (ms) until X is needed at decision time; None if no deadline.
    pub deadline_remaining_ms: Option<f64>,
    pub deadline_met: Option<bool>,
    pub prediction_error_ms: Option<f64>,
    pub fallback_used: bool,
    /// Time wasted on an aborted chosen path before falling back (ms).
    pub wasted_ms: f64,
    /// regret = actual cost - cost the oracle (perfect knowledge) would have incurred
    pub regret_ms: Option<f64>,
}

impl DecisionEvent {
    pub fn new(decision_id: u64, object_id: u64) -> Self {
        DecisionEvent {
            decision_id,
            object_id,
            timestamp_unix_ms: now_unix_ms(),
            resource_snapshot: String::new(),
            predicted_costs_ms: Vec::new(),
            chosen_action: String::new(),
            actual_cost_ms: 0.0,
            deadline_remaining_ms: None,
            deadline_met: None,
            prediction_error_ms: None,
            fallback_used: false,
            wasted_ms: 0.0,
            regret_ms: None,
        }
    }
}

impl redb::Value for DecisionEvent {
    type SelfType<'a>
        = Self
    where
        Self: 'a;
    type AsBytes<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let mut d: &[u8] = data;
        DecisionEvent {
            decision_id: take_u64(&mut d),
            object_id: take_u64(&mut d),
            timestamp_unix_ms: take_u64(&mut d),
            resource_snapshot: take_str(&mut d),
            predicted_costs_ms: take_vec_f64(&mut d),
            chosen_action: take_str(&mut d),
            actual_cost_ms: take_f64(&mut d),
            deadline_remaining_ms: take_opt_f64(&mut d),
            deadline_met: take_opt_bool(&mut d),
            prediction_error_ms: take_opt_f64(&mut d),
            fallback_used: take_bool(&mut d),
            wasted_ms: take_f64(&mut d),
            regret_ms: take_opt_f64(&mut d),
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        let mut v = Vec::new();
        push_u64(&mut v, value.decision_id);
        push_u64(&mut v, value.object_id);
        push_u64(&mut v, value.timestamp_unix_ms);
        push_str(&mut v, &value.resource_snapshot);
        push_vec_f64(&mut v, &value.predicted_costs_ms);
        push_str(&mut v, &value.chosen_action);
        push_f64(&mut v, value.actual_cost_ms);
        push_opt_f64(&mut v, value.deadline_remaining_ms);
        push_opt_bool(&mut v, value.deadline_met);
        push_opt_f64(&mut v, value.prediction_error_ms);
        push_bool(&mut v, value.fallback_used);
        push_f64(&mut v, value.wasted_ms);
        push_opt_f64(&mut v, value.regret_ms);
        v
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("gpuflux::DecisionEvent")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redb_round_trip() {
        let mut ev = DecisionEvent::new(42, 7);
        ev.resource_snapshot = "cpu=0.5,nvme=100".into();
        ev.predicted_costs_ms = vec![10.0, 20.0];
        ev.chosen_action = "move".into();
        ev.actual_cost_ms = 15.0;
        ev.deadline_remaining_ms = Some(100.0);
        ev.deadline_met = Some(true);
        ev.prediction_error_ms = Some(5.0);
        ev.fallback_used = true;
        ev.wasted_ms = 3.0;
        ev.regret_ms = Some(2.0);

        let bytes = <DecisionEvent as redb::Value>::as_bytes(&ev);
        let back = <DecisionEvent as redb::Value>::from_bytes(&bytes);
        assert_eq!(back.decision_id, ev.decision_id);
        assert_eq!(back.object_id, ev.object_id);
        assert_eq!(back.timestamp_unix_ms, ev.timestamp_unix_ms);
        assert_eq!(back.resource_snapshot, ev.resource_snapshot);
        assert_eq!(back.predicted_costs_ms, ev.predicted_costs_ms);
        assert_eq!(back.chosen_action, ev.chosen_action);
        assert_eq!(back.actual_cost_ms, ev.actual_cost_ms);
        assert_eq!(back.deadline_remaining_ms, ev.deadline_remaining_ms);
        assert_eq!(back.deadline_met, ev.deadline_met);
        assert_eq!(back.prediction_error_ms, ev.prediction_error_ms);
        assert_eq!(back.fallback_used, ev.fallback_used);
        assert_eq!(back.wasted_ms, ev.wasted_ms);
        assert_eq!(back.regret_ms, ev.regret_ms);
    }
}
