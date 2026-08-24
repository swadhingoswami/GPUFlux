use crate::object::ObjectSpec;
use crate::resource::ResourceState;

/// Coarse state "regime" suffix appended to observation-store bucket keys so
/// history is conditioned on the resource regime in which it was collected.
///
/// This is Phase 4's contention-aware upgrade: instead of one aggregate per
/// action over *all* conditions, we keep separate aggregates per regime. A
/// prediction reads the bucket matching the current regime, so a history
/// collected under heavy NVMe contention no longer pollutes the low-contention
/// estimate (and vice versa).
pub fn regime_suffix(state: &ResourceState) -> String {
    let cpu = match state.cpu_util {
        Some(u) if u < 0.33 => "cpu-lo",
        Some(u) if u < 0.66 => "cpu-med",
        Some(_) => "cpu-hi",
        None => "cpu-na",
    };
    let io = match state.nvme_latency_us {
        Some(l) if l >= 200.0 => "io-hi",
        _ => "io-lo",
    };
    let gpu = match state.gpu_util {
        Some(u) if u >= 0.5 => "gpu-hi",
        _ => "gpu-lo",
    };
    format!("{cpu}/{io}/{gpu}")
}

/// Full bucket key for `action` over `object` in the regime of `state`.
pub fn action_bucket(object: &ObjectSpec, action: &str, state: &ResourceState) -> String {
    format!(
        "{}/sim/{}/{}",
        action,
        object.size_bytes,
        regime_suffix(state)
    )
}
