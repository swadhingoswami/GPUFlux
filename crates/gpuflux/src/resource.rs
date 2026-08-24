use crate::now_unix_ms;

/// A snapshot of the machine state at decision time, normalized so the decision
/// engine never touches hardware specifics.
///
/// Fields that cannot be measured on the current platform are `None`; cost
/// models treat `None` as "no evidence, assume neutral". On the CUDA box, GPU
/// and PCIe fields are filled from NVML / CUDA runtime telemetry.
#[derive(Debug, Clone, Default)]
pub struct ResourceState {
    /// CPU utilization as a fraction in [0, 1].
    pub cpu_util: Option<f64>,
    /// GPU utilization as a fraction in [0, 1].
    pub gpu_util: Option<f64>,
    pub gpu_memory_used: Option<u64>,
    pub nvme_queue_depth: Option<u32>,
    /// Single-4KiB-read latency in microseconds.
    pub nvme_latency_us: Option<f64>,
    /// PCIe bandwidth available to the GPU, bytes/s.
    pub pcie_bandwidth: Option<f64>,
    /// Remote CPU utilization (Phase 8), fraction in [0,1]. None when no
    /// remote node is in play.
    pub remote_cpu_util: Option<f64>,
    pub captured_at_unix_ms: u64,
}

impl ResourceState {
    pub fn now() -> Self {
        ResourceState {
            captured_at_unix_ms: now_unix_ms(),
            ..Default::default()
        }
    }
}
