use std::path::PathBuf;

use crate::resource::ResourceState;

// macOS-only telemetry (Mach CPU ticks, F_NOCACHE NVMe probe). On other
// platforms the sampler reports an empty snapshot (CUDA/NVML supplies it on the
// GPU box).
#[cfg(target_os = "macos")]
use crate::executor::nocache::set_nocache;
#[cfg(target_os = "macos")]
use std::io::Read;
#[cfg(target_os = "macos")]
use std::os::unix::io::AsRawFd;
#[cfg(target_os = "macos")]
use std::path::Path;
#[cfg(target_os = "macos")]
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use std::time::Instant;

/// Samples a normalized `ResourceState`. The CUDA backend supplies its own
/// sampler (NVML) on the GPU box; this module only knows `ResourceState`.
pub trait ResourceSampler {
    fn sample(&self) -> ResourceState;
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct CpuTicks {
    user: f64,
    sys: f64,
    idle: f64,
    nice: f64,
}

/// System sampler for the development machine (macOS).
///
/// Fills what is measurable locally: CPU utilization (delta of Mach CPU ticks)
/// and NVMe latency (timed F_NOCACHE 4KiB read against a probe file). GPU and
/// PCIe fields stay `None`; the CUDA box adds them.
pub struct SystemSampler {
    probe: PathBuf,
    #[cfg(target_os = "macos")]
    prev: Mutex<Option<CpuTicks>>,
}

impl SystemSampler {
    pub fn new(probe: PathBuf) -> Self {
        Self {
            probe,
            #[cfg(target_os = "macos")]
            prev: Mutex::new(None),
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn read_cpu_ticks() -> Option<CpuTicks> {
    use std::mem;
    unsafe {
        let host = libc::mach_host_self();
        let mut info = mem::zeroed::<libc::host_cpu_load_info>();
        let mut count = (mem::size_of::<libc::host_cpu_load_info>()
            / mem::size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        let kr = libc::host_statistics(
            host,
            libc::HOST_CPU_LOAD_INFO,
            &mut info as *mut libc::host_cpu_load_info as *mut libc::integer_t,
            &mut count,
        );
        if kr != libc::KERN_SUCCESS {
            return None;
        }
        Some(CpuTicks {
            user: info.cpu_ticks[libc::CPU_STATE_USER as usize] as f64,
            sys: info.cpu_ticks[libc::CPU_STATE_SYSTEM as usize] as f64,
            idle: info.cpu_ticks[libc::CPU_STATE_IDLE as usize] as f64,
            nice: info.cpu_ticks[libc::CPU_STATE_NICE as usize] as f64,
        })
    }
}

#[cfg(target_os = "macos")]
fn sample_cpu_util(prev: &Mutex<Option<CpuTicks>>) -> Option<f64> {
    let ticks = read_cpu_ticks()?;
    let mut guard = prev.lock().unwrap();
    let util = match *guard {
        Some(p) => {
            let du = ticks.user - p.user;
            let ds = ticks.sys - p.sys;
            let di = ticks.idle - p.idle;
            let dn = ticks.nice - p.nice;
            let total = du + ds + di + dn;
            if total > 0.0 {
                Some((du + ds) / total)
            } else {
                None
            }
        }
        None => None,
    };
    *guard = Some(ticks);
    util
}

#[cfg(target_os = "macos")]
fn sample_nvme_latency_us(path: &Path) -> Option<f64> {
    let mut f = std::fs::File::open(path).ok()?;
    let _ = set_nocache(f.as_raw_fd());
    let mut buf = [0u8; 4096];
    let t0 = Instant::now();
    f.read_exact(&mut buf).ok()?;
    Some(t0.elapsed().as_secs_f64() * 1e6)
}

#[cfg(target_os = "macos")]
impl ResourceSampler for SystemSampler {
    fn sample(&self) -> ResourceState {
        let mut s = ResourceState::now();
        s.cpu_util = sample_cpu_util(&self.prev);
        s.nvme_latency_us = sample_nvme_latency_us(&self.probe);
        s.nvme_queue_depth = None;
        s.gpu_util = None;
        s.gpu_memory_used = None;
        s.pcie_bandwidth = None;
        s
    }
}

#[cfg(not(target_os = "macos"))]
impl ResourceSampler for SystemSampler {
    fn sample(&self) -> ResourceState {
        let _ = &self.probe;
        ResourceState::now()
    }
}
