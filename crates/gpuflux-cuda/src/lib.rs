//! C++/CUDA execution backend for GPUFlux.
//!
//! Rust holds the control plane; this crate is the thin execution plane. The
//! decision engine never sees CUDA: it calls `MoveExecutor`/`RecomputeExecutor`
//! (same traits as the sim backends) and receives `MoveReport`/`RecomputeReport`
//! plus a normalized `ResourceState` from `CudaSampler`. The actual CUDA work
//! lives in `cuda_backend.cu` behind a plain C ABI (`cuda_backend.h`).
//!
//! Builds ONLY on a machine with the CUDA toolkit (`nvcc`). On other machines
//! use the gpuflux sim backends.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::os::raw::c_int;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use gpuflux::error::Result;
use gpuflux::executor::{
    ExecutionControl, MoveExecutor, MoveReport, Progress, RecomputeExecutor, RecomputeReport,
};
use gpuflux::object::ObjectSpec;
use gpuflux::resource::ResourceState;
use gpuflux::telemetry::ResourceSampler;

#[repr(C)]
#[derive(Clone, Copy)]
struct ObjectDesc {
    id: u64,
    size_bytes: u64,
    loc: c_int,
    recompute_passes: c_int,
    nvme_path: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ExecutionResult {
    elapsed_ms: f64,
    bytes_moved: u64,
    success: u8,
    aborted: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ResourceSnapshot {
    gpu_util: f32,
    gpu_mem_used: f32,
    nvme_queue_depth: f32,
    nvme_latency_us: f32,
    pcie_bandwidth: f32,
}

type CheckpointFn = extern "C" fn(*const c_char, f64, f64);

unsafe extern "C" {
    fn cuda_device_count() -> c_int;
    fn cuda_snapshot(out: *mut ResourceSnapshot);
    fn cuda_move(
        obj: ObjectDesc,
        cb: Option<CheckpointFn>,
        abort_flag: *const c_int,
    ) -> ExecutionResult;
    fn cuda_recompute(
        obj: ObjectDesc,
        cb: Option<CheckpointFn>,
        abort_flag: *const c_int,
    ) -> ExecutionResult;
}

// Callback trampoline context: the current ExecutionControl (raw pointer, valid
// for the duration of the synchronous call) plus the C-side abort flag mirror.
// The C code reads the mirror at its checkpoints; our trampoline writes it when
// the Rust control requests an abort.
thread_local! {
    static CALLBACK_CTX: RefCell<Option<CallbackCtx>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy)]
struct CallbackCtx {
    control: *mut ExecutionControl<'static>,
    abort_mirror: *mut c_int,
}

unsafe impl Send for CallbackCtx {}
unsafe impl Sync for CallbackCtx {}

extern "C" fn trampoline(phase: *const c_char, elapsed_ms: f64, fraction_done: f64) {
    CALLBACK_CTX.with(|cell| {
        let Some(ctx) = cell.borrow().as_ref().copied() else {
            return;
        };
        // C string literals in the backend are static; safe to reborrow.
        let phase = unsafe { CStr::from_ptr(phase) };
        let phase_str: &'static str =
            unsafe { std::mem::transmute(std::str::from_utf8_unchecked(phase.to_bytes())) };
        let progress = Progress {
            phase: phase_str,
            elapsed: Duration::from_secs_f64(elapsed_ms / 1000.0),
            fraction_done,
            bytes_done: 0,
        };
        unsafe {
            let ctrl = &mut *ctx.control;
            if let Some(cb) = ctrl.on_checkpoint.as_mut() {
                cb(&progress);
            }
            if ctrl.abort.load(std::sync::atomic::Ordering::Relaxed) {
                *ctx.abort_mirror = 1;
            }
        }
    });
}

/// Runs a backend call, wiring `control` into the callback trampoline.
unsafe fn with_control<T>(
    control: &mut ExecutionControl,
    f: impl FnOnce(Option<CheckpointFn>, *const c_int) -> T,
) -> T {
    let mut abort_mirror: c_int = 0;
    CALLBACK_CTX.with(|cell| {
        // Erase the borrow lifetime: the control outlives the synchronous call.
        let raw: *mut ExecutionControl = control;
        let raw_static: *mut ExecutionControl<'static> = unsafe { std::mem::transmute(raw) };
        *cell.borrow_mut() = Some(CallbackCtx {
            control: raw_static,
            abort_mirror: &mut abort_mirror,
        });
    });
    let r = f(Some(trampoline), &abort_mirror);
    CALLBACK_CTX.with(|cell| {
        *cell.borrow_mut() = None;
    });
    r
}

fn desc(spec: &ObjectSpec, passes: c_int, nvme_path: &CString) -> ObjectDesc {
    ObjectDesc {
        id: spec.id,
        size_bytes: spec.size_bytes,
        loc: 2, // nvme
        recompute_passes: passes,
        nvme_path: nvme_path.as_ptr(),
    }
}

fn check(res: ExecutionResult, what: &str) -> Result<ExecutionResult> {
    if res.success == 0 {
        Err(gpuflux::Error::Backend(what.to_string()))
    } else {
        Ok(res)
    }
}

/// Number of CUDA devices visible to the process.
pub fn device_count() -> i32 {
    unsafe { cuda_device_count() }
}

/// MOVE and RECOMPUTE via the C++/CUDA backend.
pub struct CudaBackend {
    dir: PathBuf,
    passes: c_int,
}

impl CudaBackend {
    pub fn new(dir: PathBuf, passes: c_int) -> Result<Self> {
        if device_count() <= 0 {
            return Err(gpuflux::Error::Backend(
                "no CUDA device available for CudaBackend".into(),
            ));
        }
        Ok(Self { dir, passes })
    }

    fn nvme_path(&self, spec: &ObjectSpec) -> CString {
        let p = self.dir.join(format!("obj-{}.bin", spec.id));
        CString::new(p.to_str().unwrap_or("/tmp/missing")).expect("path is utf8")
    }
}

impl MoveExecutor for CudaBackend {
    fn name(&self) -> &str {
        "cuda/ssd"
    }

    fn move_to_gpu(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<MoveReport> {
        let path = self.nvme_path(spec);
        let obj = desc(spec, self.passes, &path);
        let res = unsafe { with_control(control, |cb, abort| cuda_move(obj, cb, abort)) };
        let res = check(res, "cuda_move failed")?;
        let total = Duration::from_secs_f64(res.elapsed_ms / 1000.0);
        let deadline_met = deadline.is_none_or(|d| Instant::now() <= d);
        Ok(MoveReport {
            total,
            bytes: res.bytes_moved,
            write_elapsed: total,
            read_elapsed: Duration::ZERO,
            deadline_met,
            aborted: res.aborted != 0,
        })
    }
}

impl RecomputeExecutor for CudaBackend {
    fn name(&self) -> &str {
        "cuda/kernel"
    }

    fn recompute(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<RecomputeReport> {
        let path = CString::new("").expect("empty");
        let obj = desc(spec, self.passes, &path);
        let res = unsafe { with_control(control, |cb, abort| cuda_recompute(obj, cb, abort)) };
        let res = check(res, "cuda_recompute failed")?;
        let total = Duration::from_secs_f64(res.elapsed_ms / 1000.0);
        let deadline_met = deadline.is_none_or(|d| Instant::now() <= d);
        Ok(RecomputeReport {
            total,
            bytes: res.bytes_moved,
            passes: self.passes as u32,
            deadline_met,
            aborted: res.aborted != 0,
        })
    }
}

/// Normalized telemetry sampler backed by NVML + CUDA runtime.
pub struct CudaSampler;

impl ResourceSampler for CudaSampler {
    fn sample(&self) -> ResourceState {
        let mut snap = ResourceSnapshot {
            gpu_util: -1.0,
            gpu_mem_used: -1.0,
            nvme_queue_depth: -1.0,
            nvme_latency_us: -1.0,
            pcie_bandwidth: -1.0,
        };
        unsafe { cuda_snapshot(&mut snap) };
        let mut s = ResourceState::now();
        s.gpu_util = (snap.gpu_util >= 0.0).then_some(snap.gpu_util as f64);
        s.gpu_memory_used = (snap.gpu_mem_used >= 0.0).then_some(snap.gpu_mem_used as u64);
        s.nvme_queue_depth = (snap.nvme_queue_depth >= 0.0).then_some(snap.nvme_queue_depth as u32);
        s.nvme_latency_us = (snap.nvme_latency_us >= 0.0).then_some(snap.nvme_latency_us as f64);
        s.pcie_bandwidth = (snap.pcie_bandwidth >= 0.0).then_some(snap.pcie_bandwidth as f64);
        s
    }
}
