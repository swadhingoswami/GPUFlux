use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::executor::nocache::{full_fsync, set_nocache};
use crate::executor::traits::{
    ExecutionControl, MoveExecutor, MoveReport, Progress, RecomputeExecutor, RecomputeReport,
};
use crate::object::ObjectSpec;
use crate::util::fill_xorshift;

/// Coarse I/O / fill chunk. Small enough for frequent checkpoints (fallback
/// responsiveness), large enough to avoid syscall overhead.
pub const CHUNK: usize = 16 * 1024 * 1024;

/// Emit a checkpoint callback if the control requested one.
fn emit(
    phase: &'static str,
    bytes_done: u64,
    fraction_done: f64,
    t0: Instant,
    ctrl: &mut ExecutionControl,
) {
    if let Some(cb) = ctrl.on_checkpoint.as_deref_mut() {
        cb(&Progress {
            phase,
            elapsed: t0.elapsed(),
            fraction_done,
            bytes_done,
        });
    }
}

/// Move path that persists X to a local SSD-backed file and reads it back,
/// modeling the "X already exists in storage, restore it to GPU memory" case.
///
/// Timing uses F_NOCACHE + F_FULLFSYNC on macOS so reads bypass the page cache
/// and writes flush to the device. The object buffer is generated once and
/// reused, so write/read timing is not polluted by data generation.
///
/// Executes in chunks and emits a checkpoint per chunk (fraction over
/// write+read work) so the runtime can abort mid-move if it is going to miss
/// its deadline.
pub struct SimMoveExecutor {
    dir: PathBuf,
    buffers: HashMap<u64, Vec<u8>>,
}

impl SimMoveExecutor {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            buffers: HashMap::new(),
        }
    }
}

impl MoveExecutor for SimMoveExecutor {
    fn name(&self) -> &str {
        "sim/ssd"
    }

    fn move_to_gpu(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<MoveReport> {
        let t_start = Instant::now();

        let buf = match self.buffers.entry(spec.id) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(v) => {
                let mut b = vec![0u8; spec.size_bytes as usize];
                fill_xorshift(&mut b, spec.seed());
                v.insert(b)
            }
        };

        let path = self.dir.join(format!("obj-{}.bin", spec.id));
        let total_work = (2 * spec.size_bytes) as f64;
        let len = buf.len() as u64;
        let mut aborted = false;

        let tw0 = Instant::now();
        {
            let mut f = File::create(&path)?;
            let _ = set_nocache(f.as_raw_fd());
            let mut written: u64 = 0;
            while written < len {
                if control.is_aborted() {
                    aborted = true;
                    break;
                }
                let n = ((len - written) as usize).min(CHUNK);
                f.write_all(&buf[written as usize..written as usize + n])?;
                written += n as u64;
                emit(
                    "move:write",
                    written,
                    written as f64 / total_work,
                    tw0,
                    control,
                );
                if control.is_aborted() {
                    aborted = true;
                    break;
                }
            }
            if !aborted {
                f.flush()?;
                full_fsync(f.as_raw_fd());
            }
        }
        let write_elapsed = tw0.elapsed();

        if aborted {
            let total = t_start.elapsed();
            let deadline_met = deadline.is_none_or(|d| Instant::now() <= d);
            return Ok(MoveReport {
                total,
                bytes: spec.size_bytes,
                write_elapsed,
                read_elapsed: Duration::ZERO,
                deadline_met,
                aborted: true,
            });
        }

        let tr0 = Instant::now();
        let mut out = vec![0u8; len as usize];
        let mut read: u64 = 0;
        {
            let mut f = File::open(&path)?;
            let _ = set_nocache(f.as_raw_fd());
            while read < len {
                if control.is_aborted() {
                    aborted = true;
                    break;
                }
                let n = ((len - read) as usize).min(CHUNK);
                f.read_exact(&mut out[read as usize..read as usize + n])?;
                read += n as u64;
                emit(
                    "move:read",
                    read,
                    (spec.size_bytes as f64 + read as f64) / total_work,
                    tr0,
                    control,
                );
                if control.is_aborted() {
                    aborted = true;
                    break;
                }
            }
        }
        let read_elapsed = tr0.elapsed();

        if !aborted && &out != buf {
            return Err(Error::Invalid(format!(
                "move integrity mismatch for object {}",
                spec.id
            )));
        }

        let total = t_start.elapsed();
        let deadline_met = deadline.is_none_or(|d| Instant::now() <= d);
        Ok(MoveReport {
            total,
            bytes: spec.size_bytes,
            write_elapsed,
            read_elapsed,
            deadline_met,
            aborted,
        })
    }
}

/// Recompute path that regenerates X on the CPU by running `passes` fill passes
/// over the buffer. Deterministic given (size, seed, passes); stands in for a
/// GPU kernel until the CUDA backend lands. `passes` may be fractional.
///
/// `gpu_load` simulates GPU contention (Phase 4, GPU half): on a real box a busy
/// GPU slows a kernel's effective throughput, so the same work takes
/// `1/(1-gpu_load)` longer. We model that honestly by doing *more real work*
/// (effective passes = passes/(1-gpu_load)), so the timing is genuine, not a
/// sleep.
///
/// Fills in chunks and emits a checkpoint per chunk so the runtime can abort
/// mid-recompute.
pub struct SimRecomputeExecutor {
    pub passes: f64,
    pub gpu_load: f64,
}

impl Default for SimRecomputeExecutor {
    fn default() -> Self {
        Self {
            passes: 2.0,
            gpu_load: 0.0,
        }
    }
}

impl SimRecomputeExecutor {
    /// Fill `buf` (or a prefix of it) in chunks, emitting checkpoints and
    /// respecting abort. Returns true if aborted.
    fn fill_chunked(
        buf: &mut [u8],
        seed: u64,
        t0: Instant,
        phase: &'static str,
        bytes_done: u64,
        total_work: f64,
        control: &mut ExecutionControl,
    ) -> bool {
        let mut done: u64 = 0;
        while done < buf.len() as u64 {
            if control.is_aborted() {
                return true;
            }
            let n = ((buf.len() as u64 - done) as usize).min(CHUNK);
            fill_xorshift(
                &mut buf[done as usize..done as usize + n],
                seed.wrapping_add(done),
            );
            done += n as u64;
            emit(
                phase,
                bytes_done + done,
                (bytes_done + done) as f64 / total_work,
                t0,
                control,
            );
            if control.is_aborted() {
                return true;
            }
        }
        false
    }
}

impl RecomputeExecutor for SimRecomputeExecutor {
    fn name(&self) -> &str {
        "sim/recompute"
    }

    fn recompute(
        &mut self,
        spec: &ObjectSpec,
        deadline: Option<Instant>,
        control: &mut ExecutionControl,
    ) -> Result<RecomputeReport> {
        let t0 = Instant::now();
        let effective_passes = self.passes / (1.0 - self.gpu_load).max(0.05);
        let full = effective_passes.floor() as usize;
        let frac = effective_passes - full as f64;
        let total_work = effective_passes * spec.size_bytes as f64;
        let seed = spec.seed();
        let mut buf = vec![0u8; spec.size_bytes as usize];
        let mut aborted = false;
        let mut done: u64 = 0;

        for p in 0..full {
            let start = done;
            aborted = Self::fill_chunked(
                buf.as_mut_slice(),
                seed.wrapping_add(p as u64),
                t0,
                "recompute:pass",
                start,
                total_work,
                control,
            );
            if aborted {
                break;
            }
            done = spec.size_bytes * (p as u64 + 1);
        }

        if !aborted && frac > 0.0 {
            let n = (spec.size_bytes as f64 * frac) as usize;
            aborted = Self::fill_chunked(
                &mut buf[..n],
                seed.wrapping_add(full as u64),
                t0,
                "recompute:partial",
                done,
                total_work,
                control,
            );
        }

        let total = t0.elapsed();
        let deadline_met = deadline.is_none_or(|d| Instant::now() <= d);
        Ok(RecomputeReport {
            total,
            bytes: spec.size_bytes,
            passes: full as u32,
            deadline_met,
            aborted,
        })
    }
}
