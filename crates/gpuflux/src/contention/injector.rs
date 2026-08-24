use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crate::executor::nocache::set_nocache;
use crate::util::fill_xorshift;

/// Spins `cores` threads to create sustained CPU pressure.
pub struct CpuContender {
    stop: Arc<AtomicBool>,
    _handles: Vec<JoinHandle<()>>,
}

impl CpuContender {
    pub fn start(cores: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..cores {
            let stop = Arc::clone(&stop);
            handles.push(std::thread::spawn(move || {
                let mut i = 0u64;
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    i = i
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                }
            }));
        }
        Self {
            stop,
            _handles: handles,
        }
    }
}

impl Drop for CpuContender {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Hammers the storage device with F_NOCACHE reads to create NVMe/queue
/// contention. Reads bypass the page cache, so they genuinely hit the device
/// and share its queue with the measured move path.
pub struct IoContender {
    stop: Arc<AtomicBool>,
    _handles: Vec<JoinHandle<()>>,
}

impl IoContender {
    pub fn start(path: std::path::PathBuf, readers: usize, chunk_bytes: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..readers {
            let stop = Arc::clone(&stop);
            let path = path.clone();
            handles.push(std::thread::spawn(move || {
                let mut f = match File::open(&path) {
                    Ok(f) => f,
                    Err(_) => return,
                };
                let _ = set_nocache(f.as_raw_fd());
                let mut buf = vec![0u8; chunk_bytes];
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    match f.read(&mut buf) {
                        Ok(0) | Err(_) => {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            let _ = f.seek(SeekFrom::Start(0));
                        }
                        Ok(_) => {}
                    }
                }
            }));
        }
        Self {
            stop,
            _handles: handles,
        }
    }
}

impl Drop for IoContender {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// GPU contention. No-op on the dev box (no CUDA); on the CUDA box this would
/// launch a busy kernel to occupy the GPU so recompute kernels queue behind it.
/// The interface is identical either way.
pub struct GpuContender {
    stop: Arc<AtomicBool>,
    _handles: Vec<JoinHandle<()>>,
}

impl GpuContender {
    pub fn start(enabled: bool) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        if enabled {
            let stop = Arc::clone(&stop);
            handles.push(std::thread::spawn(move || {
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    // Placeholder: launch busy CUDA kernels here on the GPU box.
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }));
        }
        Self {
            stop,
            _handles: handles,
        }
    }
}

impl Drop for GpuContender {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Writes a large pseudo-random file used as the target for IO contenders.
pub fn write_contention_file(path: &Path, size_bytes: u64) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let chunk = (64 * 1024 * 1024) as usize;
    let mut buf = vec![0u8; chunk];
    let mut remaining = size_bytes;
    let mut seed = 0x9E37_79B9_7F4A_7C15;
    while remaining > 0 {
        let n = remaining.min(chunk as u64) as usize;
        fill_xorshift(&mut buf[..n], seed);
        seed = seed.wrapping_add(1);
        f.write_all(&buf[..n])?;
        remaining -= n as u64;
    }
    f.sync_all()?;
    Ok(())
}

/// Convenience handle that starts and owns all requested contenders.
pub struct Contention {
    _cpu: CpuContender,
    _io: IoContender,
    _gpu: GpuContender,
}

impl Contention {
    pub fn start(cpu_cores: usize, io_readers: usize, io_file: &Path, gpu: bool) -> Self {
        Self {
            _cpu: CpuContender::start(cpu_cores),
            _io: IoContender::start(io_file.to_path_buf(), io_readers, 256 * 1024),
            _gpu: GpuContender::start(gpu),
        }
    }
}
