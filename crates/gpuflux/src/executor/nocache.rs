use std::os::unix::io::RawFd;

// F_NOCACHE and F_FULLFSYNC are not guaranteed to be exposed by the libc crate
// for every Apple SDK version, so we define them explicitly (values from
// <sys/fcntl.h> on macOS).
#[cfg(target_os = "macos")]
const F_NOCACHE: libc::c_int = 30;
#[cfg(target_os = "macos")]
const F_FULLFSYNC: libc::c_int = 51;

/// Bypass the OS page cache for the given file descriptor so that reads/writes
/// hit the storage device directly. Used so benchmark timings reflect device
/// speed rather than page-cache hits.
#[cfg(target_os = "macos")]
pub fn set_nocache(fd: RawFd) -> std::io::Result<()> {
    let r = unsafe { libc::fcntl(fd, F_NOCACHE, 1) };
    if r == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Flush dirty pages for this descriptor all the way to the storage device
/// (Apple's F_FULLFSYNC, which also flushes the drive's internal write cache).
#[cfg(target_os = "macos")]
pub fn full_fsync(fd: RawFd) {
    unsafe {
        libc::fcntl(fd, F_FULLFSYNC);
    }
}

// Non-macOS fallbacks: no cache control. Linux should switch to O_DIRECT when
// the CUDA/NVMe phase lands on a Linux box.
#[cfg(not(target_os = "macos"))]
pub fn set_nocache(_fd: RawFd) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn full_fsync(_fd: RawFd) {}
