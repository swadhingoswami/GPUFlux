use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda_backend.cu");
    println!("cargo:rerun-if-changed=cuda_backend.h");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".into());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());

    let obj = out.join("cuda_backend.o");
    let lib = out.join("libcuda_backend.a");

    let status = Command::new(&nvcc)
        .args([
            &format!("-arch={arch}"),
            "-O3",
            "-c",
            "cuda_backend.cu",
            "-o",
        ])
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run `{nvcc}` ({e}).\n\
                 gpuflux-cuda builds ONLY on a machine with the CUDA toolkit (NVIDIA GPU + nvcc).\n\
                 On other machines use the gpuflux Rust sim backends. Set NVCC=/path/to/nvcc and \
                 CUDA_ARCH=sm_XY to override."
            )
        });
    assert!(status.success(), "nvcc compilation failed");

    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let status = Command::new(&ar)
        .args(["crs"])
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("ar not found");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=cuda_backend");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvidia-ml");

    if let Ok(home) = env::var("CUDA_HOME") {
        let lib64 = PathBuf::from(&home).join("lib64");
        if lib64.exists() {
            println!("cargo:rustc-link-search=native={}", lib64.display());
        }
    }
}
