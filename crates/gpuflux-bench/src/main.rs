use std::path::PathBuf;

use gpuflux::Result;
use gpuflux::executor::{MoveExecutor, RecomputeExecutor, SimMoveExecutor, SimRecomputeExecutor};
use gpuflux::object::{DataLoc, ObjectSpec};
use gpuflux::observation::ObservationStore;
use gpuflux_bench::cli::{get_arg, parse_size, stats};

fn main() -> Result<()> {
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "20").parse().unwrap_or(20);
    let warmup: usize = get_arg("--warmup", "3").parse().unwrap_or(3);
    let passes: f64 = get_arg("--passes", "2").parse().unwrap_or(2.0);

    let base = std::env::temp_dir().join("gpuflux-bench");
    let dir = PathBuf::from(get_arg(
        "--dir",
        base.to_str().unwrap_or("/tmp/gpuflux-bench"),
    ));
    let store_path = PathBuf::from(get_arg(
        "--store",
        base.join("gpuflux.db").to_str().unwrap_or_default(),
    ));
    std::fs::create_dir_all(&dir)?;

    let store = ObservationStore::open(&store_path)?;
    let mut mv = SimMoveExecutor::new(dir.clone());
    let mut rc = SimRecomputeExecutor {
        passes,
        ..Default::default()
    };
    let spec = ObjectSpec::new(1, size, DataLoc::GpuMemory);

    let bucket_move = format!("move/sim/{}", size);
    let bucket_recomp = format!("recompute/sim/{}", size);

    println!("=== GPUFlux Phase 0: A->X->B benchmark ===");
    println!(
        "object size : {} bytes ({} MiB)",
        size,
        size / (1024 * 1024)
    );
    println!("iterations  : {} (warmup {})", iters, warmup);
    println!("recompute   : {} fill passes", passes);
    println!("nvme dir    : {}", dir.display());
    println!("store       : {}", store_path.display());
    println!("cache ctl   : F_NOCACHE + F_FULLFSYNC (macOS)");
    println!();

    for _ in 0..warmup {
        mv.move_to_gpu(
            &spec,
            None,
            &mut gpuflux::executor::ExecutionControl::none(),
        )?;
        rc.recompute(
            &spec,
            None,
            &mut gpuflux::executor::ExecutionControl::none(),
        )?;
    }

    let mut move_durs = Vec::new();
    let mut recomp_durs = Vec::new();
    for i in 0..iters {
        // Alternate order each iteration to reduce bias from thermal/OS effects.
        if i % 2 == 0 {
            let r = mv.move_to_gpu(
                &spec,
                None,
                &mut gpuflux::executor::ExecutionControl::none(),
            )?;
            move_durs.push(r.total);
            store.record(&bucket_move, r.total.as_secs_f64() * 1000.0)?;
            let r = rc.recompute(
                &spec,
                None,
                &mut gpuflux::executor::ExecutionControl::none(),
            )?;
            recomp_durs.push(r.total);
            store.record(&bucket_recomp, r.total.as_secs_f64() * 1000.0)?;
        } else {
            let r = rc.recompute(
                &spec,
                None,
                &mut gpuflux::executor::ExecutionControl::none(),
            )?;
            recomp_durs.push(r.total);
            store.record(&bucket_recomp, r.total.as_secs_f64() * 1000.0)?;
            let r = mv.move_to_gpu(
                &spec,
                None,
                &mut gpuflux::executor::ExecutionControl::none(),
            )?;
            move_durs.push(r.total);
            store.record(&bucket_move, r.total.as_secs_f64() * 1000.0)?;
        }
    }

    let ms = stats(&move_durs, size);
    let rs = stats(&recomp_durs, size);

    println!("op         n    mean(ms) std(ms)  min    p50    p90    p95    max    MB/s");
    println!(
        "{:<10} {:>3} {:>9.2} {:>7.2} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>7.1}",
        "move",
        ms.n,
        ms.mean_ms,
        ms.std_ms,
        ms.min_ms,
        ms.p50_ms,
        ms.p90_ms,
        ms.p95_ms,
        ms.max_ms,
        ms.mb_s
    );
    println!(
        "{:<10} {:>3} {:>9.2} {:>7.2} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>7.1}",
        "recompute",
        rs.n,
        rs.mean_ms,
        rs.std_ms,
        rs.min_ms,
        rs.p50_ms,
        rs.p90_ms,
        rs.p95_ms,
        rs.max_ms,
        rs.mb_s
    );
    println!();
    println!(
        "mean ratio  move/recompute = {:.3}",
        ms.mean_ms / rs.mean_ms
    );
    println!("p90 ratio   move/recompute = {:.3}", ms.p90_ms / rs.p90_ms);
    println!(
        "cv move     = {:.3}, cv recompute = {:.3}",
        ms.std_ms / ms.mean_ms,
        rs.std_ms / rs.mean_ms
    );
    println!();

    let am = store.aggregate(&bucket_move)?.expect("move aggregate");
    let ar = store
        .aggregate(&bucket_recomp)?
        .expect("recompute aggregate");
    println!("store aggregate readback:");
    println!(
        "  move:      n={} ewma_mean={:.2}ms ewma_std={:.2}ms p50={:.1} p90={:.1} p95={:.1}",
        am.sample_count,
        am.ewma_mean,
        am.ewma_std(),
        am.p(0.50),
        am.p(0.90),
        am.p(0.95)
    );
    println!(
        "  recompute: n={} ewma_mean={:.2}ms ewma_std={:.2}ms p50={:.1} p90={:.1} p95={:.1}",
        ar.sample_count,
        ar.ewma_mean,
        ar.ewma_std(),
        ar.p(0.50),
        ar.p(0.90),
        ar.p(0.95)
    );

    Ok(())
}
