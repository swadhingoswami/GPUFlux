use std::path::PathBuf;
use std::time::Duration;

use gpuflux::contention::Contention;
use gpuflux::decision::engine::DecisionEngine;
use gpuflux::decision::policy::{AlwaysMove, AlwaysRecompute, DeadlineAware, ExpectedCost, Policy};
use gpuflux::executor::{SimMoveExecutor, SimRecomputeExecutor};
use gpuflux::object::{DataLoc, ObjectSpec};
use gpuflux::observation::ObservationStore;
use gpuflux::prediction::{
    CurrentStateCostModel, CurrentStatePredictor, HistoricalPredictor, Predictor,
};
use gpuflux::telemetry::{ResourceSampler, SystemSampler};
use gpuflux::{Error, Result};
use gpuflux_bench::cli::{get_arg, parse_size, stats};

fn main() -> Result<()> {
    let policy = get_arg("--policy", "always_move");
    let fallback = get_arg("--fallback", "0") == "1";
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "30").parse().unwrap_or(30);
    let prime: usize = get_arg("--prime", "15").parse().unwrap_or(15);
    let passes: f64 = get_arg("--passes", "2").parse().unwrap_or(2.0);
    let lambda: f64 = get_arg("--lambda", "200").parse().unwrap_or(200.0);
    let io_readers: usize = get_arg("--io-readers", "0").parse().unwrap_or(0);
    let io_size: u64 = parse_size(&get_arg("--io-file-size", "512MiB"))?;
    let deadline_ms: f64 = {
        let s = get_arg("--deadline-ms", "260");
        s.parse()
            .map_err(|_| Error::Invalid("bad --deadline-ms".into()))?
    };

    let base = std::env::temp_dir().join("gpuflux-bench");
    let dir = PathBuf::from(get_arg(
        "--dir",
        base.to_str().unwrap_or("/tmp/gpuflux-bench"),
    ));
    let store_path = PathBuf::from(get_arg(
        "--store",
        base.join(format!("gpuflux-{}-fb{}.db", policy, fallback as u8))
            .to_str()
            .unwrap_or_default(),
    ));
    std::fs::create_dir_all(&dir)?;

    let io_file = dir.join("contention.bin");
    if io_readers > 0 && !io_file.exists() {
        println!(
            "creating contention file {} ({} MiB) ...",
            io_file.display(),
            io_size / (1024 * 1024)
        );
        gpuflux::contention::injector::write_contention_file(&io_file, io_size)?;
    }

    let store = ObservationStore::open(&store_path)?;
    let spec = ObjectSpec::new(23, size, DataLoc::GpuMemory);

    let probe = dir.join("probe.bin");
    std::fs::write(&probe, vec![0u8; 4096])?;
    let sampler = SystemSampler::new(probe);

    let model = CurrentStateCostModel {
        recompute_passes: passes,
        ..Default::default()
    };

    let engine_policy: Box<dyn Policy> = match policy.as_str() {
        "always_move" => Box::new(AlwaysMove),
        "always_recompute" => Box::new(AlwaysRecompute),
        "expected_cost" => Box::new(ExpectedCost),
        "deadline_aware" => Box::new(DeadlineAware { lambda, mu: 0.0 }),
        other => return Err(Error::Invalid(format!("unknown policy: {other}"))),
    };
    let predictor: Box<dyn Predictor> = match policy.as_str() {
        "deadline_aware" => Box::new(HistoricalPredictor::new(store.clone(), model)),
        _ => Box::new(CurrentStatePredictor { model }),
    };

    let mut engine = DecisionEngine::new(
        store.clone(),
        engine_policy,
        predictor,
        Box::new(SimMoveExecutor::new(dir.clone())),
        Box::new(SimRecomputeExecutor {
            passes,
            ..Default::default()
        }),
    );

    let _contention = Contention::start(0, io_readers, &io_file, false);

    for _ in 0..prime {
        let _ = engine.decide(&spec, sampler.sample(), Some(deadline_ms))?;
    }

    let mut actuals = Vec::with_capacity(iters);
    let mut deadline_results = Vec::with_capacity(iters);
    let mut fallbacks = 0usize;
    let mut wasted = Vec::with_capacity(iters);

    for _ in 0..iters {
        let state = sampler.sample();
        let outcome = if fallback {
            engine.decide_with_fallback(&spec, state, Some(deadline_ms))?
        } else {
            engine.decide(&spec, state, Some(deadline_ms))?
        };
        actuals.push(Duration::from_secs_f64(outcome.actual_ms / 1000.0));
        deadline_results.push(outcome.deadline_met);
        if outcome.fallback_used {
            fallbacks += 1;
        }
        wasted.push(outcome.wasted_ms);
    }

    let a = stats(&actuals, size);
    let met = deadline_results.iter().filter(|m| **m).count();
    let mean_wasted = if wasted.is_empty() {
        0.0
    } else {
        wasted.iter().sum::<f64>() / wasted.len() as f64
    };

    println!("=== GPUFlux Phase 6: replanning / fallback ===");
    println!(
        "policy        : {} (fallback={})",
        engine.policy_name(),
        fallback
    );
    println!(
        "object size   : {} bytes ({} MiB)",
        size,
        size / (1024 * 1024)
    );
    println!("recompute     : {} passes", passes);
    println!("iterations    : {} (prime {})", iters, prime);
    println!("deadline      : {:.0} ms", deadline_ms);
    println!("contention    : io={} readers", io_readers);
    println!("store         : {}", store_path.display());
    println!();
    println!(
        "chosen actual : mean={:.1}ms p50={:.1} p90={:.1} max={:.1}",
        a.mean_ms, a.p50_ms, a.p90_ms, a.max_ms
    );
    println!(
        "DEADLINE MET  : {}/{} ({:.1}%)   <- primary metric",
        met,
        deadline_results.len(),
        met as f64 / deadline_results.len() as f64 * 100.0
    );
    println!(
        "fallbacks     : {}/{} (mean wasted {:.1} ms)",
        fallbacks, iters, mean_wasted
    );
    Ok(())
}
