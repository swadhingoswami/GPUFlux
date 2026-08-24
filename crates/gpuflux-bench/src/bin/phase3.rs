use std::path::PathBuf;
use std::time::Duration;

use gpuflux::decision::engine::DecisionEngine;
use gpuflux::decision::policy::{AlwaysMove, AlwaysRecompute, ExpectedCost, Policy};
use gpuflux::executor::{SimMoveExecutor, SimRecomputeExecutor};
use gpuflux::object::{DataLoc, ObjectSpec};
use gpuflux::observation::ObservationStore;
use gpuflux::prediction::{
    CurrentStateCostModel, CurrentStatePredictor, HistoricalPredictor, Predictor,
};
use gpuflux::telemetry::{ResourceSampler, SystemSampler};
use gpuflux::{Error, Result};
use gpuflux_bench::cli::{get_arg, parse_size, stats};

fn start_cpu_burner(cores: usize) -> Vec<std::thread::JoinHandle<()>> {
    let mut handles = Vec::new();
    for _ in 0..cores {
        handles.push(std::thread::spawn(move || {
            let mut i = 0u64;
            loop {
                i = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if i % 1_000_000 == 0 {
                    std::thread::yield_now();
                }
            }
        }));
    }
    handles
}

fn main() -> Result<()> {
    let policy = get_arg("--policy", "historical_cost");
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "30").parse().unwrap_or(30);
    let prime: usize = get_arg("--prime", "20").parse().unwrap_or(20);
    let passes: f64 = get_arg("--passes", "2").parse().unwrap_or(2.0);
    let cpu_burn: usize = get_arg("--cpu-burn", "0").parse().unwrap_or(0);
    let deadline_ms: Option<f64> = {
        let s = get_arg("--deadline-ms", "");
        if s.is_empty() {
            None
        } else {
            Some(
                s.parse()
                    .map_err(|_| Error::Invalid("bad --deadline-ms".into()))?,
            )
        }
    };

    let base = std::env::temp_dir().join("gpuflux-bench");
    let dir = PathBuf::from(get_arg(
        "--dir",
        base.to_str().unwrap_or("/tmp/gpuflux-bench"),
    ));
    let store_path = PathBuf::from(get_arg(
        "--store",
        base.join(format!("gpuflux-{}.db", policy))
            .to_str()
            .unwrap_or_default(),
    ));
    std::fs::create_dir_all(&dir)?;

    let store = ObservationStore::open(&store_path)?;
    let spec = ObjectSpec::new(13, size, DataLoc::GpuMemory);

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
        "historical_cost" => Box::new(ExpectedCost),
        other => return Err(Error::Invalid(format!("unknown policy: {other}"))),
    };
    let predictor: Box<dyn Predictor> = match policy.as_str() {
        "historical_cost" => Box::new(HistoricalPredictor::new(store.clone(), model)),
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

    let burners = start_cpu_burner(cpu_burn);

    // Prime history: record BOTH actions so both buckets have samples before
    // measurement. Policy-independent, so every policy sees the same history.
    for _ in 0..prime {
        let _ = engine.decide_and_measure(&spec, sampler.sample(), None)?;
    }

    let mut actuals = Vec::with_capacity(iters);
    let mut regrets = Vec::with_capacity(iters);
    let mut pred_errors = Vec::with_capacity(iters);
    let mut deadline_results = Vec::with_capacity(iters);
    let mut moves = 0usize;
    let mut recomputes = 0usize;
    let mut samples = Vec::with_capacity(iters);

    for _ in 0..iters {
        let state = sampler.sample();
        let outcome = engine.decide_and_measure(&spec, state.clone(), deadline_ms)?;
        match outcome.action {
            gpuflux::decision::Action::Move => moves += 1,
            _ => recomputes += 1,
        }
        actuals.push(Duration::from_secs_f64(outcome.actual_ms / 1000.0));
        regrets.push(outcome.regret_ms.unwrap_or(0.0));
        pred_errors.push(outcome.prediction_error_ms.unwrap_or(0.0));
        deadline_results.push(outcome.deadline_met);
        samples.push(state);
    }

    drop(burners);

    let a = stats(&actuals, size);
    let mean_regret = regrets.iter().sum::<f64>() / regrets.len() as f64;
    let mean_pe = pred_errors.iter().sum::<f64>() / pred_errors.len() as f64;
    let mean_abs_pe = pred_errors.iter().map(|e| e.abs()).sum::<f64>() / pred_errors.len() as f64;
    let met = deadline_results.iter().filter(|m| **m).count();
    let cpu_util = samples
        .iter()
        .map(|s| s.cpu_util.unwrap_or(0.0))
        .sum::<f64>()
        / samples.len() as f64;

    println!("=== GPUFlux Phase 3: historical prediction ===");
    println!("policy        : {}", engine.policy_name());
    println!(
        "object size   : {} bytes ({} MiB)",
        size,
        size / (1024 * 1024)
    );
    println!("iterations    : {} (prime {})", iters, prime);
    println!("cpu burn      : {} cores", cpu_burn);
    println!(
        "deadline      : {:?}",
        deadline_ms.map(|m| format!("{m}ms"))
    );
    println!("observed      : cpu_util={:.2}", cpu_util);
    println!("store         : {}", store_path.display());
    println!();
    println!("decisions     : move={} recompute={}", moves, recomputes);
    println!(
        "chosen actual : mean={:.1}ms p50={:.1} p90={:.1} p95={:.1} max={:.1} cv={:.3} mb/s={:.1}",
        a.mean_ms,
        a.p50_ms,
        a.p90_ms,
        a.p95_ms,
        a.max_ms,
        a.std_ms / a.mean_ms,
        a.mb_s
    );
    println!(
        "deadline met  : {}/{} ({:.1}%)",
        met,
        deadline_results.len(),
        met as f64 / deadline_results.len() as f64 * 100.0
    );
    println!("mean regret   : {:.1} ms vs oracle", mean_regret);
    println!(
        "pred error    : mean={:+.1}ms abs={:.1}ms (actual - predicted)",
        mean_pe, mean_abs_pe
    );
    Ok(())
}
