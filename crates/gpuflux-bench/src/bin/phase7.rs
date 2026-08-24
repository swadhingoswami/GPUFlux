use std::path::PathBuf;
use std::time::Duration;

use gpuflux::contention::Contention;
use gpuflux::decision::engine::DecisionEngine;
use gpuflux::decision::policy::{AlwaysMove, AlwaysRecompute, ExpectedCost, Policy};
use gpuflux::executor::{SimMoveExecutor, SimRecomputeExecutor};
use gpuflux::object::{DataLoc, ObjectSpec};
use gpuflux::observation::ObservationStore;
use gpuflux::prediction::{
    CurrentStateCostModel, CurrentStatePredictor, HistoricalPredictor, OnlineRegressionPredictor,
    Predictor,
};
use gpuflux::telemetry::{ResourceSampler, SystemSampler};
use gpuflux::{Error, Result};
use gpuflux_bench::cli::{get_arg, parse_size, stats};

fn main() -> Result<()> {
    let policy = get_arg("--policy", "regression_cost");
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "30").parse().unwrap_or(30);
    let prime: usize = get_arg("--prime", "20").parse().unwrap_or(20);
    let passes: f64 = get_arg("--passes", "2").parse().unwrap_or(2.0);
    let io_readers: usize = get_arg("--io-readers", "0").parse().unwrap_or(0);
    let io_size: u64 = parse_size(&get_arg("--io-file-size", "512MiB"))?;
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
    let spec = ObjectSpec::new(29, size, DataLoc::GpuMemory);

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
        "expected_cost" | "historical_cost" | "regression_cost" => Box::new(ExpectedCost),
        other => return Err(Error::Invalid(format!("unknown policy: {other}"))),
    };
    let predictor: Box<dyn Predictor> = match policy.as_str() {
        "historical_cost" => Box::new(HistoricalPredictor::new(store.clone(), model)),
        "regression_cost" => Box::new(OnlineRegressionPredictor::new(model)),
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
        let _ = engine.decide_and_measure(&spec, sampler.sample(), None)?;
    }

    let mut actuals = Vec::with_capacity(iters);
    let mut regrets = Vec::with_capacity(iters);
    let mut pred_errors = Vec::with_capacity(iters);
    let mut moves = 0usize;
    let mut recomputes = 0usize;

    for _ in 0..iters {
        let outcome = engine.decide_and_measure(&spec, sampler.sample(), deadline_ms)?;
        match outcome.action {
            gpuflux::decision::Action::Move => moves += 1,
            _ => recomputes += 1,
        }
        actuals.push(Duration::from_secs_f64(outcome.actual_ms / 1000.0));
        regrets.push(outcome.regret_ms.unwrap_or(0.0));
        pred_errors.push(outcome.prediction_error_ms.unwrap_or(0.0));
    }

    let a = stats(&actuals, size);
    let mean_regret = regrets.iter().sum::<f64>() / regrets.len() as f64;
    let mean_abs_pe = pred_errors.iter().map(|e| e.abs()).sum::<f64>() / pred_errors.len() as f64;

    println!("=== GPUFlux Phase 7: online-regression prediction ===");
    println!("policy        : {}", engine.policy_name());
    println!(
        "object size   : {} bytes ({} MiB)",
        size,
        size / (1024 * 1024)
    );
    println!("iterations    : {} (prime {})", iters, prime);
    println!("contention    : io={} readers", io_readers);
    println!("store         : {}", store_path.display());
    println!();
    println!("decisions     : move={} recompute={}", moves, recomputes);
    println!(
        "chosen actual : mean={:.1}ms p50={:.1} p90={:.1} cv={:.3}",
        a.mean_ms,
        a.p50_ms,
        a.p90_ms,
        a.std_ms / a.mean_ms
    );
    println!("mean regret   : {:.1} ms vs oracle", mean_regret);
    println!(
        "pred error    : abs={:.1}ms (actual - predicted)",
        mean_abs_pe
    );

    if policy == "regression_cost" {
        let any = engine.predictor() as &dyn std::any::Any;
        if let Some(r) = any.downcast_ref::<OnlineRegressionPredictor>() {
            let (wm, wr, _rr) = r.weights();
            println!(
                "learned w     : move=[bias {:.0}, cpu {:+.1}, nvme_lat {:+.2}, queue {:+.1}] ms",
                wm[0], wm[1], wm[2], wm[3]
            );
            println!(
                "                recompute=[bias {:.0}, cpu {:+.1}, nvme_lat {:+.2}, queue {:+.1}] ms",
                wr[0], wr[1], wr[2], wr[3]
            );
        }
    }
    Ok(())
}
