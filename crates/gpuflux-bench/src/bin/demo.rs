use std::path::PathBuf;

use gpuflux::decision::engine::DecisionEngine;
use gpuflux::decision::policy::{DeadlineAware, ExpectedCost, Policy};
use gpuflux::executor::{SimMoveExecutor, SimRecomputeExecutor};
use gpuflux::object::{DataLoc, ObjectSpec};
use gpuflux::observation::ObservationStore;
use gpuflux::prediction::{
    CurrentStateCostModel, CurrentStatePredictor, HistoricalPredictor, Predictor,
};
use gpuflux::telemetry::{ResourceSampler, SystemSampler};
use gpuflux::{Error, Result};
use gpuflux_bench::cli::{get_arg, parse_size};

/// Demo: GPUFlux's full loop with SIMULATED GPU contention.
///
/// On a real box, GPU utilization comes from NVML and a busy GPU slows kernels.
/// There is no GPU here, so `gpu_load` is a parameter: it inflates the recompute
/// work by 1/(1-gpu_load) (real extra work, not a sleep) and is fed to the
/// predictor via `ResourceState.gpu_util`. The decision engine is identical to
/// the real one - it just sees a GPU-aware state and timing.
fn main() -> Result<()> {
    let policy = get_arg("--policy", "expected_cost");
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "12").parse().unwrap_or(12);
    let prime: usize = get_arg("--prime", "6").parse().unwrap_or(6);
    let passes: f64 = get_arg("--passes", "2").parse().unwrap_or(2.0);
    let gpu_load: f64 = get_arg("--gpu-load", "0.7").parse().unwrap_or(0.7);
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
    let store_path = base.join(format!("gpuflux-demo-{}.db", policy));
    std::fs::create_dir_all(&dir)?;

    let store = ObservationStore::open(&store_path)?;
    let spec = ObjectSpec::new(37, size, DataLoc::GpuMemory);

    let probe = dir.join("probe.bin");
    std::fs::write(&probe, vec![0u8; 4096])?;
    let sampler = SystemSampler::new(probe);

    // GPU-aware model: recompute rate drops with gpu_util.
    let model = CurrentStateCostModel {
        recompute_passes: passes,
        ..Default::default()
    };
    let engine_policy: Box<dyn Policy> = match policy.as_str() {
        "expected_cost" => Box::new(ExpectedCost),
        "deadline_aware" => Box::new(DeadlineAware {
            lambda: 200.0,
            mu: 0.0,
        }),
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
        Box::new(SimMoveExecutor::new(dir)),
        Box::new(SimRecomputeExecutor { passes, gpu_load }),
    );

    // Inject the simulated GPU state into every sampled snapshot.
    let observe = || {
        let mut s = sampler.sample();
        s.gpu_util = Some(gpu_load);
        s.pcie_bandwidth = Some(25e9);
        s
    };

    println!("GPUFlux demo: uncertainty-aware move vs recompute");
    println!("  object   : {} MiB", size / (1024 * 1024));
    println!("  policy   : {}", engine.policy_name());
    println!(
        "  GPU load : {:.0}% (simulated - NVML on the real box)",
        gpu_load * 100.0
    );
    println!("  deadline : {:?}", deadline_ms.map(|m| format!("{m}ms")));
    println!();
    println!("  Recompute is modeled as passes/(1-gpu_load) real fill work; the");
    println!("  predictor sees gpu_util and prices the slowdown. The engine below");
    println!("  is the same one used on real hardware.");
    println!();

    for _ in 0..prime {
        let _ = engine.decide(&spec, observe(), None)?;
    }

    let mut moves = 0usize;
    let mut recomputes = 0usize;
    let mut met = 0usize;
    let mut total_ms = 0.0;
    let mut errors = Vec::new();

    println!(
        "{:>4}  {:>6} {:>8} {:>10} {:>9}  {:>8}  {:>5}  {:>5}",
        "dec", "gpu", "cpu", "movePred", "recPred", "chosen", "actual", "met"
    );
    for i in 0..iters {
        let state = observe();
        // Show the prediction the engine will use (same predictor, no mutation).
        let pred = engine.predictor().predict(&spec, &state, deadline_ms);
        let outcome = engine.decide(&spec, state.clone(), deadline_ms)?;
        match outcome.action {
            gpuflux::decision::Action::Move => moves += 1,
            gpuflux::decision::Action::Recompute => recomputes += 1,
            gpuflux::decision::Action::RemoteRecompute => {}
        }
        if outcome.deadline_met {
            met += 1;
        }
        total_ms += outcome.actual_ms;
        errors.push(outcome.prediction_error_ms.unwrap_or(0.0));
        println!(
            "{:>4}  {:>6.2} {:>8.2} {:>10.0} {:>9.0}  {:>8}  {:>7.0}  {:>5}",
            i + 1,
            state.gpu_util.unwrap_or(0.0),
            state.cpu_util.unwrap_or(0.0),
            pred.move_est.expected_ms,
            pred.recompute_est.expected_ms,
            outcome.action.as_str(),
            outcome.actual_ms,
            if outcome.deadline_met { "yes" } else { "no" },
        );
    }

    let mean_ms = total_ms / iters as f64;
    let mean_abs_err = errors.iter().map(|e| e.abs()).sum::<f64>() / errors.len() as f64;
    println!();
    println!("  decisions  : move={} recompute={}", moves, recomputes);
    println!(
        "  deadline   : {}/{} met ({:.0}%)",
        met,
        iters,
        met as f64 / iters as f64 * 100.0
    );
    println!("  mean cost  : {:.1} ms", mean_ms);
    println!("  mean |prediction error| : {:.1} ms", mean_abs_err);
    Ok(())
}
