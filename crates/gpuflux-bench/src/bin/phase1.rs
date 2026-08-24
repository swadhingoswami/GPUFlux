use std::path::PathBuf;
use std::time::Duration;

use gpuflux::decision::engine::DecisionEngine;
use gpuflux::decision::policy::{AlwaysMove, AlwaysRecompute, Policy};
use gpuflux::executor::{SimMoveExecutor, SimRecomputeExecutor};
use gpuflux::object::{DataLoc, ObjectSpec};
use gpuflux::observation::ObservationStore;
use gpuflux::prediction::{CurrentStateCostModel, CurrentStatePredictor};
use gpuflux::resource::ResourceState;
use gpuflux::{Error, Result};
use gpuflux_bench::cli::{get_arg, parse_size, stats};

fn main() -> Result<()> {
    let policy = get_arg("--policy", "always_move");
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "30").parse().unwrap_or(30);
    let warmup: usize = get_arg("--warmup", "3").parse().unwrap_or(3);
    let passes: f64 = get_arg("--passes", "2").parse().unwrap_or(2.0);
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
    let spec = ObjectSpec::new(7, size, DataLoc::GpuMemory);

    let engine_policy: Box<dyn Policy> = match policy.as_str() {
        "always_move" => Box::new(AlwaysMove),
        "always_recompute" => Box::new(AlwaysRecompute),
        other => return Err(Error::Invalid(format!("unknown policy: {other}"))),
    };

    let mut engine = DecisionEngine::new(
        store.clone(),
        engine_policy,
        Box::new(CurrentStatePredictor {
            model: CurrentStateCostModel {
                recompute_passes: passes,
                ..Default::default()
            },
        }),
        Box::new(SimMoveExecutor::new(dir)),
        Box::new(SimRecomputeExecutor {
            passes,
            ..Default::default()
        }),
    );

    for _ in 0..warmup {
        let _ = engine.decide(&spec, ResourceState::now(), None)?;
    }

    let mut actuals = Vec::with_capacity(iters);
    let mut regrets = Vec::with_capacity(iters);
    let mut deadline_results = Vec::with_capacity(iters);
    for _ in 0..iters {
        let outcome = engine.decide_and_measure(&spec, ResourceState::now(), deadline_ms)?;
        actuals.push(Duration::from_secs_f64(outcome.actual_ms / 1000.0));
        regrets.push(outcome.regret_ms.unwrap_or(0.0));
        deadline_results.push(outcome.deadline_met);
    }

    let a = stats(&actuals, size);
    let mean_regret = regrets.iter().sum::<f64>() / regrets.len() as f64;
    let met = deadline_results.iter().filter(|m| **m).count();

    println!("=== GPUFlux Phase 1: baseline policy ===");
    println!("policy        : {}", engine.policy_name());
    println!(
        "object size   : {} bytes ({} MiB)",
        size,
        size / (1024 * 1024)
    );
    println!("iterations    : {} (warmup {})", iters, warmup);
    println!(
        "deadline      : {:?}",
        deadline_ms.map(|m| format!("{m}ms"))
    );
    println!("store         : {}", store_path.display());
    println!();
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
    println!(
        "mean regret   : {:.1} ms vs oracle (min of both paths)",
        mean_regret
    );
    println!("events logged : {}", store.event_count()?);
    Ok(())
}
