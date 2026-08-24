use std::path::PathBuf;
use std::time::Duration;

use gpuflux::decision::engine::DecisionEngine;
use gpuflux::decision::policy::{
    AlwaysMove, AlwaysRecompute, DeadlineAware, ExpectedCost, Policy, RiskAware,
};
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
    let policy = get_arg("--policy", "deadline_aware");
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "30").parse().unwrap_or(30);
    let prime: usize = get_arg("--prime", "20").parse().unwrap_or(20);
    let passes: f64 = get_arg("--passes", "2.5").parse().unwrap_or(2.5);
    let lambda: f64 = get_arg("--lambda", "200").parse().unwrap_or(200.0);
    let mu: f64 = get_arg("--mu", "0").parse().unwrap_or(0.0);
    let move_rate_mbps: f64 = get_arg("--move-rate-mbps", "2000")
        .parse()
        .unwrap_or(2000.0);
    let deadline_ms: f64 = {
        let s = get_arg("--deadline-ms", "175");
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
        base.join(format!("gpuflux-{}.db", policy))
            .to_str()
            .unwrap_or_default(),
    ));
    std::fs::create_dir_all(&dir)?;

    let store = ObservationStore::open(&store_path)?;
    let spec = ObjectSpec::new(19, size, DataLoc::GpuMemory);

    let probe = dir.join("probe.bin");
    std::fs::write(&probe, vec![0u8; 4096])?;
    let sampler = SystemSampler::new(probe);

    let model = CurrentStateCostModel {
        recompute_passes: passes,
        move_rate_bps: move_rate_mbps * 1e6,
        ..Default::default()
    };

    let engine_policy: Box<dyn Policy> = match policy.as_str() {
        "always_move" => Box::new(AlwaysMove),
        "always_recompute" => Box::new(AlwaysRecompute),
        "expected_cost" => Box::new(ExpectedCost),
        "risk_aware" => Box::new(RiskAware { mu }),
        "deadline_aware" => Box::new(DeadlineAware { lambda, mu }),
        other => return Err(Error::Invalid(format!("unknown policy: {other}"))),
    };
    let predictor: Box<dyn Predictor> = match policy.as_str() {
        "deadline_aware" | "risk_aware" => Box::new(HistoricalPredictor::new(store.clone(), model)),
        _ => Box::new(CurrentStatePredictor { model }),
    };

    let mut engine = DecisionEngine::new(
        store.clone(),
        engine_policy,
        predictor,
        Box::new(SimMoveExecutor::new(dir)),
        Box::new(SimRecomputeExecutor {
            passes,
            ..Default::default()
        }),
    );

    for _ in 0..prime {
        let _ = engine.decide_and_measure(&spec, sampler.sample(), Some(deadline_ms))?;
    }

    let mut actuals = Vec::with_capacity(iters);
    let mut regrets = Vec::with_capacity(iters);
    let mut deadline_results = Vec::with_capacity(iters);
    let mut moves = 0usize;
    let mut recomputes = 0usize;

    for _ in 0..iters {
        let outcome = engine.decide_and_measure(&spec, sampler.sample(), Some(deadline_ms))?;
        match outcome.action {
            gpuflux::decision::Action::Move => moves += 1,
            _ => recomputes += 1,
        }
        actuals.push(Duration::from_secs_f64(outcome.actual_ms / 1000.0));
        regrets.push(outcome.regret_ms.unwrap_or(0.0));
        deadline_results.push(outcome.deadline_met);
    }

    let a = stats(&actuals, size);
    let mean_regret = regrets.iter().sum::<f64>() / regrets.len() as f64;
    let met = deadline_results.iter().filter(|m| **m).count();
    let meet_rate = met as f64 / deadline_results.len() as f64 * 100.0;

    println!("=== GPUFlux Phase 5: deadline-aware risk ===");
    println!("policy        : {}", engine.policy_name());
    println!(
        "object size   : {} bytes ({} MiB)",
        size,
        size / (1024 * 1024)
    );
    println!("recompute     : {} passes", passes);
    println!("iterations    : {} (prime {})", iters, prime);
    println!(
        "deadline      : {:.0} ms (lambda={:.0} mu={:.0})",
        deadline_ms, lambda, mu
    );
    println!("store         : {}", store_path.display());
    println!();
    println!("decisions     : move={} recompute={}", moves, recomputes);
    println!(
        "chosen actual : mean={:.1}ms p50={:.1} p90={:.1} max={:.1}",
        a.mean_ms, a.p50_ms, a.p90_ms, a.max_ms
    );
    println!(
        "DEADLINE MET  : {}/{} ({:.1}%)   <- primary metric",
        met,
        deadline_results.len(),
        meet_rate
    );
    println!(
        "mean regret   : {:.1} ms vs oracle (cost only)",
        mean_regret
    );
    Ok(())
}
