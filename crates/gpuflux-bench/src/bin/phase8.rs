use std::path::PathBuf;
use std::time::Duration;

use gpuflux::contention::Contention;
use gpuflux::decision::engine::DecisionEngine;
use gpuflux::decision::policy::{DeadlineAware, ExpectedCost, Policy};
use gpuflux::executor::{SimMoveExecutor, SimRecomputeExecutor, SimRemoteExecutor};
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
    let policy = get_arg("--policy", "expected_cost");
    let size = parse_size(&get_arg("--size", "256MiB"))?;
    let iters: usize = get_arg("--iters", "30").parse().unwrap_or(30);
    let prime: usize = get_arg("--prime", "20").parse().unwrap_or(20);
    let passes: f64 = get_arg("--passes", "4").parse().unwrap_or(4.0);
    let io_readers: usize = get_arg("--io-readers", "0").parse().unwrap_or(0);
    let cpu_burn: usize = get_arg("--cpu-burn", "0").parse().unwrap_or(0);
    let io_size: u64 = parse_size(&get_arg("--io-file-size", "512MiB"))?;
    let remote_load: f64 = get_arg("--remote-load", "0.1").parse().unwrap_or(0.1);
    let remote_off = get_arg("--remote-off", "0") == "1";
    let net_latency_ms: f64 = get_arg("--net-latency-ms", "5").parse().unwrap_or(5.0);
    let net_bw_gbps: f64 = get_arg("--net-bw-gbps", "10").parse().unwrap_or(10.0);
    let remote_rate_mbps: f64 = get_arg("--remote-rate-mbps", "8400")
        .parse()
        .unwrap_or(8400.0);
    let lambda: f64 = get_arg("--lambda", "200").parse().unwrap_or(200.0);
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
    let spec = ObjectSpec::new(31, size, DataLoc::GpuMemory);

    let probe = dir.join("probe.bin");
    std::fs::write(&probe, vec![0u8; 4096])?;
    let sampler = SystemSampler::new(probe);

    let model = CurrentStateCostModel {
        recompute_passes: passes,
        remote_enabled: !remote_off,
        remote_rate_bps: remote_rate_mbps * 1e6,
        network_latency_ms: net_latency_ms,
        network_bandwidth_bps: net_bw_gbps * 1e9,
        ..Default::default()
    };
    let remote_exec = SimRemoteExecutor {
        remote_load,
        remote_rate_bps: remote_rate_mbps * 1e6,
        passes,
        network_latency_ms: net_latency_ms,
        network_bandwidth_bps: net_bw_gbps * 1e9,
    };

    let engine_policy: Box<dyn Policy> = match policy.as_str() {
        "expected_cost" => Box::new(ExpectedCost),
        "deadline_aware" => Box::new(DeadlineAware { lambda, mu: 0.0 }),
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
    )
    .with_remote_exec(Box::new(remote_exec));

    let _contention = Contention::start(cpu_burn, io_readers, &io_file, false);

    for _ in 0..prime {
        let mut state = sampler.sample();
        state.remote_cpu_util = Some(remote_load);
        let _ = engine.decide_and_measure(&spec, state, None)?;
    }

    let mut actuals = Vec::with_capacity(iters);
    let mut regrets = Vec::with_capacity(iters);
    let mut deadline_results = Vec::with_capacity(iters);
    let mut moves = 0usize;
    let mut recomputes = 0usize;
    let mut remotes = 0usize;

    for _ in 0..iters {
        let mut state = sampler.sample();
        state.remote_cpu_util = Some(remote_load);
        let outcome = engine.decide_and_measure(&spec, state, deadline_ms)?;
        match outcome.action {
            gpuflux::decision::Action::Move => moves += 1,
            gpuflux::decision::Action::Recompute => recomputes += 1,
            gpuflux::decision::Action::RemoteRecompute => remotes += 1,
        }
        actuals.push(Duration::from_secs_f64(outcome.actual_ms / 1000.0));
        regrets.push(outcome.regret_ms.unwrap_or(0.0));
        deadline_results.push(outcome.deadline_met);
    }

    let a = stats(&actuals, size);
    let mean_regret = regrets.iter().sum::<f64>() / regrets.len() as f64;
    let met = deadline_results.iter().filter(|m| **m).count();

    println!("=== GPUFlux Phase 8: remote recompute ===");
    println!("policy        : {}", engine.policy_name());
    println!(
        "object size   : {} bytes ({} MiB)",
        size,
        size / (1024 * 1024)
    );
    println!("recompute     : {} passes", passes);
    println!("iterations    : {} (prime {})", iters, prime);
    println!("contention    : cpu={} io={} readers", cpu_burn, io_readers);
    println!(
        "remote        : load={:.2} latency={:.0}ms bw={:.0}Gbps rate={:.0}MB/s",
        remote_load, net_latency_ms, net_bw_gbps, remote_rate_mbps
    );
    println!(
        "deadline      : {:?}",
        deadline_ms.map(|m| format!("{m}ms"))
    );
    println!("store         : {}", store_path.display());
    println!();
    println!(
        "decisions     : move={} recompute={} remote={}",
        moves, recomputes, remotes
    );
    println!(
        "chosen actual : mean={:.1}ms p50={:.1} p90={:.1} cv={:.3}",
        a.mean_ms,
        a.p50_ms,
        a.p90_ms,
        a.std_ms / a.mean_ms
    );
    println!(
        "DEADLINE MET  : {}/{} ({:.1}%)",
        met,
        deadline_results.len(),
        met as f64 / deadline_results.len() as f64 * 100.0
    );
    println!("mean regret   : {:.1} ms vs oracle", mean_regret);
    Ok(())
}
