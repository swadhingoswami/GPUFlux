use std::time::Duration;

use gpuflux::{Error, Result};

/// Parse `--key=value` or `--key value` style args.
pub fn get_arg(key: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        if let Some(v) = args[i].strip_prefix(key) {
            if let Some(rest) = v.strip_prefix('=') {
                return rest.to_string();
            }
            if v.is_empty() && i + 1 < args.len() {
                return args[i + 1].clone();
            }
        }
        i += 1;
    }
    default.to_string()
}

/// Parse sizes like "256MiB", "1GiB", "500MB", "100kb", "4096".
pub fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim().to_ascii_lowercase();
    let (num, mult): (&str, f64) = if let Some(r) = s.strip_suffix("gib") {
        (r, 1024f64.powi(3))
    } else if let Some(r) = s.strip_suffix("mib") {
        (r, 1024f64.powi(2))
    } else if let Some(r) = s.strip_suffix("kib") {
        (r, 1024f64)
    } else if let Some(r) = s.strip_suffix("gb") {
        (r, 1e9)
    } else if let Some(r) = s.strip_suffix("mb") {
        (r, 1e6)
    } else if let Some(r) = s.strip_suffix("b") {
        (r, 1.0)
    } else {
        (s.as_str(), 1.0)
    };
    let v: f64 = num
        .trim()
        .parse()
        .map_err(|_| Error::Invalid(format!("bad size: {s}")))?;
    Ok((v * mult) as u64)
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub n: usize,
    pub mean_ms: f64,
    pub std_ms: f64,
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
    pub mb_s: f64,
}

pub fn stats(durs: &[Duration], bytes: u64) -> Stats {
    let n = durs.len();
    let mut v: Vec<f64> = durs.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let p = |q: f64| {
        let idx = ((n as f64 - 1.0) * q).round() as usize;
        v[idx.min(n - 1)]
    };
    let total_s = durs.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let mean_s = if n > 0 { total_s / n as f64 } else { 0.0 };
    Stats {
        n,
        mean_ms: mean,
        std_ms: var.sqrt(),
        min_ms: v[0],
        p50_ms: p(0.50),
        p90_ms: p(0.90),
        p95_ms: p(0.95),
        max_ms: v[n - 1],
        // Per-operation throughput: object bytes / mean duration.
        mb_s: if mean_s > 0.0 {
            bytes as f64 / mean_s / 1e6
        } else {
            0.0
        },
    }
}
