//! Completion-time distribution helpers for deadline risk.
//!
//! We only keep a few quantiles (p50/p90/p95) in the store, not full samples.
//! `P(T > D)` is estimated by fitting a log-normal distribution through the
//! p50 and p90 quantiles (a common, robust choice for latency: log-normal
//! shapes fit completion times well and never predict negative times).

use crate::prediction::cost::CostEstimate;

/// Normal CDF via the Abramowitz–Stegun 7.1.26 erf approximation.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Log-normal fit through (p50, 0.5) and (p90, 0.9).
#[derive(Debug, Clone, Copy)]
pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    /// p90 z-score of the standard normal: Phi(1.2816) = 0.9.
    const Z90: f64 = 1.281551565545;

    /// Returns None when the quantiles do not imply a valid positive spread
    /// (e.g. a degenerate point estimate where p50 == p90).
    pub fn from_p50_p90(p50: f64, p90: f64) -> Option<Self> {
        if !(p50.is_finite() && p90.is_finite()) || p50 <= 0.0 || p90 <= p50 {
            return None;
        }
        Some(Self {
            mu: p50.ln(),
            sigma: (p90.ln() - p50.ln()) / Self::Z90,
        })
    }

    pub fn exceed_probability(&self, d: f64) -> f64 {
        if d <= 0.0 {
            return 1.0;
        }
        let z = (d.ln() - self.mu) / self.sigma;
        1.0 - normal_cdf(z)
    }
}

/// Estimated probability that a completion time with the given quantile
/// estimate exceeds deadline `d` (ms).
pub fn deadline_exceed_probability(est: &CostEstimate, d: f64) -> f64 {
    match LogNormal::from_p50_p90(est.p50_ms, est.p90_ms) {
        // Real distribution available.
        Some(dist) => dist.exceed_probability(d),
        // Degenerate (point) estimate: treat as deterministic.
        None if est.expected_ms > 0.0 => {
            if d > est.expected_ms {
                0.0
            } else {
                1.0
            }
        }
        None => 0.0,
    }
}

/// Crude uncertainty measure: p90 - p50 spread in ms. Used as the U term in
/// the decision score. Larger spread means a less reliable estimate.
pub fn uncertainty_ms(est: &CostEstimate) -> f64 {
    (est.p90_ms - est.p50_ms).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(p50: f64, p90: f64) -> CostEstimate {
        CostEstimate {
            expected_ms: p50,
            p50_ms: p50,
            p90_ms: p90,
            p95_ms: p90 + (p90 - p50),
            deadline_probability: None,
        }
    }

    #[test]
    fn lognormal_fit_is_monotonic() {
        let d = LogNormal::from_p50_p90(100.0, 200.0).unwrap();
        let p_small = d.exceed_probability(50.0);
        let p_mid = d.exceed_probability(150.0);
        let p_large = d.exceed_probability(250.0);
        assert!(p_small > p_mid && p_mid > p_large);
        assert!(p_mid > 0.0 && p_mid < 1.0);
    }

    #[test]
    fn lognormal_p90_matches_fit() {
        let d = LogNormal::from_p50_p90(100.0, 200.0).unwrap();
        assert!((d.exceed_probability(200.0) - 0.1).abs() < 0.05);
    }

    #[test]
    fn degenerate_point_estimate_threshold() {
        let est = point(100.0, 100.0);
        assert_eq!(deadline_exceed_probability(&est, 150.0), 0.0);
        assert_eq!(deadline_exceed_probability(&est, 50.0), 1.0);
    }

    #[test]
    fn inverted_quantiles_fall_back() {
        let est = point(100.0, 50.0);
        assert_eq!(deadline_exceed_probability(&est, 150.0), 0.0);
    }

    #[test]
    fn uncertainty_is_spread() {
        assert_eq!(uncertainty_ms(&point(100.0, 160.0)), 60.0);
        assert_eq!(uncertainty_ms(&point(100.0, 80.0)), 0.0);
    }
}
