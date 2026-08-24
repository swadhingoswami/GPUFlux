use crate::now_unix_ms;
use crate::observation::codec::{push_f64, push_u32, push_u64, take_f64, take_u32, take_u64};

/// Smoothing factor for the EWMA mean/variance update.
pub const DEFAULT_ALPHA: f64 = 0.1;

/// Bounded reservoir size. Quantiles are exact over the retained window, which
/// biases them toward recent history. This is deliberate: the model should
/// adapt to changing conditions. The bias is documented as a known limitation.
pub const DEFAULT_RESERVOIR: usize = 8192;

/// Fast aggregate statistics for one (operation, resource bucket, object
/// bucket) key. This is the hot-path lookup structure kept in the embedded KV
/// store; raw decision events live in a separate table.
#[derive(Debug, Clone)]
pub struct AggregateRow {
    pub sample_count: u64,
    /// EWMA mean of completion time (ms).
    pub ewma_mean: f64,
    /// EWMA variance of completion time (ms^2).
    pub ewma_variance: f64,
    /// Bounded reservoir of recent samples (ms) for p50/p90/p95.
    pub samples: Vec<f64>,
    pub deadline_ok: u64,
    pub deadline_total: u64,
    pub last_update_unix_ms: u64,
    pub model_version: u32,
}

impl Default for AggregateRow {
    fn default() -> Self {
        AggregateRow {
            sample_count: 0,
            ewma_mean: 0.0,
            ewma_variance: 0.0,
            samples: Vec::new(),
            deadline_ok: 0,
            deadline_total: 0,
            last_update_unix_ms: 0,
            model_version: 0,
        }
    }
}

impl AggregateRow {
    /// Online update:
    ///   mean_t = alpha*x + (1-alpha)*mean_{t-1}
    ///   var_t  = alpha*(x - mean_t)^2 + (1-alpha)*var_{t-1}
    ///
    /// The first sample seeds the mean directly (initializing at 0 would bias
    /// early estimates downward until enough samples accumulate).
    pub fn push(&mut self, x: f64, alpha: f64, reservoir_max: usize) {
        self.sample_count += 1;
        if self.sample_count == 1 {
            self.ewma_mean = x;
            self.ewma_variance = 0.0;
        } else {
            self.ewma_mean = alpha * x + (1.0 - alpha) * self.ewma_mean;
            let diff = x - self.ewma_mean;
            self.ewma_variance = alpha * diff * diff + (1.0 - alpha) * self.ewma_variance;
        }
        self.samples.push(x);
        if self.samples.len() > reservoir_max {
            self.samples = self.samples.iter().step_by(2).copied().collect();
        }
        self.last_update_unix_ms = now_unix_ms();
    }

    pub fn record_deadline(&mut self, met: bool) {
        self.deadline_total += 1;
        if met {
            self.deadline_ok += 1;
        }
    }

    pub fn deadline_success_rate(&self) -> Option<f64> {
        if self.deadline_total == 0 {
            None
        } else {
            Some(self.deadline_ok as f64 / self.deadline_total as f64)
        }
    }

    pub fn ewma_std(&self) -> f64 {
        self.ewma_variance.max(0.0).sqrt()
    }

    /// Quantile (0..=1) over the retained reservoir. NaN when empty.
    pub fn p(&self, q: f64) -> f64 {
        if self.samples.is_empty() {
            return f64::NAN;
        }
        let mut s = self.samples.clone();
        s.sort_by(f64::total_cmp);
        let idx = ((s.len() - 1) as f64 * q).round() as usize;
        s[idx.min(s.len() - 1)]
    }
}

impl redb::Value for AggregateRow {
    type SelfType<'a>
        = Self
    where
        Self: 'a;
    type AsBytes<'a>
        = Vec<u8>
    where
        Self: 'a;

    fn fixed_width() -> Option<usize> {
        None
    }

    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        let mut d: &[u8] = data;
        AggregateRow {
            sample_count: take_u64(&mut d),
            ewma_mean: take_f64(&mut d),
            ewma_variance: take_f64(&mut d),
            samples: crate::observation::codec::take_vec_f64(&mut d),
            deadline_ok: take_u64(&mut d),
            deadline_total: take_u64(&mut d),
            last_update_unix_ms: take_u64(&mut d),
            model_version: take_u32(&mut d),
        }
    }

    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        let mut v = Vec::new();
        push_u64(&mut v, value.sample_count);
        push_f64(&mut v, value.ewma_mean);
        push_f64(&mut v, value.ewma_variance);
        crate::observation::codec::push_vec_f64(&mut v, &value.samples);
        push_u64(&mut v, value.deadline_ok);
        push_u64(&mut v, value.deadline_total);
        push_u64(&mut v, value.last_update_unix_ms);
        push_u32(&mut v, value.model_version);
        v
    }

    fn type_name() -> redb::TypeName {
        redb::TypeName::new("gpuflux::AggregateRow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_seeds_ewma() {
        let mut row = AggregateRow::default();
        row.push(100.0, 0.1, 8192);
        assert_eq!(row.sample_count, 1);
        assert_eq!(row.ewma_mean, 100.0);
        assert_eq!(row.ewma_variance, 0.0);
        assert_eq!(row.p(0.5), 100.0);
    }

    #[test]
    fn ewma_mean_and_variance_update() {
        let mut row = AggregateRow::default();
        row.push(100.0, 0.5, 8192);
        row.push(200.0, 0.5, 8192);
        assert!((row.ewma_mean - 150.0).abs() < 1e-9);
        assert!((row.ewma_variance - 1250.0).abs() < 1e-9);
        assert_eq!(row.sample_count, 2);
    }

    #[test]
    fn quantile_over_reservoir() {
        let mut row = AggregateRow::default();
        for x in 1..=100u64 {
            row.push(x as f64, 0.1, 8192);
        }
        assert!((row.p(0.5) - 50.0).abs() < 1.5);
        assert!((row.p(0.9) - 90.0).abs() < 1.5);
        assert!((row.p(0.95) - 95.0).abs() < 1.5);
    }

    #[test]
    fn reservoir_stays_bounded() {
        let mut row = AggregateRow::default();
        for x in 0..20_000u64 {
            row.push(x as f64, 0.1, 100);
        }
        assert!(row.samples.len() <= 100);
    }

    #[test]
    fn deadline_success_rate() {
        let mut row = AggregateRow::default();
        assert_eq!(row.deadline_success_rate(), None);
        row.record_deadline(true);
        row.record_deadline(false);
        assert!((row.deadline_success_rate().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn redb_round_trip() {
        let mut row = AggregateRow::default();
        row.push(10.0, 0.1, 100);
        row.push(20.0, 0.1, 100);
        row.record_deadline(true);
        let bytes = <AggregateRow as redb::Value>::as_bytes(&row);
        let back = <AggregateRow as redb::Value>::from_bytes(&bytes);
        assert_eq!(back.sample_count, row.sample_count);
        assert_eq!(back.ewma_mean, row.ewma_mean);
        assert_eq!(back.ewma_variance, row.ewma_variance);
        assert_eq!(back.samples, row.samples);
        assert_eq!(back.deadline_ok, row.deadline_ok);
        assert_eq!(back.deadline_total, row.deadline_total);
        assert_eq!(back.last_update_unix_ms, row.last_update_unix_ms);
        assert_eq!(back.model_version, row.model_version);
    }
}
