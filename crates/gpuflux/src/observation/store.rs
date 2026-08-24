use std::path::Path;

use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};

use crate::error::Result;
use crate::observation::aggregate::{AggregateRow, DEFAULT_ALPHA, DEFAULT_RESERVOIR};
use crate::observation::event::DecisionEvent;

const OBS: TableDefinition<&str, AggregateRow> = TableDefinition::new("observations");
const EVENTS: TableDefinition<u64, DecisionEvent> = TableDefinition::new("events");

/// Embedded KV store (redb) holding two layers:
///   - fast aggregate statistics per bucket (AggregateRow)
///   - a raw decision-event log (DecisionEvent)
///
/// Keys are plain strings like "move/sim/268435456" so related buckets sort
/// adjacently and can be iterated in order.
#[derive(Clone)]
pub struct ObservationStore {
    db: std::sync::Arc<Database>,
}

impl ObservationStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = Database::create(path.as_ref()).map_err(crate::error::store_err)?;
        let store = Self {
            db: std::sync::Arc::new(db),
        };
        store.init()?;
        Ok(store)
    }

    fn init(&self) -> Result<()> {
        let txn = self.db.begin_write().map_err(crate::error::store_err)?;
        {
            txn.open_table(OBS).map_err(crate::error::store_err)?;
            txn.open_table(EVENTS).map_err(crate::error::store_err)?;
        }
        txn.commit().map_err(crate::error::store_err)?;
        Ok(())
    }

    /// Record one completion-time sample (ms) for a bucket, updating EWMA mean,
    /// EWMA variance and the bounded reservoir in a read-modify-write.
    pub fn record(&self, bucket: &str, duration_ms: f64) -> Result<()> {
        let txn = self.db.begin_write().map_err(crate::error::store_err)?;
        {
            let mut table = txn.open_table(OBS).map_err(crate::error::store_err)?;
            let mut row = match table.get(bucket).map_err(crate::error::store_err)? {
                Some(guard) => guard.value(),
                None => AggregateRow::default(),
            };
            row.push(duration_ms, DEFAULT_ALPHA, DEFAULT_RESERVOIR);
            table.insert(bucket, row).map_err(crate::error::store_err)?;
        }
        txn.commit().map_err(crate::error::store_err)?;
        Ok(())
    }

    /// Record a deadline outcome for a bucket.
    pub fn record_deadline(&self, bucket: &str, met: bool) -> Result<()> {
        let txn = self.db.begin_write().map_err(crate::error::store_err)?;
        {
            let mut table = txn.open_table(OBS).map_err(crate::error::store_err)?;
            let mut row = match table.get(bucket).map_err(crate::error::store_err)? {
                Some(guard) => guard.value(),
                None => AggregateRow::default(),
            };
            row.record_deadline(met);
            table.insert(bucket, row).map_err(crate::error::store_err)?;
        }
        txn.commit().map_err(crate::error::store_err)?;
        Ok(())
    }

    pub fn aggregate(&self, bucket: &str) -> Result<Option<AggregateRow>> {
        let txn = self.db.begin_read().map_err(crate::error::store_err)?;
        let table = txn.open_table(OBS).map_err(crate::error::store_err)?;
        match table.get(bucket).map_err(crate::error::store_err)? {
            Some(guard) => Ok(Some(guard.value())),
            None => Ok(None),
        }
    }

    pub fn log_event(&self, event: DecisionEvent) -> Result<()> {
        let txn = self.db.begin_write().map_err(crate::error::store_err)?;
        {
            let mut table = txn.open_table(EVENTS).map_err(crate::error::store_err)?;
            table
                .insert(event.decision_id, event)
                .map_err(crate::error::store_err)?;
        }
        txn.commit().map_err(crate::error::store_err)?;
        Ok(())
    }

    pub fn event_count(&self) -> Result<u64> {
        let txn = self.db.begin_read().map_err(crate::error::store_err)?;
        let table = txn.open_table(EVENTS).map_err(crate::error::store_err)?;
        table.len().map_err(crate::error::store_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> (ObservationStore, std::path::PathBuf) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("gpuflux-test-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.db");
        (ObservationStore::open(&path).unwrap(), dir)
    }

    #[test]
    fn record_aggregate_roundtrip() {
        let (store, dir) = test_store();
        store.record("move/sim/1024", 10.0).unwrap();
        store.record("move/sim/1024", 20.0).unwrap();
        store.record_deadline("move/sim/1024", true).unwrap();

        let row = store.aggregate("move/sim/1024").unwrap().unwrap();
        assert_eq!(row.sample_count, 2);
        // EWMA: first sample seeds 10.0, second: 0.1*20 + 0.9*10 = 11.0
        assert!((row.ewma_mean - 11.0).abs() < 1e-9);
        assert_eq!(row.deadline_success_rate(), Some(1.0));

        assert!(store.aggregate("nope").unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn event_log_append() {
        let (store, dir) = test_store();
        store.log_event(DecisionEvent::new(1, 1)).unwrap();
        store.log_event(DecisionEvent::new(2, 1)).unwrap();
        assert_eq!(store.event_count().unwrap(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
