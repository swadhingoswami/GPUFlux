pub mod contention;
pub mod decision;
pub mod error;
pub mod executor;
pub mod object;
pub mod observation;
pub mod prediction;
pub mod resource;
pub mod telemetry;
pub(crate) mod util;

pub use error::{Error, Result};
pub use object::{DataLoc, ObjectId, ObjectSpec};
pub use resource::ResourceState;

use std::time::{SystemTime, UNIX_EPOCH};

/// Current wall clock in milliseconds since the Unix epoch.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
