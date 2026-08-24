pub type ObjectId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataLoc {
    GpuMemory,
    HostMemory,
    Nvme,
    Remote,
    Recompute,
}

impl DataLoc {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataLoc::GpuMemory => "gpu",
            DataLoc::HostMemory => "host",
            DataLoc::Nvme => "nvme",
            DataLoc::Remote => "remote",
            DataLoc::Recompute => "recompute",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObjectSpec {
    pub id: ObjectId,
    pub size_bytes: u64,
    pub loc: DataLoc,
}

impl ObjectSpec {
    pub fn new(id: ObjectId, size_bytes: u64, loc: DataLoc) -> Self {
        Self {
            id,
            size_bytes,
            loc,
        }
    }

    /// Deterministic seed derived from the object id, used to make synthetic
    /// data reproducible across runs.
    pub fn seed(&self) -> u64 {
        self.id
    }
}
