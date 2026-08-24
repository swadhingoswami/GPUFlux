use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Store(Box<redb::Error>),
    Invalid(String),
    /// Failure reported by an execution backend (e.g. the C++/CUDA layer).
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Store(e) => write!(f, "store error: {e}"),
            Error::Invalid(m) => write!(f, "invalid input: {m}"),
            Error::Backend(m) => write!(f, "backend error: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Store(e) => Some(e),
            Error::Invalid(_) => None,
            Error::Backend(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<redb::Error> for Error {
    fn from(e: redb::Error) -> Self {
        Error::Store(Box::new(e))
    }
}

/// Convenience for `.map_err(store_err)` where the underlying redb call returns
/// a specific error type (DatabaseError, TableError, TransactionError, ...)
/// that converts into `redb::Error`.
pub(crate) fn store_err<E: Into<redb::Error>>(e: E) -> Error {
    Error::Store(Box::new(e.into()))
}

pub type Result<T> = std::result::Result<T, Error>;
