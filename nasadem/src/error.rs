use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(missing_docs, clippy::module_name_repetitions)]
pub enum NasademError {
    #[error("")]
    Io(#[from] std::io::Error),

    #[error("invalid HGT name {0}")]
    HgtName(std::path::PathBuf),

    #[error("invalid HGT file len {0} for {1}")]
    HgtLen(u64, PathBuf),
}
