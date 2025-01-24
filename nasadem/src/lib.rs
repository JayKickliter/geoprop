// #![deny(missing_docs)]

//! NASADEM elevation (`.hgt`) file format.
//!
//! # References
//!
//! 1. [30-Meter SRTM Tile Downloader](https://dwtkns.com/srtm30m)
//! 1. [HGT file layout](https://www.researchgate.net/profile/Pierre-Boulanger-4/publication/228924813/figure/fig8/AS:300852653903880@1448740270695/Description-of-a-HGT-file-structure-The-name-file-in-this-case-is-N20W100HGT.png)
//! 1. [Archive Team](http://fileformats.archiveteam.org/index.php?title=HGT&oldid=17250)
//! 1. [SRTM Collection User Guide](https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf)

pub use crate::{
    error::NasademError,
    sample::Sample,
    tile::{Tile, TileIndex},
};
pub use geo;
#[cfg(feature = "image")]
pub use image;

mod error;
mod sample;
pub(crate) mod store;
#[cfg(test)]
mod tests;
mod tile;
#[cfg(feature = "image")]
mod to_image;
pub(crate) mod util;

/// Base floating point type used for all coordinates and calculations.
///
/// Note: this _could_ be a generic parameter, but doing so makes the
/// library more complicated. While f32 vs f64 does make a measurable
/// difference when walking paths across tiles (see `Profile` type in
/// the `terrain` crate), benchmarking shows that switching NASADEMs
/// to `f32` has no effect.
pub type C = f64;

/// Bit representation of elevation samples.
pub type Elev = i16;

const ARCSEC_PER_DEG: C = 3600.0;
const HALF_ARCSEC: C = 1.0 / (2.0 * 3600.0);
