use crate::{
    geo::{Coord, Polygon},
    Elev, Tile, C,
};

/// A NASADEM elevation sample.
pub struct Sample<'a> {
    /// The parent [Tile] this grid square belongs to.
    pub(crate) tile: &'a Tile,
    /// Index into parent's elevation data corresponding to this grid
    /// square.
    pub(crate) index: usize,
}

#[allow(clippy::must_use_candidate)]
impl<'a> Sample<'a> {
    /// Sample elevation in meters.
    #[inline]
    pub fn elevation(&self) -> Elev {
        self.tile.samples.get_linear_unchecked(self.index)
    }

    /// A polygot of this samples geographic bounding box.
    #[inline]
    pub fn polygon(&self) -> Polygon {
        self.tile.xy_to_polygon(self.xy())
    }

    /// This sample's offset in the source tile's memory.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// This sample's logical row/col location in source tile.
    ///
    /// Note that (0, 0) is the NW corner.
    #[inline]
    pub fn xy(&self) -> (usize, usize) {
        self.tile.linear_to_xy(self.index)
    }

    #[inline]
    pub fn geo(&self) -> Coord<C> {
        self.tile.xy_to_geo(self.xy())
    }
}

impl<'a> std::cmp::PartialEq for Sample<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && std::ptr::eq(self, other)
    }
}

impl<'a> std::cmp::Eq for Sample<'a> {}
