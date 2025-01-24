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

impl<'a> Sample<'a> {
    #[inline]
    #[allow(clippy::must_use_candidate)]
    pub fn elevation(&self) -> Elev {
        self.tile.samples.get_linear_unchecked(self.index)
    }

    #[inline]
    #[allow(clippy::must_use_candidate)]
    pub fn polygon(&self) -> Polygon {
        self.tile.xy_to_polygon(self.xy())
    }

    #[inline]
    #[allow(clippy::must_use_candidate)]
    pub fn index(&self) -> usize {
        self.index
    }

    #[inline]
    #[allow(clippy::must_use_candidate)]
    pub fn xy(&self) -> (usize, usize) {
        self.tile.linear_to_xy(self.index)
    }

    #[inline]
    #[allow(clippy::must_use_candidate)]
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
