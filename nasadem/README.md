# NASA Digital Elevation Model (NASADEM)

This library allows you to load and query
[NASADEM](https://www.earthdata.nasa.gov/esds/competitive-programs/measures/nasadem)
tiles.

### Helpful Resources:

- [30-Meter SRTM Tile Downloader](https://dwtkns.com/srtm30m)
- [HGT File Layout](https://www.researchgate.net/profile/Pierre-Boulanger-4/publication/228924813/figure/fig8/AS:300852653903880@1448740270695/Description-of-a-HGT-file-structure-The-name-file-in-this-case-is-N20W100HGT.png)
- [Archive Team HGT Format](http://fileformats.archiveteam.org/index.php?title=HGT&oldid=17250)
- [SRTM Collection User Guide](https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf)

## Tile Layout

NASADEM height `.hgt` files use a simple, headerless binary format
with no metadata. The only variable parameter—resolution—can be
inferred from the file size.

### Example (`N51W001.hgt`)

For this example, a 3-arcsecond resolution tile (1201x1201) might be
named `N51W001.hgt`. In this case, the bottom-left (southwest) corner
of the tile is located at 51°N, 1°W. The the non-header values in the
following table represent the linear offset of each `i16` elevation
value in the flat memory array.

| **Row \ Column** |                   0 |       1 |       2 |       3 | ... |               1200 |
|-----------------:|--------------------:|--------:|--------:|--------:|----:|-------------------:|
|            **0** |       (52°N, 1°W) 0 |       1 |       2 |       3 | ... |    (52°N, 0°) 1200 |
|            **1** |                1201 |    1202 |    1203 |    1204 | ... |               2400 |
|            **2** |                2401 |    2402 |    2403 |    2404 | ... |               3600 |
|            **3** |                3601 |    3602 |    3603 |    3604 | ... |               4800 |
|          **...** |                 ... |     ... |     ... |     ... | ... |                ... |
|         **1200** | (51°N, 1°W) 1441200 | 1441201 | 1441202 | 1441203 | ... | (51°N, 0°) 1442400 |

Note that while NASADEM filenames correspond to the southwest corner of a
tile, the first elevation value in a `.hgt` file is the northwest corner.

## License

This project is licensed under one of the following licenses:

- Apache License 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT License ([LICENSE-MIT](../LICENSE-MIT) or [MIT License](http://opensource.org/licenses/MIT))

## Contributions

By default, contributions you make to this project are considered
dual-licensed under both the [Apache 2.0
License](http://www.apache.org/licenses/LICENSE-2.0) and the MIT
License, with no additional conditions.
