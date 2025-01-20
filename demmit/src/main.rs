use camino::Utf8PathBuf;
use clap::{Args, Parser, Subcommand};
use nasadem::Tile;

/// A NASADEM '.hgt' file multitool.
#[derive(Debug, Parser)]
struct Cli {
    #[command(subcommand)]
    command: SubCmd,
}

#[derive(Clone, Debug, Subcommand)]
enum SubCmd {
    /// Render NASADEM '.hgt' file as an image.
    Render(RenderArgs),
}

#[derive(Args, Clone, Debug)]
struct RenderArgs {
    /// Source NASADEM hgt file.
    src: Utf8PathBuf,

    /// Optional output file name.
    ///
    /// Image format will be based on `dest`'s extension.
    ///
    /// If not specified, a png will be written with the tile's
    /// basename in the tile's dir.
    dest: Option<Utf8PathBuf>,
}

fn render(RenderArgs { src, dest }: RenderArgs) {
    let tile = Tile::load(&src).unwrap();
    let out = dest.map_or_else(
        || {
            let mut out = src.clone();
            out.set_extension("png");
            out
        },
        |mut out| {
            if out.is_dir() {
                let name = src.file_name().unwrap();
                out.push(name);
                out.set_extension("png");
            }
            out
        },
    );

    eprintln!("writing to {out:?}");
    if let Some("png" | "tif" | "tiff") = out.extension() {
        let img = tile.to_image::<u16>();
        img.save(out).unwrap();
    } else {
        let img = tile.to_image::<u8>();
        img.save(out).unwrap();
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        SubCmd::Render(args) => render(args),
    }
}
