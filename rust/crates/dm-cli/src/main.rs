//! dpmread: command line front end over dm-reader. Same flags, report and
//! JSON as `datamatrix_reader.py`. Exit code 0 if every image decoded, 1 if
//! any failed (or could not be loaded), 2 on a usage error.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use dm_reader::timing::STAGE_NAMES;
use dm_reader::{
    Engines, Polarity, ReadOptions, ReadResult, Runner, Timings, ZXING_AVAILABLE, collect_inputs,
    default_jobs, parse_sizes, printable, read_image, report_text, to_json, write_fit_dump,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PolarityArg {
    Any,
    Dark,
    Light,
}

#[derive(Parser, Debug)]
#[command(
    name = "dpmread",
    version,
    about = "Robust Data Matrix (ECC 200) reader"
)]
struct Args {
    /// image files or folders
    inputs: Vec<PathBuf>,
    /// save variants + annotated result
    #[arg(long, value_name = "DIR")]
    debug: Option<PathBuf>,
    /// JSON output
    #[arg(long)]
    json: bool,
    /// try every method, keep best
    #[arg(long)]
    exhaustive: bool,
    /// comma separated subset of zxing,grid
    #[arg(long, default_value = "zxing,grid")]
    engines: String,
    /// print sampled module matrix
    #[arg(long)]
    matrix: bool,
    /// max seconds per image
    #[arg(long, default_value_t = 40.0)]
    budget: f64,
    /// worker threads (1 = sequential)
    #[arg(long, default_value_t = default_jobs())]
    jobs: usize,
    /// expected symbol size(s), e.g. 24x24 (repeatable; like the trained
    /// model of an industrial reader, prunes the grid search)
    #[arg(long, value_name = "RxC")]
    expect: Vec<String>,
    /// mark polarity: try both with a small inverted variant chain
    /// (default), dark-on-light only, or light-on-dark
    #[arg(long, value_enum, default_value = "any")]
    polarity: PolarityArg,
    /// read every symbol in the image: cover each read symbol and read
    /// the rest again (same budget)
    #[arg(long)]
    multi: bool,
    /// after a zxing read, skip the grid fit that gives the module matrix,
    /// RS statistics and grades (faster; the text is the same)
    #[arg(long)]
    no_analysis: bool,
    /// write each image's grid fits (the one that read, the best failed
    /// ones) with module centres and module classifier input here
    #[arg(long, value_name = "DIR")]
    dump_fits: Option<PathBuf>,
    /// do not run the module classifier on the best failed grid fits
    /// (builds with the cargo feature `ml` only)
    #[arg(long)]
    no_classifier: bool,
    /// do not try the trained symbol detector's quads first (builds with
    /// the cargo feature `ml` only)
    #[arg(long)]
    no_detector: bool,
}

/// One stderr line per image with `DPMREAD_TIMINGS` set (P4b.7 profiling):
/// `timings <file> key=seconds ...`; grid stage counters are summed over
/// the grid jobs (`noqz` etc. nest: see `dm_grid::timing::Stage`).
fn timings_line(res: &ReadResult, t: &Timings) -> String {
    let name = res
        .path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut s = format!(
        "timings {name} ok={} total={:.3} workers={} variants={:.3} \
         zxing_wall={:.3} zxing_busy={:.3} zxing_jobs={} grid_variants={:.3} \
         grid_wall={:.3} grid_busy={:.3} grid_jobs={} ml={:.3} ml_calls={} detector={:.3} \
         detector_tries={} detector_gate={:.3} detector_models={:.3} detector_crops={:.3} \
         finish={:.3}",
        u8::from(!res.data.is_empty()),
        res.elapsed,
        t.workers,
        t.variants,
        t.zxing_wall,
        t.zxing_busy,
        t.zxing_jobs,
        t.grid_variants,
        t.grid_wall,
        t.grid_busy,
        t.grid_jobs,
        t.ml,
        t.ml_calls,
        t.detector,
        t.detector_tries,
        t.detector_gate,
        t.detector_models,
        t.detector_crops,
        t.finish,
    );
    for (n, v) in STAGE_NAMES.iter().zip(t.grid) {
        s += &format!(" g_{n}={v:.3}");
    }
    s
}

fn main() -> ExitCode {
    let args = Args::parse();
    let mut inputs = args.inputs.clone();
    if inputs.is_empty() {
        print!("Image file or folder: ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        let _ = std::io::stdin().lock().read_line(&mut line);
        let p = line.trim().trim_matches('"');
        if p.is_empty() {
            return ExitCode::SUCCESS;
        }
        inputs.push(PathBuf::from(p));
    }
    let names: Vec<&str> = args
        .engines
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect();
    let engines = Engines {
        zxing: names.contains(&"zxing"),
        grid: names.contains(&"grid"),
    };
    if engines.zxing && !ZXING_AVAILABLE {
        eprintln!("warning: zxing engine unavailable (not available in this build), skipped");
    }
    if names.contains(&"libdmtx") {
        eprintln!("warning: libdmtx engine unavailable (not available in this build), skipped");
    }
    let opts = ReadOptions {
        engines,
        exhaustive: args.exhaustive,
        budget_s: args.budget,
        expect: parse_sizes(&args.expect),
        polarity: match args.polarity {
            PolarityArg::Any => Polarity::Any,
            PolarityArg::Dark => Polarity::Dark,
            PolarityArg::Light => Polarity::Light,
        },
        debug_dir: args.debug.clone(),
        multi: args.multi,
        progress: None,
        analyse: !args.no_analysis,
        fit_inputs: args.dump_fits.is_some(),
        classifier: !args.no_classifier,
        detector: !args.no_detector,
    };
    let runner = Runner::new(args.jobs.max(1));
    let mut results = Vec::new();
    let mut failed = false;
    for path in collect_inputs(&inputs) {
        match read_image(&path, &opts, &runner) {
            Ok(mut res) => {
                if let Some(dir) = &args.dump_fits {
                    if let Err(e) = write_fit_dump(dir, &res) {
                        eprintln!("{}: --dump-fits: {e}", path.display());
                        failed = true;
                    }
                    res.fits.clear();
                }
                if !args.json {
                    print!("{}", report_text(&res, args.matrix));
                }
                if let Some(t) = &res.timings {
                    eprintln!("{}", timings_line(&res, t));
                }
                failed |= res.data.is_empty();
                results.push(res);
            }
            Err(e) => {
                eprintln!("{e}");
                failed = true;
            }
        }
    }
    if args.json {
        println!("{}", to_json(&results));
    } else if results.len() > 1 {
        let ok = results.iter().filter(|r| !r.data.is_empty()).count();
        println!("\nSummary: {ok}/{} decoded", results.len());
        for r in &results {
            let name = r
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let (tag, text) = if r.data.is_empty() {
                ("FAIL", String::new())
            } else {
                ("OK  ", printable(&r.data, r.text_dec()))
            };
            println!("  {tag} {name:40} {text}");
            for (k, m) in r.more.iter().enumerate() {
                let label = format!("  symbol {}", k + 2);
                println!("  +    {label:40} {}", printable(&m.data, m.text_dec()));
            }
        }
    }
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
