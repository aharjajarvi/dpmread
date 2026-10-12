//! Read images and print one TSV line each: file, engine, variant, text,
//! attempts, seconds, near-miss size.
//!
//! cargo run --release -p dm-reader --example read_summary -- [--jobs N] [--json] IMAGE_OR_FOLDER...
//!
//! `--json`: print the `to_json` list instead; `--report`: the text report
//! (with the module matrix).

use std::path::PathBuf;

use dm_reader::{
    ReadOptions, Runner, collect_inputs, default_jobs, read_image, report_text, to_json,
};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut jobs = default_jobs();
    if let Some(i) = args.iter().position(|a| a == "--jobs") {
        jobs = args[i + 1].parse().expect("--jobs N");
        args.drain(i..i + 2);
    }
    let json = args.iter().any(|a| a == "--json");
    let report = args.iter().any(|a| a == "--report");
    args.retain(|a| a != "--json" && a != "--report");
    let runner = Runner::new(jobs);
    let mut results = Vec::new();
    let opts = ReadOptions {
        budget_s: 60.0,
        ..Default::default()
    };
    let paths: Vec<PathBuf> = args.iter().map(PathBuf::from).collect();
    for p in collect_inputs(&paths) {
        match read_image(&p, &opts, &runner) {
            Ok(r) if json => results.push(r),
            Ok(r) if report => print!("{}", report_text(&r, true)),
            Ok(r) => println!(
                "{}\t{}\t{}\t{}\t{}\t{:.2}\t{}",
                p.display(),
                r.engine,
                r.variant,
                r.text,
                r.attempts,
                r.elapsed,
                r.near_miss.map(|n| n.nm.size).unwrap_or_default()
            ),
            Err(e) => eprintln!("{e}"),
        }
    }
    if json {
        println!("{}", to_json(&results));
    }
}
