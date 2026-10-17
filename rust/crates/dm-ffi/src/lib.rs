//! uniffi bindings of `dm_reader::read_image` for apps: read an image file
//! with the CLI's options and get the CLI's JSON object, its text report and
//! the outline and module matrix of every symbol; stop a running read from
//! another thread.

use std::path::Path;
use std::sync::Mutex;

use dm_reader::{Engines, ReadResult, Runner, parse_sizes, printable, read_image, report_text};

uniffi::setup_scaffolding!();

/// Mark polarity (`--polarity`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum Polarity {
    Any,
    Dark,
    Light,
}

/// The CLI flags that affect one image; `default_options` gives the CLI
/// defaults.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Options {
    pub zxing: bool,
    pub grid: bool,
    pub exhaustive: bool,
    /// Seconds per image (`--budget`).
    pub budget_s: f64,
    /// Expected symbol sizes such as `"24x24"` (`--expect`); empty: all.
    pub expect: Vec<String>,
    pub polarity: Polarity,
    /// Read every symbol of the image (`--multi`).
    pub multi: bool,
    /// After a zxing read, fit the grid for the module matrix, RS
    /// statistics and grades; `false` (`--no-analysis`) is faster, same text.
    #[uniffi(default = true)]
    pub analyse: bool,
}

/// One read symbol: the first, then the further ones of a `multi` read.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Symbol {
    /// Text as the CLI summary prints it (control bytes as `<GS>` etc.).
    pub text: String,
    /// Outline corners in image pixels (EXIF rotation applied):
    /// `x0, y0, … x3, y3`; empty when the engine gave none.
    pub outline: Vec<f32>,
    /// Size in modules; 0 without a grid fit.
    pub rows: u32,
    pub cols: u32,
    /// Sampled module matrix, row-major `rows * cols`, 1 = dark; empty
    /// without one.
    pub modules: Vec<u8>,
}

/// Result of `Reader::read`.
#[derive(Clone, Debug, uniffi::Record)]
pub struct ReadOutput {
    pub decoded: bool,
    /// Ended by `Reader::stop`; the rest is what the read had by then.
    pub stopped: bool,
    /// The CLI's `--json` object of the image (one array element).
    pub json: String,
    /// The CLI's text report, with the module matrix as text art.
    pub report: String,
    /// Empty when not decoded.
    pub symbols: Vec<Symbol>,
}

#[derive(Debug, uniffi::Error)]
pub enum ReadError {
    /// The file could not be read or decoded as an image. (Not `message`:
    /// that clashes with `Throwable.message` in the Kotlin bindings.)
    Load { reason: String },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Load { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for ReadError {}

/// The CLI defaults: both engines, 40 s, any polarity, one symbol.
#[uniffi::export]
pub fn default_options() -> Options {
    Options {
        zxing: true,
        grid: true,
        exhaustive: false,
        budget_s: 40.0,
        expect: Vec::new(),
        polarity: Polarity::Any,
        multi: false,
        analyse: true,
    }
}

/// Worker threads the CLI uses by default on this device.
#[uniffi::export]
pub fn default_jobs() -> u32 {
    dm_reader::default_jobs() as u32
}

/// Version of the core (the workspace version).
#[uniffi::export]
pub fn core_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A reader with its own worker pool. One read runs at a time (a second
/// `read` waits); `stop` may come from any thread.
#[derive(uniffi::Object)]
pub struct Reader {
    runner: Runner,
    /// Held for the whole of a read.
    busy: Mutex<()>,
    /// A read is running; orders `stop` against the end of a read, so a
    /// stop never outlives the read it was meant for.
    running: Mutex<bool>,
}

#[uniffi::export]
impl Reader {
    /// `jobs` worker threads (1 = sequential; 0 counts as 1).
    #[uniffi::constructor]
    pub fn new(jobs: u32) -> Self {
        Reader {
            runner: Runner::new(jobs.max(1) as usize),
            busy: Mutex::new(()),
            running: Mutex::new(false),
        }
    }

    /// Read the image file at `path`. Blocks: call it off the UI thread.
    pub fn read(&self, path: String, options: Options) -> Result<ReadOutput, ReadError> {
        let _busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        *self.running.lock().unwrap_or_else(|e| e.into_inner()) = true;
        let r = read_image(Path::new(&path), &to_read_options(&options), &self.runner);
        let stopped = {
            let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
            *running = false;
            let s = self.runner.stopped();
            self.runner.stop_handle().resume();
            s
        };
        let res = r.map_err(|e| ReadError::Load {
            reason: e.to_string(),
        })?;
        Ok(output(&res, stopped))
    }

    /// End the running read soon (it returns with `stopped` set); no
    /// effect when no read is running.
    pub fn stop(&self) {
        if *self.running.lock().unwrap_or_else(|e| e.into_inner()) {
            self.runner.stop_handle().stop();
        }
    }
}

fn to_read_options(o: &Options) -> dm_reader::ReadOptions {
    dm_reader::ReadOptions {
        engines: Engines {
            zxing: o.zxing,
            grid: o.grid,
        },
        exhaustive: o.exhaustive,
        budget_s: o.budget_s,
        expect: parse_sizes(&o.expect),
        polarity: match o.polarity {
            Polarity::Any => dm_reader::Polarity::Any,
            Polarity::Dark => dm_reader::Polarity::Dark,
            Polarity::Light => dm_reader::Polarity::Light,
        },
        debug_dir: None,
        multi: o.multi,
        progress: None,
        analyse: o.analyse,
        fit_inputs: false,
        classifier: true,
    }
}

fn symbol(r: &ReadResult) -> Symbol {
    let outline = r
        .quad
        .map(|q| q.iter().flat_map(|&(x, y)| [x, y]).collect())
        .unwrap_or_default();
    let (rows, cols, modules) = match &r.fit {
        Some(f) => (
            f.geom.rows as u32,
            f.geom.cols as u32,
            f.bits.clone().unwrap_or_default(),
        ),
        None => (0, 0, Vec::new()),
    };
    Symbol {
        text: printable(&r.data, r.text_dec()),
        outline,
        rows,
        cols,
        modules,
    }
}

/// `ReadOutput` of one `read_image` result.
pub fn output(res: &ReadResult, stopped: bool) -> ReadOutput {
    let decoded = !res.data.is_empty();
    let symbols = if decoded {
        std::iter::once(res).chain(&res.more).map(symbol).collect()
    } else {
        Vec::new()
    };
    ReadOutput {
        decoded,
        stopped,
        json: serde_json::to_string_pretty(&dm_reader::to_dict(res)).unwrap_or_default(),
        report: report_text(res, true),
        symbols,
    }
}
