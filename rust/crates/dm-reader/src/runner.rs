//! Job runner (reader `Runner`, `JOB_GRACE_S`, `DEFAULT_JOBS`): one decoder
//! invocation on one variant per job, run inline (`--jobs 1`) or on a rayon
//! pool. Results are consumed in submission order, so the outcome does not
//! depend on which worker finishes first.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How long past the budget a running job is awaited.
pub const JOB_GRACE_S: f64 = 0.5;

/// A non-tiled image has 12 grid jobs with a 25 s cap: 4 workers could not
/// finish the ladder inside the default budget, so use most of the CPU.
pub fn default_jobs() -> usize {
    let cpus = std::thread::available_parallelism().map_or(2, |n| n.get());
    cpus.saturating_sub(1).clamp(1, 8)
}

pub struct Runner {
    pool: Option<rayon::ThreadPool>,
    /// One thread for `install` in inline mode, so that the rayon work in
    /// the variant builders stays sequential with `--jobs 1`.
    seq: Option<rayon::ThreadPool>,
    abort: Arc<AtomicBool>,
    /// Set by a `StopHandle`: no job starts until it is cleared.
    stop: Arc<AtomicBool>,
}

/// Stops a `Runner` from another thread (GUI Stop): running jobs are asked
/// to end through the abort flag and no job starts until `resume`, so the
/// `read_image` in progress returns soon (with whatever it has).
#[derive(Clone)]
pub struct StopHandle {
    stop: Arc<AtomicBool>,
    abort: Arc<AtomicBool>,
}

impl StopHandle {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        self.abort.store(true, Ordering::Relaxed);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Allow jobs again; call only while no `run` is in progress.
    pub fn resume(&self) {
        self.stop.store(false, Ordering::Relaxed);
        self.abort.store(false, Ordering::Relaxed);
    }
}

impl Runner {
    /// `jobs` workers; 1 (or a failed pool) runs every job inline.
    pub fn new(jobs: usize) -> Self {
        let pool = if jobs > 1 {
            match rayon::ThreadPoolBuilder::new().num_threads(jobs).build() {
                Ok(p) => Some(p),
                Err(e) => {
                    eprintln!("warning: no worker pool ({e}), running sequentially");
                    None
                }
            }
        } else {
            None
        };
        let seq = if pool.is_none() {
            rayon::ThreadPoolBuilder::new().num_threads(1).build().ok()
        } else {
            None
        };
        Runner {
            pool,
            seq,
            abort: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn stop_handle(&self) -> StopHandle {
        StopHandle {
            stop: self.stop.clone(),
            abort: self.abort.clone(),
        }
    }

    /// A `StopHandle` stopped this runner.
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Run `f` on the worker pool, so rayon work inside it (the variant
    /// builders) uses the same workers; one thread in inline mode.
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match self.pool.as_ref().or(self.seq.as_ref()) {
            Some(p) => p.install(f),
            None => f(),
        }
    }

    /// Number of workers (1 = inline).
    pub fn workers(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.current_num_threads())
    }

    /// Set while the consumer has stopped and running jobs should end
    /// (Python's `_ABORT` event); give it to the jobs' `Deadline`.
    pub fn abort_flag(&self) -> Arc<AtomicBool> {
        self.abort.clone()
    }

    /// Run `f` on every job and hand the results to `consume` in job order
    /// until it breaks. Inline mode runs each job only after the previous
    /// result was consumed and starts none after `deadline`. Pool mode
    /// starts all jobs and waits for results until `deadline + JOB_GRACE_S`;
    /// on stopping, jobs not yet started are skipped and running ones are
    /// asked to end through the abort flag and awaited. Nothing runs while
    /// stopped (`StopHandle`).
    pub fn run<J, R>(
        &self,
        jobs: &[J],
        deadline: Option<Instant>,
        f: impl Fn(&J) -> R + Sync,
        mut consume: impl FnMut(usize, R) -> ControlFlow<()>,
    ) where
        J: Sync,
        R: Send,
    {
        if self.stopped() {
            return;
        }
        let Some(pool) = &self.pool else {
            for (i, job) in jobs.iter().enumerate() {
                if deadline.is_some_and(|d| Instant::now() > d) || self.stopped() {
                    return;
                }
                if consume(i, f(job)).is_break() {
                    return;
                }
            }
            return;
        };
        let stop = deadline.map(|d| d + Duration::from_secs_f64(JOB_GRACE_S));
        let (tx, rx) = mpsc::channel::<(usize, R)>();
        let (f, abort) = (&f, &self.abort);
        pool.in_place_scope(|s| {
            for (i, job) in jobs.iter().enumerate() {
                let tx = tx.clone();
                s.spawn(move |_| {
                    if !abort.load(Ordering::Relaxed) {
                        let _ = tx.send((i, f(job)));
                    }
                });
            }
            drop(tx);
            let mut slots: Vec<Option<R>> = (0..jobs.len()).map(|_| None).collect();
            let mut next = 0;
            'consume: while next < jobs.len() {
                while let Some(r) = slots[next].take() {
                    if self.stopped() || consume(next, r).is_break() {
                        break 'consume;
                    }
                    next += 1;
                    if next == jobs.len() {
                        break 'consume;
                    }
                }
                let got = match stop {
                    None => rx.recv().ok(),
                    Some(t) => rx
                        .recv_timeout(t.saturating_duration_since(Instant::now()))
                        .ok(),
                };
                match got {
                    Some((i, r)) => slots[i] = Some(r),
                    None => break,
                }
            }
            abort.store(true, Ordering::Relaxed);
        });
        self.abort.store(false, Ordering::Relaxed);
    }
}
