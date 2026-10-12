//! Background reader: one thread owns the job queue and the `Runner` and
//! reads the images one after another, sending progress events back.
//! `stop` ends the image being read (it finishes as `Outcome::Stopped`).

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use dm_reader::{Progress, ReadOptions, ReadResult, Runner, StopHandle, read_image};

/// One image to read; `id` names its row in the events.
pub struct Job {
    pub id: u64,
    pub path: PathBuf,
    pub opts: Arc<ReadOptions>,
    /// Reader threads (the `Runner` is rebuilt when this changes).
    pub workers: usize,
}

/// Progress of the worker, in job order.
#[derive(Debug)]
pub enum Event {
    Started(u64),
    /// `(id, stage, jobs done, jobs in stage)` (see `dm_reader::Progress`).
    Stage(u64, String, usize, usize),
    Finished(u64, Outcome),
}

#[derive(Debug)]
pub enum Outcome {
    Read(Box<ReadResult>),
    /// Load error text.
    Error(String),
    /// Ended by `Worker::stop` (the partial result is dropped).
    Stopped,
}

pub struct Worker {
    jobs: Option<Sender<Job>>,
    events: Receiver<Event>,
    thread: Option<JoinHandle<()>>,
    /// Stop handle of the worker's current `Runner`.
    stop: Arc<Mutex<Option<StopHandle>>>,
}

impl Worker {
    /// Start the thread; `wake` is called after every event (the GUI's
    /// `request_repaint`).
    pub fn spawn(wake: impl Fn() + Send + Sync + 'static) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ev_tx, ev_rx) = mpsc::channel::<Event>();
        let wake = Arc::new(wake);
        let stop = Arc::new(Mutex::new(None));
        let stop2 = stop.clone();
        let thread = std::thread::Builder::new()
            .name("dm-reader".into())
            .spawn(move || run(&job_rx, &ev_tx, wake, &stop2))
            .expect("spawn reader thread");
        Worker {
            jobs: Some(job_tx),
            events: ev_rx,
            thread: Some(thread),
            stop,
        }
    }

    /// End the image being read soon (`Outcome::Stopped`). Jobs already
    /// pushed after it still run: the caller keeps its own queue.
    pub fn stop(&self) {
        if let Some(h) = self.stop.lock().ok().as_deref().and_then(Option::as_ref) {
            h.stop();
        }
    }

    /// Append a job to the queue (also while one is running).
    pub fn push(&self, job: Job) {
        if let Some(tx) = &self.jobs {
            let _ = tx.send(job);
        }
    }

    /// Events sent since the last call.
    pub fn poll(&self) -> impl Iterator<Item = Event> + '_ {
        self.events.try_iter()
    }
}

impl Drop for Worker {
    /// Close the queue, stop the image being read and wait for it.
    fn drop(&mut self) {
        self.jobs.take();
        self.stop();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn run(
    jobs: &Receiver<Job>,
    events: &Sender<Event>,
    wake: Arc<dyn Fn() + Send + Sync>,
    stop: &Mutex<Option<StopHandle>>,
) {
    let mut runner: Option<(usize, Runner)> = None;
    let send = |e: Event| {
        let _ = events.send(e);
        wake();
    };
    for job in jobs {
        let workers = job.workers.max(1);
        if runner.as_ref().is_none_or(|r| r.0 != workers) {
            let r = Runner::new(workers);
            *stop.lock().expect("stop lock") = Some(r.stop_handle());
            runner = Some((workers, r));
        }
        let runner = &runner.as_ref().expect("runner").1;
        let handle = runner.stop_handle();
        send(Event::Started(job.id));
        let mut opts = (*job.opts).clone();
        let (id, tx, w) = (job.id, events.clone(), wake.clone());
        opts.progress = Some(Progress(Arc::new(move |stage, done, total| {
            let _ = tx.send(Event::Stage(id, stage.to_string(), done, total));
            w();
        })));
        let r = read_image(&job.path, &opts, runner);
        // the lock orders this against `Worker::stop`
        let stopped = {
            let _g = stop.lock().expect("stop lock");
            let s = handle.is_stopped();
            handle.resume();
            s
        };
        let outcome = match r {
            _ if stopped => Outcome::Stopped,
            Ok(r) => Outcome::Read(Box::new(r)),
            Err(e) => Outcome::Error(e.to_string()),
        };
        send(Event::Finished(job.id, outcome));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test image in the private test data (`DM_PRIVATE_DIR`, not part of
    /// this repo); the tests using it are ignored unless `rust/tools/ci.sh`
    /// finds the data.
    fn test_image(rel: &str) -> PathBuf {
        std::env::var_os("DM_PRIVATE_DIR")
            .map(PathBuf::from)
            .expect("DM_PRIVATE_DIR not set: it points at the private test data")
            .join("testimages")
            .join(rel)
    }

    /// A test image by role from `testimages/picks.json` (the names stay in
    /// the private data).
    fn pick(key: &str) -> PathBuf {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(test_image("picks.json")).unwrap())
                .unwrap();
        test_image(v[key].as_str().unwrap())
    }

    /// Wait until `n` jobs finished; events in arrival order.
    fn collect(w: &Worker, n: usize) -> Vec<Event> {
        let mut out = Vec::new();
        let t0 = std::time::Instant::now();
        while out
            .iter()
            .filter(|e| matches!(e, Event::Finished(..)))
            .count()
            < n
        {
            assert!(t0.elapsed().as_secs() < 120, "worker timed out");
            out.extend(w.poll());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        out
    }

    #[test]
    #[ignore = "needs private test data (DM_PRIVATE_DIR)"]
    fn reads_in_order_with_progress() {
        let w = Worker::spawn(|| {});
        let opts = Arc::new(ReadOptions::default());
        let paths = [pick("gui_quick_read"), PathBuf::from("/nonexistent/x.png")];
        for (id, p) in paths.iter().enumerate() {
            w.push(Job {
                id: id as u64,
                path: p.clone(),
                opts: opts.clone(),
                workers: 2 - id,
            });
        }
        let ev = collect(&w, 2);
        assert!(matches!(ev[0], Event::Started(0)));
        let stages = ev
            .iter()
            .take_while(|e| !matches!(e, Event::Finished(..)))
            .filter(|e| matches!(e, Event::Stage(0, s, d, n) if s == "zxing" && d <= n))
            .count();
        assert!(stages > 0, "no progress events: {ev:?}");
        let finished: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                Event::Finished(id, r) => Some((*id, r)),
                _ => None,
            })
            .collect();
        assert_eq!(finished[0].0, 0);
        let Outcome::Read(r) = finished[0].1 else {
            panic!("{:?}", finished[0].1)
        };
        assert_eq!(r.engine, "zxing");
        assert!(!r.data.is_empty());
        assert_eq!(finished[1].0, 1);
        assert!(matches!(finished[1].1, Outcome::Error(_)));
    }

    #[test]
    #[ignore = "needs private test data (DM_PRIVATE_DIR)"]
    fn stop_ends_the_image_and_the_next_one_reads() {
        let w = Worker::spawn(|| {});
        let opts = Arc::new(ReadOptions::default());
        let paths = [
            // fails after the whole grid ladder (several seconds)
            pick("gui_slow_fail"),
            pick("gui_quick_read"),
        ];
        for (id, p) in paths.iter().enumerate() {
            w.push(Job {
                id: id as u64,
                path: p.clone(),
                opts: opts.clone(),
                workers: 4,
            });
        }
        let t0 = std::time::Instant::now();
        let mut ev = Vec::new();
        // stop at the first grid job
        while !ev
            .iter()
            .any(|e| matches!(e, Event::Stage(0, s, ..) if s == "grid"))
        {
            assert!(t0.elapsed().as_secs() < 60, "no grid stage");
            ev.extend(w.poll());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let t_stop = std::time::Instant::now();
        w.stop();
        let ev = collect(&w, 2);
        let fin: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                Event::Finished(id, o) => Some((*id, o)),
                _ => None,
            })
            .collect();
        assert!(matches!(fin[0], (0, Outcome::Stopped)), "{fin:?}");
        assert!(
            matches!(fin[1], (1, Outcome::Read(r)) if !r.data.is_empty()),
            "{fin:?}"
        );
        // the stopped image ended well before its own grid ladder would
        assert!(
            t_stop.elapsed().as_secs_f64() < 3.0,
            "{:?}",
            t_stop.elapsed()
        );
    }
}
