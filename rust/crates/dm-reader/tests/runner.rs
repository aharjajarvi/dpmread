//! P4.2: the runner yields results in job order whatever the worker count
//! and job durations, stops on break and at the deadline, and asks running
//! jobs to end through the abort flag.

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};

use dm_reader::Runner;

/// Job durations in ms, 0–19, from a fixed-seed LCG.
fn durations(n: usize, seed: u64) -> Vec<u64> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) % 20
        })
        .collect()
}

fn collect(runner: &Runner, jobs: &[u64]) -> Vec<(usize, u64)> {
    let mut out = Vec::new();
    runner.run(
        jobs,
        None,
        |&ms| {
            sleep(Duration::from_millis(ms));
            ms * 10
        },
        |i, r| {
            out.push((i, r));
            ControlFlow::Continue(())
        },
    );
    out
}

#[test]
fn ordered_for_any_worker_count() {
    for seed in 1..4 {
        let jobs = durations(40, seed);
        let want: Vec<(usize, u64)> = jobs.iter().map(|&ms| ms * 10).enumerate().collect();
        for workers in [1, 2, 4, 8] {
            assert_eq!(collect(&Runner::new(workers), &jobs), want, "{workers}");
        }
    }
}

#[test]
fn break_skips_the_rest_and_aborts_running_jobs() {
    for workers in [1, 4] {
        let runner = Runner::new(workers);
        let abort = runner.abort_flag();
        let started = AtomicUsize::new(0);
        let jobs: Vec<usize> = (0..200).collect();
        let mut seen = Vec::new();
        runner.run(
            &jobs,
            None,
            |&i| {
                started.fetch_add(1, Ordering::Relaxed);
                // job 7 runs until the abort flag is set (a grid job)
                while i == 7 && workers > 1 && !abort.load(Ordering::Relaxed) {
                    sleep(Duration::from_millis(1));
                }
                sleep(Duration::from_millis(2));
                i
            },
            |i, r| {
                seen.push(r);
                if i == 5 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        );
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
        assert!(started.load(Ordering::Relaxed) < 200, "{workers}");
        assert!(!abort.load(Ordering::Relaxed), "abort flag cleared");
    }
}

#[test]
fn deadline_stops_consuming() {
    for workers in [1, 4] {
        let runner = Runner::new(workers);
        let jobs: Vec<u64> = vec![100; 40];
        let t0 = Instant::now();
        let mut n = 0;
        runner.run(
            &jobs,
            Some(t0 + Duration::from_millis(100)),
            |&ms| sleep(Duration::from_millis(ms)),
            |_, ()| {
                n += 1;
                ControlFlow::Continue(())
            },
        );
        // inline: ~4 jobs before the deadline; pool: until deadline + grace
        assert!(n < 40, "{workers}: {n}");
        assert!(t0.elapsed() < Duration::from_millis(1500), "{workers}");
    }
}

/// GUI Stop: running jobs see the abort flag, the run ends early, and no
/// later run starts until `resume`.
#[test]
fn stop_handle_ends_the_run_and_blocks_until_resume() {
    for workers in [1, 4] {
        let runner = Runner::new(workers);
        let abort = runner.abort_flag();
        let handle = runner.stop_handle();
        let stopper = handle.clone();
        let t = std::thread::spawn(move || {
            sleep(Duration::from_millis(100));
            stopper.stop();
        });
        // jobs that run 10 s unless aborted
        let jobs: Vec<usize> = (0..20).collect();
        let t0 = Instant::now();
        let mut consumed = 0;
        runner.run(
            &jobs,
            None,
            |_| {
                let end = Instant::now() + Duration::from_secs(10);
                while Instant::now() < end && !abort.load(Ordering::Relaxed) {
                    sleep(Duration::from_millis(5));
                }
            },
            |_, _| {
                consumed += 1;
                ControlFlow::Continue(())
            },
        );
        t.join().unwrap();
        assert!(
            t0.elapsed() < Duration::from_secs(3),
            "{workers}: {:?}",
            t0.elapsed()
        );
        assert!(consumed < jobs.len(), "{workers}");
        assert!(runner.stopped() && handle.is_stopped());
        assert_eq!(
            collect(&runner, &[1, 2]),
            [],
            "{workers}: runs while stopped"
        );
        handle.resume();
        assert!(!abort.load(Ordering::Relaxed));
        assert_eq!(collect(&runner, &[1, 2]), [(0, 10), (1, 20)], "{workers}");
    }
}
