//! Time budget of a grid search: an absolute deadline plus a shared abort
//! flag (Python `_expired`: `time.time() > deadline` or the pool's abort
//! event).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct Deadline {
    /// `None`: no time limit.
    pub at: Option<Instant>,
    pub abort: Option<Arc<AtomicBool>>,
}

impl Deadline {
    pub fn at(at: Instant) -> Self {
        Deadline {
            at: Some(at),
            abort: None,
        }
    }

    /// No time limit, no abort flag.
    pub fn never() -> Self {
        Deadline::default()
    }

    /// Same abort flag, deadline at most `cap` from now.
    pub fn capped(&self, cap: Duration) -> Self {
        let c = Instant::now() + cap;
        Deadline {
            at: Some(self.at.map_or(c, |a| a.min(c))),
            abort: self.abort.clone(),
        }
    }

    pub fn expired(&self) -> bool {
        self.at.is_some_and(|a| Instant::now() > a)
            || self
                .abort
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
    }
}
