//! Keeps a GPU session's first run of each input shape away from every other run.
//!
//! ONNX Runtime 1.19's CUDA provider picks each convolution's cuDNN algorithm on
//! the first run of a new input shape (`cudnn_conv_algo_search` EXHAUSTIVE, its
//! default) and keeps the choice for the life of the session
//! (`onnxruntime/core/providers/cuda/nn/conv.cc`, `UpdateState`). The search
//! calls `cudnnFindConvolutionForwardAlgorithmEx`, which disturbs CUDA work on
//! other threads while it runs. When two sessions search at once, or one
//! searches while another runs, a search can come back with no algorithm.
//! `conv.cc` then stores its uninitialised `perf` struct, and every later run of
//! that session asks the arena for a garbage workspace ("Failed to allocate
//! memory for requested buffer of size 132985260658176") until the session is
//! dropped. Other runs fail meanwhile with `cudaErrorStreamCaptureUnsupported`.
//!
//! Measured on an RTX 3090 Ti with ONNX Runtime 1.19.0 and cuDNN 9.9, four
//! RetinaFace sessions started together: three kept failing every run. With
//! their first runs taken one at a time, none failed (Eagle issue 1376).
//!
//! So a gated session runs a shape it has not yet run successfully alone: it
//! takes the write side of a process-wide lock, which waits for every run in
//! flight and holds the others back. Shapes it has run before take the read
//! side and run side by side as before.
//!
//! A shape that fails its first `ALONE_ATTEMPTS` runs is treated as run: a
//! session that fails every frame would otherwise hold every other GPU session
//! back on every one of its frames. `alone/1` holds a session's creation
//! apart from runs the same way, since building a CUDA session allocates and
//! copies on the GPU too.

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

/// The lock every gated session in the process shares.
pub static GPU_RUNS: RwLock<()> = RwLock::new(());

/// How many failed runs of a shape still take the write side.
pub const ALONE_ATTEMPTS: u32 = 3;

/// Runs `f` while no gated run is in flight, holding them back meanwhile.
pub fn alone<T>(lock: &RwLock<()>, f: impl FnOnce() -> T) -> T {
    let _alone = lock.write().unwrap_or_else(|e| e.into_inner());
    f()
}

pub struct RunGate {
    lock: &'static RwLock<()>,
    gated: bool,
    /// Failed lone runs per shape; `None` once a run of the shape succeeded.
    tries: Mutex<HashMap<Vec<Vec<usize>>, Option<u32>>>,
}

impl RunGate {
    /// `gated` is false for a session with no GPU provider: it runs ungated.
    pub fn new(lock: &'static RwLock<()>, gated: bool) -> Self {
        RunGate {
            lock,
            gated,
            tries: Mutex::new(HashMap::new()),
        }
    }

    /// Runs `f` for inputs of `shapes`, alone until `f` has succeeded on those
    /// shapes once or failed on them `ALONE_ATTEMPTS` times.
    pub fn run<T, E>(
        &self,
        shapes: Vec<Vec<usize>>,
        f: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if !self.gated {
            return f();
        }

        let shared = match self
            .tries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&shapes)
        {
            Some(None) => true,
            Some(Some(failed)) => *failed >= ALONE_ATTEMPTS,
            None => false,
        };

        if shared {
            let _shared = self.lock.read().unwrap_or_else(|e| e.into_inner());
            f()
        } else {
            let result = alone(self.lock, f);
            let mut tries = self.tries.lock().unwrap_or_else(|e| e.into_inner());

            if result.is_ok() {
                tries.insert(shapes, None);
            } else {
                let entry = tries.entry(shapes).or_insert(Some(0));
                *entry = entry.map(|failed| failed + 1);
            }

            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    fn lock() -> &'static RwLock<()> {
        Box::leak(Box::new(RwLock::new(())))
    }

    // Runs `n` threads through `gate_for(i)` at once, each holding its run for
    // a while, and returns the most runs that were ever in flight together.
    fn peak(n: usize, gate_for: impl Fn(usize) -> Arc<RunGate>, shape: Vec<usize>) -> usize {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|i| {
                let gate = gate_for(i);
                let (in_flight, peak, start) = (in_flight.clone(), peak.clone(), start.clone());
                let shape = shape.clone();

                thread::spawn(move || {
                    start.wait();

                    gate.run(vec![shape], || -> Result<(), ()> {
                        let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(30));
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        peak.load(Ordering::SeqCst)
    }

    #[test]
    fn first_runs_of_different_sessions_run_one_at_a_time() {
        let lock = lock();
        let gates: Vec<_> = (0..4).map(|_| Arc::new(RunGate::new(lock, true))).collect();

        assert_eq!(peak(4, |i| gates[i].clone(), vec![1, 3, 640, 640]), 1);
    }

    #[test]
    fn runs_of_a_shape_already_run_go_side_by_side() {
        let lock = lock();
        let gate = Arc::new(RunGate::new(lock, true));
        gate.run(vec![vec![1, 3, 640, 640]], || -> Result<(), ()> { Ok(()) })
            .unwrap();

        assert_eq!(peak(4, |_| gate.clone(), vec![1, 3, 640, 640]), 4);
    }

    #[test]
    fn a_new_shape_on_a_session_runs_alone_again() {
        let lock = lock();
        let gate = Arc::new(RunGate::new(lock, true));
        gate.run(vec![vec![1, 3, 640, 640]], || -> Result<(), ()> { Ok(()) })
            .unwrap();

        assert_eq!(peak(4, |_| gate.clone(), vec![1, 3, 320, 320]), 1);
    }

    #[test]
    fn a_shape_that_keeps_failing_stops_running_alone() {
        let lock = lock();
        let gate = Arc::new(RunGate::new(lock, true));

        for _ in 0..ALONE_ATTEMPTS {
            let _ = gate.run(vec![vec![1, 3, 640, 640]], || -> Result<(), ()> { Err(()) });
        }

        assert_eq!(peak(4, |_| gate.clone(), vec![1, 3, 640, 640]), 4);
    }

    #[test]
    fn a_session_build_waits_for_runs_in_flight() {
        let lock = lock();
        let warm = Arc::new(RunGate::new(lock, true));
        warm.run(vec![vec![1]], || -> Result<(), ()> { Ok(()) })
            .unwrap();

        let in_flight = Arc::new(AtomicUsize::new(0));
        let overlapped = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(2));

        let runner = {
            let (warm, in_flight, start) = (warm.clone(), in_flight.clone(), start.clone());
            thread::spawn(move || {
                warm.run(vec![vec![1]], || -> Result<(), ()> {
                    in_flight.fetch_add(1, Ordering::SeqCst);
                    start.wait();
                    thread::sleep(Duration::from_millis(50));
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
                .unwrap();
            })
        };

        start.wait();
        alone(lock, || {
            overlapped.store(in_flight.load(Ordering::SeqCst), Ordering::SeqCst)
        });
        runner.join().unwrap();

        assert_eq!(overlapped.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_failed_first_run_leaves_the_shape_unseen() {
        let lock = lock();
        let gate = Arc::new(RunGate::new(lock, true));
        let _ = gate.run(vec![vec![1, 3, 640, 640]], || -> Result<(), ()> { Err(()) });

        assert_eq!(peak(4, |_| gate.clone(), vec![1, 3, 640, 640]), 1);
    }

    #[test]
    fn a_first_run_waits_for_runs_in_flight_on_other_sessions() {
        let lock = lock();
        let warm = Arc::new(RunGate::new(lock, true));
        warm.run(vec![vec![1]], || -> Result<(), ()> { Ok(()) })
            .unwrap();
        let cold = Arc::new(RunGate::new(lock, true));

        assert_eq!(
            peak(
                4,
                |i| if i == 0 { cold.clone() } else { warm.clone() },
                vec![1]
            ),
            3
        );
    }

    #[test]
    fn an_ungated_session_never_waits() {
        let lock = lock();
        let gates: Vec<_> = (0..4)
            .map(|_| Arc::new(RunGate::new(lock, false)))
            .collect();

        assert_eq!(peak(4, |i| gates[i].clone(), vec![1, 3, 640, 640]), 4);
    }
}
