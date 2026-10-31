//! Host threads whose calls provably overlap in time. A barrier at the end of
//! every round means each round's calls come from all four threads at once,
//! so a trace that shows them neatly serialized is a trace that lost the
//! concurrency. Each thread makes a different total so counts identify it,
//! and reports the OS thread id capture must attribute its records to.

use std::ffi::{c_int, c_uint};
use std::sync::Barrier;

use gputrace_testkit::cuda_api;

cuda_api! {
    struct Driver {
        fn cuInit(c_uint) -> c_int;
        fn cuDriverGetVersion(*mut c_int) -> c_int;
    }
}

unsafe extern "C" {
    fn gettid() -> c_int;
}

const THREADS: usize = 4;
const ROUNDS: usize = 32;
const CALLS_PER_ROUND: usize = 16;
const EXTRA_PER_THREAD: usize = 37;

fn version(cuda: &Driver) {
    let mut version = 0;
    assert_eq!(unsafe { (cuda.cuDriverGetVersion)(&mut version) }, 0);
    assert_eq!(version, 13040);
}

fn main() {
    let cuda = Driver::load();
    unsafe { assert_eq!((cuda.cuInit)(0), 0) };
    let gate = Barrier::new(THREADS);

    let reports: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|id| {
                let (cuda, gate) = (&cuda, &gate);
                scope.spawn(move || {
                    let calls = ROUNDS * CALLS_PER_ROUND + id * EXTRA_PER_THREAD;
                    let mut made = 0;
                    for _ in 0..ROUNDS {
                        for _ in 0..CALLS_PER_ROUND {
                            version(cuda);
                            made += 1;
                        }
                        gate.wait();
                    }
                    while made < calls {
                        version(cuda);
                        made += 1;
                    }
                    format!("tid={} calls={calls}", unsafe { gettid() })
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("worker"))
            .collect()
    });
    for report in reports {
        println!("{report}");
    }
}
