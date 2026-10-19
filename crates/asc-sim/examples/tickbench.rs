//! Times the stand-in factory's tick: `cargo run --release -p asc-sim --example tickbench`.
//!
//! First 100k waiting timer machines alone, then busy islands on 1, 2, 4... threads.

use std::time::Instant;

use asc_sim::stress::{Stress, StressConfig};

fn time(config: StressConfig, threads: usize, ticks: u32) -> (f64, f64, u64) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let mut s = Stress::new(config, 0);
        // Warm up past the first wave of short recipes.
        for _ in 0..200 {
            s.tick();
        }
        let (mut total, mut worst) = (0.0f64, 0.0f64);
        for _ in 0..ticks {
            let t = Instant::now();
            s.tick();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            total += ms;
            worst = worst.max(ms);
        }
        (total / ticks as f64, worst, s.checksum())
    })
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    for timers in [100_000, 1_000_000] {
        let idle = StressConfig { timers, islands: 0, per_island: 0, work: 0 };
        let (avg, worst, _) = time(idle, 1, 2_000);
        println!("{timers} timer machines, nothing else: {avg:.3} ms per tick (worst {worst:.3})");
    }
    let busy = StressConfig { timers: 0, islands: 64, per_island: 4_000, work: 200 };
    let mut base = None;
    let mut sum = None;
    let mut threads = 1;
    while threads <= cores {
        let (avg, worst, check) = time(busy, threads, 100);
        let speedup = base.map_or(1.0, |b: f64| b / avg);
        base.get_or_insert(avg);
        assert_eq!(*sum.get_or_insert(check), check, "result changed with {threads} threads");
        println!("64 busy islands on {threads:2} threads: {avg:7.2} ms per tick (worst {worst:7.2}), x{speedup:.1}");
        threads *= 2;
    }
    println!("same result on every thread count");
}
