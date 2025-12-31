//! Benchmarks for GBM simulation methods

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use stochastic_calculus::gbm::GeometricBrownianMotion;

fn benchmark_gbm_simulation(c: &mut Criterion) {
    let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
    let n_steps = 252; // 1 year of daily data
    let dt = 1.0 / 252.0;

    let mut group = c.benchmark_group("GBM Path Generation");

    for n_paths in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("Sequential", n_paths),
            n_paths,
            |b, &n| {
                b.iter(|| {
                    let mut rng = rand::thread_rng();
                    (0..n)
                        .map(|_| gbm.generate_path(&mut rng, n_steps, dt))
                        .collect::<Vec<_>>()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Parallel", n_paths),
            n_paths,
            |b, &n| {
                b.iter(|| gbm.generate_paths_parallel(n, n_steps, dt));
            },
        );
    }

    group.finish();
}

fn benchmark_terminal_sampling(c: &mut Criterion) {
    let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
    let t = 1.0;

    let mut group = c.benchmark_group("GBM Terminal Sampling");

    for n_paths in [10000, 100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("Parallel", n_paths), n_paths, |b, &n| {
            b.iter(|| gbm.sample_terminal_parallel(n, t));
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_gbm_simulation, benchmark_terminal_sampling);
criterion_main!(benches);
