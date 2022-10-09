// Copyright (c) 2022. Sebastien Soudan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http:www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Bootstrap Hypothesis Testing benchmark
#![allow(missing_docs)]

use std::time::Duration;

use bootstrap_ht::bootstrap::{non_parametric_ht, PValueType};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

fn non_parametric_ht_benchmark_f32(c: &mut Criterion) {
    const REPETITIONS: usize = 10_000;

    let mut group = c.benchmark_group("non_parametric_ht_benchmark_f32");
    for size in [10, 100, 200, 500, 1000] {
        // group.throughput(Elements(size as u64));
        let mu_a: f32 = 0.;
        let sigma_a = 1.0;

        let mu_b: f32 = 0.5;
        let sigma_b = 1.0;

        let mut rng = ChaCha8Rng::seed_from_u64(123);

        // sample a: vector of size `size` with mean `mu_a` and standard deviation `sigma_a`
        let normal_a = Normal::new(mu_a, sigma_a).unwrap();
        let s_a = normal_a
            .sample_iter(&mut rng)
            .take(size)
            .collect::<Vec<f32>>();

        // sample b: vector of size `size` with mean `mu_b` and standard deviation `sigma_b`
        let normal_b = Normal::new(mu_b, sigma_b).unwrap();
        let s_b = normal_b
            .sample_iter(&mut rng)
            .take(size)
            .collect::<Vec<f32>>();

        group.bench_with_input(BenchmarkId::from_parameter(size), &(s_a, s_b), |b, ab| {
            let (s_a, s_b) = ab;
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let test_statistic_fn = |a: &[f32], b: &[f32]| {
                let a_mean = a.iter().sum::<f32>() / a.len() as f32;
                let b_mean = b.iter().sum::<f32>() / b.len() as f32;
                (a_mean - b_mean).abs()
            };

            b.iter(|| {
                non_parametric_ht(
                    &mut rng,
                    s_a,
                    s_b,
                    test_statistic_fn,
                    PValueType::OneSidedRightTail,
                    REPETITIONS,
                )
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(20));
    targets = non_parametric_ht_benchmark_f32
}
criterion_main!(benches);
