
# Bootstrap Hypothesis Testing in Rust


![Build](https://github.com/ssoudan/bootstrap-ht/actions/workflows/rust.yml/badge.svg)
![crates.io](https://img.shields.io/crates/v/bootstrap-ht.svg)

# Description

 Sometime, we have no idea what the distribution of the test statistic is, we
 really want to be able to perform hypothesis tests, and we are willing to make the
 hypothesis that the samples we have are representative of the population.

 This is where the bootstrap hypothesis testing comes in. The idea is to generate a
 large number of samples from the null distribution (distribution the samples would
 have if H0 is true - i.e. if both samples are from the same population) and then
 compute the test statistic for each of these samples. This gives a test statistics
 sampling distribution under H0.

 We can then compute the p-value by counting the number of times the sampled test
 statistic is more 'extreme' than the test statistic for our initial samples.

 # References
 - [Bootstrap Hypothesis Testing](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bootstrap_hypothesis_testing)
 - [P-value](https://en.wikipedia.org/wiki/P-value)
 - [Stats 102A Lesson 9-2 Bootstrap Hypothesis Tests, Miles Chen](https://www.youtube.com/watch?v=s7do_F9LV-w)

# Usage 

```rust 
use bootstrap_ht::prelude::*;
use itertools::Itertools;
use rand::prelude::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

fn main() {
 let mut rng = ChaCha8Rng::seed_from_u64(42);

 let a = StandardNormal
         .sample_iter(&mut rng)
         .take(100)
         .collect::<Vec<f64>>();
 let b = StandardNormal
         .sample_iter(&mut rng)
         .take(40)
         .map(|x: f64| x + 2.0)
         .collect::<Vec<f64>>();

 let test_statistic_fn = |a: &[f64], b: &[f64]| {
  let a_max = a.iter().copied().fold(f64::NAN, f64::max);
  let b_max = b.iter().copied().fold(f64::NAN, f64::max);
  (a_max - b_max).abs()
 };

 let p_value = bootstrap::two_samples_non_parametric_ht(
  &mut rng,
  &a,
  &b,
  test_statistic_fn,
  bootstrap::PValueType::OneSidedRightTail,
  10_000,
 )
         .unwrap();
 assert_eq!(p_value, 0.0021);
}
```