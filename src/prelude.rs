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

//! Bootstrap Hypothesis Testing
//!
//! In cases, where we have no idea what the distribution of the test statistic is, we
//! still want to be able to perform hypothesis tests and we are willing to make the
//! hypothesis that the samples we have are representative of the population.
//!
//! This is where the bootstrap hypothesis testing comes in. The idea is to generate a
//! large number of samples from the null distribution (distribution the samples would
//! have if H0 is true - i.e. if both samples are from the same population) and then
//! compute the test statistic for each of these samples. This gives a test statistics
//! sampling distribution under H0.
//!
//! We can then compute the p-value by counting the number of times the sampled test
//! statistic is more 'extreme' than the test statistic for our initial samples.
//!
//! # References
//! - [Bootstrap Hypothesis Testing](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bootstrap_hypothesis_testing)
//! - [P-value](https://en.wikipedia.org/wiki/P-value)
//! - [Stats 102A Lesson 9-2 Bootstrap Hypothesis Tests, Miles Chen](https://www.youtube.com/watch?v=s7do_F9LV-w)
//!
//! # Example
//!
//! ```rust
//! use bootstrap_ht::prelude::bootstrap::{two_samples_non_parametric_ht, PValueType};
//! use itertools::Itertools;
//! use rand::prelude::Distribution;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//! use rand_distr::StandardNormal;
//!
//! let mut rng = &mut ChaCha8Rng::seed_from_u64(42);
//!
//! let a = StandardNormal
//!     .sample_iter(&mut rng)
//!     .take(100)
//!     .collect::<Vec<f64>>();
//! let b = StandardNormal
//!     .sample_iter(&mut rng)
//!     .take(100)
//!     .map(|x: f64| x + 0.7)
//!     .collect::<Vec<f64>>();
//!
//! // difference of the means
//! let test_statistic_fn = |a: &[f64], b: &[f64]| {
//!     let a_95percentile = a
//!         .iter()
//!         .sorted_by(|x, y| x.partial_cmp(y).unwrap())
//!         .nth(95)
//!         .unwrap();
//!     let b_95percentile = b
//!         .iter()
//!         .sorted_by(|x, y| x.partial_cmp(y).unwrap())
//!         .nth(95)
//!         .unwrap();
//!     (a_95percentile - b_95percentile).abs()
//! };
//!
//! let p_value = two_samples_non_parametric_ht(
//!     &mut rng,
//!     &a,
//!     &b,
//!     test_statistic_fn,
//!     PValueType::OneSidedRightTail,
//!     10_000,
//! )
//! .unwrap();
//! assert_eq!(p_value, 0.0236);
//! // p_value is small enough to reject the null hypothesis that the 95-percentiles
//! // are equal
//! ```

/// non-parametric bootstrap hypothesis test
pub mod bootstrap {
    pub use crate::bootstrap::{
        one_sample_non_parametric_ht, two_samples_non_parametric_ht, PValueType,
    };
}
