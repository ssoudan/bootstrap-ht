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

use std::fmt::Debug;

use num_traits::Float;
use rand::prelude::SliceRandom;
use rand::Rng;

use crate::Error;

// FUTURE(ssoudan) parametric bootstrap

// FUTURE(ssoudan) randomization test

/// Part of the statistic distribution to use for the p-value
/// https://en.wikipedia.org/wiki/P-value#Probability_of_obtaining_a_real-valued_test_statistic_at_least_as_extreme_as_the_one_actually_obtained
pub enum PValueType {
    /// Two-sided test - symmetric with respect to the mean of the statistic distribution
    /// 2 * min (Pr(T >= t | H0), Pr(T <= t | H0))
    TwoSided,
    /// One-sided test (right tail)
    /// Pr(T >= t | H0)
    OneSidedRightTail,
    /// One-sided test (left tail)
    /// Pr(T <= t | H0)
    OneSidedLeftTail,
}

/// Two-samples bootstrap test.
///
/// Perform a non-parametric bootstrap hypothesis test on samples `a` and `b` with the
/// given `test_statistic_fn` function.
///
/// # Description
///
/// The null hypothesis is that the two samples are drawn from the same distribution.
/// The p-value is the probability of obtaining a test statistic at least as extreme as
/// the one actually obtained (extreme being defined by `pvalue_type` - see
/// [`PValueType`]).
///
/// The test statistic is computed on the two samples and the p-value is computed by
/// comparing the test statistic to the distribution of the test statistic.
///
/// The test statistic distribution is obtained by randomly sampling with replacement from
/// the two samples `rep` times - seems that 10_000 is the norm.
///
/// Note `a` and `b` need not be of the same size.
///
/// # Example
///
/// Let's use the absolute difference of the max as the test statistic: `|max(a) -
/// max(b)|`.
///
/// ```rust
/// use bootstrap_ht::prelude::*;
/// use itertools::Itertools;
/// use rand::prelude::Distribution;
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use rand_distr::StandardNormal;
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
///
/// let a = StandardNormal
///     .sample_iter(&mut rng)
///     .take(100)
///     .collect::<Vec<f64>>();
/// let b = StandardNormal
///     .sample_iter(&mut rng)
///     .take(40)
///     .map(|x: f64| x + 2.0)
///     .collect::<Vec<f64>>();
///
/// /// absolute difference of the max
/// let test_statistic_fn = |a: &[f64], b: &[f64]| {
///     let a_max = a.iter().copied().fold(f64::NAN, f64::max);
///     let b_max = b.iter().copied().fold(f64::NAN, f64::max);
///     (a_max - b_max).abs()
/// };
///
/// let p_value = bootstrap::two_samples_non_parametric_ht(
///     &mut rng,
///     &a,
///     &b,
///     test_statistic_fn,
///     bootstrap::PValueType::OneSidedRightTail,
///     10_000,
/// )
/// .unwrap();
/// assert_eq!(p_value, 0.0021);
/// ```
pub fn two_samples_non_parametric_ht<
    R: Rng + ?Sized,
    F: Float + std::iter::Sum,
    S: Clone + Default,
>(
    mut rng: &mut R,
    a: &[S],
    b: &[S],
    test_statistic_fn: impl Fn(&[S], &[S]) -> F,
    pvalue_type: PValueType,
    rep: usize,
) -> Result<F, Error> {
    let n_a = a.len();
    if n_a == 0 {
        return Err(Error::NotEnoughSamples);
    }
    let n_b = b.len();
    if n_b == 0 {
        return Err(Error::NotEnoughSamples);
    }

    // the test statistic for the observed data
    let test_stat = test_statistic_fn(a, b);

    // the test statistic distribution of the population under the null hypothesis (a and b
    // are drawn from the same population).
    let mut test_stat_dist = vec![F::from(0.).unwrap(); rep];

    let reference = [a, b].concat();

    let mut a_ = vec![S::default(); n_a];
    let mut b_ = vec![S::default(); n_b];

    for test_stat_dist_ in test_stat_dist.iter_mut() {
        for a__ in a_.iter_mut() {
            *a__ = reference.choose(&mut rng).unwrap().clone();
        }

        for b__ in b_.iter_mut() {
            *b__ = reference.choose(&mut rng).unwrap().clone();
        }

        let test_stat_ = test_statistic_fn(&a_, &b_);
        *test_stat_dist_ = test_stat_;
    }

    // the p-value
    let p_value = compute_p_value(pvalue_type, test_stat, test_stat_dist);

    Ok(p_value)
}

/// One-sample bootstrap test.
///
/// Perform a non-parametric one-sample bootstrap hypothesis test on the population
/// generated from sample `a` with the given `test_statistic_fn` function and reference
/// statistic `mu`.
///
/// # Description
///
/// The null hypothesis is that the test statistic of the population generated by the
/// sample `a` is `test_stat`. The p-value is the probability of obtaining a test
/// statistic at least as extreme as the one provided by `mu` (extreme being defined by
/// `pvalue_type` - see [`PValueType`]).
///
/// The p-value is computed by comparing `test_stat` to the distribution of the
/// test statistic.
///
/// The test statistic distribution is obtained by randomly sampling with replacement from
/// the samples `rep` times - seems that 10_000 is the norm.
///
///
/// # Example
///
/// Let's use the mean as the test statistic: `mean(a)`.
/// ```rust
/// use bootstrap_ht::prelude::*;
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
///
/// // From p.224 - 'An introduction to the boostrap', Efron and Tibshirani
///
/// let a = [94., 197., 16., 38., 99., 141., 23.];
/// let a_mean = a.iter().sum::<f32>() / a.len() as f32;
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
///
/// let test_statistic_fn = |a: &[f32]| {
///     let a_mean = a.iter().sum::<f32>() / a.len() as f32;
///     let a_std =
///         (a.iter().map(|x| (x - a_mean).powi(2)).sum::<f32>() / (a.len() - 1) as f32).sqrt();
///
///     (a_mean - 129.) / (a_std / (a.len() as f32).sqrt())
/// };
///
/// let test_stat: f32 = test_statistic_fn(&a);
///
/// /// Got to shift the samples so their mean is the one assumed under H0
/// let a = a
///     .into_iter()
///     .map(|x| x - a_mean + 129.)
///     .collect::<Vec<f32>>();
///
/// let p_value = bootstrap::one_sample_non_parametric_ht(
///     &mut rng,
///     &a,
///     test_stat,
///     test_statistic_fn,
///     bootstrap::PValueType::OneSidedLeftTail,
///     1_000,
/// )
/// .unwrap();
/// assert_eq!(p_value, 0.092);
/// ```
pub fn one_sample_non_parametric_ht<
    R: Rng + ?Sized,
    F: Float + std::iter::Sum + Debug,
    S: Clone + Default + Debug,
>(
    mut rng: &mut R,
    a: &[S],
    test_stat: F,
    test_statistic_fn: impl Fn(&[S]) -> F,
    pvalue_type: PValueType,
    rep: usize,
) -> Result<F, Error> {
    let n_a = a.len();
    if n_a == 0 {
        return Err(Error::NotEnoughSamples);
    }

    // the test statistic distribution of the population under the null hypothesis.
    let mut test_stat_dist = vec![F::from(0.).unwrap(); rep];

    let mut a_ = vec![S::default(); n_a];
    for test_stat_dist_ in test_stat_dist.iter_mut() {
        for a__ in a_.iter_mut() {
            *a__ = a.choose(&mut rng).unwrap().clone();
        }

        let test_stat_ = test_statistic_fn(&a_);
        *test_stat_dist_ = test_stat_;
    }

    // the p-value
    let p_value = compute_p_value(pvalue_type, test_stat, test_stat_dist);

    Ok(p_value)
}

fn compute_p_value<F: Float + std::iter::Sum>(
    pvalue_type: PValueType,
    test_stat: F,
    test_stat_dist: Vec<F>,
) -> F {
    let n = F::from(test_stat_dist.len()).unwrap();

    match pvalue_type {
        PValueType::TwoSided => {
            let right_p_value =
                F::from(test_stat_dist.iter().filter(|t| **t >= test_stat).count()).unwrap() / n;

            let left_p_value =
                F::from(test_stat_dist.iter().filter(|t| **t <= test_stat).count()).unwrap() / n;

            let min = if right_p_value <= left_p_value {
                right_p_value
            } else {
                left_p_value
            };

            F::from(2.).unwrap() * min
        }
        PValueType::OneSidedRightTail => {
            F::from(test_stat_dist.iter().filter(|&t| t >= &test_stat).count()).unwrap() / n
        }
        PValueType::OneSidedLeftTail => {
            F::from(test_stat_dist.iter().filter(|&t| t <= &test_stat).count()).unwrap() / n
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    use super::*;

    #[test]
    fn test_two_samples_ht() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // absolute difference of the means
        let test_statistic_fn = |a: &[f64], b: &[f64]| {
            let a_mean = a.iter().sum::<f64>() / a.len() as f64;
            let b_mean = b.iter().sum::<f64>() / b.len() as f64;
            (a_mean - b_mean).abs()
        };

        let p_value = two_samples_non_parametric_ht(
            &mut rng,
            &a,
            &b,
            test_statistic_fn,
            PValueType::OneSidedRightTail,
            10_000,
        )
        .unwrap();
        assert_eq!(p_value, 0.0345);
        // p_value is lower than 0.05, so we reject the null hypothesis that the means are
        // equal
    }

    #[test]
    fn test_two_samples_normal_distributions_ht() {
        let mut rng = &mut ChaCha8Rng::seed_from_u64(42);
        let a = StandardNormal
            .sample_iter(&mut rng)
            .take(100)
            .collect::<Vec<f64>>();
        let b = StandardNormal
            .sample_iter(&mut rng)
            .take(40)
            .collect::<Vec<f64>>();

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // absolute difference of the means
        let test_statistic_fn = |a: &[f64], b: &[f64]| {
            let a_mean = a.iter().sum::<f64>() / a.len() as f64;
            let b_mean = b.iter().sum::<f64>() / b.len() as f64;
            (a_mean - b_mean).abs()
        };

        let p_value = two_samples_non_parametric_ht(
            &mut rng,
            &a,
            &b,
            test_statistic_fn,
            PValueType::OneSidedRightTail,
            10_000,
        )
        .unwrap();
        assert_eq!(p_value, 0.8298);
        // p_value is large enough (say, compared to 0.05) not to reject the null
        // hypothesis that the means are equal
    }

    #[test]
    fn test_two_normal_distributions_two_sided_ht() {
        let mut rng = &mut ChaCha8Rng::seed_from_u64(42);
        let a = StandardNormal
            .sample_iter(&mut rng)
            .take(100)
            .collect::<Vec<f64>>();
        let b = StandardNormal
            .sample_iter(&mut rng)
            .take(40)
            .collect::<Vec<f64>>();

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // difference of the means
        let test_statistic_fn = |a: &[f64], b: &[f64]| {
            let a_mean = a.iter().sum::<f64>() / a.len() as f64;
            let b_mean = b.iter().sum::<f64>() / b.len() as f64;
            a_mean - b_mean
        };

        let p_value = two_samples_non_parametric_ht(
            &mut rng,
            &a,
            &b,
            test_statistic_fn,
            PValueType::TwoSided,
            10_000,
        )
        .unwrap();
        assert_eq!(p_value, 0.8222);
        // p_value is large enough not to reject the null hypothesis that the means are
        // equal
    }

    #[test]
    fn test_two_different_normal_distributions_two_sided_ht() {
        let mut rng = &mut ChaCha8Rng::seed_from_u64(42);
        let a = StandardNormal
            .sample_iter(&mut rng)
            .take(100)
            .collect::<Vec<f64>>();
        let b = StandardNormal
            .sample_iter(&mut rng)
            .take(100)
            .map(|x: f64| x + 0.3)
            .collect::<Vec<f64>>();

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // difference of the means
        fn test_statistic_fn(a: &[f64], b: &[f64]) -> f64 {
            let a_mean = a.iter().sum::<f64>() / a.len() as f64;
            let b_mean = b.iter().sum::<f64>() / b.len() as f64;
            a_mean - b_mean
        }

        let p_value = two_samples_non_parametric_ht(
            &mut rng,
            &a,
            &b,
            test_statistic_fn,
            PValueType::TwoSided,
            10_000,
        )
        .unwrap();
        assert_eq!(p_value, 0.0418);
        // p_value is small enough to reject the null hypothesis that the means are equal
    }

    #[test]
    fn test_two_different_normal_distributions_one_sided_95percentile_ht() {
        let mut rng = &mut ChaCha8Rng::seed_from_u64(42);
        let a = StandardNormal
            .sample_iter(&mut rng)
            .take(100)
            .collect::<Vec<f64>>();
        let b = StandardNormal
            .sample_iter(&mut rng)
            .take(100)
            .map(|x: f64| x + 0.7)
            .collect::<Vec<f64>>();

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // difference of the means
        let test_statistic_fn = |a: &[f64], b: &[f64]| {
            let a_95percentile = a
                .iter()
                .sorted_by(|x, y| x.partial_cmp(y).unwrap())
                .nth(95 * a.len() / 100)
                .unwrap();
            let b_95percentile = b
                .iter()
                .sorted_by(|x, y| x.partial_cmp(y).unwrap())
                .nth(95 * b.len() / 100)
                .unwrap();
            (a_95percentile - b_95percentile).abs()
        };

        let p_value = two_samples_non_parametric_ht(
            &mut rng,
            &a,
            &b,
            test_statistic_fn,
            PValueType::OneSidedRightTail,
            10_000,
        )
        .unwrap();
        assert_eq!(p_value, 0.0218);
        // p_value is small enough to reject the null hypothesis that the 95-percentiles
        // are equal
    }

    #[test]
    fn test_one_sample_std_normal_mean_ht() {
        let mut rng = &mut ChaCha8Rng::seed_from_u64(42);
        let a = StandardNormal
            .sample_iter(&mut rng)
            .take(100)
            .collect::<Vec<f64>>();

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // difference of the means
        let test_statistic_fn = |a: &[f64]| {
            let a_mean = a.iter().sum::<f64>() / a.len() as f64;
            a_mean
        };

        let test_stat = test_statistic_fn(&a);

        let p_value = one_sample_non_parametric_ht(
            &mut rng,
            &a,
            test_stat,
            test_statistic_fn,
            PValueType::TwoSided,
            10_000,
        )
        .unwrap();
        assert_eq!(p_value, 0.999);
        // let's not reject H0
    }

    #[test]
    fn test_one_sample_mean_ht() {
        // From p.224 - 'An introduction to the boostrap', Efron and Tibshirani

        // H0: the mean of the population is 129.

        let a = [94., 197., 16., 38., 99., 141., 23.];
        let a_mean = a.iter().sum::<f32>() / a.len() as f32;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // 'studentized' distance to the H0 mean statistic
        let test_statistic_fn = |a: &[f32]| {
            let a_mean = a.iter().sum::<f32>() / a.len() as f32;
            let a_std =
                (a.iter().map(|x| (x - a_mean).powi(2)).sum::<f32>() / (a.len() - 1) as f32).sqrt();

            (a_mean - 129.) / (a_std / (a.len() as f32).sqrt())
        };

        let test_stat: f32 = test_statistic_fn(&a);

        // Got to shift the samples so their mean is the one assumed under H0 - namely 129.
        let a = a
            .into_iter()
            .map(|x| x - a_mean + 129.)
            .collect::<Vec<f32>>();

        // Alternatively, we could have not shifted the samples, and used a test statistic
        // measuring the distance to the empirical mean of the original sample (86.9) while
        // keeping 129 when calculating the reference test statistic.
        //
        // In the end, what we want to estimate is how 'surprising' it is for the original sample
        // to be this far from a hypothetical population mean by comparing the distribution of
        // distance to the mean for a collection of samples generated randomly from the
        // same 'data'.

        let p_value = one_sample_non_parametric_ht(
            &mut rng,
            &a,
            test_stat,
            test_statistic_fn,
            PValueType::OneSidedLeftTail,
            1_000,
        )
        .unwrap();
        assert_eq!(p_value, 0.092);
    }
}
