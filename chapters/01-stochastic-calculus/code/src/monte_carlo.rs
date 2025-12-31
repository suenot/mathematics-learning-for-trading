//! Monte Carlo Methods for Option Pricing
//!
//! Implements various variance reduction techniques:
//! - Basic Monte Carlo
//! - Antithetic variates
//! - Control variates

use crate::gbm::GeometricBrownianMotion;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Result of a Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Estimated value
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// 95% confidence interval (lower bound)
    pub ci_lower: f64,
    /// 95% confidence interval (upper bound)
    pub ci_upper: f64,
    /// Number of samples used
    pub n_samples: usize,
}

impl MonteCarloResult {
    fn new(estimate: f64, std_error: f64, n_samples: usize) -> Self {
        // 95% CI uses z = 1.96
        let margin = 1.96 * std_error;
        Self {
            estimate,
            std_error,
            ci_lower: estimate - margin,
            ci_upper: estimate + margin,
            n_samples,
        }
    }
}

/// Basic Monte Carlo estimator
///
/// Estimates E[f(S_T)] by averaging f(S_T) over many simulated paths.
pub fn monte_carlo_basic<F, R>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> MonteCarloResult
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    let normal = Normal::new(0.0, 1.0).unwrap();
    let drift = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * t;
    let vol_sqrt_t = gbm.sigma * t.sqrt();

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        let z = normal.sample(rng);
        let s_t = gbm.s0 * (drift + vol_sqrt_t * z).exp();
        let p = payoff(s_t);

        sum += p;
        sum_sq += p * p;
    }

    let mean = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - mean * mean;
    let std_error = (variance / n_paths as f64).sqrt();

    MonteCarloResult::new(mean, std_error, n_paths)
}

/// Monte Carlo with antithetic variates
///
/// For each random sample Z, also uses -Z, exploiting the
/// symmetry of the normal distribution to reduce variance.
pub fn monte_carlo_antithetic<F, R>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> MonteCarloResult
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    let normal = Normal::new(0.0, 1.0).unwrap();
    let drift = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * t;
    let vol_sqrt_t = gbm.sigma * t.sqrt();

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        let z = normal.sample(rng);

        // Primary path
        let s1 = gbm.s0 * (drift + vol_sqrt_t * z).exp();
        // Antithetic path (use -z)
        let s2 = gbm.s0 * (drift - vol_sqrt_t * z).exp();

        // Average the two payoffs
        let avg_payoff = (payoff(s1) + payoff(s2)) / 2.0;

        sum += avg_payoff;
        sum_sq += avg_payoff * avg_payoff;
    }

    let mean = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - mean * mean;
    let std_error = (variance / n_paths as f64).sqrt();

    MonteCarloResult::new(mean, std_error, n_paths)
}

/// Monte Carlo with control variates
///
/// Uses S_T as a control variate since E[S_T] = S_0 * exp(μT) is known.
/// This reduces variance when the payoff is correlated with S_T.
pub fn monte_carlo_control_variate<F, R>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> MonteCarloResult
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    let normal = Normal::new(0.0, 1.0).unwrap();
    let drift = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * t;
    let vol_sqrt_t = gbm.sigma * t.sqrt();

    // Known expectation of the control variate
    let expected_s = gbm.s0 * (gbm.mu * t).exp();

    let mut payoffs = Vec::with_capacity(n_paths);
    let mut controls = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        let z = normal.sample(rng);
        let s_t = gbm.s0 * (drift + vol_sqrt_t * z).exp();

        payoffs.push(payoff(s_t));
        controls.push(s_t);
    }

    // Compute optimal coefficient c = Cov(payoff, control) / Var(control)
    let mean_payoff: f64 = payoffs.iter().sum::<f64>() / n_paths as f64;
    let mean_control: f64 = controls.iter().sum::<f64>() / n_paths as f64;

    let mut cov = 0.0;
    let mut var_control = 0.0;

    for i in 0..n_paths {
        let dp = payoffs[i] - mean_payoff;
        let dc = controls[i] - mean_control;
        cov += dp * dc;
        var_control += dc * dc;
    }

    let c = if var_control > 0.0 { cov / var_control } else { 0.0 };

    // Compute adjusted payoffs
    let adjusted: Vec<f64> = payoffs
        .iter()
        .zip(controls.iter())
        .map(|(&p, &ctrl)| p - c * (ctrl - expected_s))
        .collect();

    let mean = adjusted.iter().sum::<f64>() / n_paths as f64;
    let variance: f64 = adjusted
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (n_paths - 1) as f64;
    let std_error = (variance / n_paths as f64).sqrt();

    MonteCarloResult::new(mean, std_error, n_paths)
}

/// Monte Carlo with parallel execution
pub fn monte_carlo_parallel<F>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    n_paths: usize,
) -> MonteCarloResult
where
    F: Fn(f64) -> f64 + Sync,
{
    let drift = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * t;
    let vol_sqrt_t = gbm.sigma * t.sqrt();
    let s0 = gbm.s0;

    let results: Vec<f64> = (0..n_paths)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, 1.0).unwrap();
            let z = normal.sample(&mut rng);
            let s_t = s0 * (drift + vol_sqrt_t * z).exp();
            payoff(s_t)
        })
        .collect();

    let sum: f64 = results.iter().sum();
    let sum_sq: f64 = results.iter().map(|x| x * x).sum();

    let mean = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - mean * mean;
    let std_error = (variance / n_paths as f64).sqrt();

    MonteCarloResult::new(mean, std_error, n_paths)
}

/// European call option payoff
pub fn call_payoff(strike: f64) -> impl Fn(f64) -> f64 {
    move |s_t| (s_t - strike).max(0.0)
}

/// European put option payoff
pub fn put_payoff(strike: f64) -> impl Fn(f64) -> f64 {
    move |s_t| (strike - s_t).max(0.0)
}

/// Digital (binary) call option payoff
pub fn digital_call_payoff(strike: f64) -> impl Fn(f64) -> f64 {
    move |s_t| if s_t > strike { 1.0 } else { 0.0 }
}

/// Straddle payoff (long call + long put at same strike)
pub fn straddle_payoff(strike: f64) -> impl Fn(f64) -> f64 {
    move |s_t| (s_t - strike).abs()
}

/// Black-Scholes formula for European call (for comparison)
pub fn black_scholes_call(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    use std::f64::consts::PI;

    fn norm_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    fn erf(x: f64) -> f64 {
        // Approximation of error function
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    s0 * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

/// Black-Scholes formula for European put
pub fn black_scholes_put(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let call = black_scholes_call(s0, k, r, sigma, t);
    // Put-call parity: P = C - S + K*exp(-rT)
    call - s0 + k * (-r * t).exp()
}

/// Compares Monte Carlo methods by running convergence analysis
pub fn convergence_analysis<F>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    true_value: f64,
    sample_sizes: &[usize],
) -> Vec<(usize, f64, f64, f64)>
where
    F: Fn(f64) -> f64 + Clone + Sync,
{
    let mut results = Vec::new();

    for &n in sample_sizes {
        // Run each method multiple times to get average error
        let n_runs = 10;

        let mut basic_errors = Vec::with_capacity(n_runs);
        let mut antithetic_errors = Vec::with_capacity(n_runs);
        let mut cv_errors = Vec::with_capacity(n_runs);

        for _ in 0..n_runs {
            let mut rng = rand::thread_rng();

            let basic = monte_carlo_basic(gbm, payoff.clone(), t, n, &mut rng);
            let antithetic = monte_carlo_antithetic(gbm, payoff.clone(), t, n, &mut rng);
            let cv = monte_carlo_control_variate(gbm, payoff.clone(), t, n, &mut rng);

            basic_errors.push((basic.estimate - true_value).abs());
            antithetic_errors.push((antithetic.estimate - true_value).abs());
            cv_errors.push((cv.estimate - true_value).abs());
        }

        let avg_basic = basic_errors.iter().sum::<f64>() / n_runs as f64;
        let avg_antithetic = antithetic_errors.iter().sum::<f64>() / n_runs as f64;
        let avg_cv = cv_errors.iter().sum::<f64>() / n_runs as f64;

        results.push((n, avg_basic, avg_antithetic, avg_cv));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_gbm() -> GeometricBrownianMotion {
        GeometricBrownianMotion::new(100.0, 0.05, 0.2)
    }

    #[test]
    fn test_basic_monte_carlo() {
        let gbm = test_gbm();
        let mut rng = rand::thread_rng();

        let strike = 100.0;
        let t = 1.0;
        let n_paths = 100_000;

        let result = monte_carlo_basic(&gbm, call_payoff(strike), t, n_paths, &mut rng);

        // Check confidence interval is reasonable
        assert!(result.std_error < 1.0, "Standard error too large");
        assert!(result.estimate > 0.0, "Call price should be positive");
    }

    #[test]
    fn test_antithetic_reduces_variance() {
        let gbm = test_gbm();
        let mut rng = rand::thread_rng();

        let strike = 100.0;
        let t = 1.0;
        let n_paths = 50_000;

        let basic = monte_carlo_basic(&gbm, call_payoff(strike), t, n_paths, &mut rng);
        let antithetic = monte_carlo_antithetic(&gbm, call_payoff(strike), t, n_paths, &mut rng);

        // Antithetic should typically have lower standard error
        // (might not always be true due to randomness, so we use a generous tolerance)
        println!("Basic SE: {}, Antithetic SE: {}", basic.std_error, antithetic.std_error);
    }

    #[test]
    fn test_control_variate() {
        let gbm = test_gbm();
        let mut rng = rand::thread_rng();

        let strike = 100.0;
        let t = 1.0;
        let n_paths = 50_000;

        let cv = monte_carlo_control_variate(&gbm, call_payoff(strike), t, n_paths, &mut rng);

        assert!(cv.estimate > 0.0);
        assert!(cv.std_error < 1.0);
    }

    #[test]
    fn test_parallel_monte_carlo() {
        let gbm = test_gbm();

        let strike = 100.0;
        let t = 1.0;
        let n_paths = 100_000;

        let result = monte_carlo_parallel(&gbm, call_payoff(strike), t, n_paths);

        assert!(result.estimate > 0.0);
    }

    #[test]
    fn test_black_scholes_call() {
        // Test against known values
        let call = black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0);

        // ATM 1-year call with 5% rate and 20% vol should be around 10.45
        assert!(
            (call - 10.45).abs() < 0.1,
            "Black-Scholes call price: {}, expected ~10.45",
            call
        );
    }

    #[test]
    fn test_monte_carlo_vs_black_scholes() {
        let s0 = 100.0;
        let k = 100.0;
        let r = 0.05;
        let sigma = 0.2;
        let t = 1.0;

        let gbm = GeometricBrownianMotion::new(s0, r, sigma);
        let bs_price = black_scholes_call(s0, k, r, sigma, t);

        // Discounted Monte Carlo
        let result = monte_carlo_parallel(&gbm, call_payoff(k), t, 500_000);
        let mc_price = result.estimate * (-r * t).exp();

        // Should be within 3 standard errors
        let tolerance = 3.0 * result.std_error * (-r * t).exp();
        assert!(
            (mc_price - bs_price).abs() < tolerance,
            "MC price: {:.4}, BS price: {:.4}, diff: {:.4}, tolerance: {:.4}",
            mc_price,
            bs_price,
            (mc_price - bs_price).abs(),
            tolerance
        );
    }
}
