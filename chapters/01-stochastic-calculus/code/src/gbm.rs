//! Geometric Brownian Motion (GBM)
//!
//! The most widely used model for asset prices in finance.
//! dS_t = μ S_t dt + σ S_t dW_t
//!
//! Solution: S_t = S_0 exp((μ - σ²/2)t + σW_t)

use crate::sde::SDE;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Geometric Brownian Motion model
///
/// Models asset prices with constant drift and volatility.
/// The price is always positive due to the exponential structure.
#[derive(Debug, Clone)]
pub struct GeometricBrownianMotion {
    /// Initial price S_0
    pub s0: f64,
    /// Drift (expected return) μ
    pub mu: f64,
    /// Volatility σ
    pub sigma: f64,
}

impl GeometricBrownianMotion {
    /// Creates a new GBM model
    ///
    /// # Arguments
    /// * `s0` - Initial price (must be positive)
    /// * `mu` - Expected return (drift)
    /// * `sigma` - Volatility (must be non-negative)
    ///
    /// # Panics
    /// Panics if s0 <= 0 or sigma < 0
    ///
    /// # Example
    /// ```
    /// use stochastic_calculus::GeometricBrownianMotion;
    ///
    /// let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
    /// ```
    pub fn new(s0: f64, mu: f64, sigma: f64) -> Self {
        assert!(s0 > 0.0, "Initial price must be positive, got {}", s0);
        assert!(sigma >= 0.0, "Volatility must be non-negative, got {}", sigma);
        Self { s0, mu, sigma }
    }

    /// Samples the price at time t using the exact analytical solution
    ///
    /// Uses: S_t = S_0 exp((μ - σ²/2)t + σ√t Z) where Z ~ N(0,1)
    ///
    /// This is more efficient than simulating the full path when only
    /// the terminal value is needed.
    pub fn sample_at_time<R: Rng>(&self, rng: &mut R, t: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.sample(rng);

        let drift = (self.mu - 0.5 * self.sigma * self.sigma) * t;
        let diffusion = self.sigma * t.sqrt() * z;

        self.s0 * (drift + diffusion).exp()
    }

    /// Generates a price path using the exact solution
    ///
    /// At each step, uses: log(S_{t+dt}) = log(S_t) + (μ - σ²/2)dt + σ√dt Z
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `n_steps` - Number of time steps
    /// * `dt` - Time step size
    ///
    /// # Returns
    /// Vector of prices S_t for t = 0, dt, 2dt, ..., n_steps*dt
    pub fn generate_path<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();
        let drift_per_step = (self.mu - 0.5 * self.sigma * self.sigma) * dt;

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.s0);

        let mut log_s = self.s0.ln();
        for _ in 0..n_steps {
            let z = normal.sample(rng);
            log_s += drift_per_step + self.sigma * sqrt_dt * z;
            path.push(log_s.exp());
        }

        path
    }

    /// Generates multiple paths in parallel
    ///
    /// # Arguments
    /// * `n_paths` - Number of paths to generate
    /// * `n_steps` - Number of time steps per path
    /// * `dt` - Time step size
    pub fn generate_paths_parallel(
        &self,
        n_paths: usize,
        n_steps: usize,
        dt: f64,
    ) -> Vec<Vec<f64>> {
        (0..n_paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                self.generate_path(&mut rng, n_steps, dt)
            })
            .collect()
    }

    /// Samples terminal values in parallel
    ///
    /// Efficient for Monte Carlo simulations where only S_T is needed.
    pub fn sample_terminal_parallel(&self, n_paths: usize, t: f64) -> Vec<f64> {
        (0..n_paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                self.sample_at_time(&mut rng, t)
            })
            .collect()
    }

    /// Expected value E[S_t] = S_0 * exp(μt)
    pub fn expected_value(&self, t: f64) -> f64 {
        self.s0 * (self.mu * t).exp()
    }

    /// Variance Var[S_t] = S_0² * exp(2μt) * (exp(σ²t) - 1)
    pub fn variance(&self, t: f64) -> f64 {
        let e_s = self.expected_value(t);
        e_s * e_s * ((self.sigma * self.sigma * t).exp() - 1.0)
    }

    /// Computes log-return: ln(S_t / S_0)
    pub fn log_return(s0: f64, st: f64) -> f64 {
        (st / s0).ln()
    }

    /// Computes simple return: (S_t - S_0) / S_0
    pub fn simple_return(s0: f64, st: f64) -> f64 {
        (st - s0) / s0
    }
}

impl SDE for GeometricBrownianMotion {
    type State = f64;

    fn drift(&self, _t: f64, x: &f64) -> f64 {
        self.mu * x
    }

    fn diffusion(&self, _t: f64, x: &f64) -> f64 {
        self.sigma * x
    }

    fn initial_state(&self) -> f64 {
        self.s0
    }
}

/// Statistics for GBM paths
pub struct GBMStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

impl GBMStatistics {
    /// Computes statistics from a vector of terminal values
    pub fn from_terminals(mut values: Vec<f64>) -> Self {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = values[0];
        let max = values[values.len() - 1];
        let median = values[values.len() / 2];

        Self { mean, std_dev, min, max, median }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbm_positive_prices() {
        let gbm = GeometricBrownianMotion::new(100.0, -0.5, 0.5);
        let mut rng = rand::thread_rng();

        let path = gbm.generate_path(&mut rng, 1000, 0.01);

        assert!(path.iter().all(|&p| p > 0.0), "All prices should be positive");
    }

    #[test]
    fn test_gbm_expected_value() {
        let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
        let t = 1.0;
        let n_simulations = 100_000;

        let terminals = gbm.sample_terminal_parallel(n_simulations, t);
        let mean = terminals.iter().sum::<f64>() / n_simulations as f64;

        let expected = gbm.expected_value(t);

        // Mean should be close to E[S_t] = S_0 * e^(μt)
        let relative_error = (mean - expected).abs() / expected;
        assert!(
            relative_error < 0.02,
            "Mean: {:.2}, Expected: {:.2}, Error: {:.2}%",
            mean,
            expected,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_gbm_log_returns_normal() {
        let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
        let mut rng = rand::thread_rng();

        let n_steps = 252;
        let dt = 1.0 / 252.0;
        let path = gbm.generate_path(&mut rng, n_steps, dt);

        // Compute log returns
        let log_returns: Vec<f64> = path
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        // Check mean of log returns ≈ (μ - σ²/2) * dt
        let expected_mean = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * dt;
        let actual_mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;

        let tolerance = 0.005;
        assert!(
            (actual_mean - expected_mean).abs() < tolerance,
            "Log return mean: {}, expected: {}",
            actual_mean,
            expected_mean
        );
    }

    #[test]
    fn test_path_length() {
        let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
        let mut rng = rand::thread_rng();

        let path = gbm.generate_path(&mut rng, 100, 0.01);
        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_parallel_generation() {
        let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
        let paths = gbm.generate_paths_parallel(1000, 50, 0.01);

        assert_eq!(paths.len(), 1000);
        assert!(paths.iter().all(|p| p.len() == 51));
    }

    #[test]
    #[should_panic(expected = "Initial price must be positive")]
    fn test_negative_s0_panics() {
        GeometricBrownianMotion::new(-100.0, 0.1, 0.2);
    }

    #[test]
    #[should_panic(expected = "Volatility must be non-negative")]
    fn test_negative_sigma_panics() {
        GeometricBrownianMotion::new(100.0, 0.1, -0.2);
    }
}
