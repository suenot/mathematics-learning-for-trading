//! Heston Stochastic Volatility Model
//!
//! The Heston model allows volatility to be stochastic:
//!
//! dS_t = μ S_t dt + √V_t S_t dW^S_t
//! dV_t = κ(θ - V_t) dt + ξ √V_t dW^V_t
//!
//! where Corr(dW^S, dW^V) = ρ

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Heston stochastic volatility model
#[derive(Debug, Clone)]
pub struct HestonModel {
    /// Initial price S_0
    pub s0: f64,
    /// Initial variance V_0
    pub v0: f64,
    /// Drift (expected return)
    pub mu: f64,
    /// Mean reversion speed (κ)
    pub kappa: f64,
    /// Long-term variance level (θ)
    pub theta: f64,
    /// Volatility of volatility (ξ)
    pub xi: f64,
    /// Correlation between price and volatility (ρ)
    pub rho: f64,
}

/// Result of Heston simulation
pub struct HestonPath {
    /// Price path
    pub prices: Vec<f64>,
    /// Variance path
    pub variances: Vec<f64>,
}

impl HestonModel {
    /// Creates a new Heston model
    ///
    /// # Arguments
    /// * `s0` - Initial price (must be positive)
    /// * `v0` - Initial variance (must be positive)
    /// * `mu` - Expected return
    /// * `kappa` - Mean reversion speed (must be positive)
    /// * `theta` - Long-term variance (must be positive)
    /// * `xi` - Volatility of volatility (must be positive)
    /// * `rho` - Correlation in [-1, 1]
    ///
    /// # Panics
    /// Panics if parameters are out of valid ranges
    pub fn new(
        s0: f64,
        v0: f64,
        mu: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
    ) -> Self {
        assert!(s0 > 0.0, "Initial price must be positive");
        assert!(v0 > 0.0, "Initial variance must be positive");
        assert!(kappa > 0.0, "Mean reversion speed must be positive");
        assert!(theta > 0.0, "Long-term variance must be positive");
        assert!(xi > 0.0, "Vol of vol must be positive");
        assert!(rho.abs() <= 1.0, "Correlation must be in [-1, 1]");

        Self { s0, v0, mu, kappa, theta, xi, rho }
    }

    /// Checks if the Feller condition is satisfied
    ///
    /// The Feller condition 2κθ > ξ² ensures the variance process
    /// stays positive (doesn't hit zero).
    pub fn feller_condition_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta > self.xi * self.xi
    }

    /// Simulates using Euler-Maruyama scheme with truncation
    ///
    /// Truncates negative variances to zero (full truncation scheme).
    pub fn simulate_euler<R: Rng>(
        &self,
        rng: &mut R,
        n_steps: usize,
        dt: f64,
    ) -> HestonPath {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();

        let mut prices = Vec::with_capacity(n_steps + 1);
        let mut variances = Vec::with_capacity(n_steps + 1);

        prices.push(self.s0);
        variances.push(self.v0);

        let mut s = self.s0;
        let mut v = self.v0;

        let sqrt_one_minus_rho_sq = (1.0 - self.rho * self.rho).sqrt();

        for _ in 0..n_steps {
            // Generate correlated Brownian increments
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);

            let dw_s = sqrt_dt * z1;
            let dw_v = sqrt_dt * (self.rho * z1 + sqrt_one_minus_rho_sq * z2);

            // Update variance with truncation at zero
            let sqrt_v = v.max(0.0).sqrt();
            let dv = self.kappa * (self.theta - v) * dt + self.xi * sqrt_v * dw_v;
            v = (v + dv).max(0.0);

            // Update price
            let ds = self.mu * s * dt + sqrt_v * s * dw_s;
            s += ds;
            s = s.max(1e-10); // Prevent zero/negative prices

            prices.push(s);
            variances.push(v);
        }

        HestonPath { prices, variances }
    }

    /// Simulates using Milstein scheme for higher accuracy
    ///
    /// Adds Milstein correction terms for both price and variance processes.
    pub fn simulate_milstein<R: Rng>(
        &self,
        rng: &mut R,
        n_steps: usize,
        dt: f64,
    ) -> HestonPath {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();

        let mut prices = Vec::with_capacity(n_steps + 1);
        let mut variances = Vec::with_capacity(n_steps + 1);

        prices.push(self.s0);
        variances.push(self.v0);

        let mut s = self.s0;
        let mut v = self.v0;

        let sqrt_one_minus_rho_sq = (1.0 - self.rho * self.rho).sqrt();

        for _ in 0..n_steps {
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);

            let dw_s = sqrt_dt * z1;
            let dw_v = sqrt_dt * (self.rho * z1 + sqrt_one_minus_rho_sq * z2);

            let sqrt_v = v.max(0.0).sqrt();

            // Milstein correction for variance: d(√V)/dV = 1/(2√V)
            // Correction term: 0.25 * ξ² * (dW² - dt)
            let dv = self.kappa * (self.theta - v) * dt
                   + self.xi * sqrt_v * dw_v
                   + 0.25 * self.xi * self.xi * (dw_v * dw_v - dt);
            v = (v + dv).max(0.0);

            // Milstein correction for price
            let ds = self.mu * s * dt
                   + sqrt_v * s * dw_s
                   + 0.5 * v * s * (dw_s * dw_s - dt);
            s += ds;
            s = s.max(1e-10);

            prices.push(s);
            variances.push(v);
        }

        HestonPath { prices, variances }
    }

    /// Simulates using QE (Quadratic Exponential) scheme
    ///
    /// The QE scheme by Andersen (2008) provides better accuracy
    /// for the variance process, especially when Feller condition is violated.
    pub fn simulate_qe<R: Rng>(
        &self,
        rng: &mut R,
        n_steps: usize,
        dt: f64,
    ) -> HestonPath {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = rand_distr::Uniform::new(0.0f64, 1.0);

        let mut prices = Vec::with_capacity(n_steps + 1);
        let mut variances = Vec::with_capacity(n_steps + 1);

        prices.push(self.s0);
        variances.push(self.v0);

        let mut log_s = self.s0.ln();
        let mut v = self.v0;

        // Precompute constants
        let exp_kappa_dt = (-self.kappa * dt).exp();
        let psi_c = 1.5; // Critical value for switching between schemes

        for _ in 0..n_steps {
            // === Variance update using QE scheme ===
            let m = self.theta + (v - self.theta) * exp_kappa_dt;
            let s2 = v * self.xi * self.xi * exp_kappa_dt * (1.0 - exp_kappa_dt) / self.kappa
                   + self.theta * self.xi * self.xi * (1.0 - exp_kappa_dt).powi(2) / (2.0 * self.kappa);
            let psi = s2 / (m * m);

            let v_new = if psi <= psi_c {
                // Use quadratic scheme
                let b2 = 2.0 / psi - 1.0 + (2.0 / psi).sqrt() * (2.0 / psi - 1.0).sqrt();
                let a = m / (1.0 + b2);
                let z = normal.sample(rng);
                a * (b2.sqrt() + z).powi(2)
            } else {
                // Use exponential scheme
                let p = (psi - 1.0) / (psi + 1.0);
                let beta = (1.0 - p) / m;
                let u: f64 = uniform.sample(rng);
                if u <= p {
                    0.0
                } else {
                    (1.0 / beta) * ((1.0 - p) / (1.0 - u)).ln()
                }
            };

            // === Price update ===
            let k0 = -self.rho * self.kappa * self.theta * dt / self.xi;
            let k1 = 0.5 * dt * (self.kappa * self.rho / self.xi - 0.5) - self.rho / self.xi;
            let k2 = 0.5 * dt * (self.kappa * self.rho / self.xi - 0.5) + self.rho / self.xi;
            let k3 = 0.5 * dt * (1.0 - self.rho * self.rho);
            let k4 = 0.5 * dt * (1.0 - self.rho * self.rho);

            let z = normal.sample(rng);
            log_s += self.mu * dt + k0 + k1 * v + k2 * v_new
                   + (k3 * v + k4 * v_new).sqrt() * z;

            v = v_new;

            prices.push(log_s.exp());
            variances.push(v);
        }

        HestonPath { prices, variances }
    }

    /// Generates multiple paths in parallel
    pub fn simulate_parallel<R: Rng + Send + Sync + Clone>(
        &self,
        n_paths: usize,
        n_steps: usize,
        dt: f64,
    ) -> Vec<HestonPath> {
        (0..n_paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                self.simulate_euler(&mut rng, n_steps, dt)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> HestonModel {
        HestonModel::new(
            100.0,  // s0
            0.04,   // v0 (20% volatility squared)
            0.05,   // mu
            2.0,    // kappa
            0.04,   // theta
            0.3,    // xi
            -0.7,   // rho (negative correlation)
        )
    }

    #[test]
    fn test_feller_condition() {
        let model = test_model();
        // 2 * 2.0 * 0.04 = 0.16, 0.3² = 0.09
        // 0.16 > 0.09, so Feller condition is satisfied
        assert!(model.feller_condition_satisfied());

        let bad_model = HestonModel::new(100.0, 0.04, 0.05, 0.5, 0.01, 0.5, -0.7);
        // 2 * 0.5 * 0.01 = 0.01, 0.5² = 0.25
        // 0.01 < 0.25, Feller condition NOT satisfied
        assert!(!bad_model.feller_condition_satisfied());
    }

    #[test]
    fn test_euler_positive_prices() {
        let model = test_model();
        let mut rng = rand::thread_rng();

        let path = model.simulate_euler(&mut rng, 1000, 0.01);

        assert!(path.prices.iter().all(|&p| p > 0.0));
        assert!(path.variances.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_milstein_positive_prices() {
        let model = test_model();
        let mut rng = rand::thread_rng();

        let path = model.simulate_milstein(&mut rng, 1000, 0.01);

        assert!(path.prices.iter().all(|&p| p > 0.0));
        assert!(path.variances.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_qe_positive_prices() {
        let model = test_model();
        let mut rng = rand::thread_rng();

        let path = model.simulate_qe(&mut rng, 1000, 0.01);

        assert!(path.prices.iter().all(|&p| p > 0.0));
        assert!(path.variances.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_variance_mean_reversion() {
        let model = test_model();

        // Average variance over many simulations should be close to theta
        let n_sims = 1000;
        let n_steps = 500;
        let dt = 0.01;

        let sum_final_var: f64 = (0..n_sims)
            .map(|_| {
                let mut rng = rand::thread_rng();
                let path = model.simulate_euler(&mut rng, n_steps, dt);
                *path.variances.last().unwrap()
            })
            .sum();

        let mean_var = sum_final_var / n_sims as f64;

        // Mean variance should be close to theta
        assert!(
            (mean_var - model.theta).abs() < 0.02,
            "Mean variance: {}, theta: {}",
            mean_var,
            model.theta
        );
    }

    #[test]
    fn test_path_lengths() {
        let model = test_model();
        let mut rng = rand::thread_rng();

        let path = model.simulate_euler(&mut rng, 100, 0.01);

        assert_eq!(path.prices.len(), 101);
        assert_eq!(path.variances.len(), 101);
    }

    #[test]
    fn test_negative_correlation() {
        let model = test_model(); // rho = -0.7
        let n_sims = 1000;
        let n_steps = 100;
        let dt = 0.01;

        // When price goes down significantly, variance should tend to go up
        let mut correlation_sum = 0.0;

        for _ in 0..n_sims {
            let mut rng = rand::thread_rng();
            let path = model.simulate_euler(&mut rng, n_steps, dt);

            // Compute correlation between log returns and variance changes
            let log_returns: Vec<f64> = path.prices.windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect();

            let var_changes: Vec<f64> = path.variances.windows(2)
                .map(|w| w[1] - w[0])
                .collect();

            let mean_r = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
            let mean_v = var_changes.iter().sum::<f64>() / var_changes.len() as f64;

            let mut cov = 0.0;
            let mut var_r = 0.0;
            let mut var_v = 0.0;

            for i in 0..log_returns.len() {
                let dr = log_returns[i] - mean_r;
                let dv = var_changes[i] - mean_v;
                cov += dr * dv;
                var_r += dr * dr;
                var_v += dv * dv;
            }

            if var_r > 0.0 && var_v > 0.0 {
                correlation_sum += cov / (var_r.sqrt() * var_v.sqrt());
            }
        }

        let avg_correlation = correlation_sum / n_sims as f64;

        // Average correlation should be negative (like rho)
        assert!(
            avg_correlation < 0.0,
            "Expected negative correlation, got {}",
            avg_correlation
        );
    }
}
