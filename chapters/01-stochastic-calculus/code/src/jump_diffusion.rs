//! Merton Jump-Diffusion Model
//!
//! Extends GBM with random jumps:
//! dS_t = (μ - λκ) S_t dt + σ S_t dW_t + S_t dJ_t
//!
//! where J_t is a compound Poisson process

use rand::Rng;
use rand_distr::{Distribution, Normal, Poisson};
use rayon::prelude::*;

/// Merton Jump-Diffusion Model
///
/// Models asset prices with both continuous diffusion and discrete jumps.
/// Jumps can represent sudden market movements like earnings announcements,
/// economic news, or market crashes.
#[derive(Debug, Clone)]
pub struct MertonJumpDiffusion {
    /// Initial price S_0
    pub s0: f64,
    /// Drift (expected return before jump adjustment)
    pub mu: f64,
    /// Diffusion volatility
    pub sigma: f64,
    /// Jump intensity (expected number of jumps per year)
    pub lambda: f64,
    /// Mean of log jump size (μ_J)
    pub mu_j: f64,
    /// Standard deviation of log jump size (σ_J)
    pub sigma_j: f64,
}

impl MertonJumpDiffusion {
    /// Creates a new Merton jump-diffusion model
    ///
    /// # Arguments
    /// * `s0` - Initial price
    /// * `mu` - Expected return
    /// * `sigma` - Diffusion volatility
    /// * `lambda` - Jump intensity (jumps per year)
    /// * `mu_j` - Mean of log jump size
    /// * `sigma_j` - Std dev of log jump size
    ///
    /// # Example
    /// ```
    /// use stochastic_calculus::MertonJumpDiffusion;
    ///
    /// // Model with 2 jumps per year, average jump of -5%, jump vol of 10%
    /// let model = MertonJumpDiffusion::new(100.0, 0.1, 0.2, 2.0, -0.05, 0.1);
    /// ```
    pub fn new(
        s0: f64,
        mu: f64,
        sigma: f64,
        lambda: f64,
        mu_j: f64,
        sigma_j: f64,
    ) -> Self {
        assert!(s0 > 0.0, "Initial price must be positive");
        assert!(sigma >= 0.0, "Volatility must be non-negative");
        assert!(lambda >= 0.0, "Jump intensity must be non-negative");
        assert!(sigma_j >= 0.0, "Jump volatility must be non-negative");

        Self { s0, mu, sigma, lambda, mu_j, sigma_j }
    }

    /// Expected jump multiplier: E[Y - 1] where Y = exp(μ_J + σ_J * Z)
    pub fn kappa(&self) -> f64 {
        (self.mu_j + 0.5 * self.sigma_j * self.sigma_j).exp() - 1.0
    }

    /// Adjusted drift: μ - λκ (to ensure E[S_t] = S_0 * e^(μt))
    pub fn adjusted_drift(&self) -> f64 {
        self.mu - self.lambda * self.kappa()
    }

    /// Simulates a single path
    pub fn simulate<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let normal_j = Normal::new(self.mu_j, self.sigma_j).unwrap();
        let sqrt_dt = dt.sqrt();

        let mu_adj = self.adjusted_drift();

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.s0);

        let mut s = self.s0;

        for _ in 0..n_steps {
            // Diffusion component
            let dw = sqrt_dt * normal.sample(rng);
            let diffusion = mu_adj * s * dt + self.sigma * s * dw;

            // Jump component
            let n_jumps = self.sample_n_jumps(rng, dt);

            let mut jump_mult = 1.0;
            for _ in 0..n_jumps {
                let log_y = normal_j.sample(rng);
                jump_mult *= log_y.exp();
            }

            s = (s + diffusion) * jump_mult;
            s = s.max(1e-10); // Prevent zero/negative prices

            path.push(s);
        }

        path
    }

    /// Samples the number of jumps in interval dt
    fn sample_n_jumps<R: Rng>(&self, rng: &mut R, dt: f64) -> usize {
        let lambda_dt = self.lambda * dt;

        if lambda_dt < 30.0 {
            // Use Poisson distribution directly
            match Poisson::new(lambda_dt) {
                Ok(poisson) => poisson.sample(rng) as usize,
                Err(_) => 0,
            }
        } else {
            // For large λdt, approximate with normal
            let normal = Normal::new(0.0, 1.0).unwrap();
            let n = lambda_dt + lambda_dt.sqrt() * normal.sample(rng);
            n.round().max(0.0) as usize
        }
    }

    /// Simulates using log-price directly (more accurate for small dt)
    pub fn simulate_log<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let normal_j = Normal::new(self.mu_j, self.sigma_j).unwrap();
        let sqrt_dt = dt.sqrt();

        // Drift for log price
        let log_drift = (self.mu - self.lambda * self.kappa() - 0.5 * self.sigma * self.sigma) * dt;

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.s0);

        let mut log_s = self.s0.ln();

        for _ in 0..n_steps {
            // Diffusion in log space
            let dw = sqrt_dt * normal.sample(rng);
            log_s += log_drift + self.sigma * dw;

            // Jumps (add log jump sizes directly)
            let n_jumps = self.sample_n_jumps(rng, dt);
            for _ in 0..n_jumps {
                log_s += normal_j.sample(rng);
            }

            path.push(log_s.exp());
        }

        path
    }

    /// Generates multiple paths in parallel
    pub fn simulate_parallel(&self, n_paths: usize, n_steps: usize, dt: f64) -> Vec<Vec<f64>> {
        (0..n_paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                self.simulate_log(&mut rng, n_steps, dt)
            })
            .collect()
    }

    /// Samples terminal values in parallel
    pub fn sample_terminal_parallel(&self, n_paths: usize, t: f64, n_steps: usize) -> Vec<f64> {
        let dt = t / n_steps as f64;

        (0..n_paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let path = self.simulate_log(&mut rng, n_steps, dt);
                *path.last().unwrap()
            })
            .collect()
    }
}

/// Statistics for jump-diffusion paths
pub struct JumpDiffusionStats {
    /// Number of jumps detected
    pub n_jumps: usize,
    /// Maximum jump size (in log terms)
    pub max_jump: f64,
    /// Minimum jump size (in log terms)
    pub min_jump: f64,
    /// Total return
    pub total_return: f64,
}

impl JumpDiffusionStats {
    /// Detects jumps in a price path using a threshold
    pub fn from_path(prices: &[f64], threshold: f64) -> Self {
        let log_returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let jumps: Vec<f64> = log_returns
            .iter()
            .filter(|&&r| r.abs() > threshold)
            .copied()
            .collect();

        let n_jumps = jumps.len();
        let max_jump = jumps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_jump = jumps.iter().copied().fold(f64::INFINITY, f64::min);
        let total_return = (prices.last().unwrap() / prices[0]).ln();

        Self {
            n_jumps,
            max_jump: if n_jumps > 0 { max_jump } else { 0.0 },
            min_jump: if n_jumps > 0 { min_jump } else { 0.0 },
            total_return,
        }
    }
}

/// Double-exponential jump-diffusion (Kou model)
///
/// Jump sizes follow a double-exponential distribution,
/// allowing for asymmetric upward and downward jumps.
#[derive(Debug, Clone)]
pub struct KouJumpDiffusion {
    /// Initial price
    pub s0: f64,
    /// Drift
    pub mu: f64,
    /// Diffusion volatility
    pub sigma: f64,
    /// Jump intensity
    pub lambda: f64,
    /// Probability of upward jump
    pub p: f64,
    /// Rate parameter for upward jumps (η1)
    pub eta1: f64,
    /// Rate parameter for downward jumps (η2)
    pub eta2: f64,
}

impl KouJumpDiffusion {
    /// Creates a new Kou model
    pub fn new(
        s0: f64,
        mu: f64,
        sigma: f64,
        lambda: f64,
        p: f64,
        eta1: f64,
        eta2: f64,
    ) -> Self {
        assert!(s0 > 0.0);
        assert!(sigma >= 0.0);
        assert!(lambda >= 0.0);
        assert!(p >= 0.0 && p <= 1.0);
        assert!(eta1 > 1.0, "η1 must be > 1 for finite expectation");
        assert!(eta2 > 0.0);

        Self { s0, mu, sigma, lambda, p, eta1, eta2 }
    }

    /// Expected jump size E[Y - 1]
    pub fn kappa(&self) -> f64 {
        self.p * self.eta1 / (self.eta1 - 1.0)
            + (1.0 - self.p) * self.eta2 / (self.eta2 + 1.0)
            - 1.0
    }

    /// Samples a double-exponential jump
    fn sample_jump<R: Rng>(&self, rng: &mut R) -> f64 {
        let uniform = rand_distr::Uniform::new(0.0f64, 1.0);
        let u: f64 = uniform.sample(rng);

        if rng.gen::<f64>() < self.p {
            // Upward jump: exponential with rate η1
            (1.0 - u).ln() / self.eta1
        } else {
            // Downward jump: negative exponential with rate η2
            -(1.0 - u).ln() / self.eta2
        }
    }

    /// Simulates a path
    pub fn simulate<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();

        let mu_adj = self.mu - self.lambda * self.kappa();
        let log_drift = (mu_adj - 0.5 * self.sigma * self.sigma) * dt;

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.s0);

        let mut log_s = self.s0.ln();

        for _ in 0..n_steps {
            let dw = sqrt_dt * normal.sample(rng);
            log_s += log_drift + self.sigma * dw;

            // Sample jumps
            let lambda_dt = self.lambda * dt;
            if lambda_dt < 30.0 {
                if let Ok(poisson) = Poisson::new(lambda_dt) {
                    let n_jumps: u64 = poisson.sample(rng);
                    for _ in 0..n_jumps {
                        log_s += self.sample_jump(rng);
                    }
                }
            }

            path.push(log_s.exp());
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merton_positive_prices() {
        let model = MertonJumpDiffusion::new(100.0, 0.1, 0.2, 2.0, -0.05, 0.1);
        let mut rng = rand::thread_rng();

        let path = model.simulate(&mut rng, 1000, 0.01);

        assert!(path.iter().all(|&p| p > 0.0), "All prices should be positive");
    }

    #[test]
    fn test_merton_log_simulation() {
        let model = MertonJumpDiffusion::new(100.0, 0.1, 0.2, 2.0, -0.05, 0.1);
        let mut rng = rand::thread_rng();

        let path = model.simulate_log(&mut rng, 1000, 0.01);

        assert_eq!(path.len(), 1001);
        assert!(path.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_jump_detection() {
        let model = MertonJumpDiffusion::new(100.0, 0.0, 0.01, 10.0, 0.0, 0.1);
        let mut rng = rand::thread_rng();

        // Many jumps, low diffusion volatility
        let path = model.simulate_log(&mut rng, 252, 1.0 / 252.0);

        // With threshold = 3 * daily vol ≈ 0.03
        let stats = JumpDiffusionStats::from_path(&path, 0.02);

        // Should detect some jumps
        assert!(stats.n_jumps > 0, "Should detect jumps");
    }

    #[test]
    fn test_kappa_calculation() {
        let model = MertonJumpDiffusion::new(100.0, 0.1, 0.2, 2.0, 0.0, 0.1);

        // When mu_j = 0: kappa = exp(0.5 * sigma_j^2) - 1
        let expected_kappa = (0.5 * 0.1 * 0.1_f64).exp() - 1.0;
        let actual_kappa = model.kappa();

        assert!(
            (actual_kappa - expected_kappa).abs() < 1e-10,
            "Kappa mismatch: {} vs {}",
            actual_kappa,
            expected_kappa
        );
    }

    #[test]
    fn test_parallel_simulation() {
        let model = MertonJumpDiffusion::new(100.0, 0.1, 0.2, 2.0, -0.05, 0.1);

        let paths = model.simulate_parallel(100, 50, 0.01);

        assert_eq!(paths.len(), 100);
        assert!(paths.iter().all(|p| p.len() == 51));
    }

    #[test]
    fn test_kou_positive_prices() {
        let model = KouJumpDiffusion::new(100.0, 0.1, 0.2, 3.0, 0.4, 3.0, 5.0);
        let mut rng = rand::thread_rng();

        let path = model.simulate(&mut rng, 500, 0.01);

        assert!(path.iter().all(|&p| p > 0.0));
    }
}
