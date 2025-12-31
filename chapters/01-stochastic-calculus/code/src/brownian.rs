//! Brownian Motion (Wiener Process) implementation
//!
//! Brownian motion is the foundation of stochastic calculus in finance.
//! It models the random component of asset price movements.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Standard Brownian motion generator
///
/// Generates paths of the Wiener process W_t with properties:
/// - W_0 = initial (typically 0)
/// - W_t - W_s ~ N(0, t-s) for s < t
/// - Independent increments
/// - Continuous paths
#[derive(Debug, Clone)]
pub struct BrownianMotion {
    /// Initial value W_0
    pub initial: f64,
    /// Standard normal distribution for generating increments
    normal: Normal<f64>,
}

impl BrownianMotion {
    /// Creates a new Brownian motion generator
    ///
    /// # Arguments
    /// * `initial` - Starting value W_0 (typically 0.0)
    ///
    /// # Example
    /// ```
    /// use stochastic_calculus::BrownianMotion;
    /// let bm = BrownianMotion::new(0.0);
    /// ```
    pub fn new(initial: f64) -> Self {
        Self {
            initial,
            normal: Normal::new(0.0, 1.0).expect("Invalid normal distribution parameters"),
        }
    }

    /// Generates a single Brownian motion path
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `n_steps` - Number of time steps
    /// * `dt` - Time step size
    ///
    /// # Returns
    /// Vector of W_t values for t = 0, dt, 2*dt, ..., n_steps*dt
    ///
    /// # Example
    /// ```
    /// use stochastic_calculus::BrownianMotion;
    ///
    /// let bm = BrownianMotion::new(0.0);
    /// let mut rng = rand::thread_rng();
    /// let path = bm.generate_path(&mut rng, 100, 0.01);
    /// assert_eq!(path.len(), 101);
    /// ```
    pub fn generate_path<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let sqrt_dt = dt.sqrt();
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.initial);

        let mut current = self.initial;
        for _ in 0..n_steps {
            // dW = sqrt(dt) * Z, where Z ~ N(0, 1)
            let dw = sqrt_dt * self.normal.sample(rng);
            current += dw;
            path.push(current);
        }

        path
    }

    /// Generates multiple Brownian motion paths in parallel
    ///
    /// Uses rayon for parallel execution across multiple CPU cores.
    ///
    /// # Arguments
    /// * `n_paths` - Number of paths to generate
    /// * `n_steps` - Number of time steps per path
    /// * `dt` - Time step size
    ///
    /// # Returns
    /// Vector of paths, each being a vector of W_t values
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

    /// Generates correlated Brownian motion increments
    ///
    /// Given a correlation coefficient rho, generates a pair (dW1, dW2) such that:
    /// - dW1, dW2 are normally distributed with variance dt
    /// - Corr(dW1, dW2) = rho
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `dt` - Time step size
    /// * `rho` - Correlation coefficient in [-1, 1]
    ///
    /// # Returns
    /// Tuple (dW1, dW2) of correlated increments
    pub fn correlated_increments<R: Rng>(&self, rng: &mut R, dt: f64, rho: f64) -> (f64, f64) {
        let sqrt_dt = dt.sqrt();
        let z1 = self.normal.sample(rng);
        let z2 = self.normal.sample(rng);

        let dw1 = sqrt_dt * z1;
        let dw2 = sqrt_dt * (rho * z1 + (1.0 - rho * rho).sqrt() * z2);

        (dw1, dw2)
    }
}

/// Computes the quadratic variation of a path
///
/// For Brownian motion, [W,W]_T should equal T (approximately).
///
/// # Arguments
/// * `path` - Vector of path values
///
/// # Returns
/// Sum of squared increments
///
/// # Example
/// ```
/// use stochastic_calculus::brownian::{BrownianMotion, quadratic_variation};
///
/// let bm = BrownianMotion::new(0.0);
/// let mut rng = rand::thread_rng();
/// let path = bm.generate_path(&mut rng, 10000, 0.001);
/// let qv = quadratic_variation(&path);
/// // qv should be approximately 10.0 (= 10000 * 0.001)
/// ```
pub fn quadratic_variation(path: &[f64]) -> f64 {
    path.windows(2)
        .map(|w| {
            let diff = w[1] - w[0];
            diff * diff
        })
        .sum()
}

/// Computes the Ito integral ∫ f(W_t) dW_t numerically
///
/// Uses left-point evaluation as per Ito's definition.
///
/// # Arguments
/// * `path` - Brownian motion path
/// * `f` - Function to integrate
/// * `_dt` - Time step (unused, kept for API consistency)
///
/// # Returns
/// Numerical approximation of the Ito integral
pub fn ito_integral<F>(path: &[f64], f: F, _dt: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    path.windows(2)
        .map(|w| {
            let w_t = w[0]; // Left endpoint (Ito's definition)
            let dw = w[1] - w[0];
            f(w_t) * dw
        })
        .sum()
}

/// Computes the Stratonovich integral ∫ f(W_t) ∘ dW_t numerically
///
/// Uses midpoint evaluation as per Stratonovich's definition.
///
/// # Arguments
/// * `path` - Brownian motion path
/// * `f` - Function to integrate
/// * `_dt` - Time step (unused, kept for API consistency)
///
/// # Returns
/// Numerical approximation of the Stratonovich integral
pub fn stratonovich_integral<F>(path: &[f64], f: F, _dt: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    path.windows(2)
        .map(|w| {
            let w_mid = (w[0] + w[1]) / 2.0; // Midpoint (Stratonovich)
            let dw = w[1] - w[0];
            f(w_mid) * dw
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brownian_starts_at_initial() {
        let bm = BrownianMotion::new(5.0);
        let mut rng = rand::thread_rng();
        let path = bm.generate_path(&mut rng, 100, 0.01);
        assert_eq!(path[0], 5.0);
    }

    #[test]
    fn test_path_length() {
        let bm = BrownianMotion::new(0.0);
        let mut rng = rand::thread_rng();
        let path = bm.generate_path(&mut rng, 100, 0.01);
        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_quadratic_variation_approximates_t() {
        let bm = BrownianMotion::new(0.0);
        let mut rng = rand::thread_rng();

        let n_steps = 10000;
        let dt = 0.001;
        let t_end = n_steps as f64 * dt;

        let path = bm.generate_path(&mut rng, n_steps, dt);
        let qv = quadratic_variation(&path);

        // Quadratic variation should be close to T
        assert!(
            (qv - t_end).abs() < 1.0,
            "QV = {}, expected ≈ {}",
            qv,
            t_end
        );
    }

    #[test]
    fn test_ito_integral_w_dw() {
        let bm = BrownianMotion::new(0.0);
        let mut rng = rand::thread_rng();

        let n_steps = 50000;
        let dt = 0.0001;
        let t_end = n_steps as f64 * dt;

        let path = bm.generate_path(&mut rng, n_steps, dt);
        let w_t = *path.last().unwrap();

        // ∫ W_t dW_t by Ito = (W_T² - T) / 2
        let ito = ito_integral(&path, |x| x, dt);
        let expected_ito = (w_t * w_t - t_end) / 2.0;

        assert!(
            (ito - expected_ito).abs() < 0.5,
            "Ito integral: got {}, expected {}",
            ito,
            expected_ito
        );
    }

    #[test]
    fn test_parallel_generation() {
        let bm = BrownianMotion::new(0.0);
        let paths = bm.generate_paths_parallel(100, 50, 0.01);
        assert_eq!(paths.len(), 100);
        assert!(paths.iter().all(|p| p.len() == 51));
    }

    #[test]
    fn test_correlated_increments() {
        let bm = BrownianMotion::new(0.0);
        let mut rng = rand::thread_rng();

        let n_samples = 10000;
        let dt = 0.01;
        let rho = 0.7;

        let mut sum_product = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for _ in 0..n_samples {
            let (dw1, dw2) = bm.correlated_increments(&mut rng, dt, rho);
            sum_product += dw1 * dw2;
            sum_sq1 += dw1 * dw1;
            sum_sq2 += dw2 * dw2;
        }

        let empirical_corr = sum_product / (sum_sq1.sqrt() * sum_sq2.sqrt());

        assert!(
            (empirical_corr - rho).abs() < 0.1,
            "Empirical correlation: {}, expected {}",
            empirical_corr,
            rho
        );
    }
}
