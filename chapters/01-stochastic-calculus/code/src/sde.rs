//! Stochastic Differential Equation (SDE) framework
//!
//! Provides traits and solvers for working with SDEs of the form:
//! dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Trait representing a Stochastic Differential Equation
///
/// An SDE has the form: dX_t = μ(t, X_t) dt + σ(t, X_t) dW_t
/// where μ is the drift and σ is the diffusion coefficient.
pub trait SDE {
    /// The state type (e.g., f64 for 1D, [f64; N] for N-dimensional)
    type State: Clone;

    /// Computes the drift coefficient μ(t, x)
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Current state
    fn drift(&self, t: f64, x: &Self::State) -> Self::State;

    /// Computes the diffusion coefficient σ(t, x)
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `x` - Current state
    fn diffusion(&self, t: f64, x: &Self::State) -> Self::State;

    /// Returns the initial state X_0
    fn initial_state(&self) -> Self::State;
}

/// Euler-Maruyama solver for SDEs
///
/// The Euler-Maruyama method is the simplest numerical scheme for SDEs:
/// X_{n+1} = X_n + μ(t_n, X_n) * Δt + σ(t_n, X_n) * ΔW_n
///
/// It has strong convergence order 0.5 and weak convergence order 1.0.
pub struct EulerMaruyama<S: SDE<State = f64>> {
    /// The SDE to solve
    pub sde: S,
    /// Time step size
    pub dt: f64,
    /// Normal distribution for generating Brownian increments
    normal: Normal<f64>,
}

impl<S: SDE<State = f64>> EulerMaruyama<S> {
    /// Creates a new Euler-Maruyama solver
    ///
    /// # Arguments
    /// * `sde` - The SDE to solve
    /// * `dt` - Time step size
    pub fn new(sde: S, dt: f64) -> Self {
        Self {
            sde,
            dt,
            normal: Normal::new(0.0, 1.0).unwrap(),
        }
    }

    /// Performs one step of the Euler-Maruyama scheme
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `t` - Current time
    /// * `x` - Current state
    ///
    /// # Returns
    /// New state after one time step
    pub fn step<R: Rng>(&self, rng: &mut R, t: f64, x: f64) -> f64 {
        let dw = self.dt.sqrt() * self.normal.sample(rng);
        let drift = self.sde.drift(t, &x);
        let diffusion = self.sde.diffusion(t, &x);

        x + drift * self.dt + diffusion * dw
    }

    /// Solves the SDE from t=0 to t=t_end
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `t_end` - Final time
    ///
    /// # Returns
    /// Vector of (time, state) pairs
    pub fn solve<R: Rng>(&self, rng: &mut R, t_end: f64) -> Vec<(f64, f64)> {
        let n_steps = (t_end / self.dt).ceil() as usize;
        let mut trajectory = Vec::with_capacity(n_steps + 1);

        let mut t = 0.0;
        let mut x = self.sde.initial_state();
        trajectory.push((t, x));

        for _ in 0..n_steps {
            x = self.step(rng, t, x);
            t += self.dt;
            trajectory.push((t, x));
        }

        trajectory
    }

    /// Solves the SDE and returns only the final value
    ///
    /// More memory-efficient when only the terminal value is needed.
    pub fn solve_terminal<R: Rng>(&self, rng: &mut R, t_end: f64) -> f64 {
        let n_steps = (t_end / self.dt).ceil() as usize;

        let mut t = 0.0;
        let mut x = self.sde.initial_state();

        for _ in 0..n_steps {
            x = self.step(rng, t, x);
            t += self.dt;
        }

        x
    }
}

/// Milstein solver for SDEs
///
/// The Milstein method adds a correction term for better accuracy:
/// X_{n+1} = X_n + μΔt + σΔW + 0.5 σσ'(ΔW² - Δt)
///
/// It has strong convergence order 1.0 (compared to 0.5 for Euler-Maruyama).
pub struct Milstein<S: SDE<State = f64>> {
    /// The SDE to solve
    pub sde: S,
    /// Time step size
    pub dt: f64,
    /// Finite difference step for computing σ'
    epsilon: f64,
    /// Normal distribution
    normal: Normal<f64>,
}

impl<S: SDE<State = f64>> Milstein<S> {
    /// Creates a new Milstein solver
    ///
    /// # Arguments
    /// * `sde` - The SDE to solve
    /// * `dt` - Time step size
    pub fn new(sde: S, dt: f64) -> Self {
        Self {
            sde,
            dt,
            epsilon: 1e-6,
            normal: Normal::new(0.0, 1.0).unwrap(),
        }
    }

    /// Computes σ'(t, x) using finite differences
    fn diffusion_derivative(&self, t: f64, x: f64) -> f64 {
        let sigma_plus = self.sde.diffusion(t, &(x + self.epsilon));
        let sigma_minus = self.sde.diffusion(t, &(x - self.epsilon));
        (sigma_plus - sigma_minus) / (2.0 * self.epsilon)
    }

    /// Performs one step of the Milstein scheme
    pub fn step<R: Rng>(&self, rng: &mut R, t: f64, x: f64) -> f64 {
        let dw = self.dt.sqrt() * self.normal.sample(rng);
        let drift = self.sde.drift(t, &x);
        let diffusion = self.sde.diffusion(t, &x);
        let diffusion_deriv = self.diffusion_derivative(t, x);

        // Milstein correction: 0.5 * σ * σ' * (ΔW² - Δt)
        let milstein_correction = 0.5 * diffusion * diffusion_deriv * (dw * dw - self.dt);

        x + drift * self.dt + diffusion * dw + milstein_correction
    }

    /// Solves the SDE from t=0 to t=t_end
    pub fn solve<R: Rng>(&self, rng: &mut R, t_end: f64) -> Vec<(f64, f64)> {
        let n_steps = (t_end / self.dt).ceil() as usize;
        let mut trajectory = Vec::with_capacity(n_steps + 1);

        let mut t = 0.0;
        let mut x = self.sde.initial_state();
        trajectory.push((t, x));

        for _ in 0..n_steps {
            x = self.step(rng, t, x);
            t += self.dt;
            trajectory.push((t, x));
        }

        trajectory
    }
}

/// A simple test SDE: Ornstein-Uhlenbeck process
/// dX_t = θ(μ - X_t)dt + σdW_t
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeck {
    /// Mean reversion speed
    pub theta: f64,
    /// Long-term mean
    pub mu: f64,
    /// Volatility
    pub sigma: f64,
    /// Initial value
    pub x0: f64,
}

impl OrnsteinUhlenbeck {
    pub fn new(theta: f64, mu: f64, sigma: f64, x0: f64) -> Self {
        Self { theta, mu, sigma, x0 }
    }
}

impl SDE for OrnsteinUhlenbeck {
    type State = f64;

    fn drift(&self, _t: f64, x: &f64) -> f64 {
        self.theta * (self.mu - x)
    }

    fn diffusion(&self, _t: f64, _x: &f64) -> f64 {
        self.sigma
    }

    fn initial_state(&self) -> f64 {
        self.x0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ou_mean_reversion() {
        // OU process should revert to mean
        let ou = OrnsteinUhlenbeck::new(2.0, 0.0, 0.1, 1.0);
        let solver = EulerMaruyama::new(ou, 0.01);

        let mut rng = rand::thread_rng();

        // Run many simulations and check mean
        let n_sims = 1000;
        let t_end = 5.0;

        let sum: f64 = (0..n_sims)
            .map(|_| solver.solve_terminal(&mut rng, t_end))
            .sum();

        let mean = sum / n_sims as f64;

        // Mean should be close to μ = 0
        assert!(
            mean.abs() < 0.2,
            "Mean = {}, expected ≈ 0.0",
            mean
        );
    }

    #[test]
    fn test_milstein_vs_euler() {
        let ou = OrnsteinUhlenbeck::new(1.0, 0.0, 0.5, 1.0);

        let euler = EulerMaruyama::new(ou.clone(), 0.001);
        let milstein = Milstein::new(ou, 0.001);

        let mut rng = rand::thread_rng();

        // For OU process with constant diffusion, Milstein = Euler
        // (since σ' = 0)
        let euler_path = euler.solve(&mut rng, 1.0);
        let milstein_path = milstein.solve(&mut rng, 1.0);

        assert_eq!(euler_path.len(), milstein_path.len());
    }
}
