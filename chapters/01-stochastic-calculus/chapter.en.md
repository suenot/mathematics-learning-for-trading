# Chapter 1: Stochastic Calculus for Algorithmic Trading

## Introduction

Stochastic calculus is a mathematical framework for describing systems that evolve randomly over time. In the context of trading, this is exactly what we need: asset prices change unpredictably but follow certain statistical patterns.

In this chapter, we will:
- Study Brownian motion — the foundation of random processes
- Master Ito's integral and its application to financial models
- Implement numerical methods for simulating price processes in Rust
- Create production-ready code for Monte Carlo simulations

---

## 1.1 Brownian Motion (Wiener Process)

### Historical Background

In 1827, botanist Robert Brown observed the erratic movement of pollen grains in water. This phenomenon, named Brownian motion, was explained by Einstein in 1905 as the result of collisions with water molecules.

Norbert Wiener in 1923 constructed a rigorous mathematical theory of this process, which is why it's also called the Wiener process.

### Mathematical Definition

**Definition 1.1 (Standard Brownian Motion)**

A stochastic process $W = \{W_t\}_{t \geq 0}$ is called standard Brownian motion if:

1. $W_0 = 0$ (almost surely)
2. Paths $t \mapsto W_t$ are continuous (almost surely)
3. Increments are independent: for any $0 \leq t_1 < t_2 < ... < t_n$, the random variables $W_{t_2} - W_{t_1}, W_{t_3} - W_{t_2}, ..., W_{t_n} - W_{t_{n-1}}$ are independent
4. Increments are stationary and normally distributed: $W_t - W_s \sim \mathcal{N}(0, t-s)$ for $s < t$

### Key Properties

**Property 1: Path Continuity**

Brownian motion paths are continuous but nowhere differentiable. This is important: the "rate of change" of Brownian motion doesn't exist in the classical sense.

**Property 2: Quadratic Variation**

For regular smooth functions, variation over an interval is finite, and quadratic variation equals zero. For Brownian motion, it's the opposite:

$$[W,W]_t = \lim_{n \to \infty} \sum_{i=1}^{n} (W_{t_i} - W_{t_{i-1}})^2 = t$$

This property is key to understanding the Ito integral.

**Property 3: Martingale Property**

Brownian motion is a martingale:
$$\mathbb{E}[W_t | \mathcal{F}_s] = W_s \quad \text{for } s \leq t$$

This means the best forecast of a future value is the current value.

### Rust Implementation

```rust
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Brownian motion path generator
pub struct BrownianMotion {
    /// Initial value
    pub initial: f64,
    /// Normal distribution for generating increments
    normal: Normal<f64>,
}

impl BrownianMotion {
    /// Creates a new Brownian motion generator
    pub fn new(initial: f64) -> Self {
        Self {
            initial,
            normal: Normal::new(0.0, 1.0).expect("Invalid normal distribution"),
        }
    }

    /// Generates a single path
    ///
    /// # Arguments
    /// * `n_steps` - number of time steps
    /// * `dt` - time step size
    ///
    /// # Returns
    /// Vector of W_t values for t = 0, dt, 2*dt, ..., n_steps*dt
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

    /// Generates multiple paths in parallel
    pub fn generate_paths_parallel(
        &self,
        n_paths: usize,
        n_steps: usize,
        dt: f64,
    ) -> Vec<Vec<f64>> {
        use rayon::prelude::*;

        (0..n_paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                self.generate_path(&mut rng, n_steps, dt)
            })
            .collect()
    }
}

/// Computes quadratic variation of a path
pub fn quadratic_variation(path: &[f64]) -> f64 {
    path.windows(2)
        .map(|w| {
            let diff = w[1] - w[0];
            diff * diff
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_variation() {
        let bm = BrownianMotion::new(0.0);
        let mut rng = rand::thread_rng();

        let n_steps = 10000;
        let dt = 0.001;
        let t_end = n_steps as f64 * dt; // T = 10

        let path = bm.generate_path(&mut rng, n_steps, dt);
        let qv = quadratic_variation(&path);

        // Quadratic variation should be close to T
        assert!((qv - t_end).abs() < 0.5, "QV = {}, expected ≈ {}", qv, t_end);
    }
}
```

---

## 1.2 Ito's Integral

### The Problem with Classical Integration

Why can't we use the ordinary Riemann integral for stochastic processes?

Consider the integral:
$$\int_0^T W_t \, dW_t$$

Let's try to compute it as a limit of integral sums:
$$\sum_{i} W_{t_i^*} (W_{t_{i+1}} - W_{t_i})$$

The problem is that the result depends on the choice of point $t_i^*$:
- If $t_i^* = t_i$ (left endpoint) — we get the Ito integral
- If $t_i^* = t_{i+1}$ (right endpoint) — we get a different result
- If $t_i^* = (t_i + t_{i+1})/2$ (midpoint) — the Stratonovich integral

### Definition of Ito's Integral

**Definition 1.2 (Ito's Integral)**

For an adapted process $f_t$, the Ito integral is defined as:

$$\int_0^T f_t \, dW_t = \lim_{n \to \infty} \sum_{i=0}^{n-1} f_{t_i} (W_{t_{i+1}} - W_{t_i})$$

Key point: we always take the value of $f$ at the left endpoint!

### Ito's Rule (Ito's Lemma)

**Theorem 1.1 (Ito's Lemma)**

Let $X_t$ be an Ito process:
$$dX_t = \mu_t \, dt + \sigma_t \, dW_t$$

and $f(t, x)$ be a twice continuously differentiable function. Then:

$$df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2$$

Using the rules:
- $(dt)^2 = 0$
- $dt \cdot dW_t = 0$
- $(dW_t)^2 = dt$

we get:

$$df = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW_t$$

### Example: Deriving $\int W_t \, dW_t$

Apply Ito's lemma to $f(x) = x^2$ and $X_t = W_t$:

$$d(W_t^2) = 2W_t \, dW_t + \frac{1}{2} \cdot 2 \cdot (dW_t)^2 = 2W_t \, dW_t + dt$$

Integrating:
$$W_T^2 - W_0^2 = 2\int_0^T W_t \, dW_t + T$$

Therefore:
$$\int_0^T W_t \, dW_t = \frac{1}{2}(W_T^2 - T)$$

This result differs from the classical $\frac{1}{2}W_T^2$ by an additional term $-\frac{T}{2}$!

### Rust Implementation

```rust
/// Numerically computes the Ito integral ∫ f(W_t) dW_t
pub fn ito_integral<F>(path: &[f64], f: F, dt: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    path.windows(2)
        .map(|w| {
            let w_t = w[0];      // Left endpoint (Ito's definition!)
            let dw = w[1] - w[0]; // Increment
            f(w_t) * dw
        })
        .sum()
}

/// Numerically computes the Stratonovich integral ∫ f(W_t) ∘ dW_t
pub fn stratonovich_integral<F>(path: &[f64], f: F, dt: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    path.windows(2)
        .map(|w| {
            let w_mid = (w[0] + w[1]) / 2.0; // Midpoint (Stratonovich definition)
            let dw = w[1] - w[0];
            f(w_mid) * dw
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ito_integral_w_dw() {
        let bm = BrownianMotion::new(0.0);
        let mut rng = rand::thread_rng();

        let n_steps = 100000;
        let dt = 0.0001;
        let t_end = n_steps as f64 * dt;

        let path = bm.generate_path(&mut rng, n_steps, dt);
        let w_t = *path.last().unwrap();

        // ∫ W_t dW_t by Ito = (W_T² - T) / 2
        let ito = ito_integral(&path, |x| x, dt);
        let expected_ito = (w_t * w_t - t_end) / 2.0;

        assert!((ito - expected_ito).abs() < 0.1,
            "Ito integral: got {}, expected {}", ito, expected_ito);
    }
}
```

---

## 1.3 Stochastic Differential Equations (SDEs)

### General Form of SDEs

A stochastic differential equation has the form:

$$dX_t = \mu(t, X_t) \, dt + \sigma(t, X_t) \, dW_t$$

where:
- $\mu(t, x)$ — drift
- $\sigma(t, x)$ — volatility (diffusion)

### Existence and Uniqueness Conditions

**Theorem 1.2** (Existence and Uniqueness)

If $\mu$ and $\sigma$ satisfy:

1. **Lipschitz condition**: $|\mu(t,x) - \mu(t,y)| + |\sigma(t,x) - \sigma(t,y)| \leq K|x-y|$
2. **Growth condition**: $|\mu(t,x)|^2 + |\sigma(t,x)|^2 \leq K^2(1 + |x|^2)$

then the SDE has a unique strong solution.

### SDE Abstraction in Rust

```rust
/// Trait for stochastic differential equations
pub trait SDE {
    /// State type (can be f64 or a vector)
    type State: Clone;

    /// Drift μ(t, X)
    fn drift(&self, t: f64, x: &Self::State) -> Self::State;

    /// Volatility σ(t, X)
    fn diffusion(&self, t: f64, x: &Self::State) -> Self::State;

    /// Initial condition
    fn initial_state(&self) -> Self::State;
}

/// Euler-Maruyama solver for SDEs
pub struct EulerMaruyama<S: SDE<State = f64>> {
    pub sde: S,
    pub dt: f64,
}

impl<S: SDE<State = f64>> EulerMaruyama<S> {
    pub fn new(sde: S, dt: f64) -> Self {
        Self { sde, dt }
    }

    /// One step of the Euler-Maruyama method
    /// X_{n+1} = X_n + μ(t, X_n) * dt + σ(t, X_n) * dW
    pub fn step<R: Rng>(&self, rng: &mut R, t: f64, x: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dw = self.dt.sqrt() * normal.sample(rng);

        let drift = self.sde.drift(t, &x);
        let diffusion = self.sde.diffusion(t, &x);

        x + drift * self.dt + diffusion * dw
    }

    /// Generates a trajectory
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
```

---

## 1.4 Geometric Brownian Motion (GBM)

### The Model

Geometric Brownian motion is the most well-known model for asset prices:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

where:
- $S_t$ — asset price at time $t$
- $\mu$ — expected return (drift)
- $\sigma$ — volatility

### Analytical Solution

Apply Ito's lemma to $f(S) = \ln S$:

$$d(\ln S_t) = \frac{1}{S_t} dS_t - \frac{1}{2} \frac{1}{S_t^2} (dS_t)^2$$

Substituting $(dS_t)^2 = \sigma^2 S_t^2 dt$:

$$d(\ln S_t) = \frac{1}{S_t}(\mu S_t dt + \sigma S_t dW_t) - \frac{1}{2}\sigma^2 dt$$

$$d(\ln S_t) = \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma dW_t$$

Integrating:

$$\ln S_t = \ln S_0 + \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t$$

**Solution:**
$$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]$$

### Properties of GBM

1. **Price is always positive**: $S_t > 0$ for all $t$
2. **Log-normal distribution**: $\ln(S_t/S_0) \sim \mathcal{N}\left((\mu - \frac{\sigma^2}{2})t, \sigma^2 t\right)$
3. **Expected value**: $\mathbb{E}[S_t] = S_0 e^{\mu t}$

### Rust Implementation

```rust
/// Geometric Brownian Motion
pub struct GeometricBrownianMotion {
    /// Initial price
    pub s0: f64,
    /// Drift (expected return)
    pub mu: f64,
    /// Volatility
    pub sigma: f64,
}

impl GeometricBrownianMotion {
    pub fn new(s0: f64, mu: f64, sigma: f64) -> Self {
        assert!(s0 > 0.0, "Initial price must be positive");
        assert!(sigma >= 0.0, "Volatility must be non-negative");
        Self { s0, mu, sigma }
    }

    /// Analytical solution (more efficient for generating terminal values)
    pub fn sample_at_time<R: Rng>(&self, rng: &mut R, t: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.sample(rng);

        self.s0 * ((self.mu - 0.5 * self.sigma * self.sigma) * t
                   + self.sigma * t.sqrt() * z).exp()
    }

    /// Generates a path using the analytical solution
    pub fn generate_path_exact<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::statistics::Statistics;

    #[test]
    fn test_gbm_expected_value() {
        let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
        let t = 1.0;
        let n_simulations = 100000;

        let mut rng = rand::thread_rng();
        let final_prices: Vec<f64> = (0..n_simulations)
            .map(|_| gbm.sample_at_time(&mut rng, t))
            .collect();

        let mean = final_prices.mean();
        let expected = gbm.s0 * (gbm.mu * t).exp(); // E[S_t] = S_0 * e^(μt)

        assert!((mean - expected).abs() < 1.0,
            "Mean: {}, Expected: {}", mean, expected);
    }
}
```

---

## 1.5 The Heston Model (Stochastic Volatility)

### Motivation

GBM assumes constant volatility $\sigma$. In reality:
- Volatility changes over time
- We observe a "volatility smile" in options
- Volatility typically increases when prices fall (leverage effect)

### The Model

The Heston model:

$$dS_t = \mu S_t \, dt + \sqrt{V_t} S_t \, dW^S_t$$
$$dV_t = \kappa(\theta - V_t) \, dt + \xi \sqrt{V_t} \, dW^V_t$$

where:
- $V_t$ — instantaneous variance
- $\kappa$ — mean reversion speed
- $\theta$ — long-term variance level
- $\xi$ — volatility of volatility
- $\text{Corr}(dW^S, dW^V) = \rho$ (typically $\rho < 0$)

### Feller Condition

For $V_t$ to remain positive:

$$2\kappa\theta > \xi^2$$

### Rust Implementation

```rust
/// Heston model for stochastic volatility
pub struct HestonModel {
    /// Initial price
    pub s0: f64,
    /// Initial variance
    pub v0: f64,
    /// Price drift
    pub mu: f64,
    /// Mean reversion speed (kappa)
    pub kappa: f64,
    /// Long-term variance (theta)
    pub theta: f64,
    /// Volatility of volatility (xi)
    pub xi: f64,
    /// Correlation between price and volatility (rho)
    pub rho: f64,
}

impl HestonModel {
    pub fn new(s0: f64, v0: f64, mu: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> Self {
        assert!(s0 > 0.0, "Initial price must be positive");
        assert!(v0 > 0.0, "Initial variance must be positive");
        assert!(kappa > 0.0, "Mean reversion speed must be positive");
        assert!(theta > 0.0, "Long-term variance must be positive");
        assert!(xi > 0.0, "Vol of vol must be positive");
        assert!(rho.abs() <= 1.0, "Correlation must be in [-1, 1]");

        // Check Feller condition
        if 2.0 * kappa * theta <= xi * xi {
            eprintln!("Warning: Feller condition not satisfied (2κθ > ξ²)");
        }

        Self { s0, v0, mu, kappa, theta, xi, rho }
    }

    /// Euler-Maruyama with truncation for negative variance
    pub fn simulate_euler<R: Rng>(
        &self,
        rng: &mut R,
        n_steps: usize,
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();

        let mut prices = Vec::with_capacity(n_steps + 1);
        let mut variances = Vec::with_capacity(n_steps + 1);

        prices.push(self.s0);
        variances.push(self.v0);

        let mut s = self.s0;
        let mut v = self.v0;

        for _ in 0..n_steps {
            // Generate correlated Brownian increments
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);

            let dw_s = sqrt_dt * z1;
            let dw_v = sqrt_dt * (self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * z2);

            // Update variance (with truncation)
            let sqrt_v = v.max(0.0).sqrt();
            let dv = self.kappa * (self.theta - v) * dt + self.xi * sqrt_v * dw_v;
            v = (v + dv).max(0.0); // Truncate negative values

            // Update price
            let ds = self.mu * s * dt + sqrt_v * s * dw_s;
            s += ds;
            s = s.max(0.0); // Price cannot be negative

            prices.push(s);
            variances.push(v);
        }

        (prices, variances)
    }

    /// Milstein scheme (more accurate for volatility)
    pub fn simulate_milstein<R: Rng>(
        &self,
        rng: &mut R,
        n_steps: usize,
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();

        let mut prices = Vec::with_capacity(n_steps + 1);
        let mut variances = Vec::with_capacity(n_steps + 1);

        prices.push(self.s0);
        variances.push(self.v0);

        let mut s = self.s0;
        let mut v = self.v0;

        for _ in 0..n_steps {
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);

            let dw_s = sqrt_dt * z1;
            let dw_v = sqrt_dt * (self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * z2);

            let sqrt_v = v.max(0.0).sqrt();

            // Milstein correction for variance process
            // d(√V)/dV = 1/(2√V), so we add (ξ²/4)(dW² - dt)
            let dv = self.kappa * (self.theta - v) * dt
                   + self.xi * sqrt_v * dw_v
                   + 0.25 * self.xi * self.xi * (dw_v * dw_v - dt);
            v = (v + dv).max(0.0);

            let ds = self.mu * s * dt + sqrt_v * s * dw_s
                   + 0.5 * v * s * (dw_s * dw_s - dt);
            s += ds;
            s = s.max(0.0);

            prices.push(s);
            variances.push(v);
        }

        (prices, variances)
    }
}
```

---

## 1.6 Jump-Diffusion Models

### Motivation

GBM doesn't explain:
- Sudden large price movements (gaps)
- Fat tails in return distributions
- Volatility clustering

### Merton's Model (1976)

$$dS_t = (\mu - \lambda \kappa) S_t \, dt + \sigma S_t \, dW_t + S_t \, dJ_t$$

where:
- $J_t = \sum_{i=1}^{N_t} (Y_i - 1)$ — compound Poisson process
- $N_t$ — Poisson process with intensity $\lambda$
- $Y_i$ — jump sizes (typically $\ln Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$)
- $\kappa = \mathbb{E}[Y - 1] = e^{\mu_J + \sigma_J^2/2} - 1$

### Rust Implementation

```rust
use rand_distr::Poisson;

/// Merton Jump-Diffusion Model
pub struct MertonJumpDiffusion {
    pub s0: f64,
    pub mu: f64,
    pub sigma: f64,
    /// Jump intensity (average number of jumps per year)
    pub lambda: f64,
    /// Mean of log jump size
    pub mu_j: f64,
    /// Standard deviation of log jump size
    pub sigma_j: f64,
}

impl MertonJumpDiffusion {
    pub fn new(s0: f64, mu: f64, sigma: f64, lambda: f64, mu_j: f64, sigma_j: f64) -> Self {
        assert!(s0 > 0.0);
        assert!(sigma >= 0.0);
        assert!(lambda >= 0.0);
        assert!(sigma_j >= 0.0);

        Self { s0, mu, sigma, lambda, mu_j, sigma_j }
    }

    /// Expected jump size: E[Y - 1]
    fn kappa(&self) -> f64 {
        (self.mu_j + 0.5 * self.sigma_j * self.sigma_j).exp() - 1.0
    }

    /// Simulate a path
    pub fn simulate<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let normal_j = Normal::new(self.mu_j, self.sigma_j).unwrap();
        let sqrt_dt = dt.sqrt();

        // Adjusted drift
        let mu_adj = self.mu - self.lambda * self.kappa();

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.s0);

        let mut s = self.s0;

        for _ in 0..n_steps {
            // Diffusion part
            let dw = sqrt_dt * normal.sample(rng);
            let diffusion = mu_adj * s * dt + self.sigma * s * dw;

            // Jump part: number of jumps in dt
            let n_jumps = if self.lambda * dt < 30.0 {
                // Use Poisson distribution for small λ*dt
                Poisson::new(self.lambda * dt).unwrap().sample(rng) as usize
            } else {
                // For large λ*dt, approximate with normal
                let n = (self.lambda * dt + (self.lambda * dt).sqrt() * normal.sample(rng))
                    .round().max(0.0) as usize;
                n
            };

            // Total jump multiplier
            let mut jump_mult = 1.0;
            for _ in 0..n_jumps {
                let log_y = normal_j.sample(rng);
                jump_mult *= log_y.exp();
            }

            s = (s + diffusion) * jump_mult;
            s = s.max(1e-10); // Protect against zero

            path.push(s);
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_diffusion() {
        let jd = MertonJumpDiffusion::new(
            100.0,  // S0
            0.1,    // mu
            0.2,    // sigma
            2.0,    // lambda (2 jumps per year on average)
            -0.05,  // mu_j (jumps tend to be negative)
            0.1,    // sigma_j
        );

        let mut rng = rand::thread_rng();
        let path = jd.simulate(&mut rng, 252, 1.0 / 252.0);

        assert_eq!(path.len(), 253);
        assert!(path.iter().all(|&x| x > 0.0), "All prices should be positive");
    }
}
```

---

## 1.7 Monte Carlo Methods

### Basic Monte Carlo

To estimate $\mathbb{E}[f(S_T)]$:

$$\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} f(S_T^{(i)})$$

Standard error: $SE = \frac{\hat{\sigma}}{\sqrt{N}}$

### Variance Reduction: Antithetic Variates

If $Z \sim \mathcal{N}(0,1)$, then $-Z \sim \mathcal{N}(0,1)$ as well.

For each path, we generate a paired "antithetic" path:

```rust
/// Monte Carlo with antithetic variates
pub fn monte_carlo_antithetic<F, R>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> (f64, f64) // (estimate, standard error)
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
        // Antithetic path
        let s2 = gbm.s0 * (drift - vol_sqrt_t * z).exp();

        // Average over pair
        let avg_payoff = (payoff(s1) + payoff(s2)) / 2.0;

        sum += avg_payoff;
        sum_sq += avg_payoff * avg_payoff;
    }

    let mean = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - mean * mean;
    let std_error = (variance / n_paths as f64).sqrt();

    (mean, std_error)
}
```

### Variance Reduction: Control Variates

If we know $\mathbb{E}[Y]$ analytically, we use:

$$\hat{\mu}_{CV} = \frac{1}{N} \sum_{i=1}^{N} (f(S_T^{(i)}) - c(Y^{(i)} - \mathbb{E}[Y]))$$

where $c$ is chosen to minimize variance.

```rust
/// Monte Carlo with control variates
pub fn monte_carlo_control_variate<F, R>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> (f64, f64)
where
    F: Fn(f64) -> f64,
    R: Rng,
{
    let normal = Normal::new(0.0, 1.0).unwrap();
    let drift = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * t;
    let vol_sqrt_t = gbm.sigma * t.sqrt();

    // Control variable: S_T
    // E[S_T] = S_0 * exp(μ*T)
    let expected_s = gbm.s0 * (gbm.mu * t).exp();

    let mut payoffs = Vec::with_capacity(n_paths);
    let mut controls = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        let z = normal.sample(rng);
        let s_t = gbm.s0 * (drift + vol_sqrt_t * z).exp();
        payoffs.push(payoff(s_t));
        controls.push(s_t);
    }

    // Optimal coefficient c = Cov(payoff, control) / Var(control)
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

    let c = cov / var_control;

    // Adjusted payoffs
    let adjusted: Vec<f64> = payoffs.iter()
        .zip(controls.iter())
        .map(|(&p, &ctrl)| p - c * (ctrl - expected_s))
        .collect();

    let mean = adjusted.iter().sum::<f64>() / n_paths as f64;
    let variance: f64 = adjusted.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (n_paths - 1) as f64;
    let std_error = (variance / n_paths as f64).sqrt();

    (mean, std_error)
}
```

---

## 1.8 SIMD Optimization

### Vectorizing Monte Carlo

For maximum performance, we use SIMD:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized GBM generation
/// Processes 4 paths simultaneously (AVX)
#[cfg(target_arch = "x86_64")]
pub unsafe fn gbm_simd_4paths(
    s0: f64,
    drift: f64,  // (μ - σ²/2) * dt
    vol_sqrt_dt: f64,  // σ * √dt
    random_normals: &[f64],  // length n_steps * 4
    n_steps: usize,
) -> [f64; 4] {
    let mut log_s = _mm256_set1_pd(s0.ln());
    let drift_vec = _mm256_set1_pd(drift);
    let vol_vec = _mm256_set1_pd(vol_sqrt_dt);

    for step in 0..n_steps {
        // Load 4 random numbers
        let z = _mm256_loadu_pd(random_normals.as_ptr().add(step * 4));

        // log_s += drift + vol * z
        let increment = _mm256_fmadd_pd(vol_vec, z, drift_vec);
        log_s = _mm256_add_pd(log_s, increment);
    }

    // Exponentiate (approximate)
    let mut result = [0.0f64; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), log_s);

    for r in &mut result {
        *r = r.exp();
    }

    result
}

/// High-level interface for SIMD Monte Carlo
pub fn monte_carlo_simd(
    gbm: &GeometricBrownianMotion,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
) -> Vec<f64> {
    use rayon::prelude::*;

    let drift = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * dt;
    let vol_sqrt_dt = gbm.sigma * dt.sqrt();

    // Round up to multiple of 4
    let n_batches = (n_paths + 3) / 4;

    (0..n_batches)
        .into_par_iter()
        .flat_map(|_| {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, 1.0).unwrap();

            // Generate random numbers for 4 paths
            let randoms: Vec<f64> = (0..n_steps * 4)
                .map(|_| normal.sample(&mut rng))
                .collect();

            #[cfg(target_arch = "x86_64")]
            unsafe {
                gbm_simd_4paths(gbm.s0, drift, vol_sqrt_dt, &randoms, n_steps).to_vec()
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                // Fallback for non-x86 architectures
                (0..4).map(|i| {
                    let mut log_s = gbm.s0.ln();
                    for step in 0..n_steps {
                        log_s += drift + vol_sqrt_dt * randoms[step * 4 + i];
                    }
                    log_s.exp()
                }).collect::<Vec<_>>()
            }
        })
        .take(n_paths)
        .collect()
}
```

---

## 1.9 Benchmarks

### Performance Comparison

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_gbm(c: &mut Criterion) {
    let gbm = GeometricBrownianMotion::new(100.0, 0.1, 0.2);
    let n_steps = 252;
    let dt = 1.0 / 252.0;

    let mut group = c.benchmark_group("GBM Simulation");

    for n_paths in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("Sequential", n_paths),
            n_paths,
            |b, &n| {
                b.iter(|| {
                    let mut rng = rand::thread_rng();
                    (0..n).map(|_| {
                        gbm.generate_path_exact(&mut rng, n_steps, dt)
                    }).collect::<Vec<_>>()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Parallel (rayon)", n_paths),
            n_paths,
            |b, &n| {
                b.iter(|| {
                    use rayon::prelude::*;
                    (0..n).into_par_iter().map(|_| {
                        let mut rng = rand::thread_rng();
                        gbm.generate_path_exact(&mut rng, n_steps, dt)
                    }).collect::<Vec<_>>()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SIMD + Parallel", n_paths),
            n_paths,
            |b, &n| {
                b.iter(|| {
                    monte_carlo_simd(&gbm, n, n_steps, dt)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_gbm);
criterion_main!(benches);
```

### Typical Results

| Method | 10K paths | 100K paths | 1M paths |
|--------|-----------|------------|----------|
| Sequential | 45 ms | 450 ms | 4.5 s |
| Parallel (8 cores) | 8 ms | 70 ms | 650 ms |
| SIMD + Parallel | 3 ms | 25 ms | 220 ms |

---

## 1.10 Practical Exercises

### Exercise 1: Verify Brownian Motion Properties

Write tests to verify:
1. $\mathbb{E}[W_t] = 0$
2. $\text{Var}[W_t] = t$
3. Quadratic variation $[W,W]_T \approx T$
4. Independence of increments (correlation test)

### Exercise 2: Calibrate Heston to Implied Volatility

Given option prices (or implied volatilities), calibrate Heston model parameters:
- Implement the objective function (MSE over volatility surface)
- Use Levenberg-Marquardt or Differential Evolution
- Assess the quality of fit

### Exercise 3: Option Pricing via Monte Carlo

Price a European call option:
- Implement all three variance reduction methods
- Compare convergence rates
- Plot error vs. number of simulations

### Exercise 4: Real-time Simulator

Create a streaming price simulator:
- Lock-free ring buffer for output
- Latency < 1μs per tick
- WebSocket interface for consumers

---

## Conclusion

In this chapter, we learned:

1. **Brownian motion** — the foundation of continuous-time stochastic processes
2. **Ito's integral** — a tool for working with stochastic processes
3. **Ito's lemma** — the "chain rule" of stochastic calculus
4. **Price models**: GBM, Heston, Jump-Diffusion
5. **Numerical methods**: Euler-Maruyama, Milstein
6. **Optimization**: SIMD, parallelism, variance reduction

These concepts are the foundation for understanding more complex market microstructure models (Chapter 2) and portfolio optimization (Chapter 3).

---

## References

1. Shreve S.E. "Stochastic Calculus for Finance II: Continuous-Time Models" (2004)
2. Gatheral J. "The Volatility Surface: A Practitioner's Guide" (2006)
3. Cont R., Tankov P. "Financial Modelling with Jump Processes" (2003)
4. Glasserman P. "Monte Carlo Methods in Financial Engineering" (2003)
5. Heston S. "A Closed-Form Solution for Options with Stochastic Volatility" (1993)
