# Chapter 3: Portfolio Optimization and Risk Management

## Metadata
- **Difficulty Level**: Intermediate → Advanced
- **Prerequisites**: Linear algebra, statistics, optimization basics
- **Implementation Language**: Rust
- **Estimated Length**: 90-120 pages

---

## Introduction

Imagine you have $1 million to invest. How should you distribute this money among different assets? Put everything in Bitcoin? Split equally among 10 stocks? Or use mathematics to find the optimal allocation?

Portfolio optimization is the science of distributing capital among assets to maximize expected return for a given level of risk (or minimize risk for a given return).

In this chapter, we will study:
1. Classical Markowitz theory
2. Modern covariance matrix estimation methods
3. Risk Parity and Hierarchical Risk Parity
4. Risk measures: VaR, CVaR, Maximum Drawdown
5. Practical aspects: transaction costs, rebalancing

---

## 3.1 Markowitz Theory: Mean-Variance Optimization

### 3.1.1 Historical Background

In 1952, Harry Markowitz published "Portfolio Selection," which revolutionized finance. He received the Nobel Prize in 1990 for this work.

**Key idea**: an investor should consider not only expected returns but also risk (volatility) and relationships between assets (correlations).

### 3.1.2 Mathematical Formulation

Let's say we have **N** assets. For each asset:
- **μᵢ** — expected return
- **σᵢ** — standard deviation (volatility)
- **ρᵢⱼ** — correlation between assets i and j

A portfolio is described by a weight vector **w = (w₁, w₂, ..., wₙ)**, where wᵢ is the fraction of capital in asset i.

**Expected portfolio return:**
```
μₚ = Σ wᵢ · μᵢ = w'μ
```

**Portfolio variance (risk):**
```
σₚ² = Σᵢ Σⱼ wᵢ · wⱼ · σᵢ · σⱼ · ρᵢⱼ = w'Σw
```

where **Σ** is the covariance matrix.

### 3.1.3 Optimization Problem

**Minimize risk for a given return:**
```
minimize    w'Σw              (minimize variance)
subject to  w'μ ≥ r_target    (return at least target)
            w'1 = 1           (weights sum to 1)
            w ≥ 0             (optional: long-only)
```

### 3.1.4 Rust Implementation

```rust
use nalgebra::{DMatrix, DVector};

/// Mean-Variance optimization structure
pub struct MeanVarianceOptimizer {
    /// Expected returns of assets
    expected_returns: DVector<f64>,
    /// Covariance matrix
    covariance: DMatrix<f64>,
    /// Number of assets
    n_assets: usize,
}

impl MeanVarianceOptimizer {
    /// Create a new optimizer
    pub fn new(expected_returns: DVector<f64>, covariance: DMatrix<f64>) -> Self {
        let n_assets = expected_returns.len();
        assert_eq!(covariance.nrows(), n_assets);
        assert_eq!(covariance.ncols(), n_assets);

        Self {
            expected_returns,
            covariance,
            n_assets,
        }
    }

    /// Minimum variance portfolio (unconstrained, allows short-selling)
    pub fn minimum_variance_unconstrained(&self) -> DVector<f64> {
        // w* = Σ⁻¹ · 1 / (1' · Σ⁻¹ · 1)
        let ones = DVector::from_element(self.n_assets, 1.0);

        // Solve Σ · x = 1 instead of explicit matrix inversion
        let cov_inv_ones = self.covariance
            .clone()
            .lu()
            .solve(&ones)
            .expect("Covariance matrix must be invertible");

        let sum = cov_inv_ones.sum();
        cov_inv_ones / sum
    }

    /// Maximum Sharpe ratio portfolio
    pub fn maximum_sharpe(&self, risk_free_rate: f64) -> DVector<f64> {
        // Excess returns
        let excess_returns: DVector<f64> = self.expected_returns
            .iter()
            .map(|r| r - risk_free_rate)
            .collect();

        // w* = Σ⁻¹ · (μ - rf) / (1' · Σ⁻¹ · (μ - rf))
        let cov_inv_excess = self.covariance
            .clone()
            .lu()
            .solve(&excess_returns)
            .expect("Covariance matrix must be invertible");

        let sum = cov_inv_excess.sum();
        cov_inv_excess / sum
    }

    /// Calculate portfolio volatility
    pub fn portfolio_volatility(&self, weights: &DVector<f64>) -> f64 {
        let variance = (weights.transpose() * &self.covariance * weights)[(0, 0)];
        variance.sqrt()
    }

    /// Calculate expected portfolio return
    pub fn portfolio_return(&self, weights: &DVector<f64>) -> f64 {
        weights.dot(&self.expected_returns)
    }

    /// Sharpe ratio
    pub fn sharpe_ratio(&self, weights: &DVector<f64>, risk_free_rate: f64) -> f64 {
        let ret = self.portfolio_return(weights);
        let vol = self.portfolio_volatility(weights);
        (ret - risk_free_rate) / vol
    }
}
```

### 3.1.5 Building the Efficient Frontier

The **Efficient Frontier** is the set of portfolios that provide maximum return for each level of risk.

```rust
impl MeanVarianceOptimizer {
    /// Build the efficient frontier
    /// Returns a vector of points (risk, return, weights)
    pub fn efficient_frontier(&self, n_points: usize) -> Vec<EfficientFrontierPoint> {
        let mut points = Vec::with_capacity(n_points);

        // Find minimum variance portfolio
        let min_var_weights = self.minimum_variance_unconstrained();
        let min_return = self.portfolio_return(&min_var_weights);

        // Maximum return — 100% in best asset
        let max_return = self.expected_returns
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Generate points between min and max return
        for i in 0..n_points {
            let target_return = min_return
                + (max_return - min_return) * (i as f64) / ((n_points - 1) as f64);

            // Solve optimization for this target return
            if let Some(weights) = self.optimize_for_target_return(target_return) {
                let risk = self.portfolio_volatility(&weights);
                let ret = self.portfolio_return(&weights);

                points.push(EfficientFrontierPoint {
                    risk,
                    expected_return: ret,
                    weights,
                });
            }
        }

        points
    }

    /// Optimize for a given target return
    fn optimize_for_target_return(&self, target_return: f64) -> Option<DVector<f64>> {
        // Analytical solution using Lagrange multipliers
        // For the unconstrained case (short-selling allowed)

        let n = self.n_assets;
        let ones = DVector::from_element(n, 1.0);

        // Invert covariance matrix
        let lu = self.covariance.clone().lu();
        let cov_inv_ones = lu.solve(&ones)?;
        let cov_inv_mu = lu.solve(&self.expected_returns)?;

        // Calculate coefficients
        let a = ones.dot(&cov_inv_ones);
        let b = ones.dot(&cov_inv_mu);
        let c = self.expected_returns.dot(&cov_inv_mu);
        let d = a * c - b * b;

        // Lagrange multipliers
        let lambda = (c - b * target_return) / d;
        let gamma = (a * target_return - b) / d;

        // Optimal weights
        let weights = &cov_inv_ones * lambda + &cov_inv_mu * gamma;

        Some(weights)
    }
}

/// Point on the efficient frontier
#[derive(Debug, Clone)]
pub struct EfficientFrontierPoint {
    pub risk: f64,
    pub expected_return: f64,
    pub weights: DVector<f64>,
}
```

---

## 3.2 Covariance Matrix Estimation

### 3.2.1 The Dimensionality Problem

A covariance matrix for N assets contains N(N+1)/2 unique parameters.

**Example:**
- 10 assets → 55 parameters
- 100 assets → 5,050 parameters
- 500 assets → 125,250 parameters

If we have T observations (trading days), when N > T the sample covariance matrix becomes singular!

**Typical situation:**
- 500 stocks in portfolio
- 252 trading days per year
- N > T → matrix is not invertible!

### 3.2.2 Sample Covariance

```rust
/// Calculate sample covariance matrix
pub fn sample_covariance(returns: &DMatrix<f64>) -> DMatrix<f64> {
    let n_observations = returns.nrows();
    let n_assets = returns.ncols();

    // Mean for each asset
    let mean: DVector<f64> = returns
        .column_iter()
        .map(|col| col.mean())
        .collect();

    // Centered returns
    let mut centered = returns.clone();
    for mut col in centered.column_iter_mut() {
        let col_mean = col.mean();
        col.iter_mut().for_each(|x| *x -= col_mean);
    }

    // Covariance matrix: (X'X) / (n-1)
    let cov = centered.transpose() * &centered;
    cov / ((n_observations - 1) as f64)
}
```

### 3.2.3 Shrinkage Estimators

**Ledoit-Wolf idea**: combine sample covariance with a "target" structured matrix.

```
Σ_shrunk = δ·F + (1-δ)·Σ_sample
```

where:
- F — target matrix (e.g., diagonal or constant correlation)
- δ — shrinkage intensity (from 0 to 1)
- δ is computed optimally from data

```rust
/// Ledoit-Wolf Shrinkage to diagonal matrix
pub struct LedoitWolfShrinkage;

impl LedoitWolfShrinkage {
    /// Calculate shrunk covariance matrix
    pub fn estimate(returns: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
        let n = returns.nrows() as f64; // T - number of observations
        let p = returns.ncols();        // N - number of assets

        // Sample covariance
        let sample_cov = sample_covariance(returns);

        // Target matrix: diagonal with average variance
        let mean_var = sample_cov.diagonal().mean();
        let target = DMatrix::from_diagonal(
            &DVector::from_element(p, mean_var)
        );

        // Calculate optimal δ
        let delta = Self::optimal_shrinkage_intensity(returns, &sample_cov, &target);

        // Shrunk estimate
        let shrunk = &target * delta + &sample_cov * (1.0 - delta);

        (shrunk, delta)
    }

    fn optimal_shrinkage_intensity(
        returns: &DMatrix<f64>,
        sample_cov: &DMatrix<f64>,
        target: &DMatrix<f64>,
    ) -> f64 {
        let n = returns.nrows() as f64;
        let p = returns.ncols();

        // Mean returns
        let mean: DVector<f64> = returns
            .column_iter()
            .map(|col| col.mean())
            .collect();

        // Calculate Ledoit-Wolf formula components
        let mut sum_pi = 0.0;
        let mut sum_gamma = 0.0;

        for i in 0..p {
            for j in 0..p {
                let s_ij = sample_cov[(i, j)];
                let f_ij = target[(i, j)];

                // pi_{ij} - asymptotic variance of s_{ij}
                let pi_ij: f64 = (0..returns.nrows())
                    .map(|t| {
                        let x_ti = returns[(t, i)] - mean[i];
                        let x_tj = returns[(t, j)] - mean[j];
                        (x_ti * x_tj - s_ij).powi(2)
                    })
                    .sum::<f64>() / n;

                sum_pi += pi_ij;
                sum_gamma += (f_ij - s_ij).powi(2);
            }
        }

        // Optimal shrinkage intensity
        let kappa = (sum_pi / sum_gamma) / n;
        kappa.clamp(0.0, 1.0)
    }
}
```

### 3.2.4 Random Matrix Theory (RMT)

If returns were pure noise (i.i.d. random variables), eigenvalues of the covariance matrix would follow the **Marchenko-Pastur distribution**.

**Distribution boundaries:**
```
λ₊ = σ² · (1 + √(N/T))²
λ₋ = σ² · (1 - √(N/T))²
```

**Idea**: eigenvalues below λ₊ are "noise" and should be filtered out.

```rust
use nalgebra::SymmetricEigen;

/// Denoise covariance matrix using RMT
pub fn denoise_covariance_rmt(
    cov: &DMatrix<f64>,
    ratio: f64  // N/T
) -> DMatrix<f64> {
    // Eigendecomposition
    let eigen = SymmetricEigen::new(cov.clone());
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Marchenko-Pastur threshold
    let sigma_sq = eigenvalues.mean();  // Variance estimate
    let lambda_plus = sigma_sq * (1.0 + ratio.sqrt()).powi(2);

    // "Clean" eigenvalues
    let n = eigenvalues.len();
    let noise_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .filter(|&l| *l < lambda_plus)
        .cloned()
        .collect();

    let mean_noise = if noise_eigenvalues.is_empty() {
        eigenvalues.min()
    } else {
        noise_eigenvalues.iter().sum::<f64>() / noise_eigenvalues.len() as f64
    };

    // Replace noisy eigenvalues with mean
    let cleaned_eigenvalues: DVector<f64> = eigenvalues
        .iter()
        .map(|&l| if l < lambda_plus { mean_noise } else { l })
        .collect();

    // Reconstruct matrix
    let diag = DMatrix::from_diagonal(&cleaned_eigenvalues);
    &eigenvectors * diag * eigenvectors.transpose()
}
```

---

## 3.3 Risk Parity

### 3.3.1 Motivation

Problem with Mean-Variance optimization: weights heavily depend on expected return estimates, which are very unstable.

**Risk Parity** solves a different problem: allocate capital so that each asset contributes **equally to total portfolio risk**.

### 3.3.2 Mathematics

**Asset i's contribution to portfolio risk:**
```
RC_i = w_i · ∂σₚ/∂w_i = w_i · (Σw)_i / σₚ
```

where σₚ = √(w'Σw) is portfolio volatility.

**Risk Parity objective:**
```
RC_1 = RC_2 = ... = RC_N = σₚ / N
```

### 3.3.3 Implementation

```rust
/// Risk Parity optimizer
pub struct RiskParityOptimizer {
    covariance: DMatrix<f64>,
    n_assets: usize,
}

impl RiskParityOptimizer {
    pub fn new(covariance: DMatrix<f64>) -> Self {
        let n_assets = covariance.nrows();
        Self { covariance, n_assets }
    }

    /// Calculate Risk Parity weights
    pub fn optimize(&self, tolerance: f64, max_iterations: usize) -> DVector<f64> {
        let n = self.n_assets;

        // Initial weights: equal
        let mut weights = DVector::from_element(n, 1.0 / n as f64);

        for iteration in 0..max_iterations {
            // Σ · w
            let sigma_w = &self.covariance * &weights;

            // Portfolio volatility
            let portfolio_variance = weights.dot(&sigma_w);
            let portfolio_vol = portfolio_variance.sqrt();

            // Risk contributions
            let risk_contributions: DVector<f64> = weights
                .iter()
                .zip(sigma_w.iter())
                .map(|(&w, &sw)| w * sw / portfolio_vol)
                .collect();

            // Target contribution: equal for all
            let target_rc = portfolio_vol / n as f64;

            // Check convergence
            let max_deviation = risk_contributions
                .iter()
                .map(|&rc| (rc - target_rc).abs())
                .fold(0.0, f64::max);

            if max_deviation < tolerance {
                println!("Converged in {} iterations", iteration + 1);
                break;
            }

            // Update weights
            let adjustment: DVector<f64> = risk_contributions
                .iter()
                .map(|&rc| target_rc / rc)
                .collect();

            weights = weights.component_mul(&adjustment);

            // Normalize (sum = 1)
            let sum = weights.sum();
            weights /= sum;
        }

        weights
    }

    /// Calculate risk contributions for given weights
    pub fn risk_contributions(&self, weights: &DVector<f64>) -> DVector<f64> {
        let sigma_w = &self.covariance * weights;
        let portfolio_vol = (weights.dot(&sigma_w)).sqrt();

        weights
            .iter()
            .zip(sigma_w.iter())
            .map(|(&w, &sw)| w * sw / portfolio_vol)
            .collect()
    }
}
```

---

## 3.4 Hierarchical Risk Parity (HRP)

### 3.4.1 The Idea

Marcos López de Prado (2016) proposed HRP — a method that:
1. Doesn't require covariance matrix inversion
2. Accounts for hierarchical structure of assets
3. Is more robust to estimation errors

### 3.4.2 Algorithm

1. **Clustering**: build a tree based on correlation matrix
2. **Quasi-diagonalization**: reorder assets according to the tree
3. **Recursive bisection**: allocate weights bottom-up

### 3.4.3 Implementation

```rust
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Hierarchical Risk Parity
pub struct HRPOptimizer {
    correlation: DMatrix<f64>,
    covariance: DMatrix<f64>,
    n_assets: usize,
}

impl HRPOptimizer {
    pub fn new(returns: &DMatrix<f64>) -> Self {
        let covariance = sample_covariance(returns);
        let correlation = Self::covariance_to_correlation(&covariance);
        let n_assets = covariance.nrows();

        Self { correlation, covariance, n_assets }
    }

    fn covariance_to_correlation(cov: &DMatrix<f64>) -> DMatrix<f64> {
        let n = cov.nrows();
        let std_devs: Vec<f64> = (0..n)
            .map(|i| cov[(i, i)].sqrt())
            .collect();

        DMatrix::from_fn(n, n, |i, j| {
            cov[(i, j)] / (std_devs[i] * std_devs[j])
        })
    }

    /// Convert correlation to distance
    fn correlation_to_distance(corr: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(corr.nrows(), corr.ncols(), |i, j| {
            ((1.0 - corr[(i, j)]) / 2.0).sqrt()
        })
    }

    /// Calculate HRP weights
    pub fn optimize(&self) -> DVector<f64> {
        // 1. Calculate distance matrix
        let distance = Self::correlation_to_distance(&self.correlation);

        // 2. Hierarchical clustering (single linkage)
        let linkage = self.hierarchical_clustering(&distance);

        // 3. Quasi-diagonalization
        let order = self.quasi_diagonalization(&linkage);

        // 4. Recursive bisection
        let weights = self.recursive_bisection(&order);

        weights
    }

    /// Single linkage clustering
    fn hierarchical_clustering(&self, distance: &DMatrix<f64>) -> Vec<(usize, usize, f64)> {
        let n = self.n_assets;
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut linkage = Vec::new();

        // Copy of distance matrix
        let mut dist = distance.clone();

        while clusters.len() > 1 {
            // Find minimum distance
            let mut min_dist = f64::INFINITY;
            let mut min_i = 0;
            let mut min_j = 0;

            for i in 0..clusters.len() {
                for j in (i+1)..clusters.len() {
                    // Minimum distance between clusters (single linkage)
                    let d = clusters[i].iter()
                        .flat_map(|&a| clusters[j].iter().map(move |&b| dist[(a, b)]))
                        .fold(f64::INFINITY, f64::min);

                    if d < min_dist {
                        min_dist = d;
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            // Merge clusters
            linkage.push((min_i, min_j, min_dist));

            let cluster_j = clusters.remove(min_j);
            clusters[min_i].extend(cluster_j);
        }

        linkage
    }

    /// Reorder assets
    fn quasi_diagonalization(&self, linkage: &[(usize, usize, f64)]) -> Vec<usize> {
        // For simplicity, return order from last cluster
        let n = self.n_assets;
        let mut order: Vec<usize> = (0..n).collect();

        // Sort by correlation with first asset
        order.sort_by(|&a, &b| {
            self.correlation[(a, 0)]
                .partial_cmp(&self.correlation[(b, 0)])
                .unwrap_or(Ordering::Equal)
        });

        order
    }

    /// Recursive bisection
    fn recursive_bisection(&self, order: &[usize]) -> DVector<f64> {
        let n = order.len();
        let mut weights = DVector::from_element(n, 1.0);

        self.bisect(&mut weights, order, 0, n);

        // Normalize
        let sum = weights.sum();
        weights / sum
    }

    fn bisect(&self, weights: &mut DVector<f64>, order: &[usize], start: usize, end: usize) {
        if end - start <= 1 {
            return;
        }

        let mid = (start + end) / 2;

        // Variances of two subgroups
        let var_left = self.cluster_variance(&order[start..mid]);
        let var_right = self.cluster_variance(&order[mid..end]);

        // Allocation inversely proportional to variance
        let alpha = var_right / (var_left + var_right);

        // Scale weights
        for i in start..mid {
            weights[order[i]] *= alpha;
        }
        for i in mid..end {
            weights[order[i]] *= 1.0 - alpha;
        }

        // Recurse
        self.bisect(weights, order, start, mid);
        self.bisect(weights, order, mid, end);
    }

    fn cluster_variance(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        // Inverse-variance weights within cluster
        let inv_vars: Vec<f64> = indices
            .iter()
            .map(|&i| 1.0 / self.covariance[(i, i)])
            .collect();

        let sum_inv_vars: f64 = inv_vars.iter().sum();
        let weights: Vec<f64> = inv_vars.iter().map(|&v| v / sum_inv_vars).collect();

        // Cluster variance
        let mut variance = 0.0;
        for (i, &idx_i) in indices.iter().enumerate() {
            for (j, &idx_j) in indices.iter().enumerate() {
                variance += weights[i] * weights[j] * self.covariance[(idx_i, idx_j)];
            }
        }

        variance
    }
}
```

---

## 3.5 Risk Measures

### 3.5.1 Value at Risk (VaR)

**VaR** is the maximum loss that won't be exceeded with a given probability (e.g., 95% or 99%).

```
VaR_α = -inf{x : P(Loss ≤ x) ≥ α}
```

**Interpretation**: "With 95% probability, our losses won't exceed VaR₉₅%"

```rust
/// Value at Risk calculator
pub struct VaRCalculator;

impl VaRCalculator {
    /// Historical VaR
    pub fn historical(returns: &[f64], confidence: f64) -> f64 {
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        -sorted[index]  // VaR is positive for losses
    }

    /// Parametric VaR (assuming normal distribution)
    pub fn parametric(mean: f64, std_dev: f64, confidence: f64) -> f64 {
        // z-score for given confidence level
        let z = Self::normal_quantile(1.0 - confidence);
        -(mean + z * std_dev)
    }

    /// Monte Carlo VaR
    pub fn monte_carlo(
        mean: f64,
        std_dev: f64,
        confidence: f64,
        n_simulations: usize,
    ) -> f64 {
        use rand::Rng;
        use rand_distr::Normal;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();

        let mut returns: Vec<f64> = (0..n_simulations)
            .map(|_| rng.sample(normal))
            .collect();

        Self::historical(&returns, confidence)
    }

    /// Standard normal quantile
    fn normal_quantile(p: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.383577518672690e2,
            -3.066479806614716e1,
            2.506628277459239e0,
        ];
        let b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
        ];
        let c = [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838e0,
            -2.549732539343734e0,
            4.374664141464968e0,
            2.938163982698783e0,
        ];
        let d = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996e0,
            3.754408661907416e0,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
    }
}
```

### 3.5.2 Conditional VaR (CVaR / Expected Shortfall)

**CVaR** is the average loss in the worst α% of cases.

```
CVaR_α = E[Loss | Loss > VaR_α]
```

**Advantages of CVaR over VaR:**
- CVaR is a coherent risk measure (subadditive)
- CVaR can be used in convex optimization

```rust
impl VaRCalculator {
    /// Historical CVaR (Expected Shortfall)
    pub fn historical_cvar(returns: &[f64], confidence: f64) -> f64 {
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let cutoff_index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;

        // Average of worst cases
        let tail_losses: f64 = sorted[..=cutoff_index].iter().sum();
        -tail_losses / (cutoff_index + 1) as f64
    }

    /// Parametric CVaR
    pub fn parametric_cvar(mean: f64, std_dev: f64, confidence: f64) -> f64 {
        let z = Self::normal_quantile(1.0 - confidence);
        let pdf_z = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();

        -(mean - std_dev * pdf_z / (1.0 - confidence))
    }
}
```

### 3.5.3 Maximum Drawdown

**Drawdown** is the decline from a peak in portfolio value.

```rust
/// Calculate Maximum Drawdown
pub fn maximum_drawdown(prices: &[f64]) -> (f64, usize, usize) {
    let mut max_dd = 0.0;
    let mut peak_idx = 0;
    let mut trough_idx = 0;

    let mut running_max = prices[0];
    let mut running_max_idx = 0;

    for (i, &price) in prices.iter().enumerate() {
        if price > running_max {
            running_max = price;
            running_max_idx = i;
        }

        let drawdown = (running_max - price) / running_max;

        if drawdown > max_dd {
            max_dd = drawdown;
            peak_idx = running_max_idx;
            trough_idx = i;
        }
    }

    (max_dd, peak_idx, trough_idx)
}

/// Calmar Ratio = Annualized Return / Max Drawdown
pub fn calmar_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    // Cumulative prices from returns
    let mut prices = vec![1.0];
    for &r in returns {
        prices.push(prices.last().unwrap() * (1.0 + r));
    }

    let (max_dd, _, _) = maximum_drawdown(&prices);

    // Annualized return
    let total_return = prices.last().unwrap() / prices[0] - 1.0;
    let n_periods = returns.len() as f64;
    let annualized_return = (1.0 + total_return).powf(periods_per_year / n_periods) - 1.0;

    if max_dd > 0.0 {
        annualized_return / max_dd
    } else {
        f64::INFINITY
    }
}
```

---

## 3.6 Transaction Costs and Rebalancing

### 3.6.1 Accounting for Transaction Costs

```rust
/// Optimization with transaction costs
pub struct TransactionCostOptimizer {
    expected_returns: DVector<f64>,
    covariance: DMatrix<f64>,
    current_weights: DVector<f64>,
    transaction_costs: DVector<f64>,  // Cost per unit traded
}

impl TransactionCostOptimizer {
    /// Optimization with costs
    ///
    /// Problem:
    /// min  w'Σw - λ·w'μ + γ·Σ|wᵢ - wᵢ⁰|·cᵢ
    pub fn optimize(
        &self,
        risk_aversion: f64,
        cost_aversion: f64,
    ) -> DVector<f64> {
        let n = self.expected_returns.len();

        // Simplified approach: gradient descent
        let mut weights = self.current_weights.clone();
        let learning_rate = 0.01;
        let max_iterations = 1000;

        for _ in 0..max_iterations {
            // Gradient of objective function
            let grad_variance = 2.0 * &self.covariance * &weights;
            let grad_return = -risk_aversion * &self.expected_returns;

            // Gradient of transaction costs (subgradient)
            let trade = &weights - &self.current_weights;
            let grad_cost: DVector<f64> = trade
                .iter()
                .zip(self.transaction_costs.iter())
                .map(|(&t, &c)| cost_aversion * c * t.signum())
                .collect();

            let gradient = grad_variance + grad_return + grad_cost;

            // Update weights
            weights -= learning_rate * gradient;

            // Project to feasible set (sum = 1, weights >= 0)
            weights = Self::project_to_simplex(&weights);
        }

        weights
    }

    /// Project onto simplex (sum = 1, all >= 0)
    fn project_to_simplex(v: &DVector<f64>) -> DVector<f64> {
        let n = v.len();
        let mut u: Vec<f64> = v.iter().cloned().collect();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap());  // Sort descending

        let mut cumsum = 0.0;
        let mut rho = 0;

        for (i, &u_i) in u.iter().enumerate() {
            cumsum += u_i;
            if u_i + (1.0 - cumsum) / (i + 1) as f64 > 0.0 {
                rho = i;
            }
        }

        let theta = (u[..=rho].iter().sum::<f64>() - 1.0) / (rho + 1) as f64;

        v.map(|x| (x - theta).max(0.0))
    }
}
```

### 3.6.2 When to Rebalance?

```rust
/// Rebalancing strategies
pub enum RebalanceStrategy {
    /// Fixed period (monthly, quarterly)
    Periodic { days: usize },
    /// Threshold-based
    Threshold { max_deviation: f64 },
    /// Combined
    Combined { days: usize, max_deviation: f64 },
}

impl RebalanceStrategy {
    pub fn should_rebalance(
        &self,
        current_weights: &DVector<f64>,
        target_weights: &DVector<f64>,
        days_since_last: usize,
    ) -> bool {
        match self {
            RebalanceStrategy::Periodic { days } => {
                days_since_last >= *days
            }
            RebalanceStrategy::Threshold { max_deviation } => {
                let deviation = (current_weights - target_weights).norm();
                deviation > *max_deviation
            }
            RebalanceStrategy::Combined { days, max_deviation } => {
                let deviation = (current_weights - target_weights).norm();
                days_since_last >= *days || deviation > *max_deviation
            }
        }
    }
}
```

---

## 3.7 Practical Example: Crypto Portfolio Optimization

```rust
use nalgebra::{DMatrix, DVector};

fn main() {
    // Historical returns (daily) for 5 cryptocurrencies
    // BTC, ETH, SOL, BNB, ADA
    let returns_data = vec![
        // ... returns data
    ];

    let n_assets = 5;
    let n_observations = returns_data.len() / n_assets;

    let returns = DMatrix::from_vec(n_observations, n_assets, returns_data);

    // 1. Sample covariance
    let sample_cov = sample_covariance(&returns);
    println!("Sample covariance:\n{:.4}", sample_cov);

    // 2. Ledoit-Wolf shrinkage
    let (shrunk_cov, delta) = LedoitWolfShrinkage::estimate(&returns);
    println!("\nShrinkage intensity δ = {:.4}", delta);

    // 3. Expected returns (mean)
    let expected_returns: DVector<f64> = returns
        .column_iter()
        .map(|col| col.mean() * 365.0)  // Annualized
        .collect();

    println!("\nExpected annual returns:");
    for (i, &r) in expected_returns.iter().enumerate() {
        println!("  Asset {}: {:.2}%", i + 1, r * 100.0);
    }

    // 4. Mean-Variance optimization
    let mv_optimizer = MeanVarianceOptimizer::new(
        expected_returns.clone(),
        shrunk_cov.clone(),
    );

    let min_var_weights = mv_optimizer.minimum_variance_unconstrained();
    println!("\nMinimum variance portfolio:");
    for (i, &w) in min_var_weights.iter().enumerate() {
        println!("  Asset {}: {:.2}%", i + 1, w * 100.0);
    }

    let max_sharpe_weights = mv_optimizer.maximum_sharpe(0.05);  // rf = 5%
    println!("\nMaximum Sharpe portfolio:");
    for (i, &w) in max_sharpe_weights.iter().enumerate() {
        println!("  Asset {}: {:.2}%", i + 1, w * 100.0);
    }

    // 5. Risk Parity
    let rp_optimizer = RiskParityOptimizer::new(shrunk_cov.clone());
    let rp_weights = rp_optimizer.optimize(1e-8, 1000);

    println!("\nRisk Parity portfolio:");
    for (i, &w) in rp_weights.iter().enumerate() {
        println!("  Asset {}: {:.2}%", i + 1, w * 100.0);
    }

    let risk_contribs = rp_optimizer.risk_contributions(&rp_weights);
    println!("\nRisk contributions:");
    for (i, &rc) in risk_contribs.iter().enumerate() {
        println!("  Asset {}: {:.2}%", i + 1, rc / risk_contribs.sum() * 100.0);
    }

    // 6. HRP
    let hrp_optimizer = HRPOptimizer::new(&returns);
    let hrp_weights = hrp_optimizer.optimize();

    println!("\nHRP portfolio:");
    for (i, &w) in hrp_weights.iter().enumerate() {
        println!("  Asset {}: {:.2}%", i + 1, w * 100.0);
    }

    // 7. Compare metrics
    println!("\n=== Portfolio Comparison ===");
    println!("{:<15} {:>10} {:>10} {:>10}",
        "Portfolio", "Return", "Vol", "Sharpe");

    for (name, weights) in [
        ("Min Var", &min_var_weights),
        ("Max Sharpe", &max_sharpe_weights),
        ("Risk Parity", &rp_weights),
        ("HRP", &hrp_weights),
    ] {
        let ret = mv_optimizer.portfolio_return(weights);
        let vol = mv_optimizer.portfolio_volatility(weights);
        let sharpe = mv_optimizer.sharpe_ratio(weights, 0.05);

        println!("{:<15} {:>9.2}% {:>9.2}% {:>10.2}",
            name, ret * 100.0, vol * 100.0, sharpe);
    }
}
```

---

## Conclusion

In this chapter, we studied:

1. **Classical Markowitz theory** — how to balance return and risk
2. **Covariance estimation problem** — shrinkage and RMT to fight noise
3. **Risk Parity** — an alternative approach without return forecasts
4. **HRP** — hierarchical method robust to estimation errors
5. **Risk measures** — VaR, CVaR, Maximum Drawdown
6. **Practical aspects** — transaction costs and rebalancing

### Key Takeaways

1. **Expected return estimates are unstable** — methods like Risk Parity avoid this problem
2. **Sample covariance overestimates risk** — use shrinkage or RMT
3. **Diversification works** — but only among uncorrelated assets
4. **Transaction costs matter** — account for them in optimization

---

## Exercises

1. Implement optimization with constraints (long-only, box bounds)
2. Add optimization with CVaR constraint
3. Test methods on real cryptocurrency data
4. Compare out-of-sample performance of different methods
5. Implement rolling window backtest

---

## Recommended Reading

1. Markowitz H. (1952) "Portfolio Selection" — Journal of Finance
2. Ledoit O., Wolf M. (2004) "A well-conditioned estimator for large-dimensional covariance matrices"
3. López de Prado M. (2016) "Building Diversified Portfolios that Outperform Out of Sample"
4. Rockafellar R.T., Uryasev S. (2000) "Optimization of Conditional Value-at-Risk"
5. Meucci A. "Risk and Asset Allocation" — comprehensive textbook

---

*Next chapter: [04. Machine Learning for Time Series](../04-ml-time-series/README.en.md)*
