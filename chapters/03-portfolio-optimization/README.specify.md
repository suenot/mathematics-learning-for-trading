# Глава 3: Портфельная оптимизация и риск-менеджмент

## Метаданные
- **Уровень сложности**: Средний → Продвинутый
- **Предварительные требования**: Линейная алгебра, статистика, основы оптимизации
- **Языки реализации**: Rust (численные методы), Python (cvxpy для сравнения), Julia
- **Расчётный объём**: 90-120 страниц

---

## Цели главы

1. Освоить классическую теорию Марковица и её ограничения
2. Изучить современные методы оценки ковариационных матриц
3. Реализовать robust portfolio optimization методы
4. Понять практические аспекты: transaction costs, constraints, rebalancing
5. Внедрить risk measures: VaR, CVaR, Maximum Drawdown

---

## Научная база

### Классические работы
1. **Markowitz H.** (1952) "Portfolio Selection" — Journal of Finance
2. **Sharpe W.** (1964) "Capital Asset Prices" — CAPM
3. **Ledoit O., Wolf M.** (2004) "A well-conditioned estimator for large-dimensional covariance matrices"
4. **Michaud R.** (1989) "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?"

### Современные исследования (2023-2025)
5. **Lee J. et al.** (2025) "Return Prediction for Mean-Variance Portfolio Selection: How Decision-Focused Learning Shapes Forecasting Models" — arXiv:2409.09684
6. **Goldberg L.R. et al.** (2024) "Portfolio optimisation via strategy-specific eigenvector shrinkage" — FSU Math
7. **Portfolio Optimization with Robust Covariance** (2024) — Gerber MAD, Nested Clustering
8. **Shrinkage covariance estimators in FX markets** (2024) — Cogent Economics & Finance

### Risk Management
9. **Rockafellar R.T., Uryasev S.** (2000) "Optimization of Conditional Value-at-Risk"
10. **López de Prado M.** (2018) "Advances in Financial Machine Learning" — HRP, risk parity

---

## Структура главы

### 3.1 Теория Марковица: Mean-Variance Optimization

**3.1.1 Математическая формулировка**

**Базовая задача:**
```
min   w'Σw                    # Minimize variance
s.t.  w'μ ≥ r_target          # Return constraint  
      w'1 = 1                  # Fully invested
      w ≥ 0                    # Long-only (optional)
```

**Лагранжиан и аналитическое решение:**
```
w* = Σ⁻¹(λμ + γ1)
```
где λ, γ — множители Лагранжа

**Efficient Frontier:**
```
Параметрическое решение: w(r) для разных target returns r
```

**3.1.2 Критическая линия алгоритм (Critical Line Algorithm)**
- Markowitz's original algorithm
- Piecewise linear efficient frontier
- Handling of inequality constraints

**3.1.3 Реализация на Rust**

```rust
use nalgebra::{DMatrix, DVector};

pub struct MeanVarianceOptimizer {
    expected_returns: DVector<f64>,
    covariance: DMatrix<f64>,
    constraints: Vec<Constraint>,
}

#[derive(Clone)]
pub enum Constraint {
    FullyInvested,                    // sum(w) = 1
    LongOnly,                         // w >= 0
    BoxBounds { lower: f64, upper: f64 }, // l <= w <= u
    SectorLimit { sector: Vec<usize>, max_weight: f64 },
}

impl MeanVarianceOptimizer {
    pub fn efficient_frontier(&self, n_points: usize) -> Vec<(f64, f64, DVector<f64>)> {
        // Returns (risk, return, weights) for each point
    }
    
    pub fn minimum_variance(&self) -> DVector<f64>;
    pub fn maximum_sharpe(&self, risk_free: f64) -> DVector<f64>;
    pub fn target_return(&self, r: f64) -> DVector<f64>;
}
```

---

### 3.2 Оценка ковариационной матрицы

**3.2.1 Проблема Curse of Dimensionality**
```
Для N активов и T наблюдений:
- Параметров: N(N+1)/2
- Если N > T: матрица сингулярна
- Типично: N = 500 акций, T = 252 (1 год) → проблема!
```

**3.2.2 Sample Covariance**
```
Σ_sample = (1/(T-1)) Σ (rₜ - r̄)(rₜ - r̄)'
```
**Проблемы:**
- High estimation error
- Extreme eigenvalues (некоторые слишком большие, некоторые слишком маленькие)
- Unstable portfolios

**3.2.3 Shrinkage Estimators**

**Ledoit-Wolf Shrinkage:**
```
Σ_shrunk = δ·F + (1-δ)·Σ_sample

где F — target matrix (например, diagonal или constant correlation)
    δ — optimal shrinkage intensity (estimated from data)
```

**Формула для δ:**
```rust
pub fn ledoit_wolf_shrinkage(returns: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
    let n = returns.nrows();  // T observations
    let p = returns.ncols();  // N assets
    
    let sample_cov = compute_sample_covariance(returns);
    let target = compute_shrinkage_target(&sample_cov);  // e.g., constant correlation
    
    // Optimal shrinkage intensity
    let delta = compute_optimal_delta(returns, &sample_cov, &target);
    
    let shrunk = delta * target + (1.0 - delta) * sample_cov;
    (shrunk, delta)
}
```

**3.2.4 Factor Models**

**Single Factor (CAPM):**
```
rᵢ = αᵢ + βᵢ·rₘ + εᵢ
Σ = β·β'·σ²ₘ + D  (где D — diagonal)
```

**Multi-Factor (Fama-French, etc.):**
```
r = α + B·f + ε
Σ = B·Σ_f·B' + D
```

**Преимущества:**
- Меньше параметров: O(NK) вместо O(N²)
- Более стабильные оценки
- Экономическая интерпретация

**3.2.5 Random Matrix Theory (RMT)**

**Marchenko-Pastur distribution:**
```
Для random matrix X (TxN) с iid entries:
Eigenvalues of XX'/T follow MP distribution с boundaries:
λ_± = σ²(1 ± √(N/T))²
```

**Eigenvalue Clipping:**
- Все eigenvalues < λ₊ считаем "noise"
- Заменяем на среднее или shrink to target

```rust
pub fn denoise_covariance(cov: &DMatrix<f64>, ratio: f64) -> DMatrix<f64> {
    let eigen = cov.symmetric_eigendecomposition();
    let (eigenvalues, eigenvectors) = (eigen.eigenvalues, eigen.eigenvectors);
    
    // Marchenko-Pastur threshold
    let lambda_plus = (1.0 + ratio.sqrt()).powi(2);
    
    // Clip eigenvalues below threshold
    let clipped: DVector<f64> = eigenvalues.map(|l| {
        if l < lambda_plus { 
            eigenvalues.mean()  // Or shrink
        } else { 
            l 
        }
    });
    
    eigenvectors * DMatrix::from_diagonal(&clipped) * eigenvectors.transpose()
}
```

**3.2.6 Gerber Covariance**

Robust covariance based on co-movements:
- MAD (Median Absolute Deviation) based
- STD based
- Более устойчива к outliers

---

### 3.3 Advanced Portfolio Optimization

**3.3.1 Robust Optimization**

**Uncertainty sets:**
```
min  max   w'Σw
 w   Σ∈U

где U — uncertainty set для covariance
```

**Black-Litterman Model:**
```
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹π + P'Ω⁻¹Q]

где:
  π — equilibrium returns (from CAPM)
  P — "pick" matrix (which assets have views)
  Q — view returns
  Ω — view uncertainty
```

**3.3.2 Risk Parity**

**Концепция:** равный вклад в риск от каждого актива

```
RC_i = w_i · (Σw)_i / √(w'Σw)

Цель: RC_1 = RC_2 = ... = RC_N = 1/N · σ_p
```

**Оптимизация:**
```rust
pub fn risk_parity(cov: &DMatrix<f64>, tol: f64, max_iter: usize) -> DVector<f64> {
    let n = cov.nrows();
    let mut w = DVector::from_element(n, 1.0 / n as f64);
    
    for _ in 0..max_iter {
        let sigma_w = cov * &w;
        let portfolio_vol = (w.dot(&sigma_w)).sqrt();
        
        // Risk contributions
        let rc: DVector<f64> = w.component_mul(&sigma_w) / portfolio_vol;
        
        // Target: equal risk
        let target_rc = portfolio_vol / n as f64;
        
        // Update weights
        let adjustment = rc.map(|r| target_rc / r);
        w = w.component_mul(&adjustment);
        w /= w.sum();  // Normalize
        
        if (rc - DVector::from_element(n, target_rc)).norm() < tol {
            break;
        }
    }
    w
}
```

**3.3.3 Hierarchical Risk Parity (HRP)**

López de Prado (2016):
1. Tree clustering на основе correlation matrix
2. Quasi-diagonalization
3. Recursive bisection для аллокации

```rust
pub fn hierarchical_risk_parity(returns: &DMatrix<f64>) -> DVector<f64> {
    let corr = compute_correlation(returns);
    let dist = correlation_to_distance(&corr);  // d = sqrt(0.5*(1-ρ))
    
    // Hierarchical clustering
    let linkage = single_linkage_clustering(&dist);
    let order = quasi_diagonalization(&linkage);
    
    // Recursive bisection
    let weights = recursive_bisection(&returns.select_columns(&order));
    
    // Reorder to original
    reorder_weights(weights, &order)
}
```

**3.3.4 Nested Clustered Optimization (NCO)**

Комбинация:
1. Cluster highly correlated assets
2. Optimize within clusters
3. Optimize across clusters

Преимущества:
- Уменьшает propagation ошибок оценки
- Более стабильные портфели

---

### 3.4 Risk Measures

**3.4.1 Value at Risk (VaR)**

```
VaR_α = -inf{x : P(L ≤ x) ≥ α}

Интерпретация: "С вероятностью α потери не превысят VaR"
```

**Методы расчёта:**
| Метод | Плюсы | Минусы |
|-------|-------|--------|
| Historical | Простой, model-free | Нужно много данных |
| Parametric (Gaussian) | Быстрый | Не захватывает fat tails |
| Monte Carlo | Flexible | Computationally intensive |

**3.4.2 Conditional VaR (CVaR / Expected Shortfall)**

```
CVaR_α = E[L | L > VaR_α]

Интерпретация: "Средние потери в worst α% случаев"
```

**Преимущества над VaR:**
- Coherent risk measure (subadditive)
- Convex optimization → можно использовать в constraints

**Оптимизация с CVaR constraint:**
```
min   w'μ
s.t.  CVaR_α(w) ≤ c
      w'1 = 1
```

Reformulation (Rockafellar-Uryasev):
```rust
pub fn cvar_optimization(
    returns: &DMatrix<f64>,
    alpha: f64,
    cvar_limit: f64,
) -> DVector<f64> {
    // Linear programming formulation
    // min  -w'μ
    // s.t. ζ + (1/(α·T)) Σᵢ zᵢ ≤ cvar_limit
    //      zᵢ ≥ -w'rᵢ - ζ  for all i
    //      zᵢ ≥ 0
    //      w'1 = 1
    
    // Use LP solver
}
```

**3.4.3 Maximum Drawdown**

```
DD(t) = (Peak_t - Value_t) / Peak_t
MDD = max DD(t)
```

**Calmar Ratio:**
```
Calmar = Annualized Return / Max Drawdown
```

---

### 3.5 Практические аспекты

**3.5.1 Transaction Costs**

```
min   w'Σw - λ·w'μ + γ·Σ|wᵢ - wᵢ⁰|·cᵢ

где:
  wᵢ⁰ — текущие веса
  cᵢ — transaction cost для актива i
```

**3.5.2 Turnover Constraints**

```
Σ|wᵢ - wᵢ⁰| ≤ τ  (maximum turnover)
```

**3.5.3 Rebalancing Frequency**

Trade-off:
- Frequent: closer to optimal, higher costs
- Rare: lower costs, drift from optimal

**Threshold-based rebalancing:**
```rust
pub fn should_rebalance(current: &DVector<f64>, target: &DVector<f64>, threshold: f64) -> bool {
    let deviation = (current - target).norm();
    deviation > threshold
}
```

**3.5.4 Portfolio Constraints**

Common constraints:
- Long-only: wᵢ ≥ 0
- Box bounds: lᵢ ≤ wᵢ ≤ uᵢ
- Sector limits: Σᵢ∈S wᵢ ≤ s_max
- Factor exposure limits
- Cardinality: |{i : wᵢ > 0}| ≤ K

---

### 3.6 Multi-Period Optimization

**3.6.1 Myopic vs Dynamic**

Myopic: оптимизируем каждый период отдельно
Dynamic: учитываем будущие периоды

**3.6.2 Stochastic Programming**

```
min E[Σₜ c'ₜxₜ]
s.t. constraints для каждого сценария
```

**3.6.3 Model Predictive Control (MPC)**

- Rolling horizon optimization
- Re-optimize каждый период с новой информацией

---

## Инструментарий

### Rust crates
```toml
[dependencies]
nalgebra = "0.32"           # Linear algebra
ndarray = "0.15"            # N-dimensional arrays
ndarray-linalg = "0.16"     # LAPACK bindings
clarabel = "0.7"            # Conic optimization solver
minilp = "0.2"              # LP solver
argmin = "0.8"              # Optimization framework
rand = "0.8"
statrs = "0.16"             # Statistics
```

### Python (для сравнения и валидации)
```python
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pyportfolioopt
```

### Julia
```julia
using Convex
using JuMP
using PortfolioOptimization
using RiskMeasures
```

---

## Практические задания

### Задание 3.1: Efficient Frontier Construction
**Цель:** Построить efficient frontier для портфеля из 50 акций
- Данные: 3 года дневных returns
- Сравнить sample cov vs Ledoit-Wolf
- Visualize: frontier, weights distribution

### Задание 3.2: Covariance Estimation Comparison
**Цель:** Benchmark разных методов estimation
- Methods: sample, shrinkage, factor, RMT, Gerber
- Metrics: out-of-sample portfolio variance, stability of weights
- Rolling window analysis

### Задание 3.3: Risk Parity Implementation
**Цель:** Реализовать и протестировать risk parity
- Compare: equal weight, min variance, risk parity, HRP
- Performance metrics: Sharpe, Calmar, turnover

### Задание 3.4: CVaR Optimization
**Цель:** Portfolio optimization с CVaR constraint
- Implement Rockafellar-Uryasev reformulation
- Backtest: compare with VaR constraint

### Задание 3.5: Crypto Portfolio Optimization
**Цель:** Применить методы к crypto assets
- Challenge: high correlation, fat tails, non-stationarity
- Test robustness of different approaches

---

## Критерии оценки

1. **Численная стабильность**: правильная работа с ill-conditioned matrices
2. **Out-of-sample performance**: реальная производительность, не in-sample fit
3. **Scalability**: работа с 1000+ активов
4. **Practical constraints**: корректная обработка bounds и других ограничений

---

## Связи с другими главами

| Глава | Связь |
|-------|-------|
| 01-stochastic-calculus | Return processes, volatility estimation |
| 02-market-microstructure | Transaction costs, execution |
| 04-ml-time-series | Return prediction, covariance forecasting |
| 05-low-latency-systems | Real-time portfolio rebalancing |

---

## Рекомендуемая литература

### Учебники
1. Fabozzi F., Markowitz H. "The Theory and Practice of Investment Management"
2. Meucci A. "Risk and Asset Allocation"
3. López de Prado M. "Advances in Financial Machine Learning"

### Papers
1. DeMiguel V. et al. (2009) "Optimal Versus Naive Diversification" — 1/N vs optimized
2. Ledoit O., Wolf M. (2017) "Nonlinear Shrinkage of the Covariance Matrix"

---

## Заметки по написанию

- Начинать с простых примеров (2-3 актива) для интуиции
- Показать реальные проблемы: estimation error, turnover, концентрация
- Включить failure cases: когда оптимизация даёт плохие результаты
- Code примеры с реальными данными (crypto или equities)
- Сравнительные таблицы методов с pros/cons
