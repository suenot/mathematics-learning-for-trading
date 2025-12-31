# Глава 3: Портфельная оптимизация и риск-менеджмент

## Метаданные
- **Уровень сложности**: Средний → Продвинутый
- **Предварительные требования**: Линейная алгебра, статистика, основы оптимизации
- **Язык реализации**: Rust
- **Расчётный объём**: 90-120 страниц

---

## Введение

Представьте, что у вас есть 1 миллион долларов для инвестирования. Как распределить эти деньги между разными активами? Положить всё в Bitcoin? Разделить поровну между 10 акциями? Или использовать математику, чтобы найти оптимальное распределение?

Портфельная оптимизация — это наука о том, как распределить капитал между активами так, чтобы получить максимальную доходность при заданном уровне риска (или минимизировать риск при заданной доходности).

В этой главе мы изучим:
1. Классическую теорию Марковица
2. Современные методы оценки ковариационных матриц
3. Risk Parity и Hierarchical Risk Parity
4. Меры риска: VaR, CVaR, Maximum Drawdown
5. Практические аспекты: транзакционные издержки, ребалансировка

---

## 3.1 Теория Марковица: Mean-Variance Optimization

### 3.1.1 Историческая справка

В 1952 году Гарри Марковиц опубликовал статью "Portfolio Selection", которая произвела революцию в финансах. За эту работу он получил Нобелевскую премию в 1990 году.

**Ключевая идея**: инвестор должен учитывать не только ожидаемую доходность активов, но и их риск (волатильность) и взаимосвязи между ними (корреляции).

### 3.1.2 Математическая формулировка

Пусть у нас есть **N** активов. Для каждого актива:
- **μᵢ** — ожидаемая доходность
- **σᵢ** — стандартное отклонение (волатильность)
- **ρᵢⱼ** — корреляция между активами i и j

Портфель описывается вектором весов **w = (w₁, w₂, ..., wₙ)**, где wᵢ — доля капитала в активе i.

**Ожидаемая доходность портфеля:**
```
μₚ = Σ wᵢ · μᵢ = w'μ
```

**Дисперсия (риск) портфеля:**
```
σₚ² = Σᵢ Σⱼ wᵢ · wⱼ · σᵢ · σⱼ · ρᵢⱼ = w'Σw
```

где **Σ** — ковариационная матрица.

### 3.1.3 Задача оптимизации

**Минимизация риска при заданной доходности:**
```
minimize    w'Σw              (минимизируем дисперсию)
subject to  w'μ ≥ r_target    (доходность не меньше целевой)
            w'1 = 1           (веса в сумме дают 1)
            w ≥ 0             (опционально: только длинные позиции)
```

### 3.1.4 Реализация на Rust

```rust
use nalgebra::{DMatrix, DVector};

/// Структура для Mean-Variance оптимизации
pub struct MeanVarianceOptimizer {
    /// Ожидаемые доходности активов
    expected_returns: DVector<f64>,
    /// Ковариационная матрица
    covariance: DMatrix<f64>,
    /// Количество активов
    n_assets: usize,
}

impl MeanVarianceOptimizer {
    /// Создание нового оптимизатора
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

    /// Портфель минимальной дисперсии (без ограничений на short-selling)
    pub fn minimum_variance_unconstrained(&self) -> DVector<f64> {
        // w* = Σ⁻¹ · 1 / (1' · Σ⁻¹ · 1)
        let ones = DVector::from_element(self.n_assets, 1.0);

        // Решаем систему Σ · x = 1 вместо явного обращения матрицы
        let cov_inv_ones = self.covariance
            .clone()
            .lu()
            .solve(&ones)
            .expect("Ковариационная матрица должна быть обратимой");

        let sum = cov_inv_ones.sum();
        cov_inv_ones / sum
    }

    /// Портфель с максимальным коэффициентом Шарпа
    pub fn maximum_sharpe(&self, risk_free_rate: f64) -> DVector<f64> {
        // Избыточная доходность
        let excess_returns: DVector<f64> = self.expected_returns
            .iter()
            .map(|r| r - risk_free_rate)
            .collect();

        // w* = Σ⁻¹ · (μ - rₓ) / (1' · Σ⁻¹ · (μ - rₓ))
        let cov_inv_excess = self.covariance
            .clone()
            .lu()
            .solve(&excess_returns)
            .expect("Ковариационная матрица должна быть обратимой");

        let sum = cov_inv_excess.sum();
        cov_inv_excess / sum
    }

    /// Расчёт волатильности портфеля
    pub fn portfolio_volatility(&self, weights: &DVector<f64>) -> f64 {
        let variance = (weights.transpose() * &self.covariance * weights)[(0, 0)];
        variance.sqrt()
    }

    /// Расчёт ожидаемой доходности портфеля
    pub fn portfolio_return(&self, weights: &DVector<f64>) -> f64 {
        weights.dot(&self.expected_returns)
    }

    /// Коэффициент Шарпа
    pub fn sharpe_ratio(&self, weights: &DVector<f64>, risk_free_rate: f64) -> f64 {
        let ret = self.portfolio_return(weights);
        let vol = self.portfolio_volatility(weights);
        (ret - risk_free_rate) / vol
    }
}
```

### 3.1.5 Построение эффективной границы

**Эффективная граница (Efficient Frontier)** — это множество портфелей, которые дают максимальную доходность при каждом уровне риска.

```rust
impl MeanVarianceOptimizer {
    /// Построение эффективной границы
    /// Возвращает вектор точек (риск, доходность, веса)
    pub fn efficient_frontier(&self, n_points: usize) -> Vec<EfficientFrontierPoint> {
        let mut points = Vec::with_capacity(n_points);

        // Находим портфель минимальной дисперсии
        let min_var_weights = self.minimum_variance_unconstrained();
        let min_return = self.portfolio_return(&min_var_weights);

        // Максимальная доходность — 100% в лучшем активе
        let max_return = self.expected_returns
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Генерируем точки между min и max доходностью
        for i in 0..n_points {
            let target_return = min_return
                + (max_return - min_return) * (i as f64) / ((n_points - 1) as f64);

            // Решаем задачу оптимизации для данной целевой доходности
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

    /// Оптимизация для заданной целевой доходности
    fn optimize_for_target_return(&self, target_return: f64) -> Option<DVector<f64>> {
        // Аналитическое решение через множители Лагранжа
        // Для случая без ограничений на short-selling

        let n = self.n_assets;
        let ones = DVector::from_element(n, 1.0);

        // Обращение ковариационной матрицы
        let lu = self.covariance.clone().lu();
        let cov_inv_ones = lu.solve(&ones)?;
        let cov_inv_mu = lu.solve(&self.expected_returns)?;

        // Вычисляем коэффициенты
        let a = ones.dot(&cov_inv_ones);
        let b = ones.dot(&cov_inv_mu);
        let c = self.expected_returns.dot(&cov_inv_mu);
        let d = a * c - b * b;

        // Множители Лагранжа
        let lambda = (c - b * target_return) / d;
        let gamma = (a * target_return - b) / d;

        // Оптимальные веса
        let weights = &cov_inv_ones * lambda + &cov_inv_mu * gamma;

        Some(weights)
    }
}

/// Точка на эффективной границе
#[derive(Debug, Clone)]
pub struct EfficientFrontierPoint {
    pub risk: f64,
    pub expected_return: f64,
    pub weights: DVector<f64>,
}
```

---

## 3.2 Оценка ковариационной матрицы

### 3.2.1 Проблема размерности

Ковариационная матрица для N активов содержит N(N+1)/2 уникальных параметров.

**Пример:**
- 10 активов → 55 параметров
- 100 активов → 5,050 параметров
- 500 активов → 125,250 параметров

Если у нас T наблюдений (дней торгов), то при N > T выборочная ковариационная матрица становится сингулярной!

**Типичная ситуация:**
- 500 акций в портфеле
- 252 торговых дня в году
- N > T → матрица необратима!

### 3.2.2 Выборочная ковариация

```rust
/// Расчёт выборочной ковариационной матрицы
pub fn sample_covariance(returns: &DMatrix<f64>) -> DMatrix<f64> {
    let n_observations = returns.nrows();
    let n_assets = returns.ncols();

    // Среднее по каждому активу
    let mean: DVector<f64> = returns
        .column_iter()
        .map(|col| col.mean())
        .collect();

    // Центрированные доходности
    let mut centered = returns.clone();
    for mut col in centered.column_iter_mut() {
        let col_mean = col.mean();
        col.iter_mut().for_each(|x| *x -= col_mean);
    }

    // Ковариационная матрица: (X'X) / (n-1)
    let cov = centered.transpose() * &centered;
    cov / ((n_observations - 1) as f64)
}
```

### 3.2.3 Shrinkage Estimators (Оценки сжатия)

**Идея Ledoit-Wolf**: комбинируем выборочную ковариацию с "целевой" структурированной матрицей.

```
Σ_shrunk = δ·F + (1-δ)·Σ_sample
```

где:
- F — целевая матрица (например, диагональная или с постоянной корреляцией)
- δ — интенсивность сжатия (от 0 до 1)
- δ вычисляется оптимально из данных

```rust
/// Ledoit-Wolf Shrinkage к диагональной матрице
pub struct LedoitWolfShrinkage;

impl LedoitWolfShrinkage {
    /// Вычисление сжатой ковариационной матрицы
    pub fn estimate(returns: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
        let n = returns.nrows() as f64; // T - количество наблюдений
        let p = returns.ncols();        // N - количество активов

        // Выборочная ковариация
        let sample_cov = sample_covariance(returns);

        // Целевая матрица: диагональная с усреднённой дисперсией
        let mean_var = sample_cov.diagonal().mean();
        let target = DMatrix::from_diagonal(
            &DVector::from_element(p, mean_var)
        );

        // Вычисление оптимального δ
        let delta = Self::optimal_shrinkage_intensity(returns, &sample_cov, &target);

        // Сжатая оценка
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

        // Среднее доходностей
        let mean: DVector<f64> = returns
            .column_iter()
            .map(|col| col.mean())
            .collect();

        // Вычисляем компоненты формулы Ledoit-Wolf
        let mut sum_pi = 0.0;
        let mut sum_gamma = 0.0;

        for i in 0..p {
            for j in 0..p {
                let s_ij = sample_cov[(i, j)];
                let f_ij = target[(i, j)];

                // pi_{ij} - асимптотическая дисперсия s_{ij}
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

        // Оптимальная интенсивность сжатия
        let kappa = (sum_pi / sum_gamma) / n;
        kappa.clamp(0.0, 1.0)
    }
}
```

### 3.2.4 Random Matrix Theory (Теория случайных матриц)

Если доходности были бы чистым шумом (i.i.d. случайные величины), собственные значения ковариационной матрицы следовали бы распределению **Марченко-Пастура**.

**Границы распределения:**
```
λ₊ = σ² · (1 + √(N/T))²
λ₋ = σ² · (1 - √(N/T))²
```

**Идея**: собственные значения ниже λ₊ — это "шум", их нужно отфильтровать.

```rust
use nalgebra::SymmetricEigen;

/// Denoise ковариационной матрицы с помощью RMT
pub fn denoise_covariance_rmt(
    cov: &DMatrix<f64>,
    ratio: f64  // N/T
) -> DMatrix<f64> {
    // Eigendecomposition
    let eigen = SymmetricEigen::new(cov.clone());
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Порог Марченко-Пастура
    let sigma_sq = eigenvalues.mean();  // Оценка дисперсии
    let lambda_plus = sigma_sq * (1.0 + ratio.sqrt()).powi(2);

    // "Очистка" собственных значений
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

    // Заменяем шумовые собственные значения на среднее
    let cleaned_eigenvalues: DVector<f64> = eigenvalues
        .iter()
        .map(|&l| if l < lambda_plus { mean_noise } else { l })
        .collect();

    // Восстанавливаем матрицу
    let diag = DMatrix::from_diagonal(&cleaned_eigenvalues);
    &eigenvectors * diag * eigenvectors.transpose()
}
```

---

## 3.3 Risk Parity (Паритет рисков)

### 3.3.1 Мотивация

Проблема Mean-Variance оптимизации: веса сильно зависят от оценок ожидаемой доходности, которые очень нестабильны.

**Risk Parity** решает другую задачу: распределить капитал так, чтобы каждый актив вносил **одинаковый вклад в общий риск** портфеля.

### 3.3.2 Математика

**Вклад актива i в риск портфеля:**
```
RC_i = w_i · ∂σₚ/∂w_i = w_i · (Σw)_i / σₚ
```

где σₚ = √(w'Σw) — волатильность портфеля.

**Цель Risk Parity:**
```
RC_1 = RC_2 = ... = RC_N = σₚ / N
```

### 3.3.3 Реализация

```rust
/// Risk Parity оптимизатор
pub struct RiskParityOptimizer {
    covariance: DMatrix<f64>,
    n_assets: usize,
}

impl RiskParityOptimizer {
    pub fn new(covariance: DMatrix<f64>) -> Self {
        let n_assets = covariance.nrows();
        Self { covariance, n_assets }
    }

    /// Вычисление весов Risk Parity
    pub fn optimize(&self, tolerance: f64, max_iterations: usize) -> DVector<f64> {
        let n = self.n_assets;

        // Начальные веса: равные
        let mut weights = DVector::from_element(n, 1.0 / n as f64);

        for iteration in 0..max_iterations {
            // Σ · w
            let sigma_w = &self.covariance * &weights;

            // Волатильность портфеля
            let portfolio_variance = weights.dot(&sigma_w);
            let portfolio_vol = portfolio_variance.sqrt();

            // Вклады в риск
            let risk_contributions: DVector<f64> = weights
                .iter()
                .zip(sigma_w.iter())
                .map(|(&w, &sw)| w * sw / portfolio_vol)
                .collect();

            // Целевой вклад: равный для всех
            let target_rc = portfolio_vol / n as f64;

            // Проверка сходимости
            let max_deviation = risk_contributions
                .iter()
                .map(|&rc| (rc - target_rc).abs())
                .fold(0.0, f64::max);

            if max_deviation < tolerance {
                println!("Сошлось за {} итераций", iteration + 1);
                break;
            }

            // Обновление весов
            let adjustment: DVector<f64> = risk_contributions
                .iter()
                .map(|&rc| target_rc / rc)
                .collect();

            weights = weights.component_mul(&adjustment);

            // Нормализация (сумма = 1)
            let sum = weights.sum();
            weights /= sum;
        }

        weights
    }

    /// Вычисление вкладов в риск для заданных весов
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

### 3.4.1 Идея

Marcos López de Prado (2016) предложил HRP — метод, который:
1. Не требует обращения ковариационной матрицы
2. Учитывает иерархическую структуру активов
3. Более устойчив к ошибкам оценивания

### 3.4.2 Алгоритм

1. **Кластеризация**: строим дерево на основе корреляционной матрицы
2. **Квази-диагонализация**: переупорядочиваем активы по дереву
3. **Рекурсивная бисекция**: распределяем веса снизу вверх

### 3.4.3 Реализация

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

    /// Преобразование корреляции в расстояние
    fn correlation_to_distance(corr: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::from_fn(corr.nrows(), corr.ncols(), |i, j| {
            ((1.0 - corr[(i, j)]) / 2.0).sqrt()
        })
    }

    /// Вычисление весов HRP
    pub fn optimize(&self) -> DVector<f64> {
        // 1. Вычисляем матрицу расстояний
        let distance = Self::correlation_to_distance(&self.correlation);

        // 2. Иерархическая кластеризация (single linkage)
        let linkage = self.hierarchical_clustering(&distance);

        // 3. Квази-диагонализация
        let order = self.quasi_diagonalization(&linkage);

        // 4. Рекурсивная бисекция
        let weights = self.recursive_bisection(&order);

        weights
    }

    /// Single linkage clustering
    fn hierarchical_clustering(&self, distance: &DMatrix<f64>) -> Vec<(usize, usize, f64)> {
        let n = self.n_assets;
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut linkage = Vec::new();

        // Копия матрицы расстояний
        let mut dist = distance.clone();

        while clusters.len() > 1 {
            // Находим минимальное расстояние
            let mut min_dist = f64::INFINITY;
            let mut min_i = 0;
            let mut min_j = 0;

            for i in 0..clusters.len() {
                for j in (i+1)..clusters.len() {
                    // Минимальное расстояние между кластерами (single linkage)
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

            // Объединяем кластеры
            linkage.push((min_i, min_j, min_dist));

            let cluster_j = clusters.remove(min_j);
            clusters[min_i].extend(cluster_j);
        }

        linkage
    }

    /// Переупорядочивание активов
    fn quasi_diagonalization(&self, linkage: &[(usize, usize, f64)]) -> Vec<usize> {
        // Для простоты возвращаем порядок из последнего кластера
        let n = self.n_assets;
        let mut order: Vec<usize> = (0..n).collect();

        // Сортируем по корреляции с первым активом
        order.sort_by(|&a, &b| {
            self.correlation[(a, 0)]
                .partial_cmp(&self.correlation[(b, 0)])
                .unwrap_or(Ordering::Equal)
        });

        order
    }

    /// Рекурсивная бисекция
    fn recursive_bisection(&self, order: &[usize]) -> DVector<f64> {
        let n = order.len();
        let mut weights = DVector::from_element(n, 1.0);

        self.bisect(&mut weights, order, 0, n);

        // Нормализация
        let sum = weights.sum();
        weights / sum
    }

    fn bisect(&self, weights: &mut DVector<f64>, order: &[usize], start: usize, end: usize) {
        if end - start <= 1 {
            return;
        }

        let mid = (start + end) / 2;

        // Дисперсии двух подгрупп
        let var_left = self.cluster_variance(&order[start..mid]);
        let var_right = self.cluster_variance(&order[mid..end]);

        // Аллокация обратно пропорциональна дисперсии
        let alpha = var_right / (var_left + var_right);

        // Масштабируем веса
        for i in start..mid {
            weights[order[i]] *= alpha;
        }
        for i in mid..end {
            weights[order[i]] *= 1.0 - alpha;
        }

        // Рекурсия
        self.bisect(weights, order, start, mid);
        self.bisect(weights, order, mid, end);
    }

    fn cluster_variance(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        // Inverse-variance веса внутри кластера
        let inv_vars: Vec<f64> = indices
            .iter()
            .map(|&i| 1.0 / self.covariance[(i, i)])
            .collect();

        let sum_inv_vars: f64 = inv_vars.iter().sum();
        let weights: Vec<f64> = inv_vars.iter().map(|&v| v / sum_inv_vars).collect();

        // Дисперсия кластера
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

## 3.5 Меры риска

### 3.5.1 Value at Risk (VaR)

**VaR** — это максимальные потери, которые не будут превышены с заданной вероятностью (например, 95% или 99%).

```
VaR_α = -inf{x : P(Loss ≤ x) ≥ α}
```

**Интерпретация**: "С вероятностью 95% наши потери не превысят VaR₉₅%"

```rust
/// Расчёт Value at Risk
pub struct VaRCalculator;

impl VaRCalculator {
    /// Исторический VaR
    pub fn historical(returns: &[f64], confidence: f64) -> f64 {
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        -sorted[index]  // VaR положительный для потерь
    }

    /// Параметрический VaR (предполагая нормальное распределение)
    pub fn parametric(mean: f64, std_dev: f64, confidence: f64) -> f64 {
        // z-score для заданного уровня доверия
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

    /// Квантиль стандартного нормального распределения
    fn normal_quantile(p: f64) -> f64 {
        // Аппроксимация Abramowitz and Stegun
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

**CVaR** — средние потери в худших α% случаев.

```
CVaR_α = E[Loss | Loss > VaR_α]
```

**Преимущества CVaR над VaR:**
- CVaR — когерентная мера риска (субаддитивная)
- CVaR можно использовать в выпуклой оптимизации

```rust
impl VaRCalculator {
    /// Исторический CVaR (Expected Shortfall)
    pub fn historical_cvar(returns: &[f64], confidence: f64) -> f64 {
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let cutoff_index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;

        // Среднее по худшим случаям
        let tail_losses: f64 = sorted[..=cutoff_index].iter().sum();
        -tail_losses / (cutoff_index + 1) as f64
    }

    /// Параметрический CVaR
    pub fn parametric_cvar(mean: f64, std_dev: f64, confidence: f64) -> f64 {
        let z = Self::normal_quantile(1.0 - confidence);
        let pdf_z = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();

        -(mean - std_dev * pdf_z / (1.0 - confidence))
    }
}
```

### 3.5.3 Maximum Drawdown

**Просадка** — падение стоимости портфеля от пика.

```rust
/// Расчёт Maximum Drawdown
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
    // Кумулятивные цены из доходностей
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

## 3.6 Транзакционные издержки и ребалансировка

### 3.6.1 Учёт транзакционных издержек

```rust
/// Оптимизация с учётом транзакционных издержек
pub struct TransactionCostOptimizer {
    expected_returns: DVector<f64>,
    covariance: DMatrix<f64>,
    current_weights: DVector<f64>,
    transaction_costs: DVector<f64>,  // Издержки на единицу торговли
}

impl TransactionCostOptimizer {
    /// Оптимизация с учётом издержек
    ///
    /// Задача:
    /// min  w'Σw - λ·w'μ + γ·Σ|wᵢ - wᵢ⁰|·cᵢ
    pub fn optimize(
        &self,
        risk_aversion: f64,
        cost_aversion: f64,
    ) -> DVector<f64> {
        let n = self.expected_returns.len();

        // Упрощённый подход: градиентный спуск
        let mut weights = self.current_weights.clone();
        let learning_rate = 0.01;
        let max_iterations = 1000;

        for _ in 0..max_iterations {
            // Градиент целевой функции
            let grad_variance = 2.0 * &self.covariance * &weights;
            let grad_return = -risk_aversion * &self.expected_returns;

            // Градиент транзакционных издержек (субградиент)
            let trade = &weights - &self.current_weights;
            let grad_cost: DVector<f64> = trade
                .iter()
                .zip(self.transaction_costs.iter())
                .map(|(&t, &c)| cost_aversion * c * t.signum())
                .collect();

            let gradient = grad_variance + grad_return + grad_cost;

            // Обновление весов
            weights -= learning_rate * gradient;

            // Проекция на допустимое множество (сумма = 1, веса >= 0)
            weights = Self::project_to_simplex(&weights);
        }

        weights
    }

    /// Проекция на симплекс (сумма = 1, все >= 0)
    fn project_to_simplex(v: &DVector<f64>) -> DVector<f64> {
        let n = v.len();
        let mut u: Vec<f64> = v.iter().cloned().collect();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap());  // Сортировка по убыванию

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

### 3.6.2 Когда ребалансировать?

```rust
/// Стратегии ребалансировки
pub enum RebalanceStrategy {
    /// Фиксированный период (ежемесячно, ежеквартально)
    Periodic { days: usize },
    /// По порогу отклонения
    Threshold { max_deviation: f64 },
    /// Комбинированная
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

## 3.7 Практический пример: оптимизация криптопортфеля

```rust
use nalgebra::{DMatrix, DVector};

fn main() {
    // Исторические доходности (дневные) для 5 криптовалют
    // BTC, ETH, SOL, BNB, ADA
    let returns_data = vec![
        // ... данные доходностей
    ];

    let n_assets = 5;
    let n_observations = returns_data.len() / n_assets;

    let returns = DMatrix::from_vec(n_observations, n_assets, returns_data);

    // 1. Выборочная ковариация
    let sample_cov = sample_covariance(&returns);
    println!("Выборочная ковариация:\n{:.4}", sample_cov);

    // 2. Ledoit-Wolf shrinkage
    let (shrunk_cov, delta) = LedoitWolfShrinkage::estimate(&returns);
    println!("\nИнтенсивность сжатия δ = {:.4}", delta);

    // 3. Ожидаемые доходности (среднее)
    let expected_returns: DVector<f64> = returns
        .column_iter()
        .map(|col| col.mean() * 365.0)  // Annualized
        .collect();

    println!("\nОжидаемые годовые доходности:");
    for (i, &r) in expected_returns.iter().enumerate() {
        println!("  Актив {}: {:.2}%", i + 1, r * 100.0);
    }

    // 4. Mean-Variance оптимизация
    let mv_optimizer = MeanVarianceOptimizer::new(
        expected_returns.clone(),
        shrunk_cov.clone(),
    );

    let min_var_weights = mv_optimizer.minimum_variance_unconstrained();
    println!("\nПортфель минимальной дисперсии:");
    for (i, &w) in min_var_weights.iter().enumerate() {
        println!("  Актив {}: {:.2}%", i + 1, w * 100.0);
    }

    let max_sharpe_weights = mv_optimizer.maximum_sharpe(0.05);  // rf = 5%
    println!("\nПортфель максимального Sharpe:");
    for (i, &w) in max_sharpe_weights.iter().enumerate() {
        println!("  Актив {}: {:.2}%", i + 1, w * 100.0);
    }

    // 5. Risk Parity
    let rp_optimizer = RiskParityOptimizer::new(shrunk_cov.clone());
    let rp_weights = rp_optimizer.optimize(1e-8, 1000);

    println!("\nRisk Parity портфель:");
    for (i, &w) in rp_weights.iter().enumerate() {
        println!("  Актив {}: {:.2}%", i + 1, w * 100.0);
    }

    let risk_contribs = rp_optimizer.risk_contributions(&rp_weights);
    println!("\nВклады в риск:");
    for (i, &rc) in risk_contribs.iter().enumerate() {
        println!("  Актив {}: {:.2}%", i + 1, rc / risk_contribs.sum() * 100.0);
    }

    // 6. HRP
    let hrp_optimizer = HRPOptimizer::new(&returns);
    let hrp_weights = hrp_optimizer.optimize();

    println!("\nHRP портфель:");
    for (i, &w) in hrp_weights.iter().enumerate() {
        println!("  Актив {}: {:.2}%", i + 1, w * 100.0);
    }

    // 7. Сравнение метрик
    println!("\n=== Сравнение портфелей ===");
    println!("{:<15} {:>10} {:>10} {:>10}",
        "Портфель", "Return", "Vol", "Sharpe");

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

## Заключение

В этой главе мы изучили:

1. **Классическую теорию Марковица** — как балансировать доходность и риск
2. **Проблему оценки ковариации** — shrinkage и RMT для борьбы с шумом
3. **Risk Parity** — альтернативный подход без прогнозов доходности
4. **HRP** — иерархический метод, устойчивый к ошибкам оценивания
5. **Меры риска** — VaR, CVaR, Maximum Drawdown
6. **Практические аспекты** — транзакционные издержки и ребалансировка

### Ключевые выводы

1. **Оценки ожидаемой доходности нестабильны** — методы типа Risk Parity обходят эту проблему
2. **Выборочная ковариация переоценивает риск** — используйте shrinkage или RMT
3. **Диверсификация работает** — но только между некоррелированными активами
4. **Транзакционные издержки важны** — учитывайте их при оптимизации

---

## Упражнения

1. Реализуйте оптимизацию с ограничениями (long-only, box bounds)
2. Добавьте оптимизацию с CVaR constraint
3. Протестируйте методы на реальных данных криптовалют
4. Сравните out-of-sample производительность разных методов
5. Реализуйте rolling window бэктест

---

## Рекомендуемая литература

1. Markowitz H. (1952) "Portfolio Selection" — Journal of Finance
2. Ledoit O., Wolf M. (2004) "A well-conditioned estimator for large-dimensional covariance matrices"
3. López de Prado M. (2016) "Building Diversified Portfolios that Outperform Out of Sample"
4. Rockafellar R.T., Uryasev S. (2000) "Optimization of Conditional Value-at-Risk"
5. Meucci A. "Risk and Asset Allocation" — исчерпывающий учебник

---

*Следующая глава: [04. Machine Learning для временных рядов](../04-ml-time-series/README.ru.md)*
