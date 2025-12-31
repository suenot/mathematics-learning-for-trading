# Глава 1: Стохастическое исчисление для алготрейдинга

## Метаданные
- **Уровень сложности**: Продвинутый
- **Предварительные требования**: Математический анализ, теория вероятностей, линейная алгебра
- **Языки реализации**: Rust (основной), Julia, Python (для сравнения)
- **Расчётный объём**: 80-120 страниц

---

## Цели главы

1. Построить математический фундамент для моделирования ценовых процессов
2. Освоить инструментарий стохастического исчисления для практических задач трейдинга
3. Реализовать численные методы в Rust с zero-allocation архитектурой
4. Понять связь между математическими моделями и реальным поведением рынков

---

## Научная база

### Фундаментальные работы
1. **Itô K.** (1944) "Stochastic Integral" — основа стохастического исчисления
2. **Black F., Scholes M.** (1973) "The Pricing of Options and Corporate Liabilities" — применение SDE в финансах
3. **Merton R.C.** (1976) "Option pricing when underlying stock returns are discontinuous" — jump-diffusion модели
4. **Cont R.** (2011) "Statistical modeling of high-frequency financial data" — IEEE Signal Processing Magazine

### Современные исследования (2023-2025)
5. **Cont R., Degond P., Xuan L.** (2025) "A Mathematical Framework for Modelling Order Book Dynamics" — SIAM J. Financial Mathematics
6. **Avellaneda M., Stoikov S.** (2008) → Extensions 2024: "Logarithmic regret in ergodic Avellaneda-Stoikov market making" — arXiv:2409.02025
7. **Hambly B., Kalsi J., Newbury J.** (2020) "From Microscopic to Macroscopic Models: Reflected SPDEs" — Applied Mathematical Finance

---

## Структура главы

### 1.1 Броуновское движение (Wiener Process)
**Теория:**
- Определение и свойства: непрерывность траекторий, независимые приращения
- Марковское свойство и сильное марковское свойство
- Квадратичная вариация: `[W,W]_t = t`
- Martingale property
- Reflection principle и hitting times

**Практика в Rust:**
```rust
// Генерация траекторий с использованием SIMD
// Box-Muller vs Ziggurat algorithm
// Lock-free реализация для параллельных симуляций
```

**Задания:**
- Реализовать генератор траекторий с cache-friendly memory layout
- Benchmark: Rust vs C++ vs Julia производительность
- Статистическая верификация свойств (тесты на нормальность приращений)

---

### 1.2 Стохастические дифференциальные уравнения (SDE)

**Теория:**
- Интеграл Ито vs интеграл Стратоновича
- Лемма Ито (Ito's Lemma) — "цепное правило" стохастического исчисления
- Формула Ито для многомерного случая
- Существование и единственность решений (условия Липшица)

**Ключевые формулы:**
```
dS_t = μS_t dt + σS_t dW_t          # Geometric Brownian Motion
dX_t = θ(μ - X_t)dt + σ dW_t        # Ornstein-Uhlenbeck
dV_t = κ(θ - V_t)dt + ξ√V_t dW_t    # CIR Process (volatility)
```

**Реализация на Rust:**
- `nalgebra` для матричных операций
- SIMD оптимизация для Monte Carlo
- Trait-based абстракции для разных SDE

---

### 1.3 Модели ценовых процессов

**1.3.1 Geometric Brownian Motion (GBM)**
- Аналитическое решение: `S_t = S_0 exp((μ - σ²/2)t + σW_t)`
- Применение: baseline модель, не захватывает fat tails

**1.3.2 Jump-Diffusion Models (Merton)**
```
dS_t = (μ - λk)S_t dt + σS_t dW_t + S_t dJ_t
```
- Compound Poisson процесс для скачков
- Калибровка к implied volatility surface

**1.3.3 Stochastic Volatility Models**
- **Heston Model:**
  ```
  dS_t = μS_t dt + √V_t S_t dW^S_t
  dV_t = κ(θ - V_t)dt + ξ√V_t dW^V_t
  Corr(dW^S, dW^V) = ρ
  ```
- **SABR Model** для FX и rates
- Численные методы: Euler-Maruyama, Milstein scheme

**1.3.4 Rough Volatility**
- Fractional Brownian Motion с H < 0.5
- Характерные особенности crypto markets
- Реализация: Cholesky decomposition для корреляций

---

### 1.4 Численные методы для SDE

**Схемы дискретизации:**
| Метод | Порядок сходимости (strong) | Порядок сходимости (weak) |
|-------|----------------------------|---------------------------|
| Euler-Maruyama | 0.5 | 1.0 |
| Milstein | 1.0 | 1.0 |
| Runge-Kutta (SDE) | 1.5 | 2.0 |

**Реализация на Rust:**
```rust
trait SDESolver<const N: usize> {
    fn step(&self, state: &mut [f64; N], dt: f64, dw: &[f64; N]);
    fn solve(&self, initial: [f64; N], t_end: f64, n_steps: usize) -> Vec<[f64; N]>;
}
```

**Monte Carlo методы:**
- Variance reduction: antithetic variates, control variates
- Importance sampling для rare events
- Quasi-Monte Carlo (Sobol sequences) — `sobol` crate

---

### 1.5 Практические приложения

**1.5.1 Симуляция limit order book dynamics**
- Связь с Hawkes processes (глава 2)
- Price impact как SDE

**1.5.2 Greeks calculation (sensitivity analysis)**
- Pathwise derivatives vs Likelihood Ratio Method
- Automatic Differentiation в Rust: `autodiff` crate

**1.5.3 Risk metrics**
- VaR и Expected Shortfall через симуляции
- Extreme Value Theory для tail risk

---

## Инструментарий

### Rust crates
```toml
[dependencies]
rand = "0.8"              # RNG
rand_distr = "0.4"        # Distributions (Normal, etc.)
nalgebra = "0.32"         # Linear algebra
ndarray = "0.15"          # N-dimensional arrays
rayon = "1.8"             # Parallel iteration
criterion = "0.5"         # Benchmarking
```

### Julia packages (для валидации и прототипирования)
```julia
using DifferentialEquations  # SDE solvers
using StochasticDiffEq       # Specialized SDE methods
using Distributions          # Statistical distributions
using QuantLib              # Financial primitives
```

### Сравнительный анализ
- Python (NumPy/SciPy) — baseline для сравнения производительности
- Показать speedup 10-100x в Rust для Monte Carlo симуляций

---

## Практические задания

### Задание 1.1: Генератор траекторий GBM
**Цель:** Реализовать высокопроизводительный генератор в Rust
- SIMD оптимизация с `packed_simd` или `std::simd`
- Параллелизация с `rayon`
- Benchmark: генерация 10^8 траекторий < 1 секунда

### Задание 1.2: Calibration Heston Model
**Цель:** Калибровать параметры Heston к опционным данным
- Входные данные: implied volatility surface
- Алгоритм: Levenberg-Marquardt или Differential Evolution
- Целевая функция: MSE по volatility surface

### Задание 1.3: Real-time price process simulation
**Цель:** Создать streaming симулятор с предсказуемой латентностью
- Lock-free ring buffer для output
- Детерминированная латентность < 1μs per tick
- Интеграция с WebSocket для live data feed

### Задание 1.4: Option Greeks via AAD
**Цель:** Automatic Adjoint Differentiation для Greeks
- Реализовать tape-based AAD
- Сравнить с finite differences по точности и скорости

---

## Критерии оценки качества главы

1. **Математическая строгость**: все утверждения с доказательствами или ссылками
2. **Практическая применимость**: каждая концепция с working кодом
3. **Производительность**: все примеры оптимизированы, с benchmarks
4. **Связь с реальностью**: примеры на реальных данных crypto/equity markets
5. **Тестируемость**: unit tests для всех алгоритмов

---

## Связи с другими главами

| Глава | Связь |
|-------|-------|
| 02-market-microstructure | SDE как baseline для более сложных point process моделей |
| 03-portfolio-optimization | Covariance estimation требует понимания процессов |
| 04-ml-time-series | Feature engineering из SDE параметров |
| 05-low-latency-systems | Real-time simulation архитектура |

---

## Рекомендуемая литература

### Учебники
1. Shreve S. "Stochastic Calculus for Finance II: Continuous-Time Models"
2. Gatheral J. "The Volatility Surface: A Practitioner's Guide"
3. Cont R., Tankov P. "Financial Modelling with Jump Processes"

### Код и репозитории
1. `rust-lang/portable-simd` — SIMD в Rust
2. `JuliaDiffEq/DifferentialEquations.jl` — reference реализации
3. QuantLib C++ — industry standard для сравнения

---

## Заметки по написанию

- Начинать каждую секцию с мотивирующего примера из реального трейдинга
- Включать "Intuition" блоки для сложных концепций
- Код должен быть production-ready, не toy examples
- Добавить "Common Pitfalls" секции (численная стабильность, etc.)
- Интерактивные Jupyter notebooks для Julia/Python частей
