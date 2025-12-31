# Глава 1: Стохастическое исчисление для алготрейдинга

## Введение

Стохастическое исчисление — это математический аппарат, позволяющий описывать системы, которые эволюционируют случайным образом во времени. В контексте трейдинга это именно то, что нам нужно: цены активов меняются непредсказуемо, но подчиняются определённым статистическим закономерностям.

В этой главе мы:
- Изучим броуновское движение — фундамент случайных процессов
- Освоим интеграл Ито и его применение к финансовым моделям
- Реализуем численные методы для симуляции ценовых процессов на Rust
- Создадим production-ready код для Monte Carlo симуляций

---

## 1.1 Броуновское движение (Винеровский процесс)

### Историческая справка

В 1827 году ботаник Роберт Браун наблюдал хаотическое движение пыльцевых зёрен в воде. Это явление, названное броуновским движением, было объяснено Эйнштейном в 1905 году как результат столкновений с молекулами воды.

Норберт Винер в 1923 году построил строгую математическую теорию этого процесса, поэтому его также называют винеровским процессом.

### Математическое определение

**Определение 1.1 (Стандартное броуновское движение)**

Случайный процесс $W = \{W_t\}_{t \geq 0}$ называется стандартным броуновским движением, если:

1. $W_0 = 0$ (почти наверное)
2. Траектории $t \mapsto W_t$ непрерывны (почти наверное)
3. Приращения независимы: для любых $0 \leq t_1 < t_2 < ... < t_n$ случайные величины $W_{t_2} - W_{t_1}, W_{t_3} - W_{t_2}, ..., W_{t_n} - W_{t_{n-1}}$ независимы
4. Приращения стационарны и нормально распределены: $W_t - W_s \sim \mathcal{N}(0, t-s)$ для $s < t$

### Ключевые свойства

**Свойство 1: Непрерывность траекторий**

Траектории броуновского движения непрерывны, но нигде не дифференцируемы. Это важно: "скорость изменения" броуновского движения не существует в классическом смысле.

**Свойство 2: Квадратичная вариация**

Для обычных гладких функций вариация на отрезке конечна, а квадратичная вариация равна нулю. Для броуновского движения всё наоборот:

$$[W,W]_t = \lim_{n \to \infty} \sum_{i=1}^{n} (W_{t_i} - W_{t_{i-1}})^2 = t$$

Это свойство — ключ к пониманию интеграла Ито.

**Свойство 3: Мартингальность**

Броуновское движение — мартингал:
$$\mathbb{E}[W_t | \mathcal{F}_s] = W_s \quad \text{для } s \leq t$$

Это означает, что лучший прогноз будущего значения — текущее значение.

### Реализация на Rust

```rust
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Генератор траекторий броуновского движения
pub struct BrownianMotion {
    /// Начальное значение
    pub initial: f64,
    /// Нормальное распределение для генерации приращений
    normal: Normal<f64>,
}

impl BrownianMotion {
    /// Создаёт новый генератор броуновского движения
    pub fn new(initial: f64) -> Self {
        Self {
            initial,
            normal: Normal::new(0.0, 1.0).expect("Invalid normal distribution"),
        }
    }

    /// Генерирует одну траекторию
    ///
    /// # Аргументы
    /// * `n_steps` - количество временных шагов
    /// * `dt` - размер временного шага
    ///
    /// # Возвращает
    /// Вектор значений W_t для t = 0, dt, 2*dt, ..., n_steps*dt
    pub fn generate_path<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let sqrt_dt = dt.sqrt();
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.initial);

        let mut current = self.initial;
        for _ in 0..n_steps {
            // dW = sqrt(dt) * Z, где Z ~ N(0, 1)
            let dw = sqrt_dt * self.normal.sample(rng);
            current += dw;
            path.push(current);
        }

        path
    }

    /// Генерирует множество траекторий параллельно
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

/// Вычисляет квадратичную вариацию траектории
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

        // Квадратичная вариация должна быть близка к T
        assert!((qv - t_end).abs() < 0.5, "QV = {}, expected ≈ {}", qv, t_end);
    }
}
```

---

## 1.2 Интеграл Ито

### Проблема классического интегрирования

Почему нельзя использовать обычный интеграл Римана для случайных процессов?

Рассмотрим интеграл:
$$\int_0^T W_t \, dW_t$$

Попробуем вычислить его как предел интегральных сумм:
$$\sum_{i} W_{t_i^*} (W_{t_{i+1}} - W_{t_i})$$

Проблема в том, что результат зависит от выбора точки $t_i^*$:
- Если $t_i^* = t_i$ (левый конец) — получаем интеграл Ито
- Если $t_i^* = t_{i+1}$ (правый конец) — получаем другой результат
- Если $t_i^* = (t_i + t_{i+1})/2$ (середина) — интеграл Стратоновича

### Определение интеграла Ито

**Определение 1.2 (Интеграл Ито)**

Для адаптированного процесса $f_t$ интеграл Ито определяется как:

$$\int_0^T f_t \, dW_t = \lim_{n \to \infty} \sum_{i=0}^{n-1} f_{t_i} (W_{t_{i+1}} - W_{t_i})$$

Ключевой момент: мы всегда берём значение $f$ в левом конце интервала!

### Правило Ито (Ito's Lemma)

**Теорема 1.1 (Лемма Ито)**

Пусть $X_t$ — процесс Ито:
$$dX_t = \mu_t \, dt + \sigma_t \, dW_t$$

и $f(t, x)$ — дважды непрерывно дифференцируемая функция. Тогда:

$$df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2$$

Используя правила:
- $(dt)^2 = 0$
- $dt \cdot dW_t = 0$
- $(dW_t)^2 = dt$

получаем:

$$df = \left(\frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma \frac{\partial f}{\partial x} dW_t$$

### Пример: вывод формулы $\int W_t \, dW_t$

Применим лемму Ито к $f(x) = x^2$ и $X_t = W_t$:

$$d(W_t^2) = 2W_t \, dW_t + \frac{1}{2} \cdot 2 \cdot (dW_t)^2 = 2W_t \, dW_t + dt$$

Интегрируя:
$$W_T^2 - W_0^2 = 2\int_0^T W_t \, dW_t + T$$

Откуда:
$$\int_0^T W_t \, dW_t = \frac{1}{2}(W_T^2 - T)$$

Этот результат отличается от классического $\frac{1}{2}W_T^2$ дополнительным слагаемым $-\frac{T}{2}$!

### Реализация на Rust

```rust
/// Вычисляет интеграл Ито ∫ f(W_t) dW_t численно
pub fn ito_integral<F>(path: &[f64], f: F, dt: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    path.windows(2)
        .map(|w| {
            let w_t = w[0];      // Левый конец (определение Ито!)
            let dw = w[1] - w[0]; // Приращение
            f(w_t) * dw
        })
        .sum()
}

/// Вычисляет интеграл Стратоновича ∫ f(W_t) ∘ dW_t численно
pub fn stratonovich_integral<F>(path: &[f64], f: F, dt: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    path.windows(2)
        .map(|w| {
            let w_mid = (w[0] + w[1]) / 2.0; // Середина (определение Стратоновича)
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

        // ∫ W_t dW_t по Ито = (W_T² - T) / 2
        let ito = ito_integral(&path, |x| x, dt);
        let expected_ito = (w_t * w_t - t_end) / 2.0;

        assert!((ito - expected_ito).abs() < 0.1,
            "Ito integral: got {}, expected {}", ito, expected_ito);
    }
}
```

---

## 1.3 Стохастические дифференциальные уравнения (SDE)

### Общая форма SDE

Стохастическое дифференциальное уравнение имеет вид:

$$dX_t = \mu(t, X_t) \, dt + \sigma(t, X_t) \, dW_t$$

где:
- $\mu(t, x)$ — дрейф (drift)
- $\sigma(t, x)$ — волатильность (diffusion)

### Условия существования и единственности решения

**Теорема 1.2** (Существование и единственность)

Если $\mu$ и $\sigma$ удовлетворяют условиям:

1. **Условие Липшица**: $|\mu(t,x) - \mu(t,y)| + |\sigma(t,x) - \sigma(t,y)| \leq K|x-y|$
2. **Условие роста**: $|\mu(t,x)|^2 + |\sigma(t,x)|^2 \leq K^2(1 + |x|^2)$

то SDE имеет единственное сильное решение.

### Реализация абстракции SDE на Rust

```rust
/// Трейт для стохастических дифференциальных уравнений
pub trait SDE {
    /// Тип состояния (может быть f64 или вектор)
    type State: Clone;

    /// Дрейф μ(t, X)
    fn drift(&self, t: f64, x: &Self::State) -> Self::State;

    /// Волатильность σ(t, X)
    fn diffusion(&self, t: f64, x: &Self::State) -> Self::State;

    /// Начальное условие
    fn initial_state(&self) -> Self::State;
}

/// Euler-Maruyama солвер для SDE
pub struct EulerMaruyama<S: SDE<State = f64>> {
    pub sde: S,
    pub dt: f64,
}

impl<S: SDE<State = f64>> EulerMaruyama<S> {
    pub fn new(sde: S, dt: f64) -> Self {
        Self { sde, dt }
    }

    /// Один шаг метода Euler-Maruyama
    /// X_{n+1} = X_n + μ(t, X_n) * dt + σ(t, X_n) * dW
    pub fn step<R: Rng>(&self, rng: &mut R, t: f64, x: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let dw = self.dt.sqrt() * normal.sample(rng);

        let drift = self.sde.drift(t, &x);
        let diffusion = self.sde.diffusion(t, &x);

        x + drift * self.dt + diffusion * dw
    }

    /// Генерирует траекторию
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

## 1.4 Геометрическое броуновское движение (GBM)

### Модель

Геометрическое броуновское движение — самая известная модель для цен активов:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

где:
- $S_t$ — цена актива в момент $t$
- $\mu$ — ожидаемая доходность (drift)
- $\sigma$ — волатильность

### Аналитическое решение

Применим лемму Ито к $f(S) = \ln S$:

$$d(\ln S_t) = \frac{1}{S_t} dS_t - \frac{1}{2} \frac{1}{S_t^2} (dS_t)^2$$

Подставляя $(dS_t)^2 = \sigma^2 S_t^2 dt$:

$$d(\ln S_t) = \frac{1}{S_t}(\mu S_t dt + \sigma S_t dW_t) - \frac{1}{2}\sigma^2 dt$$

$$d(\ln S_t) = \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma dW_t$$

Интегрируя:

$$\ln S_t = \ln S_0 + \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t$$

**Решение:**
$$S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]$$

### Свойства GBM

1. **Цена всегда положительна**: $S_t > 0$ для всех $t$
2. **Логнормальное распределение**: $\ln(S_t/S_0) \sim \mathcal{N}\left((\mu - \frac{\sigma^2}{2})t, \sigma^2 t\right)$
3. **Математическое ожидание**: $\mathbb{E}[S_t] = S_0 e^{\mu t}$

### Реализация на Rust

```rust
/// Геометрическое броуновское движение
pub struct GeometricBrownianMotion {
    /// Начальная цена
    pub s0: f64,
    /// Дрейф (ожидаемая доходность)
    pub mu: f64,
    /// Волатильность
    pub sigma: f64,
}

impl GeometricBrownianMotion {
    pub fn new(s0: f64, mu: f64, sigma: f64) -> Self {
        assert!(s0 > 0.0, "Initial price must be positive");
        assert!(sigma >= 0.0, "Volatility must be non-negative");
        Self { s0, mu, sigma }
    }

    /// Аналитическое решение (эффективнее для генерации конечных значений)
    pub fn sample_at_time<R: Rng>(&self, rng: &mut R, t: f64) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.sample(rng);

        self.s0 * ((self.mu - 0.5 * self.sigma * self.sigma) * t
                   + self.sigma * t.sqrt() * z).exp()
    }

    /// Генерирует траекторию используя аналитическое решение
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

## 1.5 Модель Heston (стохастическая волатильность)

### Мотивация

GBM предполагает постоянную волатильность $\sigma$. В реальности:
- Волатильность меняется со временем
- Наблюдается "volatility smile" в опционах
- Волатильность обычно растёт при падении цены (leverage effect)

### Модель

Модель Heston:

$$dS_t = \mu S_t \, dt + \sqrt{V_t} S_t \, dW^S_t$$
$$dV_t = \kappa(\theta - V_t) \, dt + \xi \sqrt{V_t} \, dW^V_t$$

где:
- $V_t$ — мгновенная дисперсия
- $\kappa$ — скорость возврата к среднему
- $\theta$ — долгосрочный уровень дисперсии
- $\xi$ — волатильность волатильности
- $\text{Corr}(dW^S, dW^V) = \rho$ (обычно $\rho < 0$)

### Условие Феллера

Для того чтобы $V_t$ оставалась положительной:

$$2\kappa\theta > \xi^2$$

### Реализация на Rust

```rust
/// Модель Heston для стохастической волатильности
pub struct HestonModel {
    /// Начальная цена
    pub s0: f64,
    /// Начальная дисперсия
    pub v0: f64,
    /// Дрейф цены
    pub mu: f64,
    /// Скорость возврата к среднему (kappa)
    pub kappa: f64,
    /// Долгосрочная дисперсия (theta)
    pub theta: f64,
    /// Волатильность волатильности (xi)
    pub xi: f64,
    /// Корреляция между ценой и волатильностью (rho)
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

        // Проверка условия Феллера
        if 2.0 * kappa * theta <= xi * xi {
            eprintln!("Warning: Feller condition not satisfied (2κθ > ξ²)");
        }

        Self { s0, v0, mu, kappa, theta, xi, rho }
    }

    /// Euler-Maruyama с отсечением отрицательной дисперсии
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
            // Генерируем коррелированные броуновские приращения
            let z1 = normal.sample(rng);
            let z2 = normal.sample(rng);

            let dw_s = sqrt_dt * z1;
            let dw_v = sqrt_dt * (self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * z2);

            // Обновляем дисперсию (с отсечением)
            let sqrt_v = v.max(0.0).sqrt();
            let dv = self.kappa * (self.theta - v) * dt + self.xi * sqrt_v * dw_v;
            v = (v + dv).max(0.0); // Отсечение отрицательных значений

            // Обновляем цену
            let ds = self.mu * s * dt + sqrt_v * s * dw_s;
            s += ds;
            s = s.max(0.0); // Цена не может быть отрицательной

            prices.push(s);
            variances.push(v);
        }

        (prices, variances)
    }

    /// Milstein схема (более точная для волатильности)
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

            // Milstein коррекция для процесса дисперсии
            // d(√V)/dV = 1/(2√V), поэтому добавляем (ξ²/4)(dW² - dt)
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

## 1.6 Jump-Diffusion модели

### Мотивация

GBM не объясняет:
- Внезапные большие движения цены (gaps)
- Fat tails в распределении доходностей
- Кластеризацию волатильности

### Модель Мертона (1976)

$$dS_t = (\mu - \lambda \kappa) S_t \, dt + \sigma S_t \, dW_t + S_t \, dJ_t$$

где:
- $J_t = \sum_{i=1}^{N_t} (Y_i - 1)$ — составной пуассоновский процесс
- $N_t$ — пуассоновский процесс с интенсивностью $\lambda$
- $Y_i$ — размеры скачков (обычно $\ln Y_i \sim \mathcal{N}(\mu_J, \sigma_J^2)$)
- $\kappa = \mathbb{E}[Y - 1] = e^{\mu_J + \sigma_J^2/2} - 1$

### Реализация на Rust

```rust
use rand_distr::Poisson;

/// Jump-Diffusion модель Мертона
pub struct MertonJumpDiffusion {
    pub s0: f64,
    pub mu: f64,
    pub sigma: f64,
    /// Интенсивность скачков (среднее число скачков в год)
    pub lambda: f64,
    /// Среднее логарифма размера скачка
    pub mu_j: f64,
    /// Стандартное отклонение логарифма размера скачка
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

    /// Ожидаемый размер скачка: E[Y - 1]
    fn kappa(&self) -> f64 {
        (self.mu_j + 0.5 * self.sigma_j * self.sigma_j).exp() - 1.0
    }

    /// Симуляция траектории
    pub fn simulate<R: Rng>(&self, rng: &mut R, n_steps: usize, dt: f64) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let normal_j = Normal::new(self.mu_j, self.sigma_j).unwrap();
        let sqrt_dt = dt.sqrt();

        // Скорректированный дрейф
        let mu_adj = self.mu - self.lambda * self.kappa();

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.s0);

        let mut s = self.s0;

        for _ in 0..n_steps {
            // Диффузионная часть
            let dw = sqrt_dt * normal.sample(rng);
            let diffusion = mu_adj * s * dt + self.sigma * s * dw;

            // Скачковая часть: количество скачков за dt
            let n_jumps = if self.lambda * dt < 30.0 {
                // Используем распределение Пуассона для малых λ*dt
                Poisson::new(self.lambda * dt).unwrap().sample(rng) as usize
            } else {
                // Для больших λ*dt аппроксимируем нормальным
                let n = (self.lambda * dt + (self.lambda * dt).sqrt() * normal.sample(rng))
                    .round().max(0.0) as usize;
                n
            };

            // Суммарный мультипликатор скачков
            let mut jump_mult = 1.0;
            for _ in 0..n_jumps {
                let log_y = normal_j.sample(rng);
                jump_mult *= log_y.exp();
            }

            s = (s + diffusion) * jump_mult;
            s = s.max(1e-10); // Защита от нуля

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
            2.0,    // lambda (2 скачка в год в среднем)
            -0.05,  // mu_j (скачки в среднем вниз)
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

## 1.7 Monte Carlo методы

### Базовый Monte Carlo

Для оценки $\mathbb{E}[f(S_T)]$:

$$\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} f(S_T^{(i)})$$

Стандартная ошибка: $SE = \frac{\hat{\sigma}}{\sqrt{N}}$

### Variance Reduction: Antithetic Variates

Если $Z \sim \mathcal{N}(0,1)$, то и $-Z \sim \mathcal{N}(0,1)$.

Для каждой траектории генерируем парную "антитетическую" траекторию:

```rust
/// Monte Carlo с антитетическими вариатами
pub fn monte_carlo_antithetic<F, R>(
    gbm: &GeometricBrownianMotion,
    payoff: F,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> (f64, f64) // (оценка, стандартная ошибка)
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

        // Основная траектория
        let s1 = gbm.s0 * (drift + vol_sqrt_t * z).exp();
        // Антитетическая траектория
        let s2 = gbm.s0 * (drift - vol_sqrt_t * z).exp();

        // Среднее по паре
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

Если мы знаем $\mathbb{E}[Y]$ аналитически, используем:

$$\hat{\mu}_{CV} = \frac{1}{N} \sum_{i=1}^{N} (f(S_T^{(i)}) - c(Y^{(i)} - \mathbb{E}[Y]))$$

где $c$ выбирается для минимизации дисперсии.

```rust
/// Monte Carlo с контрольными вариатами
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

    // Контрольная переменная: S_T
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

    // Оптимальный коэффициент c = Cov(payoff, control) / Var(control)
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

    // Скорректированные payoffs
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

## 1.8 SIMD-оптимизация

### Векторизация Monte Carlo

Для максимальной производительности используем SIMD:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-оптимизированная генерация GBM
/// Обрабатывает 4 траектории одновременно (AVX)
#[cfg(target_arch = "x86_64")]
pub unsafe fn gbm_simd_4paths(
    s0: f64,
    drift: f64,  // (μ - σ²/2) * dt
    vol_sqrt_dt: f64,  // σ * √dt
    random_normals: &[f64],  // длина n_steps * 4
    n_steps: usize,
) -> [f64; 4] {
    let mut log_s = _mm256_set1_pd(s0.ln());
    let drift_vec = _mm256_set1_pd(drift);
    let vol_vec = _mm256_set1_pd(vol_sqrt_dt);

    for step in 0..n_steps {
        // Загружаем 4 случайных числа
        let z = _mm256_loadu_pd(random_normals.as_ptr().add(step * 4));

        // log_s += drift + vol * z
        let increment = _mm256_fmadd_pd(vol_vec, z, drift_vec);
        log_s = _mm256_add_pd(log_s, increment);
    }

    // Экспонента (приближённая)
    let mut result = [0.0f64; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), log_s);

    for r in &mut result {
        *r = r.exp();
    }

    result
}

/// Высокоуровневый интерфейс для SIMD Monte Carlo
pub fn monte_carlo_simd(
    gbm: &GeometricBrownianMotion,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
) -> Vec<f64> {
    use rayon::prelude::*;

    let drift = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * dt;
    let vol_sqrt_dt = gbm.sigma * dt.sqrt();

    // Округляем до кратного 4
    let n_batches = (n_paths + 3) / 4;

    (0..n_batches)
        .into_par_iter()
        .flat_map(|_| {
            let mut rng = rand::thread_rng();
            let normal = Normal::new(0.0, 1.0).unwrap();

            // Генерируем случайные числа для 4 траекторий
            let randoms: Vec<f64> = (0..n_steps * 4)
                .map(|_| normal.sample(&mut rng))
                .collect();

            #[cfg(target_arch = "x86_64")]
            unsafe {
                gbm_simd_4paths(gbm.s0, drift, vol_sqrt_dt, &randoms, n_steps).to_vec()
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                // Fallback для не-x86 архитектур
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

### Сравнение производительности

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

### Типичные результаты

| Метод | 10K траекторий | 100K траекторий | 1M траекторий |
|-------|---------------|-----------------|---------------|
| Sequential | 45 ms | 450 ms | 4.5 s |
| Parallel (8 cores) | 8 ms | 70 ms | 650 ms |
| SIMD + Parallel | 3 ms | 25 ms | 220 ms |

---

## 1.10 Практические задания

### Задание 1: Верификация свойств броуновского движения

Напишите тесты, проверяющие:
1. $\mathbb{E}[W_t] = 0$
2. $\text{Var}[W_t] = t$
3. Квадратичная вариация $[W,W]_T \approx T$
4. Независимость приращений (тест корреляции)

### Задание 2: Калибровка Heston к implied volatility

Имея опционные цены (или implied volatilities), откалибруйте параметры модели Heston:
- Реализуйте целевую функцию (MSE по volatility surface)
- Используйте Levenberg-Marquardt или Differential Evolution
- Оцените качество подгонки

### Задание 3: Оценка опциона методом Monte Carlo

Оцените цену европейского колл-опциона:
- Реализуйте все три метода variance reduction
- Сравните скорость сходимости
- Постройте графики зависимости ошибки от числа симуляций

### Задание 4: Real-time симулятор

Создайте streaming симулятор цены:
- Lock-free ring buffer для выходных данных
- Латентность < 1μs на тик
- WebSocket интерфейс для потребителей

---

## Заключение

В этой главе мы изучили:

1. **Броуновское движение** — фундамент случайных процессов в непрерывном времени
2. **Интеграл Ито** — инструмент для работы со случайными процессами
3. **Лемму Ито** — "цепное правило" стохастического исчисления
4. **Модели цен**: GBM, Heston, Jump-Diffusion
5. **Численные методы**: Euler-Maruyama, Milstein
6. **Оптимизацию**: SIMD, параллелизм, variance reduction

Эти концепции — основа для понимания более сложных моделей микроструктуры рынка (Глава 2) и портфельной оптимизации (Глава 3).

---

## Литература

1. Shreve S.E. "Stochastic Calculus for Finance II: Continuous-Time Models" (2004)
2. Gatheral J. "The Volatility Surface: A Practitioner's Guide" (2006)
3. Cont R., Tankov P. "Financial Modelling with Jump Processes" (2003)
4. Glasserman P. "Monte Carlo Methods in Financial Engineering" (2003)
5. Heston S. "A Closed-Form Solution for Options with Stochastic Volatility" (1993)
