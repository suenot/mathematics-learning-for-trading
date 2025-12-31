# Глава 2: Микроструктура рынка и моделирование Order Book

## Метаданные
- **Уровень сложности**: Продвинутый
- **Предварительные требования**: Глава 1 (стохастическое исчисление), теория вероятностей, базовое понимание трейдинга
- **Языки реализации**: Rust (основной)
- **Расчётный объём**: 100-140 страниц

---

## Введение

Микроструктура рынка изучает, как конкретные торговые механизмы влияют на формирование цен. Если в главе 1 мы рассматривали цену как непрерывный случайный процесс, то здесь мы "увеличиваем масштаб" и видим, что цена меняется дискретно — каждая сделка, каждый ордер влияет на неё.

**Почему это важно для трейдера?**
- Понимание микроструктуры позволяет оптимизировать исполнение ордеров
- Market making стратегии основаны на математике order book
- Предсказание краткосрочных движений цены требует анализа потока ордеров

---

## 2.1 Анатомия Limit Order Book (LOB)

### 2.1.1 Что такое Order Book?

Order Book (книга ордеров) — это структура данных, которая хранит все активные лимитные заявки на покупку и продажу актива.

```
         ASKS (продавцы)
    ┌─────────────────────────┐
    │ $102.50  |  150 shares  │  ← Best Ask (лучшая цена продажи)
    │ $102.75  |  300 shares  │
    │ $103.00  |  500 shares  │
    └─────────────────────────┘

    ═══════════════════════════  SPREAD = $0.50

    ┌─────────────────────────┐
    │ $102.00  |  200 shares  │  ← Best Bid (лучшая цена покупки)
    │ $101.75  |  400 shares  │
    │ $101.50  |  250 shares  │
    └─────────────────────────┘
         BIDS (покупатели)
```

### 2.1.2 Типы ордеров

```rust
/// Основные типы ордеров в торговой системе
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    /// Лимитный ордер - исполняется по указанной цене или лучше
    Limit,
    /// Рыночный ордер - исполняется немедленно по лучшей доступной цене
    Market,
    /// Стоп-ордер - становится рыночным при достижении триггер-цены
    Stop,
    /// Стоп-лимит - становится лимитным при достижении триггер-цены
    StopLimit,
    /// Iceberg - показывает только часть объёма
    Iceberg,
}

/// Направление ордера
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Условия исполнения
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good Till Cancel - активен до отмены
    GTC,
    /// Immediate Or Cancel - исполнить немедленно (частично) или отменить
    IOC,
    /// Fill Or Kill - исполнить полностью или отменить
    FOK,
    /// Good Till Date - активен до указанной даты
    GTD,
}
```

### 2.1.3 Matching Engine и приоритет ордеров

Биржи используют **Price-Time Priority (FIFO)** для определения порядка исполнения:

1. **Приоритет цены**: Лучшая цена исполняется первой
2. **Приоритет времени**: При одинаковой цене — кто раньше поставил

```rust
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Instant;

/// Уровень цены с очередью ордеров
#[derive(Debug)]
pub struct PriceLevel {
    pub price: i64,           // Цена в минимальных единицах (тиках)
    pub total_volume: u64,    // Суммарный объём на уровне
    pub orders: Vec<Order>,   // Очередь ордеров (FIFO)
}

/// Отдельный ордер
#[derive(Debug, Clone)]
pub struct Order {
    pub id: u64,
    pub price: i64,
    pub volume: u64,
    pub side: Side,
    pub timestamp: Instant,
    pub order_type: OrderType,
}

impl Ord for Order {
    fn cmp(&self, other: &Self) -> Ordering {
        // Сначала по времени (раньше = приоритетнее)
        self.timestamp.cmp(&other.timestamp)
    }
}

impl PartialOrd for Order {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Order {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Order {}
```

### 2.1.4 Структура данных Order Book

```rust
use rust_decimal::Decimal;
use std::collections::BTreeMap;

/// Высокопроизводительная структура Order Book
pub struct OrderBook {
    /// Ордера на покупку (bids), отсортированы по убыванию цены
    /// BTreeMap гарантирует O(log n) операции
    bids: BTreeMap<i64, PriceLevel>,

    /// Ордера на продажу (asks), отсортированы по возрастанию цены
    asks: BTreeMap<i64, PriceLevel>,

    /// Размер тика (минимальное изменение цены)
    tick_size: Decimal,

    /// Размер лота (минимальный объём)
    lot_size: Decimal,
}

impl OrderBook {
    pub fn new(tick_size: Decimal, lot_size: Decimal) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            tick_size,
            lot_size,
        }
    }

    /// Лучшая цена покупки
    pub fn best_bid(&self) -> Option<i64> {
        self.bids.keys().next_back().copied()
    }

    /// Лучшая цена продажи
    pub fn best_ask(&self) -> Option<i64> {
        self.asks.keys().next().copied()
    }

    /// Спред в тиках
    pub fn spread(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Mid-price (средняя цена)
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) as f64 / 2.0),
            _ => None,
        }
    }

    /// Microprice - взвешенная средняя цена
    /// Более точная оценка "справедливой" цены
    pub fn microprice(&self) -> Option<f64> {
        let best_bid = self.best_bid()?;
        let best_ask = self.best_ask()?;

        let bid_vol = self.bids.get(&best_bid)?.total_volume as f64;
        let ask_vol = self.asks.get(&best_ask)?.total_volume as f64;

        // Microprice = (bid_vol * ask + ask_vol * bid) / (bid_vol + ask_vol)
        Some((bid_vol * best_ask as f64 + ask_vol * best_bid as f64)
             / (bid_vol + ask_vol))
    }
}
```

### 2.1.5 Добавление и удаление ордеров

```rust
impl OrderBook {
    /// Добавить лимитный ордер в книгу
    pub fn add_limit_order(&mut self, order: Order) -> OrderId {
        let book = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        book.entry(order.price)
            .or_insert_with(|| PriceLevel {
                price: order.price,
                total_volume: 0,
                orders: Vec::new(),
            })
            .add_order(order.clone());

        order.id
    }

    /// Отменить ордер по ID
    pub fn cancel_order(&mut self, order_id: u64, side: Side, price: i64) -> bool {
        let book = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        if let Some(level) = book.get_mut(&price) {
            if level.remove_order(order_id) {
                // Если уровень пуст, удаляем его
                if level.orders.is_empty() {
                    book.remove(&price);
                }
                return true;
            }
        }
        false
    }

    /// Исполнить рыночный ордер
    pub fn execute_market_order(&mut self, side: Side, mut volume: u64) -> Vec<Trade> {
        let mut trades = Vec::new();

        // Для покупки - забираем с asks, для продажи - с bids
        let book = match side {
            Side::Buy => &mut self.asks,
            Side::Sell => &mut self.bids,
        };

        let prices: Vec<i64> = match side {
            Side::Buy => book.keys().copied().collect(),
            Side::Sell => book.keys().rev().copied().collect(),
        };

        for price in prices {
            if volume == 0 {
                break;
            }

            if let Some(level) = book.get_mut(&price) {
                let (level_trades, remaining) = level.match_volume(volume, side);
                trades.extend(level_trades);
                volume = remaining;

                if level.orders.is_empty() {
                    book.remove(&price);
                }
            }
        }

        trades
    }
}

impl PriceLevel {
    fn add_order(&mut self, order: Order) {
        self.total_volume += order.volume;
        self.orders.push(order);
    }

    fn remove_order(&mut self, order_id: u64) -> bool {
        if let Some(pos) = self.orders.iter().position(|o| o.id == order_id) {
            let order = self.orders.remove(pos);
            self.total_volume -= order.volume;
            true
        } else {
            false
        }
    }

    fn match_volume(&mut self, mut volume: u64, aggressor_side: Side) -> (Vec<Trade>, u64) {
        let mut trades = Vec::new();

        while volume > 0 && !self.orders.is_empty() {
            let order = &mut self.orders[0];
            let fill_volume = volume.min(order.volume);

            trades.push(Trade {
                price: self.price,
                volume: fill_volume,
                aggressor_side,
            });

            order.volume -= fill_volume;
            self.total_volume -= fill_volume;
            volume -= fill_volume;

            if order.volume == 0 {
                self.orders.remove(0);
            }
        }

        (trades, volume)
    }
}

/// Запись о совершённой сделке
#[derive(Debug, Clone)]
pub struct Trade {
    pub price: i64,
    pub volume: u64,
    pub aggressor_side: Side, // Кто инициировал сделку
}

type OrderId = u64;
```

---

## 2.2 Point Processes для моделирования Order Flow

### 2.2.1 Введение в Point Processes

Point process (точечный процесс) — это математическая модель для описания случайных событий, происходящих во времени. В контексте трейдинга события — это поступающие ордера.

**Ключевое понятие — интенсивность λ(t):**
- λ(t) · dt ≈ вероятность события в интервале [t, t + dt]
- Большая интенсивность = события происходят чаще

### 2.2.2 Пуассоновский процесс

Простейшая модель — события происходят с постоянной интенсивностью λ, независимо друг от друга.

```rust
use rand::Rng;
use rand_distr::{Exp, Distribution};

/// Генератор однородного пуассоновского процесса
pub struct PoissonProcess {
    /// Интенсивность (среднее число событий в единицу времени)
    lambda: f64,
}

impl PoissonProcess {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "Интенсивность должна быть положительной");
        Self { lambda }
    }

    /// Генерация событий до времени t_end
    pub fn simulate(&self, t_end: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let exp_dist = Exp::new(self.lambda).unwrap();

        let mut events = Vec::new();
        let mut t = 0.0;

        loop {
            // Время до следующего события ~ Exp(λ)
            let inter_arrival = exp_dist.sample(&mut rng);
            t += inter_arrival;

            if t > t_end {
                break;
            }
            events.push(t);
        }

        events
    }

    /// Интенсивность (постоянная)
    pub fn intensity(&self, _t: f64) -> f64 {
        self.lambda
    }
}
```

**Проблема пуассоновского процесса:** он не захватывает **кластеризацию** событий. На реальных рынках ордера приходят "пачками" — одна сделка провоцирует другие.

### 2.2.3 Процесс Хоукса (Hawkes Process)

Процесс Хоукса — это **самовозбуждающийся** (self-exciting) процесс. Каждое событие временно увеличивает интенсивность, что приводит к кластеризации.

**Математическое определение:**

```
λ(t) = μ + Σ α · exp(-β · (t - tᵢ))
           i: tᵢ < t
```

где:
- μ — базовая интенсивность (фон)
- α — сила возбуждения (насколько событие повышает интенсивность)
- β — скорость затухания (как быстро эффект исчезает)
- tᵢ — времена прошлых событий

```rust
/// Процесс Хоукса с экспоненциальным ядром
pub struct HawkesProcess {
    /// Базовая интенсивность
    mu: f64,
    /// Параметр возбуждения
    alpha: f64,
    /// Параметр затухания
    beta: f64,
    /// История событий
    history: Vec<f64>,
}

impl HawkesProcess {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        // Проверяем условие стационарности: branching ratio < 1
        let branching_ratio = alpha / beta;
        assert!(
            branching_ratio < 1.0,
            "Branching ratio α/β = {} должен быть < 1 для стационарности",
            branching_ratio
        );

        Self {
            mu,
            alpha,
            beta,
            history: Vec::new(),
        }
    }

    /// Вычислить текущую интенсивность
    pub fn intensity(&self, t: f64) -> f64 {
        let mut lambda = self.mu;

        for &ti in &self.history {
            if ti < t {
                // Вклад события ti в интенсивность в момент t
                lambda += self.alpha * (-self.beta * (t - ti)).exp();
            }
        }

        lambda
    }

    /// Branching ratio - среднее число "детей" на одно событие
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Симуляция методом thinning (Lewis-Shedler algorithm)
    pub fn simulate(&mut self, t_end: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        self.history.clear();

        let mut t = 0.0;

        while t < t_end {
            // Верхняя граница интенсивности
            let lambda_max = self.intensity(t) + self.alpha;

            // Генерируем кандидата из Пуассона с интенсивностью lambda_max
            let u1: f64 = rng.gen();
            let dt = -u1.ln() / lambda_max;
            t += dt;

            if t > t_end {
                break;
            }

            // Acceptance-rejection
            let lambda_t = self.intensity(t);
            let u2: f64 = rng.gen();

            if u2 <= lambda_t / lambda_max {
                // Принимаем событие
                self.history.push(t);
            }
        }

        self.history.clone()
    }

    /// Оценка параметров методом максимального правдоподобия
    pub fn fit_mle(events: &[f64], t_end: f64) -> Result<Self, &'static str> {
        use std::f64::consts::E;

        if events.is_empty() {
            return Err("Нет событий для калибровки");
        }

        // Упрощённая реализация - градиентный спуск
        // В production использовать специализированные оптимизаторы

        let mut mu = events.len() as f64 / t_end * 0.5;
        let mut alpha = 0.3;
        let mut beta = 1.0;

        let learning_rate = 0.001;
        let iterations = 1000;

        for _ in 0..iterations {
            // Вычисляем градиент log-likelihood
            let (grad_mu, grad_alpha, grad_beta) =
                compute_gradients(events, t_end, mu, alpha, beta);

            // Обновляем параметры
            mu += learning_rate * grad_mu;
            alpha += learning_rate * grad_alpha;
            beta += learning_rate * grad_beta;

            // Проекция на допустимую область
            mu = mu.max(0.001);
            alpha = alpha.max(0.001);
            beta = beta.max(alpha + 0.001); // Обеспечиваем α/β < 1
        }

        Ok(Self {
            mu,
            alpha,
            beta,
            history: events.to_vec(),
        })
    }
}

/// Вычисление градиентов log-likelihood (упрощённо)
fn compute_gradients(
    events: &[f64],
    t_end: f64,
    mu: f64,
    alpha: f64,
    beta: f64
) -> (f64, f64, f64) {
    let n = events.len() as f64;

    // ∂logL/∂μ = Σ(1/λ(tᵢ)) - T
    let grad_mu = n / mu - t_end;

    // Упрощённые градиенты для демонстрации
    let grad_alpha = n / alpha - t_end / 2.0;
    let grad_beta = -n / beta + t_end / 2.0;

    (grad_mu, grad_alpha, grad_beta)
}
```

### 2.2.4 Многомерный процесс Хоукса

На реальном рынке разные типы событий влияют друг на друга:
- Покупки могут провоцировать продажи (и наоборот)
- Сделки на bid влияют на сделки на ask

```rust
/// Многомерный процесс Хоукса
pub struct MultivariateHawkes {
    /// Базовые интенсивности для каждого типа
    mu: Vec<f64>,
    /// Матрица возбуждения α[i][j] - влияние типа j на тип i
    alpha: Vec<Vec<f64>>,
    /// Матрица затухания
    beta: Vec<Vec<f64>>,
    /// История событий: (время, тип)
    history: Vec<(f64, usize)>,
    /// Количество типов событий
    dim: usize,
}

impl MultivariateHawkes {
    pub fn new(mu: Vec<f64>, alpha: Vec<Vec<f64>>, beta: Vec<Vec<f64>>) -> Self {
        let dim = mu.len();
        assert_eq!(alpha.len(), dim);
        assert_eq!(beta.len(), dim);

        Self {
            mu,
            alpha,
            beta,
            history: Vec::new(),
            dim,
        }
    }

    /// Интенсивность для типа событий k в момент t
    pub fn intensity(&self, t: f64, k: usize) -> f64 {
        let mut lambda = self.mu[k];

        for &(ti, j) in &self.history {
            if ti < t {
                lambda += self.alpha[k][j] * (-self.beta[k][j] * (t - ti)).exp();
            }
        }

        lambda
    }

    /// Суммарная интенсивность (для thinning)
    pub fn total_intensity(&self, t: f64) -> f64 {
        (0..self.dim).map(|k| self.intensity(t, k)).sum()
    }
}

/// Пример: bid-ask взаимодействие
fn create_bid_ask_hawkes() -> MultivariateHawkes {
    // 0 = события на bid, 1 = события на ask
    let mu = vec![1.0, 1.0];  // Базовые интенсивности

    // Матрица возбуждения:
    // α[0][0] = bid -> bid (автокорреляция)
    // α[0][1] = ask -> bid (cross-excitation)
    let alpha = vec![
        vec![0.3, 0.2],  // Влияние на bid
        vec![0.2, 0.3],  // Влияние на ask
    ];

    let beta = vec![
        vec![1.0, 1.0],
        vec![1.0, 1.0],
    ];

    MultivariateHawkes::new(mu, alpha, beta)
}
```

---

## 2.3 Модель Cont-Stoikov-Talreja

### 2.3.1 Основная идея

Cont, Stoikov и Talreja (2010) предложили рассматривать order book как **систему массового обслуживания** (queueing system).

Каждый ценовой уровень — это очередь, где:
- **Поступления** (arrivals): лимитные ордера
- **Уходы** (departures): отмены и исполнения

### 2.3.2 Вероятности движения цены

Ключевой вопрос: какова вероятность, что mid-price пойдёт вверх или вниз?

```rust
/// Модель Cont-Stoikov-Talreja
pub struct ContStoikovTalreja {
    /// Интенсивность прихода лимитных ордеров на уровень δ от лучшей цены
    /// λ(δ) = A * exp(-k * δ)
    arrival_a: f64,
    arrival_k: f64,

    /// Интенсивность отмен
    cancel_rate: f64,

    /// Интенсивность рыночных ордеров
    market_order_rate: f64,
}

impl ContStoikovTalreja {
    pub fn new(arrival_a: f64, arrival_k: f64, cancel_rate: f64, market_order_rate: f64) -> Self {
        Self {
            arrival_a,
            arrival_k,
            cancel_rate,
            market_order_rate,
        }
    }

    /// Интенсивность прихода лимитных ордеров на расстоянии δ тиков от лучшей цены
    pub fn limit_order_arrival_rate(&self, delta: u32) -> f64 {
        self.arrival_a * (-self.arrival_k * delta as f64).exp()
    }

    /// Вероятность движения mid-price вверх
    /// Зависит от дисбаланса объёмов на лучших уровнях
    pub fn prob_price_up(&self, bid_volume: f64, ask_volume: f64) -> f64 {
        // Упрощённая формула
        // В оригинальной статье используется более сложное выражение
        // через преобразование Лапласа

        let total = bid_volume + ask_volume;
        if total == 0.0 {
            return 0.5;
        }

        // Интуиция: больше объём на bid -> меньше шанс пробоя -> цена скорее вырастет
        bid_volume / total
    }

    /// Ожидаемое время до следующего изменения mid-price
    pub fn expected_time_to_price_change(
        &self,
        best_bid_volume: u64,
        best_ask_volume: u64
    ) -> f64 {
        // Интенсивность пробоя bid = market_order_rate * P(пробить bid)
        // Интенсивность пробоя ask = market_order_rate * P(пробить ask)

        // Упрощённо: обратная сумма интенсивностей
        let bid_vol = best_bid_volume as f64;
        let ask_vol = best_ask_volume as f64;

        let rate_bid_clear = self.market_order_rate / bid_vol.max(1.0);
        let rate_ask_clear = self.market_order_rate / ask_vol.max(1.0);

        1.0 / (rate_bid_clear + rate_ask_clear)
    }
}
```

### 2.3.3 Калибровка к реальным данным

```rust
impl ContStoikovTalreja {
    /// Калибровка параметра arrival rate из исторических данных
    pub fn calibrate_arrival_rate(
        limit_orders: &[(f64, u32)], // (время, расстояние до лучшей цены)
        t_end: f64,
    ) -> (f64, f64) {
        // Группируем ордера по расстоянию
        use std::collections::HashMap;

        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &(_time, delta) in limit_orders {
            *counts.entry(delta).or_insert(0) += 1;
        }

        // Оцениваем λ(δ) = count[δ] / T
        // Затем фитим A * exp(-k * δ)

        // Линейная регрессия на log(λ) = log(A) - k * δ
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        let mut n = 0.0;

        for (&delta, &count) in &counts {
            let lambda = count as f64 / t_end;
            if lambda > 0.0 {
                let x = delta as f64;
                let y = lambda.ln();

                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                n += 1.0;
            }
        }

        // k = -(n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x^2)
        let k = -(n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        // log(A) = (sum_y + k * sum_x) / n
        let log_a = (sum_y + k * sum_x) / n;
        let a = log_a.exp();

        (a, k)
    }
}
```

---

## 2.4 Оптимальный Market Making: модель Avellaneda-Stoikov

### 2.4.1 Постановка задачи

Market maker зарабатывает на спреде, но несёт риск inventory (позиции). Задача: найти оптимальные цены bid и ask.

**Критерий оптимизации:**
```
Максимизировать: E[-exp(-γ · W_T)]
```

где:
- W_T = X_T + q_T · S_T — финальное богатство
- X_T — денежные средства
- q_T — позиция (inventory)
- S_T — цена актива
- γ — коэффициент неприятия риска (risk aversion)

### 2.4.2 Ключевые формулы

**Reservation price (цена безразличия):**
```
r(t, q) = S_t - q · γ · σ² · (T - t)
```

**Интерпретация:**
- Если q > 0 (длинная позиция): r < S, market maker готов продать дешевле
- Если q < 0 (короткая позиция): r > S, market maker готов купить дороже
- Чем больше |q|, тем агрессивнее котировки для уменьшения позиции

**Оптимальный спред:**
```
δ⁺ + δ⁻ = γσ²(T-t) + (2/γ) · ln(1 + γ/k)
```

где k — параметр, характеризующий убывание интенсивности ордеров с расстоянием от mid-price.

### 2.4.3 Полная реализация

```rust
/// Market Making стратегия Avellaneda-Stoikov
pub struct AvellanedaStoikov {
    /// Коэффициент неприятия риска
    gamma: f64,
    /// Оценка волатильности (σ в годовом выражении)
    sigma: f64,
    /// Параметр убывания интенсивности ордеров
    k: f64,
    /// Время окончания сессии (в годах)
    t_end: f64,
    /// Текущая позиция
    inventory: i64,
    /// Текущая mid-price
    mid_price: f64,
    /// Максимальная позиция
    max_inventory: i64,
}

impl AvellanedaStoikov {
    pub fn new(
        gamma: f64,
        sigma: f64,
        k: f64,
        t_end: f64,
        max_inventory: i64,
    ) -> Self {
        Self {
            gamma,
            sigma,
            k,
            t_end,
            inventory: 0,
            mid_price: 0.0,
            max_inventory,
        }
    }

    /// Обновить текущую mid-price
    pub fn update_mid_price(&mut self, price: f64) {
        self.mid_price = price;
    }

    /// Обновить позицию
    pub fn update_inventory(&mut self, delta: i64) {
        self.inventory += delta;
    }

    /// Reservation price - "справедливая" цена с учётом inventory
    pub fn reservation_price(&self, t: f64) -> f64 {
        let time_to_end = (self.t_end - t).max(0.0);
        self.mid_price - (self.inventory as f64) * self.gamma
            * self.sigma.powi(2) * time_to_end
    }

    /// Оптимальный спред
    pub fn optimal_spread(&self, t: f64) -> f64 {
        let time_to_end = (self.t_end - t).max(0.0);

        self.gamma * self.sigma.powi(2) * time_to_end
            + (2.0 / self.gamma) * (1.0 + self.gamma / self.k).ln()
    }

    /// Вычислить котировки bid и ask
    pub fn quotes(&self, t: f64) -> (f64, f64) {
        let r = self.reservation_price(t);
        let spread = self.optimal_spread(t);

        let bid = r - spread / 2.0;
        let ask = r + spread / 2.0;

        (bid, ask)
    }

    /// Котировки с ограничением inventory
    pub fn quotes_with_inventory_control(&self, t: f64) -> QuoteDecision {
        let (mut bid, mut ask) = self.quotes(t);

        // Если позиция слишком большая - не выставляем bid
        let quote_bid = self.inventory < self.max_inventory;
        // Если позиция слишком отрицательная - не выставляем ask
        let quote_ask = self.inventory > -self.max_inventory;

        // Дополнительная корректировка при приближении к лимитам
        if self.inventory > self.max_inventory / 2 {
            // Агрессивнее продаём
            ask -= 0.0001 * (self.inventory - self.max_inventory / 2) as f64;
        }
        if self.inventory < -self.max_inventory / 2 {
            // Агрессивнее покупаем
            bid += 0.0001 * (-self.inventory - self.max_inventory / 2) as f64;
        }

        QuoteDecision {
            bid_price: if quote_bid { Some(bid) } else { None },
            ask_price: if quote_ask { Some(ask) } else { None },
        }
    }
}

/// Решение о котировках
#[derive(Debug)]
pub struct QuoteDecision {
    pub bid_price: Option<f64>,
    pub ask_price: Option<f64>,
}

impl QuoteDecision {
    pub fn to_orders(&self, volume: u64, tick_size: f64) -> Vec<Order> {
        let mut orders = Vec::new();

        if let Some(bid) = self.bid_price {
            orders.push(Order {
                id: 0, // Будет присвоен биржей
                price: (bid / tick_size).round() as i64,
                volume,
                side: Side::Buy,
                timestamp: std::time::Instant::now(),
                order_type: OrderType::Limit,
            });
        }

        if let Some(ask) = self.ask_price {
            orders.push(Order {
                id: 0,
                price: (ask / tick_size).round() as i64,
                volume,
                side: Side::Sell,
                timestamp: std::time::Instant::now(),
                order_type: OrderType::Limit,
            });
        }

        orders
    }
}
```

### 2.4.4 Оценка параметров

```rust
impl AvellanedaStoikov {
    /// Оценка волатильности методом realized variance
    pub fn estimate_volatility(prices: &[f64], dt: f64) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        // Логарифмические возвраты
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        // Realized variance
        let sum_sq: f64 = returns.iter().map(|r| r * r).sum();
        let realized_var = sum_sq / dt;  // Приводим к годовой волатильности

        realized_var.sqrt()
    }

    /// Оценка параметра k из потока ордеров
    pub fn estimate_k(
        order_distances: &[f64],  // Расстояния ордеров от mid-price
        fill_flags: &[bool],      // Был ли ордер исполнен
    ) -> f64 {
        // k определяет, как быстро убывает вероятность исполнения
        // λ(δ) = A * exp(-k * δ)

        // Логистическая регрессия P(fill | δ) = 1 / (1 + exp(k*δ - b))
        // Упрощённо: k ≈ -d(log P)/dδ

        let mut sum_delta = 0.0;
        let mut sum_delta_sq = 0.0;
        let mut sum_fill_delta = 0.0;
        let mut fills = 0.0;
        let n = order_distances.len() as f64;

        for (&delta, &filled) in order_distances.iter().zip(fill_flags) {
            sum_delta += delta;
            sum_delta_sq += delta * delta;
            if filled {
                sum_fill_delta += delta;
                fills += 1.0;
            }
        }

        // Эвристическая оценка
        if fills > 0.0 {
            let mean_fill_delta = sum_fill_delta / fills;
            let mean_delta = sum_delta / n;

            // k обратно пропорционален среднему расстоянию исполненных ордеров
            1.0 / mean_fill_delta.max(0.0001)
        } else {
            1.0  // Значение по умолчанию
        }
    }
}
```

---

## 2.5 LOB Симулятор

### 2.5.1 Архитектура симулятора

```rust
use std::collections::VecDeque;

/// События в симуляторе
#[derive(Debug, Clone)]
pub enum SimulationEvent {
    /// Пришёл лимитный ордер
    LimitOrder { side: Side, price: i64, volume: u64 },
    /// Пришёл рыночный ордер
    MarketOrder { side: Side, volume: u64 },
    /// Отмена ордера
    CancelOrder { order_id: u64 },
    /// Изменение mid-price
    PriceMove { direction: i32 },
}

/// Симулятор LOB на основе Hawkes процессов
pub struct LOBSimulator {
    /// Order book
    book: OrderBook,
    /// Процесс для bid ордеров
    bid_limit_process: HawkesProcess,
    /// Процесс для ask ордеров
    ask_limit_process: HawkesProcess,
    /// Процесс для рыночных ордеров
    market_order_process: HawkesProcess,
    /// Процесс для отмен
    cancel_process: HawkesProcess,
    /// Текущее время симуляции
    current_time: f64,
    /// Текущая mid-price
    mid_price: f64,
    /// История событий
    event_log: Vec<(f64, SimulationEvent)>,
    /// Генератор случайных чисел
    rng: rand::rngs::ThreadRng,
}

impl LOBSimulator {
    pub fn new(initial_mid_price: f64) -> Self {
        Self {
            book: OrderBook::new(
                rust_decimal::Decimal::new(1, 2),  // tick = 0.01
                rust_decimal::Decimal::new(1, 0),  // lot = 1
            ),
            bid_limit_process: HawkesProcess::new(10.0, 0.3, 1.0),
            ask_limit_process: HawkesProcess::new(10.0, 0.3, 1.0),
            market_order_process: HawkesProcess::new(2.0, 0.5, 2.0),
            cancel_process: HawkesProcess::new(5.0, 0.2, 0.5),
            current_time: 0.0,
            mid_price: initial_mid_price,
            event_log: Vec::new(),
            rng: rand::thread_rng(),
        }
    }

    /// Симулировать до времени t_end
    pub fn simulate(&mut self, t_end: f64) -> &[(f64, SimulationEvent)] {
        use rand::Rng;

        while self.current_time < t_end {
            // Находим время следующего события (минимум по всем процессам)
            let (next_time, event_type) = self.next_event(t_end);

            if next_time > t_end {
                break;
            }

            self.current_time = next_time;

            // Обрабатываем событие
            let event = self.generate_event(event_type);
            self.process_event(&event);
            self.event_log.push((self.current_time, event));
        }

        &self.event_log
    }

    /// Найти время следующего события
    fn next_event(&mut self, t_max: f64) -> (f64, EventType) {
        use rand::Rng;

        // Для каждого процесса генерируем потенциальное время
        // и выбираем минимальное

        let bid_time = self.sample_next_event_time(&self.bid_limit_process, t_max);
        let ask_time = self.sample_next_event_time(&self.ask_limit_process, t_max);
        let market_time = self.sample_next_event_time(&self.market_order_process, t_max);
        let cancel_time = self.sample_next_event_time(&self.cancel_process, t_max);

        let mut min_time = t_max;
        let mut event_type = EventType::None;

        if bid_time < min_time {
            min_time = bid_time;
            event_type = EventType::BidLimit;
        }
        if ask_time < min_time {
            min_time = ask_time;
            event_type = EventType::AskLimit;
        }
        if market_time < min_time {
            min_time = market_time;
            event_type = EventType::MarketOrder;
        }
        if cancel_time < min_time {
            min_time = cancel_time;
            event_type = EventType::Cancel;
        }

        (min_time, event_type)
    }

    fn sample_next_event_time(&mut self, process: &HawkesProcess, t_max: f64) -> f64 {
        use rand::Rng;

        // Thinning algorithm
        let lambda_max = process.intensity(self.current_time) + process.alpha;
        let u: f64 = self.rng.gen();
        let dt = -u.ln() / lambda_max;
        let candidate = self.current_time + dt;

        if candidate > t_max {
            return f64::INFINITY;
        }

        let lambda_t = process.intensity(candidate);
        let accept_prob = lambda_t / lambda_max;

        if self.rng.gen::<f64>() < accept_prob {
            candidate
        } else {
            f64::INFINITY
        }
    }

    fn generate_event(&mut self, event_type: EventType) -> SimulationEvent {
        use rand::Rng;

        match event_type {
            EventType::BidLimit => {
                // Генерируем цену на случайном расстоянии от mid
                let delta = self.rng.gen_range(1..=10);
                let price = ((self.mid_price - delta as f64 * 0.01) * 100.0) as i64;
                let volume = self.rng.gen_range(1..=100);

                SimulationEvent::LimitOrder {
                    side: Side::Buy,
                    price,
                    volume,
                }
            }
            EventType::AskLimit => {
                let delta = self.rng.gen_range(1..=10);
                let price = ((self.mid_price + delta as f64 * 0.01) * 100.0) as i64;
                let volume = self.rng.gen_range(1..=100);

                SimulationEvent::LimitOrder {
                    side: Side::Sell,
                    price,
                    volume,
                }
            }
            EventType::MarketOrder => {
                let side = if self.rng.gen() { Side::Buy } else { Side::Sell };
                let volume = self.rng.gen_range(1..=50);

                SimulationEvent::MarketOrder { side, volume }
            }
            EventType::Cancel => {
                SimulationEvent::CancelOrder { order_id: 0 }
            }
            EventType::None => panic!("Invalid event type"),
        }
    }

    fn process_event(&mut self, event: &SimulationEvent) {
        match event {
            SimulationEvent::LimitOrder { side, price, volume } => {
                let order = Order {
                    id: self.rng.gen(),
                    price: *price,
                    volume: *volume,
                    side: *side,
                    timestamp: std::time::Instant::now(),
                    order_type: OrderType::Limit,
                };
                self.book.add_limit_order(order);
            }
            SimulationEvent::MarketOrder { side, volume } => {
                let trades = self.book.execute_market_order(*side, *volume);
                // Обновляем mid-price после сделок
                if let Some(mid) = self.book.mid_price() {
                    self.mid_price = mid / 100.0;
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EventType {
    None,
    BidLimit,
    AskLimit,
    MarketOrder,
    Cancel,
}
```

---

## 2.6 Извлечение признаков из LOB

### 2.6.1 Основные признаки

```rust
/// Признаки, извлекаемые из order book
#[derive(Debug, Clone)]
pub struct LOBFeatures {
    /// Mid-price
    pub mid_price: f64,
    /// Microprice (взвешенная средняя)
    pub microprice: f64,
    /// Спред
    pub spread: f64,
    /// Дисбаланс объёмов на лучших уровнях
    pub imbalance_l1: f64,
    /// Дисбаланс объёмов на первых 5 уровнях
    pub imbalance_l5: f64,
    /// Глубина bid (суммарный объём на N уровнях)
    pub bid_depth: f64,
    /// Глубина ask
    pub ask_depth: f64,
    /// Давление на книгу (интеграл объёма по уровням)
    pub book_pressure: f64,
}

impl OrderBook {
    /// Извлечь признаки из текущего состояния книги
    pub fn extract_features(&self, levels: usize) -> Option<LOBFeatures> {
        let best_bid = self.best_bid()?;
        let best_ask = self.best_ask()?;

        let mid_price = (best_bid + best_ask) as f64 / 2.0;
        let spread = (best_ask - best_bid) as f64;

        // Объёмы на лучших уровнях
        let bid_vol_l1 = self.bids.get(&best_bid)?.total_volume as f64;
        let ask_vol_l1 = self.asks.get(&best_ask)?.total_volume as f64;

        let microprice = (bid_vol_l1 * best_ask as f64 + ask_vol_l1 * best_bid as f64)
            / (bid_vol_l1 + ask_vol_l1);

        // Imbalance L1
        let imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / (bid_vol_l1 + ask_vol_l1);

        // Суммируем объёмы на первых N уровнях
        let bid_levels: Vec<_> = self.bids.iter().rev().take(levels).collect();
        let ask_levels: Vec<_> = self.asks.iter().take(levels).collect();

        let bid_depth: f64 = bid_levels.iter()
            .map(|(_, level)| level.total_volume as f64)
            .sum();
        let ask_depth: f64 = ask_levels.iter()
            .map(|(_, level)| level.total_volume as f64)
            .sum();

        let imbalance_l5 = if bid_depth + ask_depth > 0.0 {
            (bid_depth - ask_depth) / (bid_depth + ask_depth)
        } else {
            0.0
        };

        // Book pressure: взвешенная сумма объёмов
        // Веса уменьшаются с расстоянием от лучшей цены
        let mut book_pressure = 0.0;

        for (i, (price, level)) in bid_levels.iter().enumerate() {
            let weight = 1.0 / (1.0 + i as f64);
            book_pressure += weight * level.total_volume as f64;
        }
        for (i, (price, level)) in ask_levels.iter().enumerate() {
            let weight = 1.0 / (1.0 + i as f64);
            book_pressure -= weight * level.total_volume as f64;
        }

        Some(LOBFeatures {
            mid_price,
            microprice,
            spread,
            imbalance_l1,
            imbalance_l5,
            bid_depth,
            ask_depth,
            book_pressure,
        })
    }
}
```

### 2.6.2 Динамические признаки

```rust
/// Признаки, требующие истории
pub struct DynamicFeatures {
    /// История mid-price
    mid_prices: VecDeque<f64>,
    /// История imbalance
    imbalances: VecDeque<f64>,
    /// Максимальный размер истории
    max_history: usize,
}

impl DynamicFeatures {
    pub fn new(max_history: usize) -> Self {
        Self {
            mid_prices: VecDeque::with_capacity(max_history),
            imbalances: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Добавить новое наблюдение
    pub fn update(&mut self, features: &LOBFeatures) {
        if self.mid_prices.len() >= self.max_history {
            self.mid_prices.pop_front();
            self.imbalances.pop_front();
        }

        self.mid_prices.push_back(features.mid_price);
        self.imbalances.push_back(features.imbalance_l1);
    }

    /// Realized volatility (за последние N наблюдений)
    pub fn realized_volatility(&self, n: usize) -> f64 {
        let prices: Vec<_> = self.mid_prices.iter()
            .rev()
            .take(n + 1)
            .copied()
            .collect();

        if prices.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[0] / w[1]).ln())  // Обратный порядок из-за rev()
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Тренд imbalance (производная)
    pub fn imbalance_trend(&self, n: usize) -> f64 {
        let values: Vec<_> = self.imbalances.iter()
            .rev()
            .take(n)
            .copied()
            .collect();

        if values.len() < 2 {
            return 0.0;
        }

        // Простая линейная регрессия
        let n_f = values.len() as f64;
        let sum_x = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum();
        let sum_xx: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        // slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)
        (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x + 1e-10)
    }
}
```

---

## 2.7 Работа с реальными данными

### 2.7.1 Подключение к бирже через WebSocket

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};

/// Сообщение обновления глубины рынка (Binance format)
#[derive(Debug, Deserialize)]
pub struct DepthUpdate {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: u64,
    #[serde(rename = "u")]
    pub final_update_id: u64,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,  // [price, quantity]
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
}

/// WebSocket клиент для получения данных order book
pub struct BinanceWSClient {
    symbol: String,
    book: OrderBook,
    last_update_id: u64,
}

impl BinanceWSClient {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_lowercase(),
            book: OrderBook::new(
                rust_decimal::Decimal::new(1, 8),
                rust_decimal::Decimal::new(1, 8),
            ),
            last_update_id: 0,
        }
    }

    /// Запустить получение данных
    pub async fn run<F>(&mut self, mut callback: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(&OrderBook, &LOBFeatures),
    {
        let url = format!(
            "wss://stream.binance.com:9443/ws/{}@depth@100ms",
            self.symbol
        );

        let (ws_stream, _) = connect_async(&url).await?;
        let (mut write, mut read) = ws_stream.split();

        println!("Подключено к {}", url);

        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(update) = serde_json::from_str::<DepthUpdate>(&text) {
                        self.apply_update(&update);

                        if let Some(features) = self.book.extract_features(5) {
                            callback(&self.book, &features);
                        }
                    }
                }
                Ok(Message::Ping(data)) => {
                    write.send(Message::Pong(data)).await?;
                }
                Err(e) => {
                    eprintln!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn apply_update(&mut self, update: &DepthUpdate) {
        // Применяем обновления к книге
        for bid in &update.bids {
            let price: f64 = bid[0].parse().unwrap_or(0.0);
            let volume: f64 = bid[1].parse().unwrap_or(0.0);
            let price_ticks = (price * 1e8) as i64;
            let volume_units = (volume * 1e8) as u64;

            // volume = 0 означает удаление уровня
            // volume > 0 означает обновление/добавление
            // (упрощённая логика)
        }

        // Аналогично для asks
        for ask in &update.asks {
            let price: f64 = ask[0].parse().unwrap_or(0.0);
            let volume: f64 = ask[1].parse().unwrap_or(0.0);
            // ...
        }

        self.last_update_id = update.final_update_id;
    }
}
```

### 2.7.2 Пример использования

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = BinanceWSClient::new("btcusdt");

    let mut strategy = AvellanedaStoikov::new(
        0.01,   // gamma
        0.02,   // sigma (2% волатильность)
        1.0,    // k
        1.0,    // t_end (1 год = 1.0)
        10,     // max_inventory
    );

    let start_time = std::time::Instant::now();

    client.run(|book, features| {
        // Обновляем стратегию
        strategy.update_mid_price(features.mid_price);

        // Получаем котировки
        let t = start_time.elapsed().as_secs_f64() / (365.25 * 24.0 * 3600.0);
        let quotes = strategy.quotes_with_inventory_control(t);

        println!(
            "Mid: {:.2}, Spread: {:.4}, Imbalance: {:.3}, Quotes: {:?}",
            features.mid_price,
            features.spread,
            features.imbalance_l1,
            quotes
        );
    }).await?;

    Ok(())
}
```

---

## 2.8 Практические задания

### Задание 2.1: Реализация Order Book

Создайте полную реализацию order book с поддержкой:
- Всех типов ордеров (limit, market, stop)
- Различных TimeInForce (GTC, IOC, FOK)
- Matching engine с price-time priority

**Критерии:**
- Обработка одного сообщения < 1 мкс
- Корректная работа с граничными случаями

### Задание 2.2: Калибровка Hawkes процесса

Используя исторические данные сделок:
1. Загрузите данные торгов за 1 день
2. Калибруйте параметры Hawkes процесса (μ, α, β)
3. Проверьте качество fit через Q-Q plot

### Задание 2.3: Backtesting Market Making

1. Реализуйте симулятор с реалистичным order flow
2. Протестируйте стратегию Avellaneda-Stoikov
3. Постройте equity curve и вычислите Sharpe ratio
4. Проанализируйте чувствительность к параметру γ

### Задание 2.4: Real-time бот

Создайте paper trading бота:
1. Подключение к testnet биржи
2. Автоматическое обновление котировок
3. Risk мониторинг (inventory, P&L)
4. Graceful shutdown

---

## 2.9 Часто допускаемые ошибки

### Ошибка 1: Игнорирование задержки

```rust
// ❌ НЕПРАВИЛЬНО: предполагаем мгновенное исполнение
let (bid, ask) = strategy.quotes(t);
place_order(bid, volume);  // К моменту исполнения цена уже изменится!

// ✅ ПРАВИЛЬНО: учитываем задержку
let estimated_latency = 0.001; // 1ms в годовых единицах
let (bid, ask) = strategy.quotes(t + estimated_latency);
```

### Ошибка 2: Неправильный учёт inventory

```rust
// ❌ НЕПРАВИЛЬНО: не обновляем inventory при частичном исполнении
fn on_fill(&mut self, trade: &Trade) {
    // Забыли обновить inventory!
}

// ✅ ПРАВИЛЬНО
fn on_fill(&mut self, trade: &Trade) {
    match trade.side {
        Side::Buy => self.inventory += trade.volume as i64,
        Side::Sell => self.inventory -= trade.volume as i64,
    }
}
```

### Ошибка 3: Branching ratio >= 1

```rust
// ❌ НЕПРАВИЛЬНО: процесс нестационарен, взорвётся
let hawkes = HawkesProcess::new(1.0, 2.0, 1.0);  // α/β = 2 > 1

// ✅ ПРАВИЛЬНО
let hawkes = HawkesProcess::new(1.0, 0.5, 1.0);  // α/β = 0.5 < 1
```

---

## Заключение

В этой главе мы изучили:

1. **Структуру Order Book** и эффективные структуры данных для её представления
2. **Point processes** (Пуассон, Хоукс) для моделирования потока ордеров
3. **Модель Cont-Stoikov-Talreja** для анализа динамики LOB
4. **Стратегию Avellaneda-Stoikov** для оптимального market making
5. **Практические аспекты** работы с реальными биржевыми данными

Эти знания являются фундаментом для создания собственных торговых стратегий, работающих на уровне микроструктуры рынка.

---

## Рекомендуемая литература

1. **Cartea Á., Jaimungal S., Penalva J.** "Algorithmic and High-Frequency Trading" (2015)
2. **Lehalle C.-A., Laruelle S.** "Market Microstructure in Practice" (2018)
3. **Bouchaud J.-P. et al.** "Trades, Quotes and Prices" (2018)
4. **Avellaneda M., Stoikov S.** "High-frequency trading in a limit order book" (2008)
5. **Cont R., Stoikov S., Talreja R.** "A stochastic model for order book dynamics" (2010)
