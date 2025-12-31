# Глава 2: Микроструктура рынка и моделирование Order Book

## Метаданные
- **Уровень сложности**: Продвинутый
- **Предварительные требования**: Глава 1 (стохастическое исчисление), теория очередей, point processes
- **Языки реализации**: Rust (основной), Go (для сетевых компонентов), Julia
- **Расчётный объём**: 100-140 страниц

---

## Цели главы

1. Понять математику микроструктуры рынка на уровне отдельных ордеров
2. Освоить моделирование limit order book с использованием point processes
3. Реализовать market making стратегию Avellaneda-Stoikov и её расширения
4. Создать симулятор order book для backtesting с реалистичной динамикой
5. Научиться работать с реальными L2/L3 данными бирж

---

## Научная база

### Фундаментальные работы
1. **Avellaneda M., Stoikov S.** (2008) "High-frequency trading in a limit order book" — Quantitative Finance, 8(3)
2. **Cont R., Stoikov S., Talreja R.** (2010) "A stochastic model for order book dynamics" — Operations Research
3. **Guéant O., Lehalle C.-A., Fernandez-Tapia J.** (2013) "Dealing with the inventory risk" — Mathematics and Financial Economics
4. **Cartea Á., Jaimungal S., Penalva J.** (2015) "Algorithmic and High-Frequency Trading" — Cambridge University Press

### Современные исследования (2023-2025)
5. **Cont R., Degond P., Xuan L.** (2025) "A Mathematical Framework for Modelling Order Book Dynamics" — SIAM J. Financial Mathematics, 16(1):123-166
6. **Jain K., Firoozye N., Kochems J., Treleaven P.** (2024) "Limit Order Book dynamics and order size modelling using Compound Hawkes Process" — Finance Research Letters
7. **Zhang Z., Zohren S.** (2021) "Deep Learning for Market by Order Data" — Applied Mathematical Finance
8. **Morariu-Patrichi M., Pakkanen M.S.** (2022) "State-dependent Hawkes processes for LOB modelling" — Quantitative Finance
9. Новые работы 2024 по reinforcement learning для market making (PLOS One, arXiv)

### Point Processes
10. **Hawkes A.G.** (2018) "Hawkes processes and their applications to finance: a review" — Quantitative Finance
11. **Bacry E., Mastromatteo I., Muzy J.-F.** (2015) "Hawkes processes in finance" — Market Microstructure and Liquidity

---

## Структура главы

### 2.1 Анатомия Limit Order Book

**2.1.1 Базовые концепции**
- Order types: market, limit, stop, iceberg, FOK, IOC
- Price-time priority (FIFO matching)
- Bid-ask spread, mid-price, microprice
- Tick size и его влияние на динамику

**2.1.2 Математическое описание**
```
Состояние LOB: X(t) = (a(t), b(t)) 
где:
  a(t) = (a₁(t), ..., aₖ(t)) — объёмы на ask уровнях
  b(t) = (b₁(t), ..., bₖ(t)) — объёмы на bid уровнях
```

**2.1.3 Структуры данных для LOB**
```rust
#[repr(C, align(64))]  // Cache-line aligned
pub struct PriceLevel {
    price: Decimal,      // Fixed-point arithmetic
    volume: u64,
    order_count: u32,
    _padding: [u8; 20],  // Fill to 64 bytes
}

pub struct OrderBook {
    bids: BTreeMap<Decimal, PriceLevel>,  // Sorted, O(log n) operations
    asks: BTreeMap<Decimal, PriceLevel>,
    // Или custom skip-list для O(1) best bid/ask access
}
```

---

### 2.2 Point Processes для моделирования Order Flow

**2.2.1 Poisson Processes**
- Homogeneous vs inhomogeneous Poisson
- Calibration к реальным данным
- Ограничения: не захватывают clustering

**2.2.2 Hawkes Processes**
```
Интенсивность: λ(t) = μ + Σᵢ α·φ(t - tᵢ)
где φ(t) = β·exp(-β·t) — kernel function
```

**Свойства:**
- Self-exciting: события порождают новые события
- Branching ratio: n = α/β (должен быть < 1 для стационарности)
- Multivariate версия для cross-excitation между bid/ask

**Реализация на Rust:**
```rust
pub struct HawkesProcess {
    mu: f64,           // Base intensity
    alpha: f64,        // Excitation
    beta: f64,         // Decay
    history: Vec<f64>, // Event times
}

impl HawkesProcess {
    pub fn intensity(&self, t: f64) -> f64 {
        let mut lambda = self.mu;
        for &ti in &self.history {
            if ti < t {
                lambda += self.alpha * self.beta * (-self.beta * (t - ti)).exp();
            }
        }
        lambda
    }
    
    pub fn simulate(&mut self, t_end: f64) -> Vec<f64>;
    pub fn calibrate_mle(&mut self, events: &[f64]) -> Result<(), CalibrationError>;
}
```

**2.2.3 Compound Hawkes Processes**
- Моделирование размеров ордеров вместе со временем
- Jain et al. (2024) framework

**2.2.4 State-Dependent Hawkes**
- Интенсивность зависит от состояния LOB
- Morariu-Patrichi & Pakkanen (2022)

---

### 2.3 Модель Cont-Stoikov-Talreja

**Основные идеи:**
- LOB как система массового обслуживания (queueing system)
- Birth-death process для очередей на каждом уровне цены
- Laplace transform методы для аналитических результатов

**Ключевые вероятности:**
1. P(mid-price увеличится | текущее состояние LOB)
2. P(исполнение на bid до движения ask)
3. P("making the spread")

**Калибровка:**
- Intensity функции из эмпирических данных
- Arrival rates: λ(δ) = A·exp(-k·δ) — экспоненциальное убывание

---

### 2.4 Оптимальный Market Making: Avellaneda-Stoikov

**2.4.1 Постановка задачи**
```
Максимизировать: E[-exp(-γ·W_T)]
где W_T = X_T + q_T·S_T — терминальное богатство
     q_T — inventory в момент T
     γ — risk aversion
```

**2.4.2 Reservation price и optimal spread**
```
r(t, s, q) = s - q·γ·σ²·(T-t)    # Reservation price

δ⁺ + δ⁻ = γσ²(T-t) + (2/γ)·ln(1 + γ/k)    # Optimal spread
```

**Интерпретация:**
- При q > 0 (длинная позиция): склонны продавать дешевле
- При q < 0 (короткая позиция): склонны покупать дороже
- Spread сжимается к концу торговой сессии

**2.4.3 Hamilton-Jacobi-Bellman уравнение**
```
∂V/∂t + (σ²/2)·∂²V/∂s² 
    + max_{δ⁺}[λ(δ⁺)·(V(t,s,q-1,x+s+δ⁺) - V)]
    + max_{δ⁻}[λ(δ⁻)·(V(t,s,q+1,x-s+δ⁻) - V)] = 0
```

**2.4.4 Расширения модели**

| Расширение | Автор(ы) | Ключевая идея |
|------------|----------|---------------|
| Inventory constraints | Guéant et al. (2013) | Bounded inventory |
| Non-martingale mid-price | Various | Directional bets |
| Multiple assets | Guéant & Manziuk (2017) | Corporate bonds MM |
| Reinforcement Learning | PLOS One (2022) | Adaptive γ parameter |
| Ergodic formulation | arXiv 2024 | Online learning |

---

### 2.5 Реализация Market Making стратегии

**2.5.1 Архитектура системы**
```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Market Data    │────>│   Strategy   │────>│  Order Manager  │
│  (WebSocket)    │     │   Engine     │     │                 │
└─────────────────┘     └──────────────┘     └─────────────────┘
        │                      │                     │
        v                      v                     v
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Order Book     │     │   Risk       │     │  Exchange       │
│  Reconstruction │     │   Monitor    │     │  Connector      │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

**2.5.2 Rust реализация core components**

```rust
pub struct AvellanedaStoikov {
    gamma: f64,           // Risk aversion
    sigma: f64,           // Volatility estimate
    k: f64,               // Order arrival decay
    t_end: f64,           // Session end time
    inventory: i32,       // Current position
    mid_price: f64,       // Current mid price
}

impl AvellanedaStoikov {
    pub fn reservation_price(&self, t: f64) -> f64 {
        self.mid_price - (self.inventory as f64) * self.gamma 
            * self.sigma.powi(2) * (self.t_end - t)
    }
    
    pub fn optimal_spread(&self, t: f64) -> f64 {
        self.gamma * self.sigma.powi(2) * (self.t_end - t)
            + (2.0 / self.gamma) * (1.0 + self.gamma / self.k).ln()
    }
    
    pub fn quotes(&self, t: f64) -> (f64, f64) {
        let r = self.reservation_price(t);
        let spread = self.optimal_spread(t);
        let bid = r - spread / 2.0;
        let ask = r + spread / 2.0;
        (bid, ask)
    }
}
```

**2.5.3 Parameter estimation**
- Volatility: realized variance с bias correction
- k parameter: MLE из order arrival данных
- Risk aversion γ: hyperparameter, tune через backtesting

---

### 2.6 LOB Simulation Framework

**2.6.1 Требования к симулятору**
- Реалистичная order arrival динамика
- Price impact awareness
- Latency modeling
- Детерминированное воспроизведение (для debugging)

**2.6.2 Agent-based vs Stochastic**
| Подход | Плюсы | Минусы |
|--------|-------|--------|
| Agent-based | Интуитивный, гибкий | Много параметров |
| Stochastic | Математически строгий | Может не захватить все эффекты |
| Hybrid | Лучшее из обоих | Сложность |

**2.6.3 Реализация симулятора**
```rust
pub trait OrderBookSimulator {
    fn step(&mut self, dt: f64) -> Vec<OrderBookEvent>;
    fn insert_order(&mut self, order: Order) -> ExecutionReport;
    fn get_state(&self) -> OrderBookSnapshot;
}

pub struct HawkesLOBSimulator {
    bid_process: HawkesProcess,
    ask_process: HawkesProcess,
    cancel_process: HawkesProcess,
    market_order_process: HawkesProcess,
    book: OrderBook,
}
```

---

### 2.7 Работа с реальными данными

**2.7.1 Источники L2/L3 данных**
- Crypto: Binance, Bybit, Hyperliquid WebSocket feeds
- Equities: NYSE OpenBook, NASDAQ ITCH
- Futures: CME Market Data

**2.7.2 Data engineering pipeline**
```
Raw tick data → Parsing → Validation → Book reconstruction → Feature extraction
     │
     v
   Storage: Arctic/TimescaleDB/ClickHouse
```

**2.7.3 Feature extraction из LOB**
| Feature | Формула | Интерпретация |
|---------|---------|---------------|
| Microprice | (bid_vol × ask_price + ask_vol × bid_price) / (bid_vol + ask_vol) | Better mid estimate |
| Order imbalance | (bid_vol - ask_vol) / (bid_vol + ask_vol) | Directional pressure |
| Weighted mid | Σ(price × vol) / Σ(vol) for top N levels | Depth-weighted |
| Book pressure | Integral of volume over price levels | Supply/demand balance |

---

## Инструментарий

### Rust crates
```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }  # Async runtime
tungstenite = "0.20"      # WebSocket
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rust_decimal = "1.32"     # Fixed-point decimals
crossbeam = "0.8"         # Lock-free data structures
dashmap = "5.5"           # Concurrent hashmap
hdrhistogram = "7.5"      # Latency histograms
```

### Go для networking (optional)
```go
// High-performance WebSocket multiplexer
// Zero-copy deserialization
```

### Julia для research
```julia
using HawkesProcesses
using MarketMicrostructure  # If available
using Plots
```

---

## Практические задания

### Задание 2.1: Order Book Reconstruction
**Цель:** Реконструировать LOB из потока сообщений биржи
- Входные данные: Binance depth stream (1000ms snapshots + updates)
- Требования: <100μs per message processing
- Проверка: cross-validation с snapshot-ами

### Задание 2.2: Hawkes Process Calibration
**Цель:** Калибровать Hawkes процесс к trade data
- Данные: 1 день торгов BTC-USDT
- Метод: Maximum Likelihood Estimation
- Валидация: Q-Q plots, KS test

### Задание 2.3: Avellaneda-Stoikov Backtesting
**Цель:** Backtest market making стратегии
- Симулятор с реалистичным price impact
- P&L analysis, Sharpe ratio
- Sensitivity analysis по параметрам (γ, k)

### Задание 2.4: Real-time Market Making Bot
**Цель:** Запустить paper trading бот
- Подключение к testnet биржи
- Real-time parameter adaptation
- Risk monitoring dashboard

---

## Критерии оценки

1. **Математическая корректность**: верная реализация HJB, правильные Hawkes интенсивности
2. **Производительность**: обработка 100k+ messages/sec
3. **Robustness**: handling disconnects, order rejection, partial fills
4. **Реалистичность**: симуляции соответствуют stylized facts реальных рынков

---

## Связи с другими главами

| Глава | Связь |
|-------|-------|
| 01-stochastic-calculus | SDE для mid-price dynamics |
| 03-portfolio-optimization | Multi-asset market making |
| 04-ml-time-series | Deep learning для order flow prediction |
| 05-low-latency-systems | Production infrastructure |

---

## Рекомендуемая литература

### Учебники
1. Cartea Á., Jaimungal S., Penalva J. "Algorithmic and High-Frequency Trading" (2015)
2. Lehalle C.-A., Laruelle S. "Market Microstructure in Practice" (2018)
3. Bouchaud J.-P. et al. "Trades, Quotes and Prices" (2018)

### Онлайн ресурсы
1. Oxford Man Institute papers
2. arXiv q-fin (quantitative finance) section
3. SSRN finance papers

---

## Заметки по написанию

- Включить реальные примеры из crypto markets (Binance, Bybit)
- Показать типичные ошибки: неправильный inventory management, игнорирование latency
- Добавить секцию "When NOT to use Avellaneda-Stoikov" (trending markets, etc.)
- Case study: market making во время high volatility events
- Code должен быть готов к production, с proper error handling
