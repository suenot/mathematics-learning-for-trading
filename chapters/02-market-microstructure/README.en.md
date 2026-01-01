# Chapter 2: Market Microstructure and Order Book Modeling

## Metadata
- **Difficulty Level**: Advanced
- **Prerequisites**: Chapter 1 (Stochastic Calculus), Probability Theory, Basic Trading Knowledge
- **Implementation Languages**: Rust (primary)
- **Estimated Volume**: 100-140 pages

---

## Introduction

Market microstructure studies how specific trading mechanisms affect price formation. While Chapter 1 treated price as a continuous stochastic process, here we "zoom in" and see that prices change discretely — each trade, each order affects them.

**Why is this important for traders?**
- Understanding microstructure enables optimal order execution
- Market making strategies are built on order book mathematics
- Short-term price prediction requires order flow analysis

---

## 2.1 Anatomy of the Limit Order Book (LOB)

### 2.1.1 What is an Order Book?

An Order Book is a data structure that stores all active limit buy and sell orders for an asset.

```
         ASKS (sellers)
    ┌─────────────────────────┐
    │ $102.50  |  150 shares  │  ← Best Ask (best sell price)
    │ $102.75  |  300 shares  │
    │ $103.00  |  500 shares  │
    └─────────────────────────┘

    ═══════════════════════════  SPREAD = $0.50

    ┌─────────────────────────┐
    │ $102.00  |  200 shares  │  ← Best Bid (best buy price)
    │ $101.75  |  400 shares  │
    │ $101.50  |  250 shares  │
    └─────────────────────────┘
         BIDS (buyers)
```

### 2.1.2 Order Types

```rust
/// Main order types in a trading system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    /// Limit order - executes at specified price or better
    Limit,
    /// Market order - executes immediately at best available price
    Market,
    /// Stop order - becomes market order when trigger price is reached
    Stop,
    /// Stop-limit - becomes limit order when trigger price is reached
    StopLimit,
    /// Iceberg - shows only a portion of the volume
    Iceberg,
}

/// Order direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Time in force conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good Till Cancel - active until cancelled
    GTC,
    /// Immediate Or Cancel - execute immediately (partially) or cancel
    IOC,
    /// Fill Or Kill - execute completely or cancel
    FOK,
    /// Good Till Date - active until specified date
    GTD,
}
```

### 2.1.3 Matching Engine and Order Priority

Exchanges use **Price-Time Priority (FIFO)** to determine execution order:

1. **Price Priority**: Better price executes first
2. **Time Priority**: At same price — earlier order wins

```rust
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Instant;

/// Price level with order queue
#[derive(Debug)]
pub struct PriceLevel {
    pub price: i64,           // Price in minimum units (ticks)
    pub total_volume: u64,    // Total volume at this level
    pub orders: Vec<Order>,   // Order queue (FIFO)
}

/// Individual order
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
        // Sort by time (earlier = higher priority)
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

### 2.1.4 Order Book Data Structure

```rust
use rust_decimal::Decimal;
use std::collections::BTreeMap;

/// High-performance Order Book structure
pub struct OrderBook {
    /// Buy orders (bids), sorted by descending price
    /// BTreeMap guarantees O(log n) operations
    bids: BTreeMap<i64, PriceLevel>,

    /// Sell orders (asks), sorted by ascending price
    asks: BTreeMap<i64, PriceLevel>,

    /// Tick size (minimum price change)
    tick_size: Decimal,

    /// Lot size (minimum volume)
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

    /// Best bid price
    pub fn best_bid(&self) -> Option<i64> {
        self.bids.keys().next_back().copied()
    }

    /// Best ask price
    pub fn best_ask(&self) -> Option<i64> {
        self.asks.keys().next().copied()
    }

    /// Spread in ticks
    pub fn spread(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Mid-price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) as f64 / 2.0),
            _ => None,
        }
    }

    /// Microprice - volume-weighted average price
    /// More accurate estimate of "fair" price
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

### 2.1.5 Adding and Removing Orders

```rust
impl OrderBook {
    /// Add a limit order to the book
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

    /// Cancel an order by ID
    pub fn cancel_order(&mut self, order_id: u64, side: Side, price: i64) -> bool {
        let book = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        if let Some(level) = book.get_mut(&price) {
            if level.remove_order(order_id) {
                // If level is empty, remove it
                if level.orders.is_empty() {
                    book.remove(&price);
                }
                return true;
            }
        }
        false
    }

    /// Execute a market order
    pub fn execute_market_order(&mut self, side: Side, mut volume: u64) -> Vec<Trade> {
        let mut trades = Vec::new();

        // For buy - take from asks, for sell - take from bids
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

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    pub price: i64,
    pub volume: u64,
    pub aggressor_side: Side, // Who initiated the trade
}

type OrderId = u64;
```

---

## 2.2 Point Processes for Order Flow Modeling

### 2.2.1 Introduction to Point Processes

A point process is a mathematical model for describing random events occurring in time. In trading, events are incoming orders.

**Key concept — intensity λ(t):**
- λ(t) · dt ≈ probability of an event in interval [t, t + dt]
- Higher intensity = events occur more frequently

### 2.2.2 Poisson Process

The simplest model — events occur with constant intensity λ, independently of each other.

```rust
use rand::Rng;
use rand_distr::{Exp, Distribution};

/// Homogeneous Poisson process generator
pub struct PoissonProcess {
    /// Intensity (average number of events per unit time)
    lambda: f64,
}

impl PoissonProcess {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "Intensity must be positive");
        Self { lambda }
    }

    /// Generate events until time t_end
    pub fn simulate(&self, t_end: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let exp_dist = Exp::new(self.lambda).unwrap();

        let mut events = Vec::new();
        let mut t = 0.0;

        loop {
            // Time to next event ~ Exp(λ)
            let inter_arrival = exp_dist.sample(&mut rng);
            t += inter_arrival;

            if t > t_end {
                break;
            }
            events.push(t);
        }

        events
    }

    /// Intensity (constant)
    pub fn intensity(&self, _t: f64) -> f64 {
        self.lambda
    }
}
```

**Problem with Poisson process:** it doesn't capture event **clustering**. In real markets, orders arrive in bursts — one trade triggers others.

### 2.2.3 Hawkes Process

The Hawkes process is a **self-exciting** process. Each event temporarily increases intensity, leading to clustering.

**Mathematical definition:**

```
λ(t) = μ + Σ α · exp(-β · (t - tᵢ))
           i: tᵢ < t
```

where:
- μ — baseline intensity (background)
- α — excitation strength (how much an event raises intensity)
- β — decay rate (how quickly the effect disappears)
- tᵢ — times of past events

```rust
/// Hawkes process with exponential kernel
pub struct HawkesProcess {
    /// Baseline intensity
    mu: f64,
    /// Excitation parameter
    alpha: f64,
    /// Decay parameter
    beta: f64,
    /// Event history
    history: Vec<f64>,
}

impl HawkesProcess {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        // Check stationarity condition: branching ratio < 1
        let branching_ratio = alpha / beta;
        assert!(
            branching_ratio < 1.0,
            "Branching ratio α/β = {} must be < 1 for stationarity",
            branching_ratio
        );

        Self {
            mu,
            alpha,
            beta,
            history: Vec::new(),
        }
    }

    /// Compute current intensity
    pub fn intensity(&self, t: f64) -> f64 {
        let mut lambda = self.mu;

        for &ti in &self.history {
            if ti < t {
                // Contribution of event ti to intensity at time t
                lambda += self.alpha * (-self.beta * (t - ti)).exp();
            }
        }

        lambda
    }

    /// Branching ratio - average number of "children" per event
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Simulation via thinning (Lewis-Shedler algorithm)
    pub fn simulate(&mut self, t_end: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        self.history.clear();

        let mut t = 0.0;

        while t < t_end {
            // Upper bound on intensity
            let lambda_max = self.intensity(t) + self.alpha;

            // Generate candidate from Poisson with intensity lambda_max
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
                // Accept event
                self.history.push(t);
            }
        }

        self.history.clone()
    }

    /// Maximum likelihood parameter estimation
    pub fn fit_mle(events: &[f64], t_end: f64) -> Result<Self, &'static str> {
        if events.is_empty() {
            return Err("No events for calibration");
        }

        // Simplified implementation - gradient descent
        // In production, use specialized optimizers

        let mut mu = events.len() as f64 / t_end * 0.5;
        let mut alpha = 0.3;
        let mut beta = 1.0;

        let learning_rate = 0.001;
        let iterations = 1000;

        for _ in 0..iterations {
            // Compute log-likelihood gradient
            let (grad_mu, grad_alpha, grad_beta) =
                compute_gradients(events, t_end, mu, alpha, beta);

            // Update parameters
            mu += learning_rate * grad_mu;
            alpha += learning_rate * grad_alpha;
            beta += learning_rate * grad_beta;

            // Project onto feasible region
            mu = mu.max(0.001);
            alpha = alpha.max(0.001);
            beta = beta.max(alpha + 0.001); // Ensure α/β < 1
        }

        Ok(Self {
            mu,
            alpha,
            beta,
            history: events.to_vec(),
        })
    }
}

/// Compute log-likelihood gradients (simplified)
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

    // Simplified gradients for demonstration
    let grad_alpha = n / alpha - t_end / 2.0;
    let grad_beta = -n / beta + t_end / 2.0;

    (grad_mu, grad_alpha, grad_beta)
}
```

### 2.2.4 Multivariate Hawkes Process

In real markets, different event types influence each other:
- Buys can trigger sells (and vice versa)
- Trades on bid affect trades on ask

```rust
/// Multivariate Hawkes process
pub struct MultivariateHawkes {
    /// Baseline intensities for each type
    mu: Vec<f64>,
    /// Excitation matrix α[i][j] - effect of type j on type i
    alpha: Vec<Vec<f64>>,
    /// Decay matrix
    beta: Vec<Vec<f64>>,
    /// Event history: (time, type)
    history: Vec<(f64, usize)>,
    /// Number of event types
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

    /// Intensity for event type k at time t
    pub fn intensity(&self, t: f64, k: usize) -> f64 {
        let mut lambda = self.mu[k];

        for &(ti, j) in &self.history {
            if ti < t {
                lambda += self.alpha[k][j] * (-self.beta[k][j] * (t - ti)).exp();
            }
        }

        lambda
    }

    /// Total intensity (for thinning)
    pub fn total_intensity(&self, t: f64) -> f64 {
        (0..self.dim).map(|k| self.intensity(t, k)).sum()
    }
}

/// Example: bid-ask interaction
fn create_bid_ask_hawkes() -> MultivariateHawkes {
    // 0 = bid events, 1 = ask events
    let mu = vec![1.0, 1.0];  // Baseline intensities

    // Excitation matrix:
    // α[0][0] = bid -> bid (autocorrelation)
    // α[0][1] = ask -> bid (cross-excitation)
    let alpha = vec![
        vec![0.3, 0.2],  // Effect on bid
        vec![0.2, 0.3],  // Effect on ask
    ];

    let beta = vec![
        vec![1.0, 1.0],
        vec![1.0, 1.0],
    ];

    MultivariateHawkes::new(mu, alpha, beta)
}
```

---

## 2.3 The Cont-Stoikov-Talreja Model

### 2.3.1 Core Idea

Cont, Stoikov, and Talreja (2010) proposed viewing the order book as a **queueing system**.

Each price level is a queue where:
- **Arrivals**: limit orders
- **Departures**: cancellations and executions

### 2.3.2 Price Movement Probabilities

Key question: what's the probability that mid-price goes up or down?

```rust
/// Cont-Stoikov-Talreja Model
pub struct ContStoikovTalreja {
    /// Limit order arrival intensity at level δ from best price
    /// λ(δ) = A * exp(-k * δ)
    arrival_a: f64,
    arrival_k: f64,

    /// Cancellation rate
    cancel_rate: f64,

    /// Market order intensity
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

    /// Limit order arrival rate at δ ticks from best price
    pub fn limit_order_arrival_rate(&self, delta: u32) -> f64 {
        self.arrival_a * (-self.arrival_k * delta as f64).exp()
    }

    /// Probability of mid-price moving up
    /// Depends on volume imbalance at best levels
    pub fn prob_price_up(&self, bid_volume: f64, ask_volume: f64) -> f64 {
        // Simplified formula
        // Original paper uses more complex expression
        // via Laplace transform

        let total = bid_volume + ask_volume;
        if total == 0.0 {
            return 0.5;
        }

        // Intuition: more volume on bid -> less chance of breakthrough -> price more likely to rise
        bid_volume / total
    }

    /// Expected time to next mid-price change
    pub fn expected_time_to_price_change(
        &self,
        best_bid_volume: u64,
        best_ask_volume: u64
    ) -> f64 {
        // Bid clear rate = market_order_rate * P(clear bid)
        // Ask clear rate = market_order_rate * P(clear ask)

        // Simplified: inverse of sum of rates
        let bid_vol = best_bid_volume as f64;
        let ask_vol = best_ask_volume as f64;

        let rate_bid_clear = self.market_order_rate / bid_vol.max(1.0);
        let rate_ask_clear = self.market_order_rate / ask_vol.max(1.0);

        1.0 / (rate_bid_clear + rate_ask_clear)
    }
}
```

### 2.3.3 Calibration to Real Data

```rust
impl ContStoikovTalreja {
    /// Calibrate arrival rate parameter from historical data
    pub fn calibrate_arrival_rate(
        limit_orders: &[(f64, u32)], // (time, distance from best price)
        t_end: f64,
    ) -> (f64, f64) {
        // Group orders by distance
        use std::collections::HashMap;

        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &(_time, delta) in limit_orders {
            *counts.entry(delta).or_insert(0) += 1;
        }

        // Estimate λ(δ) = count[δ] / T
        // Then fit A * exp(-k * δ)

        // Linear regression on log(λ) = log(A) - k * δ
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

## 2.4 Optimal Market Making: Avellaneda-Stoikov Model

### 2.4.1 Problem Formulation

A market maker earns on the spread but carries inventory risk. Goal: find optimal bid and ask prices.

**Optimization criterion:**
```
Maximize: E[-exp(-γ · W_T)]
```

where:
- W_T = X_T + q_T · S_T — terminal wealth
- X_T — cash
- q_T — inventory (position)
- S_T — asset price
- γ — risk aversion coefficient

### 2.4.2 Key Formulas

**Reservation price (indifference price):**
```
r(t, q) = S_t - q · γ · σ² · (T - t)
```

**Interpretation:**
- If q > 0 (long position): r < S, market maker willing to sell cheaper
- If q < 0 (short position): r > S, market maker willing to buy higher
- Larger |q| means more aggressive quotes to reduce position

**Optimal spread:**
```
δ⁺ + δ⁻ = γσ²(T-t) + (2/γ) · ln(1 + γ/k)
```

where k characterizes how order intensity decays with distance from mid-price.

### 2.4.3 Full Implementation

```rust
/// Avellaneda-Stoikov Market Making Strategy
pub struct AvellanedaStoikov {
    /// Risk aversion coefficient
    gamma: f64,
    /// Volatility estimate (σ annualized)
    sigma: f64,
    /// Order arrival decay parameter
    k: f64,
    /// Session end time (in years)
    t_end: f64,
    /// Current position
    inventory: i64,
    /// Current mid-price
    mid_price: f64,
    /// Maximum position
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

    /// Update current mid-price
    pub fn update_mid_price(&mut self, price: f64) {
        self.mid_price = price;
    }

    /// Update position
    pub fn update_inventory(&mut self, delta: i64) {
        self.inventory += delta;
    }

    /// Reservation price - "fair" price accounting for inventory
    pub fn reservation_price(&self, t: f64) -> f64 {
        let time_to_end = (self.t_end - t).max(0.0);
        self.mid_price - (self.inventory as f64) * self.gamma
            * self.sigma.powi(2) * time_to_end
    }

    /// Optimal spread
    pub fn optimal_spread(&self, t: f64) -> f64 {
        let time_to_end = (self.t_end - t).max(0.0);

        self.gamma * self.sigma.powi(2) * time_to_end
            + (2.0 / self.gamma) * (1.0 + self.gamma / self.k).ln()
    }

    /// Compute bid and ask quotes
    pub fn quotes(&self, t: f64) -> (f64, f64) {
        let r = self.reservation_price(t);
        let spread = self.optimal_spread(t);

        let bid = r - spread / 2.0;
        let ask = r + spread / 2.0;

        (bid, ask)
    }

    /// Quotes with inventory control
    pub fn quotes_with_inventory_control(&self, t: f64) -> QuoteDecision {
        let (mut bid, mut ask) = self.quotes(t);

        // If position too large - don't quote bid
        let quote_bid = self.inventory < self.max_inventory;
        // If position too negative - don't quote ask
        let quote_ask = self.inventory > -self.max_inventory;

        // Additional adjustment when approaching limits
        if self.inventory > self.max_inventory / 2 {
            // Sell more aggressively
            ask -= 0.0001 * (self.inventory - self.max_inventory / 2) as f64;
        }
        if self.inventory < -self.max_inventory / 2 {
            // Buy more aggressively
            bid += 0.0001 * (-self.inventory - self.max_inventory / 2) as f64;
        }

        QuoteDecision {
            bid_price: if quote_bid { Some(bid) } else { None },
            ask_price: if quote_ask { Some(ask) } else { None },
        }
    }
}

/// Quote decision
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
                id: 0, // Will be assigned by exchange
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

### 2.4.4 Parameter Estimation

```rust
impl AvellanedaStoikov {
    /// Volatility estimation via realized variance
    pub fn estimate_volatility(prices: &[f64], dt: f64) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        // Log returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        // Realized variance
        let sum_sq: f64 = returns.iter().map(|r| r * r).sum();
        let realized_var = sum_sq / dt;  // Annualize

        realized_var.sqrt()
    }

    /// Estimate k parameter from order flow
    pub fn estimate_k(
        order_distances: &[f64],  // Order distances from mid-price
        fill_flags: &[bool],      // Was order filled
    ) -> f64 {
        // k determines how quickly fill probability decays
        // λ(δ) = A * exp(-k * δ)

        // Logistic regression P(fill | δ) = 1 / (1 + exp(k*δ - b))
        // Simplified: k ≈ -d(log P)/dδ

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

        // Heuristic estimate
        if fills > 0.0 {
            let mean_fill_delta = sum_fill_delta / fills;
            let mean_delta = sum_delta / n;

            // k inversely proportional to mean distance of filled orders
            1.0 / mean_fill_delta.max(0.0001)
        } else {
            1.0  // Default value
        }
    }
}
```

---

## 2.5 LOB Simulator

### 2.5.1 Simulator Architecture

```rust
use std::collections::VecDeque;

/// Simulation events
#[derive(Debug, Clone)]
pub enum SimulationEvent {
    /// Limit order arrived
    LimitOrder { side: Side, price: i64, volume: u64 },
    /// Market order arrived
    MarketOrder { side: Side, volume: u64 },
    /// Order cancellation
    CancelOrder { order_id: u64 },
    /// Mid-price change
    PriceMove { direction: i32 },
}

/// Hawkes-based LOB Simulator
pub struct LOBSimulator {
    /// Order book
    book: OrderBook,
    /// Bid order process
    bid_limit_process: HawkesProcess,
    /// Ask order process
    ask_limit_process: HawkesProcess,
    /// Market order process
    market_order_process: HawkesProcess,
    /// Cancellation process
    cancel_process: HawkesProcess,
    /// Current simulation time
    current_time: f64,
    /// Current mid-price
    mid_price: f64,
    /// Event log
    event_log: Vec<(f64, SimulationEvent)>,
    /// Random number generator
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

    /// Simulate until time t_end
    pub fn simulate(&mut self, t_end: f64) -> &[(f64, SimulationEvent)] {
        use rand::Rng;

        while self.current_time < t_end {
            // Find next event time (minimum across all processes)
            let (next_time, event_type) = self.next_event(t_end);

            if next_time > t_end {
                break;
            }

            self.current_time = next_time;

            // Process event
            let event = self.generate_event(event_type);
            self.process_event(&event);
            self.event_log.push((self.current_time, event));
        }

        &self.event_log
    }

    /// Find next event time
    fn next_event(&mut self, t_max: f64) -> (f64, EventType) {
        use rand::Rng;

        // Generate potential time for each process
        // and select minimum

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
                // Generate price at random distance from mid
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
                // Update mid-price after trades
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

## 2.6 Feature Extraction from LOB

### 2.6.1 Basic Features

```rust
/// Features extracted from order book
#[derive(Debug, Clone)]
pub struct LOBFeatures {
    /// Mid-price
    pub mid_price: f64,
    /// Microprice (weighted average)
    pub microprice: f64,
    /// Spread
    pub spread: f64,
    /// Volume imbalance at best levels
    pub imbalance_l1: f64,
    /// Volume imbalance at first 5 levels
    pub imbalance_l5: f64,
    /// Bid depth (total volume at N levels)
    pub bid_depth: f64,
    /// Ask depth
    pub ask_depth: f64,
    /// Book pressure (integral of volume over price levels)
    pub book_pressure: f64,
}

impl OrderBook {
    /// Extract features from current book state
    pub fn extract_features(&self, levels: usize) -> Option<LOBFeatures> {
        let best_bid = self.best_bid()?;
        let best_ask = self.best_ask()?;

        let mid_price = (best_bid + best_ask) as f64 / 2.0;
        let spread = (best_ask - best_bid) as f64;

        // Volumes at best levels
        let bid_vol_l1 = self.bids.get(&best_bid)?.total_volume as f64;
        let ask_vol_l1 = self.asks.get(&best_ask)?.total_volume as f64;

        let microprice = (bid_vol_l1 * best_ask as f64 + ask_vol_l1 * best_bid as f64)
            / (bid_vol_l1 + ask_vol_l1);

        // Imbalance L1
        let imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / (bid_vol_l1 + ask_vol_l1);

        // Sum volumes at first N levels
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

        // Book pressure: weighted sum of volumes
        // Weights decrease with distance from best price
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

### 2.6.2 Dynamic Features

```rust
/// Features requiring history
pub struct DynamicFeatures {
    /// Mid-price history
    mid_prices: VecDeque<f64>,
    /// Imbalance history
    imbalances: VecDeque<f64>,
    /// Maximum history size
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

    /// Add new observation
    pub fn update(&mut self, features: &LOBFeatures) {
        if self.mid_prices.len() >= self.max_history {
            self.mid_prices.pop_front();
            self.imbalances.pop_front();
        }

        self.mid_prices.push_back(features.mid_price);
        self.imbalances.push_back(features.imbalance_l1);
    }

    /// Realized volatility (over last N observations)
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
            .map(|w| (w[0] / w[1]).ln())  // Reverse order due to rev()
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Imbalance trend (derivative)
    pub fn imbalance_trend(&self, n: usize) -> f64 {
        let values: Vec<_> = self.imbalances.iter()
            .rev()
            .take(n)
            .copied()
            .collect();

        if values.len() < 2 {
            return 0.0;
        }

        // Simple linear regression
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

## 2.7 Working with Real Data

### 2.7.1 Connecting to Exchange via WebSocket

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};

/// Depth update message (Binance format)
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

/// WebSocket client for order book data
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

    /// Start receiving data
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

        println!("Connected to {}", url);

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
        // Apply updates to the book
        for bid in &update.bids {
            let price: f64 = bid[0].parse().unwrap_or(0.0);
            let volume: f64 = bid[1].parse().unwrap_or(0.0);
            let price_ticks = (price * 1e8) as i64;
            let volume_units = (volume * 1e8) as u64;

            // volume = 0 means remove level
            // volume > 0 means update/add
            // (simplified logic)
        }

        // Similarly for asks
        for ask in &update.asks {
            let price: f64 = ask[0].parse().unwrap_or(0.0);
            let volume: f64 = ask[1].parse().unwrap_or(0.0);
            // ...
        }

        self.last_update_id = update.final_update_id;
    }
}
```

### 2.7.2 Usage Example

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = BinanceWSClient::new("btcusdt");

    let mut strategy = AvellanedaStoikov::new(
        0.01,   // gamma
        0.02,   // sigma (2% volatility)
        1.0,    // k
        1.0,    // t_end (1 year = 1.0)
        10,     // max_inventory
    );

    let start_time = std::time::Instant::now();

    client.run(|book, features| {
        // Update strategy
        strategy.update_mid_price(features.mid_price);

        // Get quotes
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

## 2.8 Practical Exercises

### Exercise 2.1: Order Book Implementation

Create a complete order book implementation with support for:
- All order types (limit, market, stop)
- Various TimeInForce (GTC, IOC, FOK)
- Matching engine with price-time priority

**Criteria:**
- Processing one message < 1 μs
- Correct handling of edge cases

### Exercise 2.2: Hawkes Process Calibration

Using historical trade data:
1. Load one day of trading data
2. Calibrate Hawkes process parameters (μ, α, β)
3. Verify fit quality via Q-Q plot

### Exercise 2.3: Market Making Backtesting

1. Implement simulator with realistic order flow
2. Test Avellaneda-Stoikov strategy
3. Build equity curve and compute Sharpe ratio
4. Analyze sensitivity to γ parameter

### Exercise 2.4: Real-time Bot

Create a paper trading bot:
1. Connect to exchange testnet
2. Automatic quote updates
3. Risk monitoring (inventory, P&L)
4. Graceful shutdown

---

## 2.9 Common Mistakes

### Mistake 1: Ignoring Latency

```rust
// ❌ WRONG: assuming instant execution
let (bid, ask) = strategy.quotes(t);
place_order(bid, volume);  // Price will change by execution time!

// ✅ CORRECT: account for latency
let estimated_latency = 0.001; // 1ms in annual units
let (bid, ask) = strategy.quotes(t + estimated_latency);
```

### Mistake 2: Incorrect Inventory Tracking

```rust
// ❌ WRONG: not updating inventory on partial fill
fn on_fill(&mut self, trade: &Trade) {
    // Forgot to update inventory!
}

// ✅ CORRECT
fn on_fill(&mut self, trade: &Trade) {
    match trade.side {
        Side::Buy => self.inventory += trade.volume as i64,
        Side::Sell => self.inventory -= trade.volume as i64,
    }
}
```

### Mistake 3: Branching Ratio >= 1

```rust
// ❌ WRONG: process is non-stationary, will explode
let hawkes = HawkesProcess::new(1.0, 2.0, 1.0);  // α/β = 2 > 1

// ✅ CORRECT
let hawkes = HawkesProcess::new(1.0, 0.5, 1.0);  // α/β = 0.5 < 1
```

---

## Conclusion

In this chapter we learned:

1. **Order Book structure** and efficient data structures for representation
2. **Point processes** (Poisson, Hawkes) for order flow modeling
3. **Cont-Stoikov-Talreja model** for LOB dynamics analysis
4. **Avellaneda-Stoikov strategy** for optimal market making
5. **Practical aspects** of working with real exchange data

This knowledge forms the foundation for creating your own trading strategies operating at the market microstructure level.

---

## Recommended Reading

1. **Cartea Á., Jaimungal S., Penalva J.** "Algorithmic and High-Frequency Trading" (2015)
2. **Lehalle C.-A., Laruelle S.** "Market Microstructure in Practice" (2018)
3. **Bouchaud J.-P. et al.** "Trades, Quotes and Prices" (2018)
4. **Avellaneda M., Stoikov S.** "High-frequency trading in a limit order book" (2008)
5. **Cont R., Stoikov S., Talreja R.** "A stochastic model for order book dynamics" (2010)
