# Chapter 4: Machine Learning for Financial Time Series

## Introduction

Machine Learning (ML) has fundamentally transformed how we analyze financial markets. However, applying ML in trading differs significantly from classical problems — data is non-stationary, signal-to-noise ratio is low, and mistakes cost real money.

In this chapter, we'll build a production-ready machine learning system for financial time series, using Rust for high-performance feature engineering and inference.

---

## 4.1 ML Challenges in Finance

### 4.1.1 Key Problems

| Problem | Description | Solution |
|---------|-------------|----------|
| Non-stationarity | Data distribution changes over time | Walk-forward validation, regime detection |
| Low SNR | Most price movements are noise | Careful feature engineering, ensembles |
| Lookahead bias | Using future information during training | Strict time-based splits |
| Survivorship bias | Data only from surviving assets | Include delisted securities |
| Data snooping | Multiple hypothesis testing | Out-of-sample validation, Bonferroni correction |
| Regime changes | Different market conditions | Regime-aware models |

### 4.1.2 Walk-Forward Validation

Classical random train/test splits are **unacceptable** for time series:

```rust
/// Walk-forward validation with embargo period
/// Prevents lookahead bias through a time gap between train and test
pub struct WalkForwardValidator {
    n_splits: usize,
    embargo_periods: usize,
}

impl WalkForwardValidator {
    pub fn new(n_splits: usize, embargo_periods: usize) -> Self {
        Self { n_splits, embargo_periods }
    }

    /// Generates indices for train/test splits
    pub fn split(&self, data_len: usize) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
        let mut splits = Vec::with_capacity(self.n_splits);
        let fold_size = data_len / (self.n_splits + 1);

        for i in 0..self.n_splits {
            let train_end = (i + 1) * fold_size;
            let test_start = train_end + self.embargo_periods;
            let test_end = (test_start + fold_size).min(data_len);

            if test_start < data_len {
                splits.push((0..train_end, test_start..test_end));
            }
        }

        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_forward_splits() {
        let validator = WalkForwardValidator::new(3, 5);
        let splits = validator.split(100);

        // Verify that test always comes after train + embargo
        for (train, test) in &splits {
            assert!(train.end + 5 <= test.start);
        }
    }
}
```

### 4.1.3 Triple Barrier Labeling

Labeling method by Marcos López de Prado — instead of simple "up/down", we consider three barriers:

```rust
/// Triple barrier labeling result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TripleBarrierLabel {
    /// Price reached upper barrier (profit target)
    TakeProfit = 1,
    /// Price reached lower barrier (stop loss)
    StopLoss = -1,
    /// Time horizon expired without hitting barriers
    TimeOut = 0,
}

/// Triple Barrier Labeling for creating ML targets
pub struct TripleBarrierLabeler {
    /// Threshold for take profit (e.g., 0.02 = 2%)
    upper_barrier: f64,
    /// Threshold for stop loss (e.g., -0.02 = -2%)
    lower_barrier: f64,
    /// Maximum horizon in periods
    max_horizon: usize,
}

impl TripleBarrierLabeler {
    pub fn new(upper_barrier: f64, lower_barrier: f64, max_horizon: usize) -> Self {
        Self {
            upper_barrier,
            lower_barrier: lower_barrier.abs() * -1.0, // Ensure negative value
            max_horizon,
        }
    }

    /// Labels a single data point
    pub fn label_single(&self, prices: &[f64], start_idx: usize) -> Option<TripleBarrierLabel> {
        if start_idx + self.max_horizon >= prices.len() {
            return None;
        }

        let entry_price = prices[start_idx];
        let mut first_upper_hit: Option<usize> = None;
        let mut first_lower_hit: Option<usize> = None;

        for i in 1..=self.max_horizon {
            let current_price = prices[start_idx + i];
            let return_pct = (current_price - entry_price) / entry_price;

            // Record first touch of each barrier
            if first_upper_hit.is_none() && return_pct >= self.upper_barrier {
                first_upper_hit = Some(i);
            }
            if first_lower_hit.is_none() && return_pct <= self.lower_barrier {
                first_lower_hit = Some(i);
            }
        }

        // Determine which barrier was hit first
        match (first_upper_hit, first_lower_hit) {
            (Some(up), Some(down)) => {
                if up <= down {
                    Some(TripleBarrierLabel::TakeProfit)
                } else {
                    Some(TripleBarrierLabel::StopLoss)
                }
            }
            (Some(_), None) => Some(TripleBarrierLabel::TakeProfit),
            (None, Some(_)) => Some(TripleBarrierLabel::StopLoss),
            (None, None) => Some(TripleBarrierLabel::TimeOut),
        }
    }

    /// Labels the entire price array
    pub fn label_all(&self, prices: &[f64]) -> Vec<Option<TripleBarrierLabel>> {
        (0..prices.len())
            .map(|i| self.label_single(prices, i))
            .collect()
    }
}
```

---

## 4.2 Feature Engineering in Rust

### 4.2.1 Why Rust for Features?

- **Speed**: Real-time computation for live trading
- **Memory**: Zero-copy operations for large datasets
- **Reliability**: Compiler prevents race conditions
- **Integration**: Easily embeds into streaming pipeline

### 4.2.2 Technical Indicators

```rust
use std::collections::VecDeque;

/// High-performance technical indicators calculator
/// Uses ring buffer for streaming computations
pub struct TechnicalIndicators {
    /// Buffer for rolling calculations
    price_buffer: VecDeque<f64>,
    /// Maximum window size
    max_window: usize,
}

impl TechnicalIndicators {
    pub fn new(max_window: usize) -> Self {
        Self {
            price_buffer: VecDeque::with_capacity(max_window),
            max_window,
        }
    }

    /// Adds new price to the buffer
    pub fn update(&mut self, price: f64) {
        if self.price_buffer.len() >= self.max_window {
            self.price_buffer.pop_front();
        }
        self.price_buffer.push_back(price);
    }

    /// Simple Moving Average
    pub fn sma(&self, period: usize) -> Option<f64> {
        if self.price_buffer.len() < period {
            return None;
        }

        let sum: f64 = self.price_buffer
            .iter()
            .rev()
            .take(period)
            .sum();

        Some(sum / period as f64)
    }

    /// Exponential Moving Average
    /// alpha = 2 / (period + 1)
    pub fn ema(&self, period: usize) -> Option<f64> {
        if self.price_buffer.len() < period {
            return None;
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let prices: Vec<f64> = self.price_buffer.iter().copied().collect();

        let mut ema = prices[0];
        for &price in &prices[1..] {
            ema = alpha * price + (1.0 - alpha) * ema;
        }

        Some(ema)
    }

    /// Relative Strength Index (RSI)
    pub fn rsi(&self, period: usize) -> Option<f64> {
        if self.price_buffer.len() < period + 1 {
            return None;
        }

        let prices: Vec<f64> = self.price_buffer
            .iter()
            .rev()
            .take(period + 1)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return Some(100.0);
        }

        let rs = avg_gain / avg_loss;
        Some(100.0 - 100.0 / (1.0 + rs))
    }

    /// Bollinger Bands: returns (lower, middle, upper)
    pub fn bollinger_bands(&self, period: usize, num_std: f64) -> Option<(f64, f64, f64)> {
        if self.price_buffer.len() < period {
            return None;
        }

        let prices: Vec<f64> = self.price_buffer
            .iter()
            .rev()
            .take(period)
            .copied()
            .collect();

        let mean = prices.iter().sum::<f64>() / period as f64;

        let variance = prices
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / period as f64;

        let std_dev = variance.sqrt();

        Some((
            mean - num_std * std_dev,  // lower
            mean,                       // middle
            mean + num_std * std_dev,  // upper
        ))
    }

    /// MACD (Moving Average Convergence Divergence)
    /// Returns (macd_line, signal_line, histogram)
    pub fn macd(
        &self,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize
    ) -> Option<(f64, f64, f64)> {
        let fast_ema = self.ema(fast_period)?;
        let slow_ema = self.ema(slow_period)?;

        let macd_line = fast_ema - slow_ema;

        // For full MACD, need history of macd_line
        // Simplified version: signal = EMA(macd_line)
        // In production, use a separate buffer for macd_line
        let signal_line = macd_line * 0.9; // Simplification
        let histogram = macd_line - signal_line;

        Some((macd_line, signal_line, histogram))
    }
}
```

### 4.2.3 Microstructure Features

```rust
/// Features from market microstructure (order book, trades)
pub struct MicrostructureFeatures;

impl MicrostructureFeatures {
    /// Order Book Imbalance
    /// Shows imbalance between bid and ask volumes
    /// Values from -1 (ask dominant) to +1 (bid dominant)
    pub fn order_imbalance(bid_volume: f64, ask_volume: f64) -> f64 {
        let total = bid_volume + ask_volume;
        if total == 0.0 {
            return 0.0;
        }
        (bid_volume - ask_volume) / total
    }

    /// Microprice
    /// More accurate estimate of "fair" price than mid price
    /// Weights prices by opposite volumes
    pub fn microprice(bid: f64, ask: f64, bid_volume: f64, ask_volume: f64) -> f64 {
        let total_volume = bid_volume + ask_volume;
        if total_volume == 0.0 {
            return (bid + ask) / 2.0;
        }
        // Large ask_volume pushes price toward bid, and vice versa
        (bid * ask_volume + ask * bid_volume) / total_volume
    }

    /// Spread in basis points
    pub fn spread_bps(bid: f64, ask: f64) -> f64 {
        let mid = (bid + ask) / 2.0;
        if mid == 0.0 {
            return 0.0;
        }
        (ask - bid) / mid * 10000.0
    }

    /// VWAP Deviation
    /// Current price deviation from VWAP
    pub fn vwap_deviation(
        prices: &[f64],
        volumes: &[f64],
        current_price: f64
    ) -> f64 {
        let total_volume: f64 = volumes.iter().sum();
        if total_volume == 0.0 {
            return 0.0;
        }

        let vwap: f64 = prices
            .iter()
            .zip(volumes.iter())
            .map(|(&p, &v)| p * v)
            .sum::<f64>() / total_volume;

        (current_price - vwap) / vwap
    }

    /// Kyle's Lambda — price impact estimate
    /// Linear regression: returns ~ signed_volume
    pub fn kyle_lambda(returns: &[f64], signed_volumes: &[f64]) -> f64 {
        let n = returns.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let sum_xy: f64 = returns
            .iter()
            .zip(signed_volumes.iter())
            .map(|(&r, &v)| r * v)
            .sum();

        let sum_x: f64 = signed_volumes.iter().sum();
        let sum_y: f64 = returns.iter().sum();
        let sum_x2: f64 = signed_volumes.iter().map(|&v| v * v).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Trade Flow Imbalance over period
    /// Ratio of buy volume to total volume
    pub fn trade_flow_imbalance(buy_volume: f64, sell_volume: f64) -> f64 {
        let total = buy_volume + sell_volume;
        if total == 0.0 {
            return 0.5;
        }
        buy_volume / total
    }
}
```

---

## 4.3 Deep Learning Architectures

### 4.3.1 LSTM for Time Series

LSTM (Long Short-Term Memory) solves the vanishing gradient problem and can capture long-term dependencies.

```rust
/// Structure for storing LSTM layer parameters
/// In production, use established libraries (tch-rs, burn)
/// This is an educational forward pass implementation
pub struct LSTMCell {
    /// Input size
    input_size: usize,
    /// Hidden state size
    hidden_size: usize,
    /// Weights for all gates (combined)
    weights_ih: Vec<Vec<f64>>, // 4*hidden_size x input_size
    weights_hh: Vec<Vec<f64>>, // 4*hidden_size x hidden_size
    bias_ih: Vec<f64>,         // 4*hidden_size
    bias_hh: Vec<f64>,         // 4*hidden_size
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        // Xavier initialization
        let scale_ih = (6.0 / (input_size + hidden_size) as f64).sqrt();
        let scale_hh = (6.0 / (hidden_size * 2) as f64).sqrt();

        Self {
            input_size,
            hidden_size,
            weights_ih: Self::random_matrix(4 * hidden_size, input_size, scale_ih),
            weights_hh: Self::random_matrix(4 * hidden_size, hidden_size, scale_hh),
            bias_ih: vec![0.0; 4 * hidden_size],
            bias_hh: vec![0.0; 4 * hidden_size],
        }
    }

    fn random_matrix(rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
        use std::f64::consts::PI;
        let mut matrix = vec![vec![0.0; cols]; rows];
        let mut seed = 42u64;

        for row in matrix.iter_mut() {
            for val in row.iter_mut() {
                // Simple LCG for reproducibility
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let u1 = (seed as f64) / (u64::MAX as f64);
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let u2 = (seed as f64) / (u64::MAX as f64);
                // Box-Muller transform
                *val = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * scale;
            }
        }
        matrix
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    /// Forward pass through a single LSTM cell
    /// Returns (h_new, c_new)
    pub fn forward(
        &self,
        input: &[f64],
        h_prev: &[f64],
        c_prev: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let hs = self.hidden_size;

        // Compute gates = W_ih @ input + b_ih + W_hh @ h_prev + b_hh
        let mut gates = vec![0.0; 4 * hs];

        // W_ih @ input + b_ih
        for i in 0..4 * hs {
            gates[i] = self.bias_ih[i] + self.bias_hh[i];
            for j in 0..self.input_size {
                gates[i] += self.weights_ih[i][j] * input[j];
            }
            for j in 0..hs {
                gates[i] += self.weights_hh[i][j] * h_prev[j];
            }
        }

        // Split into 4 gates
        let i_gate: Vec<f64> = gates[0..hs].iter().map(|&x| Self::sigmoid(x)).collect();
        let f_gate: Vec<f64> = gates[hs..2*hs].iter().map(|&x| Self::sigmoid(x)).collect();
        let g_gate: Vec<f64> = gates[2*hs..3*hs].iter().map(|&x| Self::tanh(x)).collect();
        let o_gate: Vec<f64> = gates[3*hs..4*hs].iter().map(|&x| Self::sigmoid(x)).collect();

        // c_new = f * c_prev + i * g
        let c_new: Vec<f64> = (0..hs)
            .map(|i| f_gate[i] * c_prev[i] + i_gate[i] * g_gate[i])
            .collect();

        // h_new = o * tanh(c_new)
        let h_new: Vec<f64> = (0..hs)
            .map(|i| o_gate[i] * Self::tanh(c_new[i]))
            .collect();

        (h_new, c_new)
    }
}

/// LSTM network for time series prediction
pub struct LSTMPredictor {
    cell: LSTMCell,
    output_weights: Vec<f64>,
    output_bias: f64,
}

impl LSTMPredictor {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            cell: LSTMCell::new(input_size, hidden_size),
            output_weights: vec![0.1; hidden_size],
            output_bias: 0.0,
        }
    }

    /// Prediction for a sequence
    pub fn predict(&self, sequence: &[Vec<f64>]) -> f64 {
        let hs = self.cell.hidden_size;
        let mut h = vec![0.0; hs];
        let mut c = vec![0.0; hs];

        // Pass through entire sequence
        for input in sequence {
            let (h_new, c_new) = self.cell.forward(input, &h, &c);
            h = h_new;
            c = c_new;
        }

        // Linear output layer
        let mut output = self.output_bias;
        for (i, &w) in self.output_weights.iter().enumerate() {
            output += w * h[i];
        }

        output
    }
}
```

### 4.3.2 Transformer for Financial Data

```rust
/// Positional Encoding for Transformer
pub struct PositionalEncoding {
    encodings: Vec<Vec<f64>>,
    d_model: usize,
}

impl PositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut encodings = vec![vec![0.0; d_model]; max_len];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / 10000_f64.powf((2 * (i / 2)) as f64 / d_model as f64);
                encodings[pos][i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }

        Self { encodings, d_model }
    }

    pub fn get(&self, position: usize) -> &[f64] {
        &self.encodings[position]
    }
}

/// Scaled Dot-Product Attention
pub struct Attention {
    d_k: f64,
}

impl Attention {
    pub fn new(d_model: usize) -> Self {
        Self { d_k: (d_model as f64).sqrt() }
    }

    /// Softmax over last dimension
    fn softmax(scores: &mut [f64]) {
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = scores.iter().map(|&x| (x - max).exp()).sum();

        for score in scores.iter_mut() {
            *score = (*score - max).exp() / exp_sum;
        }
    }

    /// Computes attention for a single head
    /// Q, K, V — query, key, value matrices
    pub fn forward(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        value: &[Vec<f64>],
        mask: Option<&[Vec<bool>]>,
    ) -> Vec<Vec<f64>> {
        let seq_len = query.len();
        let d = query[0].len();

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let mut scores = vec![vec![0.0; seq_len]; seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                let dot: f64 = (0..d)
                    .map(|k| query[i][k] * key[j][k])
                    .sum();
                scores[i][j] = dot / self.d_k;
            }

            // Apply causal mask (for trading — can't look into the future!)
            if let Some(m) = mask {
                for j in 0..seq_len {
                    if m[i][j] {
                        scores[i][j] = f64::NEG_INFINITY;
                    }
                }
            }

            // Softmax
            Self::softmax(&mut scores[i]);
        }

        // Weighted sum: scores @ V
        let mut output = vec![vec![0.0; d]; seq_len];
        for i in 0..seq_len {
            for k in 0..d {
                for j in 0..seq_len {
                    output[i][k] += scores[i][j] * value[j][k];
                }
            }
        }

        output
    }
}

/// Causal mask generator for autoregressive models
pub fn generate_causal_mask(size: usize) -> Vec<Vec<bool>> {
    let mut mask = vec![vec![false; size]; size];
    for i in 0..size {
        for j in (i + 1)..size {
            mask[i][j] = true; // Mask future positions
        }
    }
    mask
}
```

---

## 4.4 Volatility Forecasting

### 4.4.1 Realized Volatility

```rust
/// Realized volatility calculator
pub struct RealizedVolatility {
    /// Number of periods per year (252 for daily data)
    annualization_factor: f64,
}

impl RealizedVolatility {
    pub fn new(periods_per_year: f64) -> Self {
        Self {
            annualization_factor: periods_per_year,
        }
    }

    /// Classic realized variance (sum of squared returns)
    pub fn realized_variance(&self, returns: &[f64]) -> f64 {
        returns.iter().map(|r| r * r).sum()
    }

    /// Realized volatility (square root of RV)
    pub fn realized_volatility(&self, returns: &[f64]) -> f64 {
        self.realized_variance(returns).sqrt()
    }

    /// Annualized realized volatility
    pub fn annualized_volatility(&self, returns: &[f64]) -> f64 {
        let rv = self.realized_volatility(returns);
        let n_periods = returns.len() as f64;
        rv * (self.annualization_factor / n_periods).sqrt()
    }

    /// Parkinson volatility (uses high-low range)
    /// More efficient estimator than close-to-close
    pub fn parkinson_volatility(&self, highs: &[f64], lows: &[f64]) -> f64 {
        let n = highs.len() as f64;
        let sum: f64 = highs
            .iter()
            .zip(lows.iter())
            .map(|(&h, &l)| {
                let log_ratio = (h / l).ln();
                log_ratio * log_ratio
            })
            .sum();

        (sum / (4.0 * n * 2.0_f64.ln())).sqrt()
    }

    /// Garman-Klass volatility (uses OHLC)
    pub fn garman_klass_volatility(
        &self,
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> f64 {
        let n = opens.len() as f64;

        let sum: f64 = (0..opens.len())
            .map(|i| {
                let hl = (highs[i] / lows[i]).ln();
                let co = (closes[i] / opens[i]).ln();
                0.5 * hl * hl - (2.0 * 2.0_f64.ln() - 1.0) * co * co
            })
            .sum();

        (sum / n).sqrt()
    }
}

/// HAR-RV model (Heterogeneous Autoregressive model of Realized Volatility)
/// RV_t = beta_0 + beta_d * RV_{t-1} + beta_w * RV_week + beta_m * RV_month + epsilon
pub struct HarRvModel {
    beta_0: f64,  // Intercept
    beta_d: f64,  // Daily coefficient
    beta_w: f64,  // Weekly coefficient
    beta_m: f64,  // Monthly coefficient
}

impl HarRvModel {
    /// Creates model with default parameters (typical values)
    pub fn default() -> Self {
        Self {
            beta_0: 0.0001,
            beta_d: 0.35,
            beta_w: 0.30,
            beta_m: 0.25,
        }
    }

    /// Creates model with specified parameters
    pub fn new(beta_0: f64, beta_d: f64, beta_w: f64, beta_m: f64) -> Self {
        Self { beta_0, beta_d, beta_w, beta_m }
    }

    /// Predicts RV for next period
    pub fn predict(&self, rv_daily: f64, rv_weekly: f64, rv_monthly: f64) -> f64 {
        self.beta_0
            + self.beta_d * rv_daily
            + self.beta_w * rv_weekly
            + self.beta_m * rv_monthly
    }

    /// Computes RV components from history
    pub fn compute_components(rv_history: &[f64]) -> Option<(f64, f64, f64)> {
        if rv_history.len() < 22 {
            return None;
        }

        let rv_daily = rv_history[rv_history.len() - 1];

        let rv_weekly: f64 = rv_history
            .iter()
            .rev()
            .take(5)
            .sum::<f64>() / 5.0;

        let rv_monthly: f64 = rv_history
            .iter()
            .rev()
            .take(22)
            .sum::<f64>() / 22.0;

        Some((rv_daily, rv_weekly, rv_monthly))
    }
}
```

---

## 4.5 Reinforcement Learning for Trading

### 4.5.1 Trading Environment

```rust
/// Trading environment state
#[derive(Debug, Clone)]
pub struct TradingState {
    /// Normalized balance (relative to initial)
    pub balance_ratio: f64,
    /// Current position (-1.0 to 1.0)
    pub position: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Market features (prices, indicators, etc.)
    pub features: Vec<f64>,
}

/// Agent action
#[derive(Debug, Clone, Copy)]
pub struct TradingAction {
    /// Target position from -1.0 (full short) to 1.0 (full long)
    pub target_position: f64,
}

/// Trading environment for RL agent
pub struct TradingEnvironment {
    /// Historical data
    prices: Vec<f64>,
    features: Vec<Vec<f64>>,
    /// Environment parameters
    initial_balance: f64,
    commission_rate: f64,
    max_position: f64,
    /// Current state
    current_step: usize,
    balance: f64,
    position: f64,
    entry_price: f64,
}

impl TradingEnvironment {
    pub fn new(
        prices: Vec<f64>,
        features: Vec<Vec<f64>>,
        initial_balance: f64,
        commission_rate: f64,
    ) -> Self {
        Self {
            prices,
            features,
            initial_balance,
            commission_rate,
            max_position: 1.0,
            current_step: 0,
            balance: initial_balance,
            position: 0.0,
            entry_price: 0.0,
        }
    }

    /// Reset environment to initial state
    pub fn reset(&mut self) -> TradingState {
        self.current_step = 0;
        self.balance = self.initial_balance;
        self.position = 0.0;
        self.entry_price = 0.0;

        self.get_state()
    }

    /// Execute action and get reward
    pub fn step(&mut self, action: TradingAction) -> (TradingState, f64, bool) {
        let old_price = self.prices[self.current_step];
        let target_position = action.target_position.clamp(-self.max_position, self.max_position);

        // Calculate trading costs
        let trade_size = (target_position - self.position).abs();
        let trade_cost = trade_size * old_price * self.commission_rate;

        // Update position
        self.position = target_position;
        self.balance -= trade_cost;

        // Move to next step
        self.current_step += 1;
        let new_price = self.prices[self.current_step];

        // Calculate reward (PnL for step)
        let price_return = (new_price - old_price) / old_price;
        let pnl = self.position * price_return * self.initial_balance;
        let reward = pnl - trade_cost;

        // Check episode termination
        let done = self.current_step >= self.prices.len() - 1;

        (self.get_state(), reward / self.initial_balance, done)
    }

    fn get_state(&self) -> TradingState {
        let current_price = self.prices[self.current_step];
        let unrealized_pnl = if self.position != 0.0 && self.entry_price != 0.0 {
            self.position * (current_price - self.entry_price) / self.entry_price
        } else {
            0.0
        };

        TradingState {
            balance_ratio: self.balance / self.initial_balance,
            position: self.position,
            unrealized_pnl,
            features: self.features[self.current_step].clone(),
        }
    }

    /// Episode final result
    pub fn get_total_return(&self) -> f64 {
        (self.balance - self.initial_balance) / self.initial_balance
    }
}

/// Simple Q-table for discrete actions
pub struct SimpleQLearner {
    /// Q-values: state_bucket -> action -> value
    q_table: std::collections::HashMap<i32, [f64; 3]>,
    /// Learning rate
    alpha: f64,
    /// Discount factor
    gamma: f64,
    /// Exploration rate
    epsilon: f64,
}

impl SimpleQLearner {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64) -> Self {
        Self {
            q_table: std::collections::HashMap::new(),
            alpha,
            gamma,
            epsilon,
        }
    }

    /// State discretization to bucket
    fn discretize_state(&self, state: &TradingState) -> i32 {
        // Simplified discretization by first feature
        let feature = state.features.get(0).copied().unwrap_or(0.0);
        (feature * 10.0).round() as i32
    }

    /// Action selection (epsilon-greedy)
    /// 0 = sell, 1 = hold, 2 = buy
    pub fn select_action(&self, state: &TradingState) -> usize {
        let state_bucket = self.discretize_state(state);

        // Exploration
        let random: f64 = rand_simple();
        if random < self.epsilon {
            return (random * 3.0) as usize;
        }

        // Exploitation
        let q_values = self.q_table.get(&state_bucket).unwrap_or(&[0.0; 3]);
        q_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(1)
    }

    /// Q-value update
    pub fn update(
        &mut self,
        state: &TradingState,
        action: usize,
        reward: f64,
        next_state: &TradingState,
        done: bool,
    ) {
        let state_bucket = self.discretize_state(state);
        let next_bucket = self.discretize_state(next_state);

        let next_max_q = if done {
            0.0
        } else {
            *self.q_table
                .get(&next_bucket)
                .unwrap_or(&[0.0; 3])
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0)
        };

        let q_values = self.q_table.entry(state_bucket).or_insert([0.0; 3]);
        let old_q = q_values[action];
        q_values[action] = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q);
    }

    /// Convert discrete action to TradingAction
    pub fn action_to_position(action: usize) -> TradingAction {
        let target = match action {
            0 => -1.0,  // Sell
            1 => 0.0,   // Hold
            2 => 1.0,   // Buy
            _ => 0.0,
        };
        TradingAction { target_position: target }
    }
}

/// Simple random number generator
fn rand_simple() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f64) / (u32::MAX as f64)
}
```

---

## 4.6 Deployment and Inference

### 4.6.1 Feature Pipeline for Production

```rust
use std::collections::VecDeque;

/// Production-ready feature pipeline
/// Computes all features in real-time
pub struct FeaturePipeline {
    /// Technical indicators
    tech_indicators: TechnicalIndicators,
    /// Buffer for returns
    returns_buffer: VecDeque<f64>,
    /// Buffer for volumes
    volume_buffer: VecDeque<f64>,
    /// Last price
    last_price: Option<f64>,
    /// Lookback window size
    lookback: usize,
}

impl FeaturePipeline {
    pub fn new(lookback: usize) -> Self {
        Self {
            tech_indicators: TechnicalIndicators::new(lookback),
            returns_buffer: VecDeque::with_capacity(lookback),
            volume_buffer: VecDeque::with_capacity(lookback),
            last_price: None,
            lookback,
        }
    }

    /// Update pipeline with new tick
    /// Returns feature vector if enough data accumulated
    pub fn update(&mut self, tick: &MarketTick) -> Option<FeatureVector> {
        // Update buffers
        if let Some(last) = self.last_price {
            let ret = (tick.price - last) / last;
            self.returns_buffer.push_back(ret);
            if self.returns_buffer.len() > self.lookback {
                self.returns_buffer.pop_front();
            }
        }

        self.volume_buffer.push_back(tick.volume);
        if self.volume_buffer.len() > self.lookback {
            self.volume_buffer.pop_front();
        }

        self.tech_indicators.update(tick.price);
        self.last_price = Some(tick.price);

        // Check readiness
        if self.returns_buffer.len() < self.lookback {
            return None;
        }

        // Generate features
        Some(self.compute_features(tick))
    }

    fn compute_features(&self, tick: &MarketTick) -> FeatureVector {
        let mut features = Vec::with_capacity(20);

        // Price-based features
        if let Some(sma_20) = self.tech_indicators.sma(20) {
            features.push((tick.price - sma_20) / sma_20); // Price vs SMA
        } else {
            features.push(0.0);
        }

        if let Some(rsi) = self.tech_indicators.rsi(14) {
            features.push((rsi - 50.0) / 50.0); // Normalized RSI
        } else {
            features.push(0.0);
        }

        if let Some((lower, mid, upper)) = self.tech_indicators.bollinger_bands(20, 2.0) {
            let bb_position = (tick.price - lower) / (upper - lower);
            features.push(bb_position * 2.0 - 1.0); // -1 to 1
        } else {
            features.push(0.0);
        }

        // Returns statistics
        let returns: Vec<f64> = self.returns_buffer.iter().copied().collect();
        features.push(returns.iter().sum::<f64>()); // Cumulative return
        features.push(self.compute_std(&returns));   // Volatility
        features.push(self.compute_skewness(&returns));
        features.push(self.compute_kurtosis(&returns));

        // Microstructure
        features.push(MicrostructureFeatures::order_imbalance(
            tick.bid_volume, tick.ask_volume
        ));
        features.push(MicrostructureFeatures::spread_bps(tick.bid, tick.ask));

        // Volume features
        let volumes: Vec<f64> = self.volume_buffer.iter().copied().collect();
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        features.push(tick.volume / avg_volume - 1.0); // Relative volume

        FeatureVector { values: features }
    }

    fn compute_std(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    fn compute_skewness(&self, data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std = self.compute_std(data);
        if std == 0.0 {
            return 0.0;
        }
        let m3 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
        m3
    }

    fn compute_kurtosis(&self, data: &[f64]) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std = self.compute_std(data);
        if std == 0.0 {
            return 0.0;
        }
        let m4 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
        m4 - 3.0 // Excess kurtosis
    }
}

/// Market data tick
#[derive(Debug, Clone)]
pub struct MarketTick {
    pub timestamp: u64,
    pub price: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
}

/// Feature vector for model
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub values: Vec<f64>,
}

impl FeatureVector {
    /// Feature normalization
    pub fn normalize(&mut self, means: &[f64], stds: &[f64]) {
        for (i, val) in self.values.iter_mut().enumerate() {
            if i < means.len() && i < stds.len() && stds[i] > 0.0 {
                *val = (*val - means[i]) / stds[i];
            }
        }
    }

    /// Convert to flat array for inference
    pub fn to_array(&self) -> Vec<f32> {
        self.values.iter().map(|&x| x as f32).collect()
    }
}
```

---

## 4.7 Metrics and Model Evaluation

```rust
/// Metrics for evaluating trading models
pub struct TradingMetrics;

impl TradingMetrics {
    /// Sharpe Ratio
    /// (mean_return - risk_free) / std_return * sqrt(periods_per_year)
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return 0.0;
        }

        let excess_return = mean - risk_free_rate / periods_per_year;
        excess_return / std * periods_per_year.sqrt()
    }

    /// Sortino Ratio (only downside risk)
    pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        // Only negative deviations
        let downside_variance: f64 = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>() / returns.len() as f64;
        let downside_std = downside_variance.sqrt();

        if downside_std == 0.0 {
            return f64::INFINITY;
        }

        let excess_return = mean - risk_free_rate / periods_per_year;
        excess_return / downside_std * periods_per_year.sqrt()
    }

    /// Maximum Drawdown
    pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut peak = equity_curve[0];
        let mut max_dd = 0.0;

        for &equity in equity_curve {
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }

    /// Calmar Ratio (annual return / max drawdown)
    pub fn calmar_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
        // Compute equity curve
        let mut equity = vec![1.0];
        for &r in returns {
            equity.push(equity.last().unwrap() * (1.0 + r));
        }

        let total_return = equity.last().unwrap() - 1.0;
        let n_years = returns.len() as f64 / periods_per_year;
        let annual_return = (1.0 + total_return).powf(1.0 / n_years) - 1.0;

        let max_dd = Self::max_drawdown(&equity);

        if max_dd == 0.0 {
            return f64::INFINITY;
        }

        annual_return / max_dd
    }

    /// Profit Factor (gross profit / gross loss)
    pub fn profit_factor(returns: &[f64]) -> f64 {
        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

        if gross_loss == 0.0 {
            return f64::INFINITY;
        }

        gross_profit / gross_loss
    }

    /// Win Rate
    pub fn win_rate(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        wins as f64 / returns.len() as f64
    }
}
```

---

## Practical Assignments

### Assignment 4.1: Walk-Forward Backtesting Framework
**Goal:** Build a proper backtesting pipeline
- Implement purged K-fold cross-validation
- Add embargo periods
- Model transaction costs
- Calculate metrics: Sharpe, Sortino, Calmar

### Assignment 4.2: LSTM vs Transformer
**Goal:** Compare architectures on crypto data
- Dataset: BTC-USDT 1-minute data, 2 years
- Target: 5-minute movement direction
- Metrics: accuracy, precision, recall, F1, profit factor

### Assignment 4.3: Volatility Forecasting Pipeline
**Goal:** Production-ready volatility prediction
- HAR baseline
- Deep HAR
- Integration with options pricing

### Assignment 4.4: RL Trading Agent
**Goal:** Train and deploy an RL agent
- Environment with realistic costs
- PPO vs DQN comparison
- Paper trading evaluation

### Assignment 4.5: End-to-End ML System
**Goal:** Full pipeline from data to trading
- Feature engineering in Rust
- Model training
- Low latency inference
- Latency benchmarks

---

## Connections with Other Chapters

| Chapter | Connection |
|---------|------------|
| 01-stochastic-calculus | Feature engineering from SDE |
| 02-market-microstructure | LOB features, order flow prediction |
| 03-portfolio-optimization | ML-based return/cov prediction |
| 05-low-latency-systems | Real-time inference architecture |

---

## Recommended Reading

1. Lopez de Prado M. "Advances in Financial Machine Learning" (2018)
2. Lopez de Prado M. "Machine Learning for Asset Managers" (2020)
3. Jansen S. "Machine Learning for Algorithmic Trading" (2020)
4. Vaswani et al. "Attention Is All You Need" (2017)
5. Hochreiter S., Schmidhuber J. "Long Short-Term Memory" (1997)
