# Chapter 4: Machine Learning for Trading (Simple Explanation)

## What is Machine Learning?

Imagine you're teaching a dog to fetch a ball. At first, the dog doesn't understand what to do. But you show it many times: "fetch the ball — get a treat!" After many attempts, the dog **learns by itself** what to do.

**Machine Learning** is when we teach a computer the same way we teach a dog. But instead of a ball — we use price data, and instead of treats — correct answers.

---

## Why is This Difficult with Money?

### Imagine a School Cafeteria

Let's say you want to guess how long the line will be in the cafeteria tomorrow.

**Problem 1: Everything Changes**
- On Monday they served pizza — huge line
- On Tuesday — oatmeal — almost empty
- And on Wednesday, a new chef arrived!

The market is the same — the rules keep changing. What worked yesterday might not work tomorrow.

**Problem 2: Lots of Noise**
Imagine trying to hear your friend at a disco. Music is blasting, everyone is shouting — that's **noise**. Your friend's voice is the **signal**.

The market is similar: most price movements are noise. Finding the real signal is very hard!

**Problem 3: No Peeking at the Future**
It's like preparing for a test with the answer key. You'll ace the test, but in real life, there won't be answers!

When we teach a computer using historical data, we must pretend we don't know the future.

---

## How to Properly Teach a Computer to Trade?

### The "Step by Step" Method (Walk-Forward)

Imagine learning to ride a bike:
1. First, you learn with training wheels (training)
2. Then try without them in the backyard (testing)
3. If you fall — learn some more
4. Then try on the street

```rust
/// "Step by step" validator
/// Like a teacher who explains first, then tests
pub struct WalkForwardValidator {
    /// How many times we'll test
    num_tests: usize,
    /// Break between learning and testing (to prevent peeking)
    break_days: usize,
}

impl WalkForwardValidator {
    pub fn new(num_tests: usize, break_days: usize) -> Self {
        Self { num_tests, break_days }
    }

    /// Split data into parts: learn → break → test
    pub fn split(&self, total_days: usize) -> Vec<(usize, usize, usize, usize)> {
        let mut splits = Vec::new();
        let chunk_size = total_days / (self.num_tests + 1);

        for i in 0..self.num_tests {
            let learn_end = (i + 1) * chunk_size;
            let test_start = learn_end + self.break_days;
            let test_end = test_start + chunk_size;

            // (learning start, learning end, test start, test end)
            splits.push((0, learn_end, test_start, test_end.min(total_days)));
        }

        splits
    }
}

// Example usage:
// Imagine we have data for 100 days
// validator.split(100) will divide them:
// 1. Learn on days 0-25, test on 30-55
// 2. Learn on days 0-50, test on 55-80
// and so on...
```

---

## The Three Barriers Method

### Analogy: Playing with a Ball

Imagine you're bouncing a ball in a room:
- **Ceiling** — if the ball hits the ceiling, you win! (price rose enough)
- **Floor** — if it hits the floor, you lose... (price fell)
- **Time** — if nothing happens in 10 seconds — it's a tie (price barely changed)

```rust
/// Three barriers: win, lose, or draw
#[derive(Debug, Clone, Copy)]
pub enum GameResult {
    Win,   // Price rose enough (ball hit ceiling)
    Lose,  // Price fell too much (ball hit floor)
    Draw,  // Time ran out (ball floated in air)
}

/// The "three barriers" game
pub struct ThreeBarriersGame {
    /// How much price must rise to win (e.g., 2%)
    ceiling_percent: f64,
    /// How much price can fall before losing (e.g., -2%)
    floor_percent: f64,
    /// How long we wait
    wait_time: usize,
}

impl ThreeBarriersGame {
    pub fn new(win_threshold: f64, lose_threshold: f64, max_wait: usize) -> Self {
        Self {
            ceiling_percent: win_threshold,
            floor_percent: -lose_threshold.abs(),
            wait_time: max_wait,
        }
    }

    /// Let's play! See what happens to the price
    pub fn play(&self, prices: &[f64], start: usize) -> Option<GameResult> {
        // Need enough data ahead
        if start + self.wait_time >= prices.len() {
            return None;
        }

        let start_price = prices[start];

        // Check each following day
        for day in 1..=self.wait_time {
            let current_price = prices[start + day];
            let change = (current_price - start_price) / start_price * 100.0;

            // Hit the ceiling?
            if change >= self.ceiling_percent {
                return Some(GameResult::Win);
            }

            // Hit the floor?
            if change <= self.floor_percent {
                return Some(GameResult::Lose);
            }
        }

        // Time's up — it's a draw
        Some(GameResult::Draw)
    }
}

// Example:
// let game = ThreeBarriersGame::new(2.0, 2.0, 10);
// If price rises 2% within 10 days — Win
// If it falls 2% — Lose
// If it stays in the corridor — Draw
```

---

## Indicators: Decision-Making Helpers

### RSI — The "Fatigue" Indicator

Imagine a long-distance runner:
- If they've been running very fast for a long time, they'll probably **get tired** and slow down
- If they've been barely moving for a while, maybe they've **rested** and will speed up

RSI shows how "tired" the market is from rising or falling.

```rust
/// Market fatigue indicator (RSI)
/// Like an energy level sensor for an athlete
pub struct FatigueIndicator {
    /// How many days to look back
    lookback_days: usize,
}

impl FatigueIndicator {
    pub fn new(days: usize) -> Self {
        Self { lookback_days: days }
    }

    /// Calculate "fatigue" level from 0 to 100
    /// 0-30 = very tired from falling (maybe time to rise?)
    /// 70-100 = very tired from rising (maybe time to fall?)
    /// 30-70 = normal state
    pub fn calculate(&self, prices: &[f64]) -> Option<f64> {
        if prices.len() < self.lookback_days + 1 {
            return None;
        }

        let recent_prices = &prices[prices.len() - self.lookback_days - 1..];

        // Count how many ups and downs there were
        let mut energy_gained = 0.0;  // Energy from rising
        let mut energy_spent = 0.0;   // Energy from falling

        for i in 1..recent_prices.len() {
            let change = recent_prices[i] - recent_prices[i - 1];
            if change > 0.0 {
                energy_gained += change;
            } else {
                energy_spent += change.abs();
            }
        }

        // Average energy
        let avg_gain = energy_gained / self.lookback_days as f64;
        let avg_loss = energy_spent / self.lookback_days as f64;

        // If there were no declines — maximum fatigue from rising
        if avg_loss == 0.0 {
            return Some(100.0);
        }

        // RSI formula
        let strength = avg_gain / avg_loss;
        Some(100.0 - 100.0 / (1.0 + strength))
    }

    /// Simple explanation of the result
    pub fn explain(rsi: f64) -> &'static str {
        if rsi < 30.0 {
            "Market is very tired of falling. It might start rising soon!"
        } else if rsi > 70.0 {
            "Market is very tired of rising. It might start falling soon!"
        } else {
            "Market is in a normal state. Nothing special."
        }
    }
}
```

### Moving Average — "Smoothed" Price

Imagine you're watching the weather:
- Yesterday it was 50°F
- Today it's 70°F
- Does this mean the weather changed drastically?

**Moving Average** is like the average temperature for a week. It shows the **trend**, not random jumps.

```rust
/// Moving average — "noise smoother"
/// Like glasses that remove ripples from an image
pub struct SmoothingGlasses {
    /// How many days to average
    smooth_period: usize,
}

impl SmoothingGlasses {
    pub fn new(period: usize) -> Self {
        Self { smooth_period: period }
    }

    /// Put on the "glasses" and look at prices
    /// Return the smoothed price
    pub fn look(&self, prices: &[f64]) -> Option<f64> {
        if prices.len() < self.smooth_period {
            return None;
        }

        // Take the last N days and average them
        let recent = &prices[prices.len() - self.smooth_period..];
        let sum: f64 = recent.iter().sum();
        Some(sum / self.smooth_period as f64)
    }

    /// Compare current price with smoothed one
    pub fn compare_with_current(&self, prices: &[f64]) -> Option<String> {
        let smooth = self.look(prices)?;
        let current = *prices.last()?;

        if current > smooth * 1.02 {
            Some(String::from("Price is ABOVE average! Could be an uptrend."))
        } else if current < smooth * 0.98 {
            Some(String::from("Price is BELOW average! Could be a downtrend."))
        } else {
            Some(String::from("Price is near average. Trend unclear."))
        }
    }
}

// Example:
// Prices: [100, 102, 98, 105, 103, 108, 110]
// 5-day average: (98+105+103+108+110)/5 = 104.8
// Current price 110 > 104.8 → probably going up!
```

---

## Neural Networks: The Computer's Brain

### LSTM — Dory's Memory Problem Solved

Remember the movie "Finding Nemo"? Dory the fish forgot everything in seconds.

Regular neural networks are like Dory. They forget what happened earlier.

**LSTM** (Long Short-Term Memory) is a neural network with good memory. It remembers important things from the past!

```rust
/// LSTM memory cell
/// Like a cell in your brain that decides:
/// - What to remember?
/// - What to forget?
/// - What to tell others?
pub struct MemoryCell {
    /// Forget gate: what to throw out of memory?
    forget_gate: f64,
    /// Input gate: what new things to remember?
    input_gate: f64,
    /// Output gate: what to tell?
    output_gate: f64,
    /// Long-term memory
    long_memory: f64,
    /// Short-term memory (what I remember right now)
    short_memory: f64,
}

impl MemoryCell {
    pub fn new() -> Self {
        Self {
            forget_gate: 0.0,
            input_gate: 0.0,
            output_gate: 0.0,
            long_memory: 0.0,
            short_memory: 0.0,
        }
    }

    /// Process new information
    /// Like when someone tells you something new
    pub fn process(&mut self, new_info: f64) -> f64 {
        // 1. Decide what to forget from old memory
        // (e.g., forget what you had for breakfast a week ago)
        self.forget_gate = Self::think(new_info + self.short_memory);

        // 2. Decide what new stuff is worth remembering
        // (e.g., remember your friend's birthday)
        self.input_gate = Self::think(new_info);

        // 3. Update long-term memory
        let new_memory_candidate = (new_info * 2.0 - 1.0).tanh(); // from -1 to 1
        self.long_memory = self.forget_gate * self.long_memory
                         + self.input_gate * new_memory_candidate;

        // 4. Decide what's important now (output)
        self.output_gate = Self::think(new_info + self.long_memory);

        // 5. Update short-term memory
        self.short_memory = self.output_gate * self.long_memory.tanh();

        self.short_memory
    }

    /// The "think" function — turns a number into probability (0 to 1)
    fn think(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

// Example usage:
// let mut brain = MemoryCell::new();
// brain.process(100.0);  // Price 100
// brain.process(105.0);  // Price rose to 105
// brain.process(103.0);  // Dropped a bit to 103
// Now brain remembers: there was growth, then a small drop
```

---

## Reinforcement Learning: Teaching a Robot to Trade

### Analogy: Learning Through Gaming

Imagine you're learning to play a video game:
- Score points — feels good! (reward)
- Lose a life — bummer! (penalty)
- Over time you understand how to play better

```rust
/// Trading student bot
/// Like a child learning to make decisions
pub struct TradingStudent {
    /// Money in pocket (relative to start)
    wallet: f64,
    /// What we're holding (-1 = sold, 0 = nothing, 1 = bought)
    holding: f64,
    /// Experience table: situation -> action -> how good it was
    experience: std::collections::HashMap<String, [f64; 3]>,
    /// How fast we learn (0.1 = slow, 0.9 = fast)
    learning_speed: f64,
    /// How often we try new things (0.1 = rarely, 0.9 = often)
    curiosity: f64,
}

impl TradingStudent {
    pub fn new() -> Self {
        Self {
            wallet: 1.0,
            holding: 0.0,
            experience: std::collections::HashMap::new(),
            learning_speed: 0.1,
            curiosity: 0.2,
        }
    }

    /// Look at the situation and decide what to do
    pub fn decide(&self, situation: &str) -> Action {
        // Sometimes try a random action (exploring)
        if Self::random() < self.curiosity {
            return match (Self::random() * 3.0) as usize {
                0 => Action::Sell,
                1 => Action::Wait,
                _ => Action::Buy,
            };
        }

        // Usually do what gave more points before
        let scores = self.experience
            .get(situation)
            .unwrap_or(&[0.0, 0.0, 0.0]);

        if scores[0] > scores[1] && scores[0] > scores[2] {
            Action::Sell
        } else if scores[2] > scores[0] && scores[2] > scores[1] {
            Action::Buy
        } else {
            Action::Wait
        }
    }

    /// Learn from the result
    pub fn learn(&mut self, situation: &str, action: Action, reward: f64) {
        let action_idx = match action {
            Action::Sell => 0,
            Action::Wait => 1,
            Action::Buy => 2,
        };

        let scores = self.experience
            .entry(situation.to_string())
            .or_insert([0.0, 0.0, 0.0]);

        // Update action score
        // New score = old + learning_speed * (reward - old_score)
        let old_score = scores[action_idx];
        scores[action_idx] = old_score + self.learning_speed * (reward - old_score);
    }

    fn random() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        (nanos as f64) / (u32::MAX as f64)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Sell,  // Sell (think price will fall)
    Wait,  // Do nothing
    Buy,   // Buy (think price will rise)
}

// How it works:
// 1. Bot looks at situation (e.g., "RSI is low")
// 2. Decides: buy, sell, or wait
// 3. Gets a reward or penalty
// 4. Remembers: "when RSI is low, better to buy!"
```

---

## How to Measure Success?

### Sharpe Ratio: Reward for Risk

Imagine two students:
- Pete gets A's, but sometimes F's too (high variance)
- Mary gets B's consistently (low variance)

Who's doing better? Depends on how we value consistency!

**Sharpe Ratio** shows: how much profit we get per unit of risk.

```rust
/// Success calculator
pub struct SuccessCalculator;

impl SuccessCalculator {
    /// Sharpe Ratio
    /// Higher is better for profit/risk balance
    /// > 1.0 = good
    /// > 2.0 = excellent!
    /// < 0 = bad, losing money
    pub fn sharpe_ratio(daily_returns: &[f64]) -> f64 {
        if daily_returns.is_empty() {
            return 0.0;
        }

        // Average daily profit
        let avg: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;

        // How much profit "jumps" (standard deviation)
        let variance: f64 = daily_returns
            .iter()
            .map(|r| (r - avg).powi(2))
            .sum::<f64>() / daily_returns.len() as f64;
        let volatility = variance.sqrt();

        if volatility == 0.0 {
            return 0.0;
        }

        // Annualize (252 trading days)
        avg / volatility * (252.0_f64).sqrt()
    }

    /// Explain the result
    pub fn explain_sharpe(ratio: f64) -> &'static str {
        if ratio < 0.0 {
            "Uh oh! We're losing money. Something needs to change!"
        } else if ratio < 0.5 {
            "Meh. There's profit, but risk is high."
        } else if ratio < 1.0 {
            "Not bad! Profit roughly equals risk."
        } else if ratio < 2.0 {
            "Good! Profit is higher than risk."
        } else {
            "Excellent! High profit with low risk!"
        }
    }

    /// Win percentage
    pub fn win_rate(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        wins as f64 / returns.len() as f64 * 100.0
    }

    /// Maximum drawdown (biggest "dip")
    /// Like the longest losing streak in a game
    pub fn max_drawdown(portfolio_values: &[f64]) -> f64 {
        if portfolio_values.is_empty() {
            return 0.0;
        }

        let mut peak = portfolio_values[0];
        let mut max_drop = 0.0;

        for &value in portfolio_values {
            if value > peak {
                peak = value;
            }
            let current_drop = (peak - value) / peak * 100.0;
            if current_drop > max_drop {
                max_drop = current_drop;
            }
        }

        max_drop
    }
}

// Example:
// Daily returns: [0.01, -0.005, 0.02, -0.01, 0.015, 0.008]
// Average: 0.63%
// Volatility: 1.1%
// Sharpe = 0.63 / 1.1 * sqrt(252) ≈ 9.1 (very good!)
```

---

## Golden Rules of Machine Learning for Trading

### 1. No Peeking at the Future!
Like on a test — you can't look at the answers beforehand.

### 2. Data Constantly Changes
Today's market is not like yesterday's. You need to keep relearning.

### 3. Most Movements Are Noise
Not every price movement means something. You need to find real signals.

### 4. Simple is Better Than Complex
Often a simple strategy works better than a super-smart neural network.

### 5. Test on New Data
If a strategy only works on old data — it's useless!

---

## What's Next?

After this chapter, you understand:
- How a computer "learns" to trade
- Why you can't peek at the future
- What indicators are (RSI, moving averages)
- How neural network memory works (LSTM)
- How to train a trading robot through gaming
- How to measure success (Sharpe ratio)

In the next chapters, we'll learn how to make all of this **super fast** — because on the exchange, every millisecond counts!

---

## Glossary

| Term | Simple Explanation |
|------|-------------------|
| **Machine Learning** | Teaching a computer through examples |
| **Neural Network** | A program similar to a brain |
| **LSTM** | A neural network with good memory |
| **RSI** | Market "fatigue" indicator |
| **Moving Average** | "Smoothed" price |
| **Sharpe Ratio** | Reward for risk |
| **Walk-Forward** | "Step by step" testing |
| **Reinforcement Learning** | Learning through rewards and penalties |
