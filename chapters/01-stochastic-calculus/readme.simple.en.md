# Stochastic Calculus: Explained Simply

## What Is This All About?

Imagine watching a drunk person trying to walk home. They take steps, but don't know exactly where — sometimes left, sometimes right, randomly. This is called a **random walk**.

Now imagine this person takes steps so tiny and fast that their movement becomes continuous — like they're floating along a random path. This is **Brownian motion**!

---

## Why Does This Matter for Trading?

Stock and cryptocurrency prices behave like that drunk person:
- We don't know where the price will go next
- But we can describe the probabilities of different outcomes
- And use this to make decisions

---

## Brownian Motion: A Real-Life Story

### Pollen in Water

In 1827, botanist Robert Brown looked through a microscope at pollen in water. He saw that tiny pollen particles constantly jiggled in random directions.

**Why?** Because millions of invisible water molecules constantly push the pollen particle from all sides. Each push is random, and the result is chaotic movement.

### An Analogy to Understand It

Imagine a **soccer ball in a stadium** where thousands of fans simultaneously throw tennis balls at it from all directions. The ball will move chaotically because the hits come randomly.

Similarly, a stock price: thousands of traders buy and sell, pushing the price up and down.

---

## Key Properties of Brownian Motion

### 1. Starts at Zero

We start counting from zero: $W_0 = 0$.

**Analogy**: You stand in place before starting your random walk.

### 2. Continuity

The path is continuous — no "teleportation" from one point to another.

**Analogy**: Like a string being pulled — it doesn't break, it bends smoothly.

### 3. Independent Increments

Where the price goes in the next second doesn't depend on where it went before.

**Analogy**: When you flip a coin, the result doesn't depend on previous flips. Each time is a new randomness.

### 4. Normal Distribution

If you wait for time $t$, the change $W_t$ will be distributed like a "bell curve" (normally) with spread $\sqrt{t}$.

**Analogy**: If you roll a die many times and add up the results — you get a bell curve. The more rolls, the wider the bell.

---

## Ito's Integral: Why Do We Need It?

### The Problem

In regular math, if you know the rate of change of something, you can find the value itself using an integral.

But Brownian motion **has no speed**! It's too "jittery" — the path is never smooth.

### Ito's Solution

Japanese mathematician Ito invented a special way of integrating that works for random processes.

**Banking Analogy**:

Imagine you put money in a bank with an interest rate that changes randomly every second.
- Regular integral doesn't work because the rate "jumps"
- Ito's integral says: "Use the rate that was at the **beginning** of each small period"

It's like the bank saying: "We'll pay you interest at the rate from the start of the day, even if it changed afterward."

---

## Ito's Lemma: The "Chain Rule" for Randomness

### Regular Chain Rule

In school math, if $y = x^2$ and $x$ changes, then:
$$\frac{dy}{dx} = 2x$$

### The Problem with Randomness

But if $x$ is Brownian motion, an **extra term** appears!

### Ito's Formula

If $f(W_t)$ is a function of Brownian motion, then:

$$df = f'(W_t) \cdot dW_t + \frac{1}{2} f''(W_t) \cdot dt$$

**Where does the extra term come from?**

Because $(dW)^2 = dt$, not zero!

### Analogy

Imagine riding a bicycle on a very bumpy road (Brownian motion).
- On a smooth road, you'd just ride
- But on bumps, you constantly bounce, and this adds "extra" distance

This "bump effect" is the extra term in Ito's formula.

---

## Price Models

### Geometric Brownian Motion (GBM)

**Formula**:
$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

**What does this mean in simple words?**

- $S_t$ — stock price
- $\mu$ — "trend": where the price tends to go on average (up or down)
- $\sigma$ — the "range" of random fluctuations (volatility)
- $dW_t$ — random "push"

**Analogy — a hot air balloon**:

Imagine a hot air balloon that:
- On average rises upward (trend $\mu > 0$)
- But wind constantly blows it left and right ($\sigma \cdot dW$)

The stronger the wind ($\sigma$), the less predictable the trajectory.

### Why Multiply by $S_t$?

Because a stock price can't become negative!

If $S_t = 100$ and volatility is 20%, fluctuations are around $\pm 20$.
If $S_t = 10$, fluctuations are around $\pm 2$.

**Analogy**: A rich person can lose more money in absolute terms, but both lose the same **percentage**.

---

## Stochastic Volatility (Heston Model)

### The Problem with GBM

In reality, volatility isn't constant! Sometimes the market is calm, sometimes it's stormy.

### Heston's Solution

Let volatility also change randomly:

$$dS_t = \mu S_t \, dt + \sqrt{V_t} S_t \, dW^S_t$$
$$dV_t = \kappa(\theta - V_t) \, dt + \xi \sqrt{V_t} \, dW^V_t$$

**What does this mean?**

- $V_t$ — current volatility (it's random!)
- $\theta$ — "normal" volatility level
- $\kappa$ — how fast volatility returns to normal
- $\xi$ — how much volatility itself jumps

**Analogy — weather**:

- Temperature ($S$) changes randomly
- But the "variability of weather" ($V$) also changes!
- In summer, weather is more stable; in winter, more variable
- Over time, weather returns to "normal" variability

---

## Jumps (Jump-Diffusion)

### The Problem

GBM assumes price changes smoothly. But sometimes there are sudden jumps — good news, bad news, crashes.

### Merton's Solution

Add random jumps:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t + S_t \, dJ_t$$

where $J_t$ is the jump process.

**Analogy — pedestrian and bus**:

Imagine walking (regular movement) and sometimes jumping on a bus (jump). The bus can take you forward or backward, and you don't know when it will appear.

---

## Monte Carlo: When Formulas Don't Help

### The Idea

If it's impossible to calculate something with a formula, you can **simulate** many random paths and see what happens on average.

**Analogy — experiment**:

To find the probability of winning a board game, you can:
1. Derive a formula (hard!)
2. Or play 10,000 games and count how many times you won

### Example in Rust

```rust
// Imagine we want to find the average stock price after one year
// Starting price = 100, growth = 10%, volatility = 20%

use rand::Rng;
use rand_distr::{Normal, Distribution};

fn simulate_price(s0: f64, mu: f64, sigma: f64, t: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Random number from normal distribution
    let z = normal.sample(&mut rng);

    // GBM formula (exact solution)
    s0 * ((mu - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp()
}

fn main() {
    let s0 = 100.0;      // Starting price
    let mu = 0.10;       // 10% annual growth
    let sigma = 0.20;    // 20% volatility
    let t = 1.0;         // 1 year

    let n_simulations = 100_000;

    // Simulate many paths
    let sum: f64 = (0..n_simulations)
        .map(|_| simulate_price(s0, mu, sigma, t))
        .sum();

    let average = sum / n_simulations as f64;

    println!("Average price after one year: {:.2}", average);
    // We expect about 110.52 (= 100 * e^0.10)
}
```

---

## Practical Example: Simulating Bitcoin Price

```rust
use rand::Rng;
use rand_distr::{Normal, Distribution};

/// Generates a price path with jumps
fn simulate_bitcoin_path(
    initial_price: f64,
    days: usize,
) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Bitcoin parameters (approximate)
    let mu = 0.5;        // 50% annual growth (historically)
    let sigma = 0.8;     // 80% volatility (very high!)
    let dt = 1.0 / 365.0; // One day

    let mut prices = Vec::with_capacity(days + 1);
    prices.push(initial_price);

    let mut price = initial_price;

    for _ in 0..days {
        let z = normal.sample(&mut rng);

        // Daily change
        let daily_return = (mu - 0.5 * sigma * sigma) * dt
                          + sigma * dt.sqrt() * z;

        price *= daily_return.exp();
        prices.push(price);
    }

    prices
}

fn main() {
    let btc_price = 50_000.0; // Starting price $50,000
    let days = 365;           // Simulate one year

    // Generate several paths
    for i in 1..=5 {
        let path = simulate_bitcoin_path(btc_price, days);
        let final_price = path.last().unwrap();

        println!("Path {}: ${:.0} -> ${:.0} ({:+.1}%)",
            i,
            btc_price,
            final_price,
            (final_price / btc_price - 1.0) * 100.0
        );
    }
}
```

**Example output**:
```
Path 1: $50000 -> $127543 (+155.1%)
Path 2: $50000 -> $23891 (-52.2%)
Path 3: $50000 -> $89234 (+78.5%)
Path 4: $50000 -> $41023 (-18.0%)
Path 5: $50000 -> $234521 (+369.0%)
```

See the range! This is the high volatility of cryptocurrencies.

---

## Key Terms (Cheat Sheet)

| Term | Simple Explanation |
|------|-------------------|
| **Brownian motion** | Random walk in continuous time |
| **Volatility (σ)** | How much the price jumps around |
| **Drift (μ)** | Where the price tends to go on average |
| **Ito's integral** | Way to sum up random changes |
| **Ito's lemma** | Rule for functions of random processes |
| **Monte Carlo** | Estimation method using many simulations |
| **GBM** | Basic model for asset prices |
| **Stochastic volatility** | When volatility itself is random |

---

## Why Does a Trader Need All This?

1. **Risk assessment**: How much can I lose with 95% probability?
2. **Option pricing**: How much is the right to buy a stock worth next month?
3. **Strategy creation**: How to trade while accounting for randomness?
4. **Backtesting**: Simulate thousands of scenarios to test a strategy

---

## What's Next?

In the next chapter, we'll study **market microstructure**:
- How the order book works
- Why the price "jumps" between bid and ask
- How market makers make money

Understanding stochastic calculus will help you understand how to model order flow and make real-time decisions!
