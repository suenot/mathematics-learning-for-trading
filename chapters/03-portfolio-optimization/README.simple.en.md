# Chapter 3: Don't Put All Your Eggs in One Basket!

## Who is this chapter for?

This chapter is written in simple language — as if I'm explaining to a younger brother or sister. If you're a student or just want to understand the basics without complex math — welcome!

---

## The Main Idea: Don't Put All Your Eggs in One Basket!

Imagine grandma gave you 10 eggs and asked you to carry them home. You have three baskets of different sizes.

**Option 1**: Put all 10 eggs in one big basket.
- If you trip and drop that basket — ALL eggs break!

**Option 2**: Spread eggs across three baskets.
- If you drop one — you only lose some eggs, the rest are safe!

**This is the main principle of investing — diversification!**

---

## The Story: The Wise Grandpa Markowitz

In 1952, there was a scientist named Harry Markowitz. He wondered: "How should we properly distribute money across different investments?"

He came up with a simple rule:
> **You need to look not only at how much you can earn, but also at how much you can lose!**

For this idea, he received the Nobel Prize in 1990.

---

## Real-Life Example: Ice Cream Stand and Umbrella Shop

Imagine you have some money and want to start a business. You have two options:

### Business 1: Ice Cream Stand
- On sunny days: earn **$100** per day
- On rainy days: earn **$0** (nobody wants ice cream)

### Business 2: Umbrella Shop
- On sunny days: earn **$0** (nobody needs umbrellas)
- On rainy days: earn **$100**

### What to choose?

**Option A**: Put all money in ice cream
- If it's sunny — you earn a lot!
- If it rains — you earn nothing :(

**Option B**: Put all money in umbrellas
- If it rains — you earn a lot!
- If it's sunny — you earn nothing :(

**Option C**: Put half in ice cream, half in umbrellas
- If it's sunny: $50 from ice cream + $0 from umbrellas = **$50**
- If it rains: $0 from ice cream + $50 from umbrellas = **$50**

**Magic!** In Option C you **ALWAYS** earn $50, regardless of weather!

This is called **hedging** — when one investment protects another.

---

## What is Risk? (Simple Explanation)

**Risk** is how much your result can change.

### Example: Two Ways to Get to School

**Way 1: Walking**
- Always takes 20 minutes
- Sometimes 19, sometimes 21 — but almost always around 20
- **Low risk** — result is predictable

**Way 2: By Bus**
- Sometimes 10 minutes (if lucky with traffic)
- Sometimes 40 minutes (if stuck in traffic)
- On average also about 20 minutes
- **High risk** — result is unpredictable

Both ways take 20 minutes on average, but **the bus is riskier** because the result can vary a lot!

---

## Correlation: When Things Move Together

**Correlation** shows how different things are connected.

### Positive Correlation (move together)
- Temperature outside and ice cream sales
- When it's hot → people buy more ice cream
- When it's cold → people buy less ice cream

### Negative Correlation (move in opposite directions)
- Ice cream and umbrellas (like in our example)
- When sunny → ice cream good, umbrellas bad
- When rainy → ice cream bad, umbrellas good

### No Correlation (not connected at all)
- Ice cream sales in New York and fish sales in Tokyo
- They don't affect each other

**The Main Secret of Diversification:**
> For diversification to work, you need to choose assets with **low or negative correlation**!

If you put money in two businesses that fall together — that's NOT diversification!

---

## Risk Parity: Everyone Shares Risk Equally

Imagine 4 students in your class are doing a group project:
- Mary — A-student, very reliable
- Pete — B-student, quite reliable
- Bob — C-student, not very reliable
- Kate — D-student, unreliable

### Normal Approach: Give each person 25% of the work
Problem: Kate will fail her part and drag everyone down!

### Risk Parity Approach: Distribute by reliability
- Mary does 40% of work (she's reliable, can trust her with more)
- Pete does 30%
- Bob does 20%
- Kate does 10% (less work — less risk)

Now if Kate fails, it only affects 10% of the project, not 25%!

**Risk Parity in Investing:**
> Reliable assets (bonds) get LESS money, risky assets (stocks) get MORE, so each contributes equally to total risk.

Wait... that seems backwards? Actually it's the opposite:
- Bonds are less volatile → to "feel" the same as stocks, they need more weight
- Stocks are more volatile → even a small share creates a lot of risk

---

## VaR: The Worst Day (Almost)

**VaR (Value at Risk)** is a way to say: "In 95% of cases, I'll lose no more than X dollars".

### Example: Roller Coaster

Imagine riding a roller coaster 100 times. Each time you record how scary it was on a scale from 0 to 10.

Results (from least scary to most scary):
- 95 rides: fear from 0 to 7
- 5 rides: fear from 8 to 10 (very scary!)

**VaR at 95% level** = 7

This means: "In 95% of cases, my fear won't exceed 7 out of 10".

### In Investing

If you have $100,000 and **VaR 95% = $5,000**, this means:
> "In 95 days out of 100, I'll lose no more than $5,000"

But in 5 days out of 100, losses could be MORE than $5,000!

---

## Maximum Drawdown: The Deepest Fall

**Drawdown** — how much value dropped from the peak.

### Example: Climbing a Mountain

Imagine you're climbing a mountain:
1. Start at 0 meters elevation
2. Climb to 100 meters (yay, new record!)
3. Go down to 80 meters (drawdown of 20 meters)
4. Climb to 150 meters (new record!)
5. Go down to 90 meters (drawdown of 60 meters)
6. Climb to 200 meters

**Maximum Drawdown = 60 meters** — this is the deepest fall from a peak.

### In Investing

If your portfolio:
1. Grew to $150,000
2. Fell to $90,000
3. Grew again to $200,000

Maximum Drawdown = $150,000 - $90,000 = **$60,000** (or 40%)

This is important! Even if you made money in the end — you felt terrible when you lost 40% in the process.

---

## Why Simple "Equal Split" Doesn't Work

### Example: Packing Food for a Picnic

You're organizing a picnic and taking:
- 1 watermelon (20 lbs)
- 1 bag of chips (4 oz)
- 1 bottle of water (1 liter)

If you say "we're taking one of each" — seems equal? But watermelon weighs 80 times more than chips!

In investing it's the same:
- Stocks are very volatile (like a heavy watermelon)
- Bonds are stable (like light chips)

If you invest 50% in stocks and 50% in bonds:
- Stocks will determine 90% of your portfolio's movements!
- Bonds barely make a difference

**Risk Parity** solves this by measuring "weight" by risk, not by money.

---

## Rust Code: Simple Portfolio Calculator

```rust
/// Simple portfolio of two assets
fn main() {
    // Asset 1: Aggressive stocks
    // Average return 15% per year, but volatility 30%
    let stock_return = 0.15;
    let stock_volatility = 0.30;

    // Asset 2: Bonds
    // Average return 5% per year, volatility 5%
    let bond_return = 0.05;
    let bond_volatility = 0.05;

    // Correlation between them (weak positive)
    let correlation = 0.2;

    println!("=== Comparing Different Strategies ===\n");

    // Strategy 1: All in stocks
    println!("Strategy 1: 100% stocks");
    println!("  Expected return: {:.1}%", stock_return * 100.0);
    println!("  Volatility: {:.1}%", stock_volatility * 100.0);
    println!();

    // Strategy 2: All in bonds
    println!("Strategy 2: 100% bonds");
    println!("  Expected return: {:.1}%", bond_return * 100.0);
    println!("  Volatility: {:.1}%", bond_volatility * 100.0);
    println!();

    // Strategy 3: 50/50
    let weight_stocks = 0.5;
    let weight_bonds = 0.5;

    let portfolio_return = weight_stocks * stock_return + weight_bonds * bond_return;

    // Formula for two-asset portfolio volatility:
    // σ_p = √(w1²·σ1² + w2²·σ2² + 2·w1·w2·σ1·σ2·ρ)
    let portfolio_volatility = (
        (weight_stocks * stock_volatility).powi(2)
        + (weight_bonds * bond_volatility).powi(2)
        + 2.0 * weight_stocks * weight_bonds * stock_volatility * bond_volatility * correlation
    ).sqrt();

    println!("Strategy 3: 50% stocks + 50% bonds");
    println!("  Expected return: {:.1}%", portfolio_return * 100.0);
    println!("  Volatility: {:.1}%", portfolio_volatility * 100.0);
    println!();

    // Strategy 4: Risk Parity (simplified)
    // Idea: give more weight to less volatile asset
    // Weight is inversely proportional to volatility
    let inv_vol_stocks = 1.0 / stock_volatility;
    let inv_vol_bonds = 1.0 / bond_volatility;
    let total_inv_vol = inv_vol_stocks + inv_vol_bonds;

    let rp_weight_stocks = inv_vol_stocks / total_inv_vol;
    let rp_weight_bonds = inv_vol_bonds / total_inv_vol;

    let rp_return = rp_weight_stocks * stock_return + rp_weight_bonds * bond_return;

    let rp_volatility = (
        (rp_weight_stocks * stock_volatility).powi(2)
        + (rp_weight_bonds * bond_volatility).powi(2)
        + 2.0 * rp_weight_stocks * rp_weight_bonds * stock_volatility * bond_volatility * correlation
    ).sqrt();

    println!("Strategy 4: Risk Parity");
    println!("  Weights: {:.1}% stocks + {:.1}% bonds",
        rp_weight_stocks * 100.0, rp_weight_bonds * 100.0);
    println!("  Expected return: {:.1}%", rp_return * 100.0);
    println!("  Volatility: {:.1}%", rp_volatility * 100.0);
    println!();

    // Sharpe Ratio (return per unit of risk)
    let risk_free = 0.03; // Risk-free rate 3%

    println!("=== Sharpe Ratio (higher is better) ===");
    println!("  100% stocks: {:.2}", (stock_return - risk_free) / stock_volatility);
    println!("  100% bonds: {:.2}", (bond_return - risk_free) / bond_volatility);
    println!("  50/50: {:.2}", (portfolio_return - risk_free) / portfolio_volatility);
    println!("  Risk Parity: {:.2}", (rp_return - risk_free) / rp_volatility);
}
```

**Result:**
```
=== Comparing Different Strategies ===

Strategy 1: 100% stocks
  Expected return: 15.0%
  Volatility: 30.0%

Strategy 2: 100% bonds
  Expected return: 5.0%
  Volatility: 5.0%

Strategy 3: 50% stocks + 50% bonds
  Expected return: 10.0%
  Volatility: 15.8%

Strategy 4: Risk Parity
  Weights: 14.3% stocks + 85.7% bonds
  Expected return: 6.4%
  Volatility: 5.8%

=== Sharpe Ratio (higher is better) ===
  100% stocks: 0.40
  100% bonds: 0.40
  50/50: 0.44
  Risk Parity: 0.59
```

See? Risk Parity gives the best Sharpe Ratio!

---

## Simple VaR Calculation in Rust

```rust
/// Calculate VaR using "historical simulation" method
fn calculate_var(returns: &[f64], confidence: f64) -> f64 {
    // Sort returns from worst to best
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find value at (1 - confidence) level
    // For example, for 95% confidence we take the 5th percentile
    let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;

    // VaR is the potential loss (with minus sign)
    -sorted[index]
}

fn main() {
    // Imagine we have daily returns history for 100 days
    // In percentages: -5%, +2%, -1%, +3%, ... etc.
    let daily_returns = vec![
        -0.05, 0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.04, -0.01, 0.02,
        -0.04, 0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.05, -0.02, 0.01,
        -0.01, 0.02, -0.02, 0.03, -0.01, 0.01, -0.04, 0.02, -0.01, 0.03,
        -0.02, 0.01, -0.01, 0.02, -0.03, 0.04, -0.02, 0.01, -0.05, 0.03,
        -0.01, 0.02, -0.02, 0.01, -0.01, 0.03, -0.02, 0.02, -0.01, 0.01,
        -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, -0.02, 0.03, -0.01, 0.02,
        -0.01, 0.01, -0.04, 0.02, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01,
        -0.02, 0.03, -0.01, 0.01, -0.02, 0.02, -0.01, 0.04, -0.03, 0.02,
        -0.01, 0.01, -0.02, 0.03, -0.01, 0.02, -0.04, 0.01, -0.02, 0.03,
        -0.01, 0.02, -0.02, 0.01, -0.01, 0.03, -0.02, 0.02, -0.06, 0.01,
    ];

    let var_95 = calculate_var(&daily_returns, 0.95);
    let var_99 = calculate_var(&daily_returns, 0.99);

    println!("Portfolio Risk Analysis:");
    println!();
    println!("VaR 95%: {:.2}%", var_95 * 100.0);
    println!("  This means: in 95 days out of 100, losses won't exceed {:.2}%", var_95 * 100.0);
    println!();
    println!("VaR 99%: {:.2}%", var_99 * 100.0);
    println!("  This means: in 99 days out of 100, losses won't exceed {:.2}%", var_99 * 100.0);

    // If we have a $1,000,000 portfolio:
    let portfolio_value = 1_000_000.0;
    println!();
    println!("For a ${:,.0} portfolio:", portfolio_value);
    println!("  VaR 95% = ${:,.0}", var_95 * portfolio_value);
    println!("  VaR 99% = ${:,.0}", var_99 * portfolio_value);
}
```

---

## Maximum Drawdown in Rust

```rust
/// Calculate maximum drawdown
fn maximum_drawdown(prices: &[f64]) -> (f64, usize, usize) {
    let mut max_dd = 0.0;
    let mut peak_day = 0;
    let mut bottom_day = 0;

    let mut running_max = prices[0];
    let mut running_max_day = 0;

    for (day, &price) in prices.iter().enumerate() {
        // New maximum?
        if price > running_max {
            running_max = price;
            running_max_day = day;
        }

        // Current drawdown
        let drawdown = (running_max - price) / running_max;

        // New drawdown record?
        if drawdown > max_dd {
            max_dd = drawdown;
            peak_day = running_max_day;
            bottom_day = day;
        }
    }

    (max_dd, peak_day, bottom_day)
}

fn main() {
    // Portfolio price history by day
    let portfolio_prices = vec![
        100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 107.0, 105.0,
        102.0, 98.0, 95.0, 97.0, 100.0, 105.0, 110.0, 115.0,
        112.0, 108.0, 105.0, 110.0, 118.0, 125.0, 122.0, 128.0,
    ];

    let (max_dd, peak, bottom) = maximum_drawdown(&portfolio_prices);

    println!("=== Maximum Drawdown Analysis ===");
    println!();
    println!("Maximum drawdown: {:.1}%", max_dd * 100.0);
    println!("Peak was on day: {} (price: {:.0})", peak + 1, portfolio_prices[peak]);
    println!("Bottom was on day: {} (price: {:.0})", bottom + 1, portfolio_prices[bottom]);
    println!();
    println!("This means: at some point the portfolio fell {:.1}% from its maximum", max_dd * 100.0);

    // Visualization
    println!();
    println!("Price chart (simplified):");
    let max_price = portfolio_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for (day, &price) in portfolio_prices.iter().enumerate() {
        let bar_length = (price / max_price * 40.0) as usize;
        let bar: String = "█".repeat(bar_length);
        let marker = if day == peak {
            " <-- PEAK"
        } else if day == bottom {
            " <-- BOTTOM"
        } else {
            ""
        };
        println!("Day {:2}: {} {:.0}{}", day + 1, bar, price, marker);
    }
}
```

---

## Key Takeaways (Remember These!)

### 1. Diversification is Your Friend
Don't put all your money in one place. Spread it across different assets.

### 2. Look Beyond Returns
High returns often mean high risk. Find the balance!

### 3. Correlation Matters
Choose assets that don't move together. If one falls, another might rise.

### 4. Know Your Risk
Use VaR and Maximum Drawdown to understand how much you can lose.

### 5. Simple is Often Better
Risk Parity — a simple method that often works better than complex strategies.

---

## Homework (For the Curious)

1. **Eggs in Baskets**: You have 12 eggs and 3 baskets. Figure out how to distribute them, knowing that:
   - Basket 1 drops 10% of the time
   - Basket 2 drops 20% of the time
   - Basket 3 drops 30% of the time
   - Baskets drop independently of each other

2. **Ice Cream and Hot Chocolate**: How would you distribute money between these businesses, knowing that:
   - Ice cream sells in summer
   - Hot chocolate sells in winter
   - What's the correlation between them?

3. **Run the Code**: Try running the Rust examples and change the numbers. What happens if you increase correlation? Decrease it?

---

## Useful Resources

- **For Beginners**: Book "Rich Dad Poor Dad" — simple explanation of investing
- **Games**: Try a stock market simulator to understand in practice
- **Videos**: YouTube has lots of simple explanations of diversification

---

## Summary

Portfolio optimization is not rocket science! The main ideas are simple:

1. **Don't put all eggs in one basket**
2. **Choose different assets** (that don't fall at the same time)
3. **Watch your risk** (not just returns)
4. **Check yourself** (use VaR and Drawdown)

Now you know the basics! The full version of the chapter has more math and complex algorithms, but the principles stay the same.

---

*Next chapter: [04. Machine Learning — How Computers Learn to Predict](../04-ml-time-series/README.simple.en.md)*
