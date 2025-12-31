# Chapter 2: How an Exchange Works (Simple Explanation)

## Who is this chapter for?

This chapter is written so that a school student can understand it. We'll use examples from everyday life instead of complex math.

---

## What is an Exchange?

Imagine a **school trading fair** where kids swap collectible cards.

- Peter wants to **sell** a rare card for $10
- Mary wants to **buy** such a card but is only willing to pay $9

The difference of $1 between them is called the **spread**.

An exchange is like a huge organized fair where instead of cards, people trade stocks, currencies, and cryptocurrencies.

---

## Order Book — The Queue of Buyers and Sellers

### Imagine a Cafeteria Queue

In a cafeteria, there are **two queues**:
- A queue of **buyers** wanting cookies
- A queue of **sellers** (the lunch lady with cookies)

```
SELLERS (want to sell):
┌─────────────────────────┐
│ Cookie for $5 - 3 pcs   │  ← Cheapest price
│ Cookie for $5.50 - 5 pcs│
│ Cookie for $6 - 2 pcs   │
└─────────────────────────┘

═══════════════════════════  Gap = $0.50 (spread)

BUYERS (want to buy):
┌─────────────────────────┐
│ Will pay $4.50 - Tommy  │  ← Highest buy price
│ Will pay $4 - Sarah     │
│ Will pay $3.50 - Mike   │
└─────────────────────────┘
```

### How Does a Trade Happen?

If **Jake** comes and says: "I want a cookie **right now** at any price!" — he buys for $5 (the cheapest selling price).

If **Emma** comes and says: "I'm selling a cookie **right now** at any price!" — she sells for $4.50 (the highest buying price).

---

## Types of Orders — How to Ask for a Cookie

### 1. Limit Order
**"I want to buy, but only at THIS price!"**

Real-life example: "I'll buy this game, but only for $30, not more. I'll wait until someone agrees."

```rust
// Limit buy order
struct LimitOrder {
    price: 30,         // Price I'm willing to pay
    quantity: 1,       // Amount
    side: "Buy",       // I'm buying
}
```

### 2. Market Order
**"I want to buy RIGHT NOW at any price!"**

Real-life example: "I really need an umbrella, it's raining! I'll pay whatever you ask, just give it to me fast!"

```rust
// Market buy order
struct MarketOrder {
    quantity: 1,       // Amount
    side: "Buy",       // I'm buying
    // No price! Will buy at the first available price
}
```

---

## What is a Market Maker?

### Analogy: A Lemonade Stand

Imagine a **lemonade stand** that buys lemons wholesale for $0.40 and sells lemonade for $0.50.

- The stand is **always ready to buy** (if you have lemons)
- The stand is **always ready to sell** (if you want lemonade)
- The $0.10 difference is their **profit** (spread)

**A market maker on an exchange does the same thing:**
- Places a buy order at a low price
- Places a sell order at a high price
- Earns on the difference

```rust
// Market maker places two orders
fn make_quotes(current_price: f64) -> (f64, f64) {
    let bid = current_price - 0.5;  // Willing to buy lower
    let ask = current_price + 0.5;  // Willing to sell higher
    (bid, ask)
}

// Example: price is now 100
// Market maker: "I'll buy at 99.5, sell at 100.5"
```

### The Market Maker's Problem: "Stuck with Inventory"

Imagine you run a lemonade stand:
- In the morning, you bought 100 cups of lemonade for $0.40 each
- Suddenly, everyone stopped drinking lemonade!
- You have 100 cups that nobody's buying
- And the market price dropped to $0.30

**You're losing money!** This is called **inventory risk**.

---

## The Hawkes Process — Why Events Come in Bursts

### Example: Applause in a Theater

In a theater:
1. One person starts clapping
2. The neighbor hears and also starts clapping
3. Then a few more people
4. In a second, the whole audience is clapping!

One clap **triggered** a wave of new claps.

### The Same Happens on an Exchange:

1. Someone bought 1000 shares
2. Others notice: "Oh, someone's buying! I should buy too!"
3. A wave of buying begins
4. The price rises

The **Hawkes Process** is a mathematical way to describe such "chain reactions."

```rust
// Simple example: event intensity
struct HawkesProcess {
    base_rate: f64,     // Base event rate (1 per minute)
    excitation: f64,    // How much one event speeds up the next
    decay: f64,         // How quickly the effect fades
}

impl HawkesProcess {
    // After each event, probability of next one increases
    fn event_happened(&mut self) {
        // Intensity temporarily increases
        // Then gradually returns to normal
    }
}
```

### Visually:

```
Event Intensity:

    ^
    │     /\        /\
    │    /  \      /  \    /\
    │   /    \    /    \  /  \
────┴──/──────\──/──────\/────\───> Time
       ↑       ↑         ↑
     Event   Event     Event
```

Each event creates a "spike" that then fades away.

---

## The Avellaneda-Stoikov Model: The Smart Stand

### The Problem

You run a stand and want to make money, but don't want to "get stuck with inventory."

### The Solution: Shift Your Prices!

**If you have too much inventory:**
- Sell CHEAPER (to get rid of it faster)
- Buy at HIGHER prices (to buy less)

**If you have little inventory:**
- Sell at HIGHER prices (no rush to sell)
- Buy CHEAPER (want to buy more)

### In Code:

```rust
struct SmartMarketMaker {
    inventory: i32,     // How much stock we have
    risk_level: f64,    // How risk-averse (0.1 = brave, 1.0 = cautious)
}

impl SmartMarketMaker {
    fn calculate_prices(&self, market_price: f64) -> (f64, f64) {
        // Shift the "fair" price based on inventory
        let adjusted_price = market_price
            - (self.inventory as f64) * self.risk_level;

        // If inventory = +10, adjusted_price is below market
        // This means we want to sell!

        // If inventory = -10, adjusted_price is above market
        // This means we want to buy!

        let spread = 1.0;  // Our profit
        let bid = adjusted_price - spread / 2.0;
        let ask = adjusted_price + spread / 2.0;

        (bid, ask)
    }
}

fn main() {
    let mut mm = SmartMarketMaker {
        inventory: 0,
        risk_level: 0.1,
    };

    let market_price = 100.0;

    // No inventory - prices are symmetric
    println!("{:?}", mm.calculate_prices(market_price));
    // Prints: (99.5, 100.5)

    // Accumulated 10 units of inventory
    mm.inventory = 10;
    println!("{:?}", mm.calculate_prices(market_price));
    // Prints: (98.5, 99.5) - willing to sell cheaper!

    // We owe -10 units
    mm.inventory = -10;
    println!("{:?}", mm.calculate_prices(market_price));
    // Prints: (100.5, 101.5) - willing to buy at higher price!
}
```

---

## Why Order Book Balance Matters

### Analogy: A Seesaw

```
           Buyers                  Sellers
             ↓                        ↓
        ┌────┴────┐              ┌────┴────┐
        │ 100 pcs │              │  50 pcs │
        └─────────┘              └─────────┘
              \                    /
               \                  /
                \       △        /
                 \     / \      /
                  \___/   \____/

        Seesaw tilts left - price will go UP
```

- If there are **more buyers** — price is likely to rise
- If there are **more sellers** — price is likely to fall

### In Code:

```rust
fn predict_price_direction(bid_volume: u64, ask_volume: u64) -> &'static str {
    let total = bid_volume + ask_volume;
    let imbalance = (bid_volume as f64 - ask_volume as f64) / total as f64;

    // imbalance ranges from -1 to +1
    // +1 = all buyers
    // -1 = all sellers

    if imbalance > 0.3 {
        "Price will likely go UP"
    } else if imbalance < -0.3 {
        "Price will likely go DOWN"
    } else {
        "Unclear, roughly balanced"
    }
}
```

---

## Microprice — A More Accurate "Average" Price

### Problem with Regular Average Price

Bid = 99, Ask = 101
Regular average = (99 + 101) / 2 = 100

But what if:
- There are 1000 lots on the Bid
- There are only 10 lots on the Ask

The price is likely to go UP (few sellers), so the "fair" price is closer to 101!

### Microprice Accounts for Volume:

```rust
fn microprice(bid: f64, ask: f64, bid_volume: f64, ask_volume: f64) -> f64 {
    // Weighted average
    // Larger volume "pulls" the price toward itself
    (bid_volume * ask + ask_volume * bid) / (bid_volume + ask_volume)
}

fn main() {
    let bid = 99.0;
    let ask = 101.0;

    // Equal volumes
    let mp1 = microprice(bid, ask, 100.0, 100.0);
    println!("Equal volumes: {}", mp1);  // 100.0

    // More buyers
    let mp2 = microprice(bid, ask, 1000.0, 10.0);
    println!("More buyers: {}", mp2);  // 99.02 (closer to bid)

    // More sellers
    let mp3 = microprice(bid, ask, 10.0, 1000.0);
    println!("More sellers: {}", mp3);  // 100.98 (closer to ask)
}
```

---

## Practical Example: Simple Simulator

Let's make a simple exchange simulator game:

```rust
use rand::Rng;

// Order book (very simplified)
struct SimpleOrderBook {
    best_bid: f64,       // Best buy price
    best_ask: f64,       // Best sell price
    bid_volume: u64,     // Volume at bid
    ask_volume: u64,     // Volume at ask
}

impl SimpleOrderBook {
    fn new(mid_price: f64) -> Self {
        Self {
            best_bid: mid_price - 0.5,
            best_ask: mid_price + 0.5,
            bid_volume: 100,
            ask_volume: 100,
        }
    }

    fn mid_price(&self) -> f64 {
        (self.best_bid + self.best_ask) / 2.0
    }

    fn spread(&self) -> f64 {
        self.best_ask - self.best_bid
    }

    // Simulate a random order
    fn random_order(&mut self) {
        let mut rng = rand::thread_rng();

        // 50% chance buy or sell
        if rng.gen_bool(0.5) {
            // Someone BOUGHT at market
            // This "eats" volume on the ask
            self.ask_volume = self.ask_volume.saturating_sub(10);

            // If volume is depleted, price rises
            if self.ask_volume == 0 {
                self.best_bid += 1.0;
                self.best_ask += 1.0;
                self.ask_volume = 100;
                self.bid_volume = 100;
            }
        } else {
            // Someone SOLD at market
            self.bid_volume = self.bid_volume.saturating_sub(10);

            if self.bid_volume == 0 {
                self.best_bid -= 1.0;
                self.best_ask -= 1.0;
                self.ask_volume = 100;
                self.bid_volume = 100;
            }
        }
    }
}

// Simple market maker
struct SimpleMarketMaker {
    inventory: i32,
    cash: f64,
    pnl_history: Vec<f64>,
}

impl SimpleMarketMaker {
    fn new() -> Self {
        Self {
            inventory: 0,
            cash: 10000.0,
            pnl_history: Vec::new(),
        }
    }

    // Calculate current profit
    fn pnl(&self, current_price: f64) -> f64 {
        self.cash + (self.inventory as f64) * current_price - 10000.0
    }

    // Decide on quotes
    fn decide(&mut self, book: &SimpleOrderBook) -> (f64, f64) {
        let mid = book.mid_price();

        // Simple strategy: shift prices based on inventory
        let shift = (self.inventory as f64) * 0.1;

        let my_bid = mid - 0.5 - shift;
        let my_ask = mid + 0.5 - shift;

        (my_bid, my_ask)
    }

    // When our order is filled
    fn on_fill(&mut self, side: &str, price: f64, volume: i32) {
        match side {
            "buy" => {
                self.inventory += volume;
                self.cash -= price * (volume as f64);
            }
            "sell" => {
                self.inventory -= volume;
                self.cash += price * (volume as f64);
            }
            _ => {}
        }
    }
}

fn main() {
    let mut book = SimpleOrderBook::new(100.0);
    let mut mm = SimpleMarketMaker::new();

    println!("Starting simulation!");
    println!("Initial capital: ${}", mm.cash);
    println!();

    // Simulate 20 steps
    for step in 1..=20 {
        // Random market change
        book.random_order();

        // Market maker decides what prices to quote
        let (my_bid, my_ask) = mm.decide(&book);

        // Simple fill logic:
        // If our bid >= market's best_ask, we buy
        // If our ask <= market's best_bid, we sell

        if my_bid >= book.best_ask {
            mm.on_fill("buy", book.best_ask, 1);
            println!("Step {}: BOUGHT at {:.2}", step, book.best_ask);
        }

        if my_ask <= book.best_bid {
            mm.on_fill("sell", book.best_bid, 1);
            println!("Step {}: SOLD at {:.2}", step, book.best_bid);
        }

        let pnl = mm.pnl(book.mid_price());
        println!(
            "Step {}: Price={:.2}, Inventory={}, P&L={:.2}",
            step, book.mid_price(), mm.inventory, pnl
        );
    }

    println!();
    println!("=== Results ===");
    println!("Final inventory: {}", mm.inventory);
    println!("Final P&L: ${:.2}", mm.pnl(book.mid_price()));
}
```

---

## Key Ideas of This Chapter (Remember These!)

### 1. Order Book is Two Queues
- Buyers (bids) want to buy cheaper
- Sellers (asks) want to sell higher
- A trade happens when prices cross

### 2. Market Maker Earns on the Spread
- Buys lower, sells higher
- But risks "getting stuck with inventory"

### 3. Market Events Come in Waves
- One purchase triggers others
- This is described by the Hawkes process

### 4. Volume Balance Predicts Price Direction
- More buyers — price goes up
- More sellers — price goes down

### 5. A Smart Market Maker Shifts Prices
- Too much inventory? Sell cheaper!
- Too little inventory? Buy cheaper!

---

## Homework

### Exercise 1: Draw an Order Book
Draw an order book on paper for your favorite game/cards. Include:
- 3 buy price levels
- 3 sell price levels
- The spread

### Exercise 2: Play Market Maker
Ask a friend or parent to play a game:
1. You're trading candy
2. You're the market maker: name buy and sell prices
3. Your friend is a random buyer/seller
4. See if you can make a profit!

### Exercise 3: Run the Simulator
If you know how to code:
1. Copy the simulator code above
2. Run it several times
3. Try changing parameters (shift) and see how P&L changes

---

## Glossary

| Term | Simple Explanation |
|------|-------------------|
| **Order Book** | List of all buy and sell orders |
| **Bid** | Price someone is willing to BUY at |
| **Ask** | Price someone is willing to SELL at |
| **Spread** | Difference between best sell and buy prices |
| **Market Maker** | Someone who places both buy AND sell orders |
| **Inventory** | How much stock you currently have |
| **P&L** | Profit and Loss — how much you've made or lost |
| **Hawkes Process** | Math for "chain reaction" events |
| **Microprice** | More accurate average price accounting for volumes |

---

*Now you know how an exchange works inside! In the next chapters, we'll learn to build portfolios and predict prices.*
