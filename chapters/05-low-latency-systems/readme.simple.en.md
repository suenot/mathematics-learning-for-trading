# Chapter 5: Fast Trading Systems — Explained for Students

## What is "Low Latency"?

Imagine you're playing an online video game. When you press the "jump" button, your character should jump **instantly**. If there's a whole second between pressing and jumping — the game is unplayable! This delay is called **latency**.

In the world of stock trading, it's exactly the same. When a computer sees a good price and wants to buy, it must do it **very, very fast** — faster than everyone else!

### How Fast?

Let's compare speeds:

| What happens | How long it takes |
|--------------|-------------------|
| Blink your eye | 300-400 milliseconds |
| Regular online store | 1-3 seconds |
| Professional trader | 1 millisecond |
| Super-fast trading system | 0.00001 seconds! |

**A super-fast system works 30 million times faster than you blink!**

---

## Analogy: Restaurant Kitchen

Imagine that a trading system is like a kitchen in a very popular restaurant.

### Regular Kitchen (slow system):

```
Order arrives → Chef walks to warehouse → Searches for ingredients →
→ Carries to kitchen → Starts cooking → Dish ready!

Time: 30 minutes
```

### Fast Food Kitchen (fast system):

```
Order arrives → All ingredients ALREADY at hand →
→ Start cooking immediately → Dish ready!

Time: 2 minutes
```

**The secret of speed: everything needed must be prepared and within reach!**

---

## Main Speed Secrets

### 1. Prepare Everything in Advance (Pre-allocation)

**Bad:** Every time we need a plate — we go to the store to buy one.

**Good:** We bought 1000 plates in advance, they're on the shelf.

```rust
// In code it looks like this:

// BAD — create new order each time
fn process_order() {
    let order = Order::new();  // Slow! Allocating memory
    // ...
}

// GOOD — orders are already ready, take from the "stack"
struct Kitchen {
    prepared_orders: Vec<Order>,  // 1000 prepared templates
}

fn process_order(kitchen: &mut Kitchen) {
    let order = kitchen.prepared_orders.pop();  // Instant!
    // ...
}
```

### 2. Don't Get in Each Other's Way (Lock-free)

**Bad:** Two chefs want to use one pan. One waits while the other finishes.

**Good:** Each chef has their own pan!

```
Chef 1: [own pan] → cooking
Chef 2: [own pan] → cooking

Nobody waits for anyone!
```

In programming, this is called **lock-free** (without locks).

### 3. Keep Everything Close (Cache optimization)

**Bad:** Salt is on the first floor, pepper on the second, oil in the basement.

**Good:** Salt, pepper, and oil are all on the same shelf!

```rust
// Data is stored together in memory — CPU reads it quickly
struct TradingData {
    price: u64,      // Price
    quantity: u64,   // Quantity
    time: u64,       // Time
    // All together = fast access!
}
```

---

## Analogy: Conveyor Belt

Imagine a conveyor belt at a toy factory:

```
[Parts] → [Assembly] → [Painting] → [Packaging] → [Done!]
   ↓          ↓            ↓             ↓
  1 sec      1 sec        1 sec         1 sec
```

### SPSC Queue (like a conveyor)

**SPSC** = Single Producer, Single Consumer (one puts, one takes)

It's like a conveyor belt:
- **One worker** puts parts on the belt
- **Another worker** takes them from the other end

```
Worker 1                               Worker 2
    ↓                                      ↓
  PUTS →  [□][□][□][□][□][□]  → TAKES
          ←── belt moves ──→
```

**Why is this fast?** Nobody collides! One only puts, the other only takes.

```rust
// Simple queue
let queue = SPSCQueue::new();

// Thread 1: puts data
queue.push(MarketData { price: 100 });

// Thread 2: takes data
let data = queue.pop();  // Got it!
```

---

## Analogy: Order Book

Imagine **two stacks of cards**:

### Green Stack (buyers):
```
┌──────────────┐
│ Will buy at 99│  ← most generous buyer on top
├──────────────┤
│ Will buy at 98│
├──────────────┤
│ Will buy at 97│
└──────────────┘
```

### Red Stack (sellers):
```
┌───────────────┐
│ Will sell at 101│  ← cheapest seller on top
├───────────────┤
│ Will sell at 102│
├───────────────┤
│ Will sell at 103│
└───────────────┘
```

**When someone wants to buy or sell:**
1. Look at the top card of the needed stack
2. If price matches — deal!
3. Remove the card

```rust
// In code
struct OrderBook {
    bids: Vec<PriceLevel>,  // Green stack (buyers)
    asks: Vec<PriceLevel>,  // Red stack (sellers)
}

// Best buy price
fn best_bid(&self) -> u64 {
    self.bids[0].price  // Just take the first element!
}
```

---

## Analogy: Why Internet Speed Matters

Imagine you're sending a letter to a friend:

### Regular Mail:
```
You → Post office → Sorting → Delivery → Friend
                    3-5 days
```

### Messenger:
```
You → Internet → Friend
       0.1 seconds
```

In trading, **special network settings** are used:

```rust
// Disable "bundling" of small messages
socket.set_nodelay(true);  // Send immediately!

// Increase the data "pipe"
socket.set_buffer_size(4_000_000);  // 4 megabytes
```

---

## Analogy: Why Pin a Program to a CPU Core?

**CPU (processor)** is like a group of chefs in a kitchen.

### Bad: Chef runs between stoves
```
Chef → Stove 1 → Stove 2 → Stove 1 → Stove 3
         ↑                              ↑
    Forgot where they stopped!    Remembering again!
```

### Good: Each chef at their own stove
```
Chef 1 → Stove 1 (always here!)
Chef 2 → Stove 2 (always here!)
```

```rust
// Pin program to core #2
CpuAffinity::pin_to_core(2);

// Now the program ALWAYS works on this core
// and doesn't "forget" its data
```

---

## Measuring Speed

### How to Know if the System is Fast?

It's like measuring a runner's time with a stopwatch:

```rust
// Start the "stopwatch"
let start = Instant::now();

// Do the work
process_order();

// Stop the stopwatch
let time = start.elapsed();
println!("Took {} nanoseconds", time.as_nanos());
```

### What are Percentiles?

Imagine 100 students ran 100 meters:

- **p50 (median)** — time achieved by 50 out of 100 people
- **p99** — time achieved by 99 out of 100 people
- **p99.9** — time achieved by 999 out of 1000 people

```
Running results:
  Best:    10 seconds
  p50:     12 seconds  (half are faster)
  p99:     15 seconds  (almost all are faster)
  Worst:   20 seconds
```

**In trading, p99 matters!** Because even rare "slowdowns" can cost money.

---

## Failure Protection (Circuit Breaker)

### Analogy: Automatic Switch

At home, there's a **circuit breaker** in the electrical panel. If the current is too high — it turns off the electricity to prevent a fire.

```
Normal: [ON] ──electricity──▶ [Devices work]

Problem: [Too much current!] → [Breaker tripped!] → [Safe]
```

In a trading system, a **Circuit Breaker** works the same way:

```rust
struct CircuitBreaker {
    failures: u32,      // How many errors
    threshold: u32,     // Maximum errors (e.g., 5)
    is_open: bool,      // Is the "breaker" tripped
}

// If 5 errors in a row — we stop!
if self.failures >= 5 {
    self.is_open = true;  // "Breaker" tripped
    println!("Too many errors! Pausing.");
}
```

---

## The Main Rule: Measure, Then Improve!

### Bad:
```
"I think this part is slow..."
*Spend a week speeding it up*
"Oops, the problem was somewhere else!"
```

### Good:
```
1. Measure ALL parts of the system
2. Find the slowest one
3. Speed up exactly that part
4. Measure again
```

---

## Summary: Main Speed Secrets

| Secret | Analogy | In Code |
|--------|---------|---------|
| Pre-allocation | Plates bought in advance | `Vec::with_capacity(1000)` |
| Lock-free | Each chef has own pan | `SPSCQueue`, `AtomicU64` |
| Cache locality | Everything on one shelf | `#[repr(C, align(64))]` |
| CPU pinning | Chef at own stove | `pin_to_core(2)` |
| Measurement | Stopwatch for runner | `Instant::now()` |
| Circuit breaker | Breaker in panel | `CircuitBreaker` |

---

## Fun Facts

1. **Speed of light** limits trading! A signal from New York to London takes ~30 milliseconds. That's why companies place servers as close to the exchange as possible.

2. **The fastest systems** run on special chips (FPGA), where the program is "hardwired" directly into the hardware!

3. **A nanosecond** is so short that light travels only 30 centimeters (about 1 foot) in that time.

---

## Try It Yourself!

### Experiment 1: Measure Speed

```rust
use std::time::Instant;

fn main() {
    let start = Instant::now();

    // Count to a million
    let mut sum = 0u64;
    for i in 0..1_000_000 {
        sum += i;
    }

    let elapsed = start.elapsed();
    println!("Sum: {}", sum);
    println!("Time: {:?}", elapsed);
}
```

### Experiment 2: Compare Speeds

```rust
use std::time::Instant;

fn main() {
    // Method 1: Create vector each time
    let start = Instant::now();
    for _ in 0..10000 {
        let v: Vec<u64> = Vec::with_capacity(100);
        // use v
    }
    println!("Method 1: {:?}", start.elapsed());

    // Method 2: Reuse one vector
    let start = Instant::now();
    let mut v: Vec<u64> = Vec::with_capacity(100);
    for _ in 0..10000 {
        v.clear();  // Clear but don't deallocate
        // use v
    }
    println!("Method 2: {:?}", start.elapsed());
}
```

---

## Questions to Think About

1. **Why can't we just make a computer 100 times faster?**
   *Hint: think about physical limitations*

2. **What happens if two threads write to the same variable at the same time?**
   *Hint: who "wins"?*

3. **Why is it important to store data close together in memory?**
   *Hint: imagine you're looking for books in a library*

---

*Now you know how the fastest trading systems in the world work!*
