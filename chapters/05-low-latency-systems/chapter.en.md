# Chapter 5: Low-Latency Systems and Practical Implementation

## Introduction

In the world of algorithmic trading, **speed is everything**. A difference of a few microseconds can mean the difference between profit and loss. In this chapter, we'll dive deep into the architecture of ultra-low-latency trading systems and learn to write code that operates at the edge of modern hardware capabilities.

**What is a low-latency system?**

Latency is the time from receiving market data to sending an order to the exchange. In modern HFT systems, this time is measured in microseconds (μs) or even nanoseconds (ns).

```
1 second      = 1,000 milliseconds (ms)
1 millisecond = 1,000 microseconds (μs)
1 microsecond = 1,000 nanoseconds (ns)
```

**Target metrics:**
- Retail trading: 100-500 ms
- Institutional: 1-10 ms
- HFT: 1-100 μs
- Ultra-HFT: <1 μs

---

## 5.1 Low-Latency Trading System Architecture

### 5.1.1 System Components

A typical trading system consists of the following components:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Trading System Architecture                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Market    │───▶│   Order     │───▶│  Strategy   │              │
│  │   Data      │    │   Book      │    │   Engine    │              │
│  │   Handler   │    │             │    │             │              │
│  └─────────────┘    └─────────────┘    └──────┬──────┘              │
│        │                                       │                      │
│        │         ┌─────────────┐              │                      │
│        │         │    Risk     │◀─────────────┤                      │
│        │         │   Manager   │              │                      │
│        │         └─────────────┘              │                      │
│        │                                       │                      │
│        ▼                                       ▼                      │
│  ┌─────────────┐                      ┌─────────────┐              │
│  │  Exchange   │◀─────────────────────│   Order     │              │
│  │  Connector  │                      │   Router    │              │
│  └─────────────┘                      └─────────────┘              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**Component descriptions:**

1. **Market Data Handler** — receives data from exchange (prices, volumes, trades)
2. **Order Book** — stores current state of the order book
3. **Strategy Engine** — makes trading decisions
4. **Risk Manager** — checks limits and risks
5. **Order Router** — routes orders to exchange
6. **Exchange Connector** — network connection to exchange

### 5.1.2 Latency Budget

Each component has its own time "budget":

| Component | Target Latency | Critical Path? |
|-----------|----------------|----------------|
| Network I/O | <10 μs | Yes |
| Market data parsing | <1 μs | Yes |
| Order book update | <100 ns | Yes |
| Strategy decision | <1 μs | Yes |
| Risk check | <100 ns | Yes |
| Order serialization | <500 ns | Yes |
| Logging | N/A | No (async) |
| Monitoring | N/A | No (async) |

**Total tick-to-trade target: <15 μs**

### 5.1.3 Design Principles

```rust
// Low-latency design principles in code

// 1. Single-threaded hot path — no locks
fn process_market_data(data: &MarketData) -> Option<Order> {
    // Entire critical path in one thread
    // No mutex, no contention
    update_order_book(data);
    let signal = calculate_signal();
    if signal.is_strong() {
        Some(create_order(signal))
    } else {
        None
    }
}

// 2. Pre-allocation — no allocations in critical path
struct PreallocatedBuffers {
    orders: Vec<Order>,      // Pre-allocated 1000 orders
    messages: Vec<Message>,  // Pre-allocated 10000 messages
    current_order: usize,
    current_message: usize,
}

impl PreallocatedBuffers {
    fn new() -> Self {
        Self {
            orders: vec![Order::default(); 1000],
            messages: vec![Message::default(); 10000],
            current_order: 0,
            current_message: 0,
        }
    }

    #[inline(always)]
    fn get_order(&mut self) -> &mut Order {
        let order = &mut self.orders[self.current_order];
        self.current_order = (self.current_order + 1) % 1000;
        order
    }
}

// 3. Cache optimization — data locality
#[repr(C, align(64))]  // Cache line alignment
struct HotData {
    best_bid: u64,
    best_ask: u64,
    mid_price: u64,
    spread: u64,
    // Everything fits in one cache line (64 bytes)
}
```

---

## 5.2 Memory Management and Cache Optimization

### 5.2.1 Why Memory Matters

Memory access is one of the main causes of latency:

```
Access Type          | Latency
---------------------|----------
L1 cache             | ~1 ns
L2 cache             | ~3 ns
L3 cache             | ~10 ns
RAM                  | ~100 ns
SSD                  | ~100 μs
Network (localhost)  | ~500 μs
```

**Conclusion:** The difference between L1 cache and RAM is 100x! That's why proper data organization is critical.

### 5.2.2 Cache-Aligned Structures

```rust
use std::sync::atomic::{AtomicU32, AtomicU64};

/// Structure aligned to cache line (64 bytes on x86)
/// This prevents "false sharing" between CPU cores
#[repr(C, align(64))]
pub struct OrderBookLevel {
    pub price: u64,          // 8 bytes - price in fixed-point format
    pub quantity: u64,       // 8 bytes - quantity
    pub order_count: u32,    // 4 bytes - number of orders at level
    pub update_seq: u32,     // 4 bytes - sequence number
    _padding: [u8; 40],      // Padding to 64 bytes
}

impl OrderBookLevel {
    pub const fn new(price: u64, quantity: u64) -> Self {
        Self {
            price,
            quantity,
            order_count: 1,
            update_seq: 0,
            _padding: [0u8; 40],
        }
    }

    /// Update level — inlined for speed
    #[inline(always)]
    pub fn update(&mut self, quantity: u64, order_count: u32) {
        self.quantity = quantity;
        self.order_count = order_count;
        self.update_seq += 1;
    }
}

/// Per-CPU-core state
/// Alignment prevents false sharing
#[repr(C, align(64))]
pub struct PerCoreState {
    pub messages_processed: AtomicU64,
    pub last_update_ns: AtomicU64,
    pub errors: AtomicU32,
    _padding: [u8; 44],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_line_size() {
        // Verify structures occupy exactly 64 bytes
        assert_eq!(std::mem::size_of::<OrderBookLevel>(), 64);
        assert_eq!(std::mem::size_of::<PerCoreState>(), 64);
    }
}
```

### 5.2.3 Object Pool — Object Reuse

```rust
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Object pool to avoid allocations in hot path
///
/// # Usage Example
/// ```
/// let pool: ObjectPool<Order, 1000> = ObjectPool::new();
///
/// // Get object from pool (no allocation!)
/// let order = pool.acquire().unwrap();
/// order.price = 100;
///
/// // Return to pool
/// pool.release(order);
/// ```
pub struct ObjectPool<T: Default, const N: usize> {
    storage: UnsafeCell<[Option<T>; N]>,
    free_indices: UnsafeCell<Vec<usize>>,
    count: AtomicUsize,
}

// Safety: we control access via atomic operations
unsafe impl<T: Default + Send, const N: usize> Send for ObjectPool<T, N> {}
unsafe impl<T: Default + Send, const N: usize> Sync for ObjectPool<T, N> {}

impl<T: Default + Clone, const N: usize> ObjectPool<T, N> {
    /// Create new pool with pre-allocated objects
    pub fn new() -> Self {
        let storage: [Option<T>; N] = std::array::from_fn(|_| Some(T::default()));
        let free_indices: Vec<usize> = (0..N).collect();

        Self {
            storage: UnsafeCell::new(storage),
            free_indices: UnsafeCell::new(free_indices),
            count: AtomicUsize::new(N),
        }
    }

    /// Acquire object from pool
    /// Returns None if pool is empty
    #[inline(always)]
    pub fn acquire(&self) -> Option<PooledObject<T, N>> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }

        // Safe since we're the only writer in this thread
        unsafe {
            let indices = &mut *self.free_indices.get();
            if let Some(idx) = indices.pop() {
                self.count.fetch_sub(1, Ordering::Relaxed);
                let storage = &mut *self.storage.get();
                let obj = storage[idx].take()?;
                return Some(PooledObject {
                    obj: Some(obj),
                    index: idx,
                    pool: self,
                });
            }
        }
        None
    }

    /// Return object to pool (called automatically via Drop)
    fn release(&self, index: usize, obj: T) {
        unsafe {
            let storage = &mut *self.storage.get();
            storage[index] = Some(obj);
            let indices = &mut *self.free_indices.get();
            indices.push(index);
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of available objects
    pub fn available(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// Pooled object — automatically returned on drop
pub struct PooledObject<'a, T: Default, const N: usize> {
    obj: Option<T>,
    index: usize,
    pool: &'a ObjectPool<T, N>,
}

impl<'a, T: Default, const N: usize> std::ops::Deref for PooledObject<'a, T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.obj.as_ref().unwrap()
    }
}

impl<'a, T: Default, const N: usize> std::ops::DerefMut for PooledObject<'a, T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.obj.as_mut().unwrap()
    }
}

impl<'a, T: Default, const N: usize> Drop for PooledObject<'a, T, N> {
    fn drop(&mut self) {
        if let Some(obj) = self.obj.take() {
            self.pool.release(self.index, obj);
        }
    }
}
```

### 5.2.4 Arena Allocator for Messages

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// Arena allocator — ultra-fast allocation for temporary data
///
/// Idea: allocate a large memory block and "slice" it as needed.
/// At end of cycle, just reset the pointer — deallocation is instant!
pub struct MessageArena {
    buffer: Box<[u8]>,
    offset: AtomicUsize,
    capacity: usize,
}

impl MessageArena {
    /// Create arena of given size
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity].into_boxed_slice(),
            offset: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Allocate memory block
    /// Returns None if no space
    #[inline(always)]
    pub fn alloc(&self, size: usize) -> Option<&mut [u8]> {
        // Align to 8 bytes for performance
        let aligned_size = (size + 7) & !7;

        let old = self.offset.fetch_add(aligned_size, Ordering::Relaxed);

        if old + aligned_size <= self.capacity {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    self.buffer.as_ptr().add(old) as *mut u8,
                    size
                ))
            }
        } else {
            // Rollback if not enough space
            self.offset.fetch_sub(aligned_size, Ordering::Relaxed);
            None
        }
    }

    /// Allocate typed object
    #[inline(always)]
    pub fn alloc_obj<T: Sized>(&self) -> Option<&mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        // Account for alignment
        let current = self.offset.load(Ordering::Relaxed);
        let aligned_start = (current + align - 1) & !(align - 1);
        let needed = aligned_start - current + size;

        let slice = self.alloc(needed)?;
        let ptr = slice.as_mut_ptr().wrapping_add(aligned_start - current);

        unsafe { Some(&mut *(ptr as *mut T)) }
    }

    /// Reset arena — instant deallocation of all memory
    #[inline(always)]
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }

    /// Bytes used
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Bytes available
    pub fn available(&self) -> usize {
        self.capacity - self.used()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let arena = MessageArena::new(1024);

        // Allocate several blocks
        let block1 = arena.alloc(100).unwrap();
        assert_eq!(block1.len(), 100);

        let block2 = arena.alloc(200).unwrap();
        assert_eq!(block2.len(), 200);

        // Reset — instant!
        arena.reset();
        assert_eq!(arena.used(), 0);
    }
}
```

---

## 5.3 Lock-Free Data Structures

### 5.3.1 Why Lock-Free?

Regular mutex/locks have serious problems for low-latency:

1. **Contention** — threads wait for each other
2. **Context switch** — switching takes ~1-10 μs
3. **Priority inversion** — low-priority thread blocks high-priority
4. **Unpredictability** — latency can vary widely

Lock-free structures use atomic operations instead of locks.

### 5.3.2 SPSC Queue (Single Producer Single Consumer)

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

/// Lock-free queue for one producer and one consumer
///
/// This is the fastest possible queue for scenarios
/// where one thread writes, another reads.
///
/// # Example
/// ```
/// let queue: SPSCQueue<MarketData, 1024> = SPSCQueue::new();
///
/// // Producer thread
/// queue.push(MarketData { price: 100 }).unwrap();
///
/// // Consumer thread
/// if let Some(data) = queue.pop() {
///     process(data);
/// }
/// ```
pub struct SPSCQueue<T, const N: usize> {
    buffer: UnsafeCell<[Option<T>; N]>,
    head: AtomicUsize,  // Write position (producer)
    tail: AtomicUsize,  // Read position (consumer)
}

// Safety: different threads work with different positions
unsafe impl<T: Send, const N: usize> Send for SPSCQueue<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for SPSCQueue<T, N> {}

impl<T, const N: usize> SPSCQueue<T, N> {
    /// Create new queue
    pub fn new() -> Self {
        Self {
            buffer: UnsafeCell::new(std::array::from_fn(|_| None)),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Add element to queue
    /// Returns Err(value) if queue is full
    #[inline(always)]
    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % N;

        // Check if full
        if next_head == self.tail.load(Ordering::Acquire) {
            return Err(value);
        }

        // Write value
        unsafe {
            (*self.buffer.get())[head] = Some(value);
        }

        // Publish new head position
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Remove element from queue
    /// Returns None if queue is empty
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);

        // Check if empty
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }

        // Read value
        let value = unsafe {
            (*self.buffer.get())[tail].take()
        };

        // Publish new tail position
        self.tail.store((tail + 1) % N, Ordering::Release);
        value
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Relaxed) == self.head.load(Ordering::Relaxed)
    }

    /// Number of elements in queue
    #[inline(always)]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        if head >= tail {
            head - tail
        } else {
            N - tail + head
        }
    }

    /// Is queue full?
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % N;
        next_head == self.tail.load(Ordering::Relaxed)
    }
}

impl<T, const N: usize> Default for SPSCQueue<T, N> {
    fn default() -> Self {
        Self::new()
    }
}
```

### 5.3.3 Lock-Free Order Book

```rust
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Price level in order book
#[derive(Debug, Clone, Default)]
pub struct PriceLevel {
    pub price: u64,       // Price in fixed-point (multiplied by 10^8)
    pub quantity: u64,    // Total volume
    pub order_count: u32, // Number of orders
}

/// High-performance Order Book
///
/// Uses BTreeMap for O(log n) operations
/// with additional optimizations for frequent operations
pub struct OrderBook {
    bids: BTreeMap<std::cmp::Reverse<u64>, PriceLevel>,  // Descending order
    asks: BTreeMap<u64, PriceLevel>,                      // Ascending order
    sequence: AtomicU64,

    // Cached values for fast access
    cached_best_bid: AtomicU64,
    cached_best_ask: AtomicU64,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            sequence: AtomicU64::new(0),
            cached_best_bid: AtomicU64::new(0),
            cached_best_ask: AtomicU64::new(u64::MAX),
        }
    }

    /// Update bid level
    #[inline(always)]
    pub fn update_bid(&mut self, price: u64, quantity: u64) {
        if quantity == 0 {
            // Remove level
            self.bids.remove(&std::cmp::Reverse(price));
        } else {
            // Add/update level
            self.bids.insert(
                std::cmp::Reverse(price),
                PriceLevel {
                    price,
                    quantity,
                    order_count: 1,
                }
            );
        }

        // Update best bid cache
        if let Some((key, _)) = self.bids.first_key_value() {
            self.cached_best_bid.store(key.0, Ordering::Relaxed);
        } else {
            self.cached_best_bid.store(0, Ordering::Relaxed);
        }

        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Update ask level
    #[inline(always)]
    pub fn update_ask(&mut self, price: u64, quantity: u64) {
        if quantity == 0 {
            self.asks.remove(&price);
        } else {
            self.asks.insert(
                price,
                PriceLevel {
                    price,
                    quantity,
                    order_count: 1,
                }
            );
        }

        // Update best ask cache
        if let Some((key, _)) = self.asks.first_key_value() {
            self.cached_best_ask.store(*key, Ordering::Relaxed);
        } else {
            self.cached_best_ask.store(u64::MAX, Ordering::Relaxed);
        }

        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Best bid (highest buy price)
    #[inline(always)]
    pub fn best_bid(&self) -> Option<(u64, u64)> {
        self.bids.first_key_value()
            .map(|(k, v)| (k.0, v.quantity))
    }

    /// Best ask (lowest sell price)
    #[inline(always)]
    pub fn best_ask(&self) -> Option<(u64, u64)> {
        self.asks.first_key_value()
            .map(|(k, v)| (*k, v.quantity))
    }

    /// Fast access to best bid via cache
    #[inline(always)]
    pub fn best_bid_cached(&self) -> u64 {
        self.cached_best_bid.load(Ordering::Relaxed)
    }

    /// Fast access to best ask via cache
    #[inline(always)]
    pub fn best_ask_cached(&self) -> u64 {
        self.cached_best_ask.load(Ordering::Relaxed)
    }

    /// Mid price — average between best bid and ask
    #[inline(always)]
    pub fn mid_price(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2),
            _ => None,
        }
    }

    /// Spread — difference between best ask and bid
    #[inline(always)]
    pub fn spread(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    /// Market depth (sum of volumes at N best levels)
    pub fn depth(&self, levels: usize) -> (u64, u64) {
        let bid_depth: u64 = self.bids.values()
            .take(levels)
            .map(|l| l.quantity)
            .sum();

        let ask_depth: u64 = self.asks.values()
            .take(levels)
            .map(|l| l.quantity)
            .sum();

        (bid_depth, ask_depth)
    }

    /// Sequence number for tracking updates
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_operations() {
        let mut book = OrderBook::new();

        // Add levels
        book.update_bid(100_00000000, 1000);  // 100.00 @ 1000
        book.update_bid(99_50000000, 2000);   // 99.50 @ 2000
        book.update_ask(100_50000000, 500);   // 100.50 @ 500
        book.update_ask(101_00000000, 1500);  // 101.00 @ 1500

        // Check best bid/ask
        assert_eq!(book.best_bid(), Some((100_00000000, 1000)));
        assert_eq!(book.best_ask(), Some((100_50000000, 500)));

        // Check spread
        assert_eq!(book.spread(), Some(50000000)); // 0.50
    }
}
```

### 5.3.4 SeqLock for Read-Heavy Data

```rust
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::cell::UnsafeCell;

/// SeqLock — optimized for frequent reads, rare writes
///
/// Perfect for data that is:
/// - Read very frequently (millions of times per second)
/// - Written rarely (single writer)
/// - Needs snapshot consistency
///
/// # How it works
/// 1. Writer increments sequence by 1 (odd = write in progress)
/// 2. Writer writes data
/// 3. Writer increments sequence by 1 (even = write complete)
/// 4. Reader checks sequence before and after reading
/// 5. If sequence changed or is odd — retry read
pub struct SeqLock<T: Copy> {
    sequence: AtomicU64,
    data: UnsafeCell<T>,
}

unsafe impl<T: Copy + Send> Send for SeqLock<T> {}
unsafe impl<T: Copy + Send> Sync for SeqLock<T> {}

impl<T: Copy> SeqLock<T> {
    pub fn new(data: T) -> Self {
        Self {
            sequence: AtomicU64::new(0),
            data: UnsafeCell::new(data),
        }
    }

    /// Write data (single writer only!)
    pub fn write(&self, value: T) {
        let seq = self.sequence.load(Ordering::Relaxed);

        // Mark start of write (odd number)
        self.sequence.store(seq + 1, Ordering::Release);

        // Write data
        unsafe { *self.data.get() = value; }

        // Mark end of write (even number)
        self.sequence.store(seq + 2, Ordering::Release);
    }

    /// Read data (lock-free, many readers allowed)
    #[inline(always)]
    pub fn read(&self) -> T {
        loop {
            // Read sequence
            let seq1 = self.sequence.load(Ordering::Acquire);

            // If odd — writer is writing, wait
            if seq1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }

            // Read data
            let value = unsafe { *self.data.get() };

            // Memory fence for correctness
            fence(Ordering::Acquire);

            // Check if sequence changed
            let seq2 = self.sequence.load(Ordering::Relaxed);

            if seq1 == seq2 {
                return value;
            }
            // Sequence changed — retry
        }
    }

    /// Try to read (returns None if writer is active)
    #[inline(always)]
    pub fn try_read(&self) -> Option<T> {
        let seq1 = self.sequence.load(Ordering::Acquire);

        if seq1 & 1 != 0 {
            return None;
        }

        let value = unsafe { *self.data.get() };
        fence(Ordering::Acquire);
        let seq2 = self.sequence.load(Ordering::Relaxed);

        if seq1 == seq2 {
            Some(value)
        } else {
            None
        }
    }
}

/// Example: shared market state
#[derive(Copy, Clone, Default)]
pub struct MarketState {
    pub best_bid: u64,
    pub best_ask: u64,
    pub last_trade_price: u64,
    pub volume_24h: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seqlock() {
        let lock = SeqLock::new(MarketState {
            best_bid: 100,
            best_ask: 101,
            last_trade_price: 100,
            volume_24h: 1000,
        });

        // Read
        let state = lock.read();
        assert_eq!(state.best_bid, 100);

        // Write
        lock.write(MarketState {
            best_bid: 102,
            best_ask: 103,
            last_trade_price: 102,
            volume_24h: 1100,
        });

        // Read updated data
        let state = lock.read();
        assert_eq!(state.best_bid, 102);
    }
}
```

---

## 5.4 Network I/O Optimization

### 5.4.1 TCP Tuning for Low Latency

```rust
use std::net::TcpStream;
use std::os::unix::io::AsRawFd;

/// TCP socket settings for low-latency
pub struct TcpTuning;

impl TcpTuning {
    /// Apply low-latency settings to socket
    pub fn apply(stream: &TcpStream) -> std::io::Result<()> {
        use socket2::{Socket, TcpKeepalive};
        use std::time::Duration;

        // Convert to socket2::Socket for advanced settings
        let socket = Socket::from(stream.try_clone()?);

        // 1. Disable Nagle's algorithm
        // Nagle buffers small packets — this increases latency!
        socket.set_nodelay(true)?;

        // 2. Increase socket buffers
        socket.set_recv_buffer_size(4 * 1024 * 1024)?;  // 4MB
        socket.set_send_buffer_size(4 * 1024 * 1024)?;  // 4MB

        // 3. Configure keepalive
        let keepalive = TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10));
        socket.set_tcp_keepalive(&keepalive)?;

        Ok(())
    }

    /// Linux-specific optimizations
    #[cfg(target_os = "linux")]
    pub fn apply_linux_optimizations(stream: &TcpStream) -> std::io::Result<()> {
        let fd = stream.as_raw_fd();

        unsafe {
            // TCP_QUICKACK — disable delayed ACKs
            let quickack: libc::c_int = 1;
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_QUICKACK,
                &quickack as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );

            // TCP_NODELAY again for reliability
            let nodelay: libc::c_int = 1;
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_NODELAY,
                &nodelay as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );

            // SO_BUSY_POLL — active polling instead of interrupts
            let busy_poll: libc::c_int = 50;  // microseconds
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_BUSY_POLL,
                &busy_poll as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }

        Ok(())
    }
}

/// Create low-latency TCP connection
pub fn create_low_latency_connection(addr: &str) -> std::io::Result<TcpStream> {
    let stream = TcpStream::connect(addr)?;

    TcpTuning::apply(&stream)?;

    #[cfg(target_os = "linux")]
    TcpTuning::apply_linux_optimizations(&stream)?;

    Ok(stream)
}
```

### 5.4.2 WebSocket Client for Crypto Exchanges

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};

/// Market data message from Binance
#[derive(Debug, Deserialize)]
pub struct BinanceDepthUpdate {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<(String, String)>,  // [price, quantity]
    #[serde(rename = "a")]
    pub asks: Vec<(String, String)>,
}

/// Trade message
#[derive(Debug, Deserialize)]
pub struct BinanceTrade {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub quantity: String,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

/// Binance channel subscription
#[derive(Serialize)]
struct SubscribeMessage {
    method: String,
    params: Vec<String>,
    id: u64,
}

/// Low-latency WebSocket client for Binance
pub struct BinanceWebSocket {
    url: String,
}

impl BinanceWebSocket {
    pub fn new() -> Self {
        Self {
            url: "wss://stream.binance.com:9443/ws".to_string(),
        }
    }

    /// Connect and process messages
    pub async fn connect_and_subscribe<F>(
        &self,
        symbols: &[&str],
        mut handler: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(MarketEvent) + Send,
    {
        // Connect
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Build subscriptions
        let mut params = Vec::new();
        for symbol in symbols {
            let s = symbol.to_lowercase();
            params.push(format!("{}@depth@100ms", s));  // Order book updates
            params.push(format!("{}@trade", s));        // Trades
        }

        let subscribe = SubscribeMessage {
            method: "SUBSCRIBE".to_string(),
            params,
            id: 1,
        };

        // Send subscription
        let msg = serde_json::to_string(&subscribe)?;
        write.send(Message::Text(msg)).await?;

        // Process messages
        while let Some(msg) = read.next().await {
            match msg? {
                Message::Text(text) => {
                    // Try to parse different message types
                    if let Ok(depth) = serde_json::from_str::<BinanceDepthUpdate>(&text) {
                        handler(MarketEvent::Depth(depth));
                    } else if let Ok(trade) = serde_json::from_str::<BinanceTrade>(&text) {
                        handler(MarketEvent::Trade(trade));
                    }
                }
                Message::Ping(data) => {
                    // Respond to ping instantly
                    write.send(Message::Pong(data)).await?;
                }
                Message::Close(_) => break,
                _ => {}
            }
        }

        Ok(())
    }
}

/// Market events
pub enum MarketEvent {
    Depth(BinanceDepthUpdate),
    Trade(BinanceTrade),
}

impl Default for BinanceWebSocket {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## 5.5 CPU Affinity and Scheduling

### 5.5.1 Thread Pinning to CPU Cores

```rust
use std::thread;

/// CPU Affinity — pin thread to specific core
///
/// Why this matters:
/// 1. Avoid thread migration between cores (expensive operation)
/// 2. Improve cache locality
/// 3. Predictable latency
pub struct CpuAffinity;

impl CpuAffinity {
    /// Pin current thread to core
    #[cfg(target_os = "linux")]
    pub fn pin_to_core(core_id: usize) -> bool {
        unsafe {
            let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut cpuset);
            libc::CPU_SET(core_id, &mut cpuset);

            let result = libc::sched_setaffinity(
                0,  // current thread
                std::mem::size_of::<libc::cpu_set_t>(),
                &cpuset,
            );

            result == 0
        }
    }

    /// Get number of CPU cores
    pub fn num_cores() -> usize {
        num_cpus::get()
    }

    /// Get number of physical cores (without hyperthreading)
    pub fn num_physical_cores() -> usize {
        num_cpus::get_physical()
    }
}

/// Thread priority settings
pub struct ThreadPriority;

impl ThreadPriority {
    /// Set real-time priority (requires root)
    #[cfg(target_os = "linux")]
    pub fn set_realtime(priority: i32) -> bool {
        unsafe {
            let param = libc::sched_param {
                sched_priority: priority,
            };

            let result = libc::sched_setscheduler(
                0,
                libc::SCHED_FIFO,
                &param,
            );

            result == 0
        }
    }

    /// Set high priority (doesn't require root)
    #[cfg(target_os = "linux")]
    pub fn set_high() -> bool {
        unsafe {
            libc::setpriority(libc::PRIO_PROCESS, 0, -20) == 0
        }
    }
}

/// Setup trading threads
pub fn setup_trading_threads() {
    // Critical path — separate isolated core
    thread::Builder::new()
        .name("strategy".to_string())
        .spawn(move || {
            // Pin to core 2 (usually isolated)
            CpuAffinity::pin_to_core(2);
            // Maximum priority
            ThreadPriority::set_realtime(99);

            // Main strategy loop
            run_strategy_loop();
        })
        .expect("Failed to spawn strategy thread");

    // Market data handler — another core
    thread::Builder::new()
        .name("market_data".to_string())
        .spawn(move || {
            CpuAffinity::pin_to_core(3);
            ThreadPriority::set_realtime(98);

            run_market_data_loop();
        })
        .expect("Failed to spawn market data thread");

    // Logging — low priority, any core
    thread::Builder::new()
        .name("logger".to_string())
        .spawn(move || {
            // Don't pin — let OS decide
            run_logging_loop();
        })
        .expect("Failed to spawn logger thread");
}

fn run_strategy_loop() {
    // TODO: implement
}

fn run_market_data_loop() {
    // TODO: implement
}

fn run_logging_loop() {
    // TODO: implement
}
```

### 5.5.2 Linux System Settings

```bash
#!/bin/bash
# System setup script for low-latency trading

# 1. CPU core isolation from scheduler
# Add to /etc/default/grub: GRUB_CMDLINE_LINUX="isolcpus=2,3,4,5"

# 2. Disable CPU power saving
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > $cpu
done

# 3. Network settings
sysctl -w net.core.rmem_max=26214400
sysctl -w net.core.wmem_max=26214400
sysctl -w net.ipv4.tcp_rmem="4096 87380 26214400"
sysctl -w net.ipv4.tcp_wmem="4096 65536 26214400"
sysctl -w net.ipv4.tcp_low_latency=1
sysctl -w net.ipv4.tcp_timestamps=0
sysctl -w net.ipv4.tcp_sack=0

# 4. Huge Pages to reduce TLB misses
echo 1024 > /proc/sys/vm/nr_hugepages

# 5. Disable swap
swapoff -a

# 6. IRQ affinity — bind NIC interrupts to core 0
# (to not disturb isolated cores)
for irq in $(cat /proc/interrupts | grep eth0 | awk '{print $1}' | tr -d ':'); do
    echo 1 > /proc/irq/$irq/smp_affinity
done

echo "Low-latency tuning applied!"
```

---

## 5.6 Message Parsing and Serialization

### 5.6.1 Zero-Copy Parsing

```rust
/// Zero-copy parsing of binary messages
///
/// Idea: instead of copying data — just "look" at it
/// through properly aligned pointer

/// Raw market data message (binary protocol)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct RawMarketDataMessage {
    pub msg_type: u8,
    pub symbol_id: u32,
    pub price: i64,       // Fixed-point, 8 decimal places
    pub quantity: u64,
    pub timestamp_ns: u64,
}

impl RawMarketDataMessage {
    /// Parse without copying — O(1)!
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() >= std::mem::size_of::<Self>() {
            // Safety: checked size, struct is packed
            Some(unsafe { &*(bytes.as_ptr() as *const Self) })
        } else {
            None
        }
    }

    /// Convert price to f64
    #[inline(always)]
    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 100_000_000.0
    }

    /// Check message type
    #[inline(always)]
    pub fn is_trade(&self) -> bool {
        self.msg_type == 1
    }

    #[inline(always)]
    pub fn is_quote(&self) -> bool {
        self.msg_type == 2
    }
}

/// Serialization without allocations
impl RawMarketDataMessage {
    #[inline(always)]
    pub fn to_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_parsing() {
        // Create message
        let msg = RawMarketDataMessage {
            msg_type: 1,
            symbol_id: 12345,
            price: 50000_00000000,  // 50000.00
            quantity: 1000,
            timestamp_ns: 1234567890,
        };

        // Serialize
        let bytes = msg.to_bytes();

        // Parse without copying
        let parsed = RawMarketDataMessage::from_bytes(bytes).unwrap();

        assert_eq!(parsed.symbol_id, 12345);
        assert_eq!(parsed.price_f64(), 50000.0);
    }
}
```

### 5.6.2 FIX Protocol Parser

```rust
/// FIX Protocol — standard for financial message exchange
///
/// Format: Tag=Value|Tag=Value|...
/// Delimiter: SOH (0x01)

/// Parsed FIX message
pub struct FixMessage<'a> {
    raw: &'a [u8],
    fields: Vec<(u32, &'a [u8])>,
}

/// Standard FIX tags
pub mod fix_tags {
    pub const MSG_TYPE: u32 = 35;
    pub const SENDER_COMP_ID: u32 = 49;
    pub const TARGET_COMP_ID: u32 = 56;
    pub const SYMBOL: u32 = 55;
    pub const SIDE: u32 = 54;
    pub const ORDER_QTY: u32 = 38;
    pub const PRICE: u32 = 44;
    pub const ORD_TYPE: u32 = 40;
    pub const CL_ORD_ID: u32 = 11;
    pub const EXEC_TYPE: u32 = 150;
    pub const ORD_STATUS: u32 = 39;
}

impl<'a> FixMessage<'a> {
    /// Parse FIX message
    #[inline]
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        let mut fields = Vec::with_capacity(32);
        let mut pos = 0;

        while pos < data.len() {
            // Find '='
            let eq_pos = find_byte(&data[pos..], b'=')?;

            // Parse tag
            let tag = parse_u32(&data[pos..pos + eq_pos])?;
            pos += eq_pos + 1;

            // Find SOH delimiter
            let soh_pos = find_byte(&data[pos..], 0x01)?;

            // Value
            let value = &data[pos..pos + soh_pos];
            pos += soh_pos + 1;

            fields.push((tag, value));
        }

        Some(Self { raw: data, fields })
    }

    /// Get field value by tag
    #[inline(always)]
    pub fn get(&self, tag: u32) -> Option<&'a [u8]> {
        self.fields.iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, v)| *v)
    }

    /// Get value as string
    #[inline(always)]
    pub fn get_str(&self, tag: u32) -> Option<&'a str> {
        self.get(tag).and_then(|v| std::str::from_utf8(v).ok())
    }

    /// Get value as number
    #[inline(always)]
    pub fn get_u64(&self, tag: u32) -> Option<u64> {
        self.get(tag).and_then(|v| parse_u64(v))
    }

    /// Get value as f64
    #[inline(always)]
    pub fn get_f64(&self, tag: u32) -> Option<f64> {
        self.get_str(tag).and_then(|s| s.parse().ok())
    }

    /// Message type
    pub fn msg_type(&self) -> Option<&'a str> {
        self.get_str(fix_tags::MSG_TYPE)
    }

    /// Symbol
    pub fn symbol(&self) -> Option<&'a str> {
        self.get_str(fix_tags::SYMBOL)
    }
}

/// Fast byte search
#[inline(always)]
fn find_byte(data: &[u8], byte: u8) -> Option<usize> {
    memchr::memchr(byte, data)
}

/// Fast u32 parsing
#[inline(always)]
fn parse_u32(bytes: &[u8]) -> Option<u32> {
    let mut result = 0u32;
    for &b in bytes {
        if b < b'0' || b > b'9' {
            return None;
        }
        result = result.wrapping_mul(10).wrapping_add((b - b'0') as u32);
    }
    Some(result)
}

/// Fast u64 parsing
#[inline(always)]
fn parse_u64(bytes: &[u8]) -> Option<u64> {
    let mut result = 0u64;
    for &b in bytes {
        if b < b'0' || b > b'9' {
            return None;
        }
        result = result.wrapping_mul(10).wrapping_add((b - b'0') as u64);
    }
    Some(result)
}

/// FIX message builder
pub struct FixMessageBuilder {
    buffer: Vec<u8>,
}

impl FixMessageBuilder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(512),
        }
    }

    /// Add field
    pub fn field(mut self, tag: u32, value: &str) -> Self {
        use std::fmt::Write;
        write!(&mut self.buffer, "{}={}\x01", tag, value).unwrap();
        self
    }

    /// Add numeric field
    pub fn field_num(mut self, tag: u32, value: u64) -> Self {
        use std::fmt::Write;
        write!(&mut self.buffer, "{}={}\x01", tag, value).unwrap();
        self
    }

    /// Add price field
    pub fn field_price(mut self, tag: u32, value: f64) -> Self {
        use std::fmt::Write;
        write!(&mut self.buffer, "{}={:.8}\x01", tag, value).unwrap();
        self
    }

    /// Build message
    pub fn build(self) -> Vec<u8> {
        self.buffer
    }
}

impl Default for FixMessageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_parsing() {
        // Example FIX message (SOH = 0x01)
        let msg = b"35=D\x0149=CLIENT\x0156=EXCHANGE\x0155=BTCUSD\x0154=1\x0138=100\x0144=50000.00\x01";

        let parsed = FixMessage::parse(msg).unwrap();

        assert_eq!(parsed.msg_type(), Some("D"));
        assert_eq!(parsed.symbol(), Some("BTCUSD"));
        assert_eq!(parsed.get_str(fix_tags::SIDE), Some("1"));
    }

    #[test]
    fn test_fix_building() {
        let msg = FixMessageBuilder::new()
            .field(fix_tags::MSG_TYPE, "D")
            .field(fix_tags::SYMBOL, "BTCUSD")
            .field_num(fix_tags::ORDER_QTY, 100)
            .field_price(fix_tags::PRICE, 50000.0)
            .build();

        let parsed = FixMessage::parse(&msg).unwrap();
        assert_eq!(parsed.symbol(), Some("BTCUSD"));
    }
}
```

---

## 5.7 Profiling and Benchmarking

### 5.7.1 Latency Measurement

```rust
use std::time::Instant;

/// Latency profiler with HDR histogram
pub struct LatencyProfiler {
    samples: Vec<u64>,
    capacity: usize,
}

impl LatencyProfiler {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Record measurement (in nanoseconds)
    #[inline(always)]
    pub fn record(&mut self, latency_ns: u64) {
        if self.samples.len() < self.capacity {
            self.samples.push(latency_ns);
        }
    }

    /// Measure function execution time
    #[inline(always)]
    pub fn measure<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_nanos() as u64;
        self.record(elapsed);
        result
    }

    /// Statistics
    pub fn stats(&mut self) -> LatencyStats {
        self.samples.sort_unstable();

        let count = self.samples.len();
        if count == 0 {
            return LatencyStats::default();
        }

        LatencyStats {
            count,
            min: self.samples[0],
            max: self.samples[count - 1],
            mean: self.samples.iter().sum::<u64>() / count as u64,
            p50: self.samples[count * 50 / 100],
            p90: self.samples[count * 90 / 100],
            p99: self.samples[count * 99 / 100],
            p999: self.samples[count * 999 / 1000],
        }
    }

    /// Clear data
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

#[derive(Debug, Default)]
pub struct LatencyStats {
    pub count: usize,
    pub min: u64,
    pub max: u64,
    pub mean: u64,
    pub p50: u64,
    pub p90: u64,
    pub p99: u64,
    pub p999: u64,
}

impl std::fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Latency Statistics ({} samples):", self.count)?;
        writeln!(f, "  Min:   {:>10} ns", self.min)?;
        writeln!(f, "  Mean:  {:>10} ns", self.mean)?;
        writeln!(f, "  p50:   {:>10} ns", self.p50)?;
        writeln!(f, "  p90:   {:>10} ns", self.p90)?;
        writeln!(f, "  p99:   {:>10} ns", self.p99)?;
        writeln!(f, "  p99.9: {:>10} ns", self.p999)?;
        writeln!(f, "  Max:   {:>10} ns", self.max)
    }
}

/// Macro for convenient measurement
#[macro_export]
macro_rules! measure_latency {
    ($profiler:expr, $code:expr) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let elapsed = start.elapsed().as_nanos() as u64;
        $profiler.record(elapsed);
        result
    }};
}
```

### 5.7.2 CPU Cycle Counting

```rust
/// CPU cycle counter for ultra-precise measurements
pub struct CpuCycleCounter;

impl CpuCycleCounter {
    /// Read TSC (Time Stamp Counter)
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn rdtsc() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    /// RDTSCP — more precise version with barrier
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn rdtscp() -> u64 {
        let mut aux: u32 = 0;
        unsafe { core::arch::x86_64::__rdtscp(&mut aux) }
    }

    /// Convert cycles to nanoseconds
    /// cpu_freq_ghz — CPU frequency in GHz
    #[inline(always)]
    pub fn cycles_to_ns(cycles: u64, cpu_freq_ghz: f64) -> f64 {
        cycles as f64 / cpu_freq_ghz
    }

    /// Measure cycles for function execution
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn measure_cycles<F, R>(f: F) -> (R, u64)
    where
        F: FnOnce() -> R,
    {
        let start = Self::rdtscp();
        let result = f();
        let end = Self::rdtscp();
        (result, end - start)
    }
}

/// Detect CPU frequency
#[cfg(target_os = "linux")]
pub fn get_cpu_freq_ghz() -> Option<f64> {
    use std::fs;

    // Read from /proc/cpuinfo
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;

    for line in cpuinfo.lines() {
        if line.starts_with("cpu MHz") {
            let mhz: f64 = line
                .split(':')
                .nth(1)?
                .trim()
                .parse()
                .ok()?;
            return Some(mhz / 1000.0);
        }
    }
    None
}
```

---

## 5.8 Production Architecture

### 5.8.1 Error Handling

```rust
use std::fmt;

/// Trading system error types
#[derive(Debug)]
pub enum TradingError {
    /// Network error
    Network(std::io::Error),
    /// Parse error
    Parse(String),
    /// Risk limit exceeded
    RiskLimit { limit: f64, attempted: f64 },
    /// Order rejected
    OrderRejected { reason: String },
    /// System overload
    SystemOverload,
    /// Disconnected from exchange
    Disconnected,
    /// Timeout
    Timeout,
}

impl fmt::Display for TradingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradingError::Network(e) => write!(f, "Network error: {}", e),
            TradingError::Parse(s) => write!(f, "Parse error: {}", s),
            TradingError::RiskLimit { limit, attempted } => {
                write!(f, "Risk limit exceeded: {} > {}", attempted, limit)
            }
            TradingError::OrderRejected { reason } => {
                write!(f, "Order rejected: {}", reason)
            }
            TradingError::SystemOverload => write!(f, "System overload"),
            TradingError::Disconnected => write!(f, "Disconnected from exchange"),
            TradingError::Timeout => write!(f, "Operation timed out"),
        }
    }
}

impl std::error::Error for TradingError {}

/// Result type for trading system
pub type TradingResult<T> = Result<T, TradingError>;

/// Fast error codes for hot path (no allocations)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastError {
    Ok = 0,
    NetworkTimeout = 1,
    ParseFailed = 2,
    RiskLimit = 3,
    QueueFull = 4,
    InvalidOrder = 5,
}

impl FastError {
    #[inline(always)]
    pub fn is_ok(self) -> bool {
        self == FastError::Ok
    }

    #[inline(always)]
    pub fn is_err(self) -> bool {
        self != FastError::Ok
    }
}
```

### 5.8.2 Circuit Breaker

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Circuit Breaker — protection from cascading failures
///
/// States:
/// - Closed: normal operation
/// - Open: block requests (after N errors)
/// - HalfOpen: try to recover (after timeout)
pub struct CircuitBreaker {
    failure_count: AtomicU32,
    last_failure_ms: AtomicU64,
    state: AtomicU8,  // 0=Closed, 1=Open, 2=HalfOpen

    failure_threshold: u32,
    recovery_time_ms: u64,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_time_ms: u64) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            last_failure_ms: AtomicU64::new(0),
            state: AtomicU8::new(0),
            failure_threshold,
            recovery_time_ms,
        }
    }

    /// Check if execution is allowed
    #[inline(always)]
    pub fn allow(&self) -> bool {
        match self.state.load(Ordering::Relaxed) {
            0 => true,  // Closed — allowed
            1 => {      // Open — check timeout
                let now = current_time_ms();
                let last = self.last_failure_ms.load(Ordering::Relaxed);

                if now - last > self.recovery_time_ms {
                    // Transition to HalfOpen
                    self.state.store(2, Ordering::Release);
                    true
                } else {
                    false
                }
            }
            2 => true,  // HalfOpen — allow one attempt
            _ => false,
        }
    }

    /// Record successful execution
    #[inline(always)]
    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.state.store(0, Ordering::Release);  // -> Closed
    }

    /// Record failure
    #[inline(always)]
    pub fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        if count >= self.failure_threshold {
            self.state.store(1, Ordering::Release);  // -> Open
            self.last_failure_ms.store(current_time_ms(), Ordering::Relaxed);
        }
    }

    /// Current state
    pub fn state(&self) -> CircuitBreakerState {
        match self.state.load(Ordering::Relaxed) {
            0 => CircuitBreakerState::Closed,
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
```

### 5.8.3 Graceful Shutdown

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Graceful shutdown management
pub struct ShutdownController {
    shutdown_requested: Arc<AtomicBool>,
}

impl ShutdownController {
    pub fn new() -> Self {
        Self {
            shutdown_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get token for checking shutdown
    pub fn token(&self) -> ShutdownToken {
        ShutdownToken {
            shutdown_requested: Arc::clone(&self.shutdown_requested),
        }
    }

    /// Initiate shutdown
    pub fn shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
    }

    /// Install Ctrl+C handler
    pub fn install_signal_handler(&self) {
        let shutdown = Arc::clone(&self.shutdown_requested);

        ctrlc::set_handler(move || {
            println!("\nShutdown signal received...");
            shutdown.store(true, Ordering::Release);
        }).expect("Failed to set Ctrl+C handler");
    }
}

impl Default for ShutdownController {
    fn default() -> Self {
        Self::new()
    }
}

/// Token for checking shutdown request
#[derive(Clone)]
pub struct ShutdownToken {
    shutdown_requested: Arc<AtomicBool>,
}

impl ShutdownToken {
    /// Check if shutdown is requested
    #[inline(always)]
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Acquire)
    }
}

/// Example usage in main loop
pub fn run_trading_loop(token: ShutdownToken) {
    println!("Trading loop started. Press Ctrl+C to stop.");

    while !token.is_shutdown_requested() {
        // Process market data
        // Execute strategy
        // Send orders

        std::thread::sleep(std::time::Duration::from_micros(100));
    }

    println!("Trading loop stopped gracefully.");
}
```

---

## 5.9 Complete Example: Trading System

```rust
//! Complete example of a low-latency trading system

use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Trading system configuration
pub struct TradingConfig {
    pub symbol: String,
    pub max_position: f64,
    pub risk_limit: f64,
    pub strategy_core: usize,
    pub market_data_core: usize,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            max_position: 1.0,
            risk_limit: 1000.0,
            strategy_core: 2,
            market_data_core: 3,
        }
    }
}

/// Trading system
pub struct TradingSystem {
    config: TradingConfig,
    order_book: Arc<std::sync::RwLock<OrderBook>>,
    shutdown: ShutdownController,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl TradingSystem {
    pub fn new(config: TradingConfig) -> Self {
        Self {
            config,
            order_book: Arc::new(std::sync::RwLock::new(OrderBook::new())),
            shutdown: ShutdownController::new(),
            circuit_breaker: Arc::new(CircuitBreaker::new(5, 5000)),
        }
    }

    /// Start system
    pub fn run(&self) {
        // Install signal handlers
        self.shutdown.install_signal_handler();

        let token = self.shutdown.token();
        let order_book = Arc::clone(&self.order_book);
        let circuit_breaker = Arc::clone(&self.circuit_breaker);
        let strategy_core = self.config.strategy_core;

        // Strategy thread
        let strategy_handle = thread::Builder::new()
            .name("strategy".to_string())
            .spawn(move || {
                #[cfg(target_os = "linux")]
                CpuAffinity::pin_to_core(strategy_core);

                Self::strategy_loop(token, order_book, circuit_breaker);
            })
            .expect("Failed to spawn strategy thread");

        // Wait for completion
        strategy_handle.join().expect("Strategy thread panicked");

        println!("Trading system shut down.");
    }

    fn strategy_loop(
        token: ShutdownToken,
        order_book: Arc<std::sync::RwLock<OrderBook>>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) {
        let mut profiler = LatencyProfiler::new(10000);
        let mut iteration = 0u64;

        while !token.is_shutdown_requested() {
            // Check circuit breaker
            if !circuit_breaker.allow() {
                thread::sleep(Duration::from_millis(100));
                continue;
            }

            // Measure latency of one iteration
            profiler.measure(|| {
                // 1. Read order book
                let book = order_book.read().unwrap();

                // 2. Calculate signal
                if let (Some((bid, _)), Some((ask, _))) = (book.best_bid(), book.best_ask()) {
                    let mid = (bid + ask) / 2;
                    let spread = ask - bid;

                    // Simple strategy: if spread is large — opportunity exists
                    if spread > 100_000 {  // > 0.001
                        // Order would be sent here
                        let _order_price = mid;
                    }
                }
            });

            iteration += 1;

            // Every 10000 iterations output statistics
            if iteration % 10000 == 0 {
                let stats = profiler.stats();
                println!("Iteration {}: p50={} ns, p99={} ns",
                    iteration, stats.p50, stats.p99);
                profiler.reset();
            }

            // Small pause to not overload CPU
            std::hint::spin_loop();
        }
    }
}

fn main() {
    println!("Starting Low-Latency Trading System...\n");

    let config = TradingConfig::default();
    println!("Symbol: {}", config.symbol);
    println!("Max Position: {}", config.max_position);
    println!("Risk Limit: {}", config.risk_limit);
    println!();

    let system = TradingSystem::new(config);
    system.run();
}
```

---

## 5.10 Practical Exercises

### Exercise 5.1: SPSC Queue Implementation

**Goal:** Implement a lock-free SPSC queue and compare with `crossbeam-channel`.

**Requirements:**
- Generic type support
- Batch operations (push_batch, pop_batch)
- Benchmark: target <50 ns per operation

```rust
// Template for implementation
pub struct BatchSPSCQueue<T, const N: usize> {
    // TODO: implement
}

impl<T, const N: usize> BatchSPSCQueue<T, N> {
    pub fn push_batch(&self, items: &[T]) -> usize {
        // TODO: return number of successfully added items
        todo!()
    }

    pub fn pop_batch(&self, buffer: &mut [T]) -> usize {
        // TODO: return number of extracted items
        todo!()
    }
}
```

### Exercise 5.2: High-Performance Order Book

**Goal:** Build an order book supporting 1M+ orders.

**Requirements:**
- O(1) for best bid/ask
- O(log n) for add/remove
- Support L2 and L3 data
- Target: <100 ns per operation

### Exercise 5.3: Market Data Handler

**Goal:** Connect to Binance WebSocket and update order book in real-time.

**Requirements:**
- Connect to testnet
- Parse depth and trade messages
- Measure tick-to-update latency
- Handle reconnect

### Exercise 5.4: Complete Trading System

**Goal:** Integrate all components into a working system.

**Requirements:**
- Market data → Order book → Strategy → Order management
- End-to-end latency profiling
- Paper trading on testnet
- Graceful shutdown

### Exercise 5.5: Production Hardening

**Goal:** Prepare system for production.

**Requirements:**
- Comprehensive error handling
- Monitoring with Prometheus metrics
- Alerting on critical events
- Configuration management (TOML/YAML)

---

## Conclusion

In this chapter, we learned:

1. **Architecture** of low-latency trading systems
2. **Memory management** — cache alignment, object pools, arena allocators
3. **Lock-free structures** — SPSC queues, SeqLock
4. **Network optimization** — TCP tuning, WebSocket
5. **CPU optimization** — affinity, scheduling, NUMA
6. **Profiling** — latency measurement, benchmarking
7. **Production patterns** — error handling, circuit breaker, graceful shutdown

**Key principles:**
- Measure before optimizing
- Avoid allocations in hot path
- Use lock-free where possible
- Pin critical threads to cores
- Always have a graceful degradation plan

---

## Recommended Reading

### Books
1. "Systems Performance" — Brendan Gregg
2. "The Art of Multiprocessor Programming" — Herlihy, Shavit
3. "Algorithmic and High-Frequency Trading" — Cartea, Jaimungal, Penalva

### Online Resources
1. LMAX Disruptor pattern
2. Aeron messaging documentation
3. Jane Street tech blog
4. Two Sigma engineering blog

---

*Next chapter: Information Theory and Kelly Criterion*
