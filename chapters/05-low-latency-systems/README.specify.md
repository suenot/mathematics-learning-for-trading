# Глава 5: Системы низкой задержки и практическая реализация

## Метаданные
- **Уровень сложности**: Продвинутый (системное программирование)
- **Предварительные требования**: Rust (intermediate+), OS concepts, networking
- **Языки реализации**: Rust (основной), C/C++ (для сравнения), Go (networking)
- **Расчётный объём**: 100-130 страниц

---

## Цели главы

1. Понять архитектуру ultra-low-latency trading систем
2. Освоить техники оптимизации на уровне CPU, памяти и сети
3. Реализовать lock-free структуры данных для trading
4. Построить production-grade order management system
5. Изучить profiling и benchmarking для latency-critical систем

---

## Научная база и индустриальные практики

### Фундаментальные ресурсы
1. **"Algorithmic and High-Frequency Trading"** — Cartea, Jaimungal, Penalva (2015)
2. **Intel Optimization Manual** — низкоуровневая оптимизация
3. **Linux kernel documentation** — networking, scheduling

### Современные практики (2024-2025)
4. **"High-Frequency Trading Systems in Rust"** — Crossley (2025)
5. **Digital One Agency** — Rust + FIX API production systems
6. **Luca Sbardella** — "Rust for HFT" blog series (2025)
7. **TickGrinder** — open-source Rust trading platform
8. **Hummingbot** — market making framework architecture
9. GitHub: `high-frequency-trading` topic в Rust

### Networking & Systems
10. **DPDK** (Data Plane Development Kit) — kernel bypass
11. **io_uring** — async I/O в Linux
12. **Aeron** — high-performance messaging

---

## Структура главы

### 5.1 Архитектура Low-Latency Trading System

**5.1.1 Компоненты системы**

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Trading System Architecture                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Market    │───>│   Order     │───>│  Strategy   │              │
│  │   Data      │    │   Book      │    │   Engine    │              │
│  │   Handler   │    │             │    │             │              │
│  └─────────────┘    └─────────────┘    └──────┬──────┘              │
│        │                                       │                      │
│        │         ┌─────────────┐              │                      │
│        │         │    Risk     │<─────────────┤                      │
│        │         │   Manager   │              │                      │
│        │         └─────────────┘              │                      │
│        │                                       │                      │
│        v                                       v                      │
│  ┌─────────────┐                      ┌─────────────┐              │
│  │  Exchange   │<─────────────────────│   Order     │              │
│  │  Connector  │                      │   Router    │              │
│  └─────────────┘                      └─────────────┘              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**5.1.2 Latency Budget**

| Component | Target Latency | Critical Path? |
|-----------|---------------|----------------|
| Network I/O | <10 μs | Yes |
| Market data parsing | <1 μs | Yes |
| Order book update | <100 ns | Yes |
| Strategy decision | <1 μs | Yes |
| Risk check | <100 ns | Yes |
| Order serialization | <500 ns | Yes |
| Logging | N/A | No (async) |
| Monitoring | N/A | No (async) |

**Total tick-to-trade target: <15 μs**

**5.1.3 Design Principles**

1. **Single-threaded hot path** — no locks, no contention
2. **Pre-allocation** — no allocations in critical path
3. **Cache optimization** — data locality
4. **Branch prediction friendly** — predictable control flow
5. **Async off critical path** — logging, monitoring separate

---

### 5.2 Memory Management и Cache Optimization

**5.2.1 Memory Layout**

```rust
// Cache-line aligned structure (64 bytes on x86)
#[repr(C, align(64))]
pub struct OrderBookLevel {
    pub price: u64,          // 8 bytes - fixed-point
    pub quantity: u64,       // 8 bytes
    pub order_count: u32,    // 4 bytes
    pub update_seq: u32,     // 4 bytes
    _padding: [u8; 40],      // Pad to 64 bytes
}

// Ensure no false sharing between threads
#[repr(C, align(64))]
pub struct PerCoreState {
    pub messages_processed: u64,
    pub last_update_ns: u64,
    _padding: [u8; 48],
}
```

**5.2.2 Object Pools**

```rust
use std::cell::UnsafeCell;

pub struct ObjectPool<T, const N: usize> {
    storage: UnsafeCell<[Option<T>; N]>,
    free_list: AtomicPtr<Node>,
}

impl<T: Default, const N: usize> ObjectPool<T, N> {
    pub fn new() -> Self {
        let mut storage = [(); N].map(|_| None);
        // Pre-allocate all objects
        for slot in storage.iter_mut() {
            *slot = Some(T::default());
        }
        // Initialize free list
        // ...
    }
    
    #[inline(always)]
    pub fn acquire(&self) -> Option<PooledObject<T>> {
        // Lock-free acquire from free list
        // ...
    }
    
    #[inline(always)]
    pub fn release(&self, obj: PooledObject<T>) {
        // Return to free list
        // ...
    }
}
```

**5.2.3 Arena Allocator для Messages**

```rust
pub struct MessageArena {
    buffer: Box<[u8]>,
    offset: AtomicUsize,
    capacity: usize,
}

impl MessageArena {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity].into_boxed_slice(),
            offset: AtomicUsize::new(0),
            capacity,
        }
    }
    
    #[inline(always)]
    pub fn alloc(&self, size: usize) -> Option<&mut [u8]> {
        let aligned_size = (size + 7) & !7;  // 8-byte alignment
        let old = self.offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if old + aligned_size <= self.capacity {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    self.buffer.as_ptr().add(old) as *mut u8,
                    size
                ))
            }
        } else {
            None
        }
    }
    
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }
}
```

---

### 5.3 Lock-Free Data Structures

**5.3.1 SPSC Queue (Single Producer Single Consumer)**

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

pub struct SPSCQueue<T, const N: usize> {
    buffer: UnsafeCell<[Option<T>; N]>,
    head: AtomicUsize,  // Write position (producer)
    tail: AtomicUsize,  // Read position (consumer)
}

unsafe impl<T: Send, const N: usize> Send for SPSCQueue<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for SPSCQueue<T, N> {}

impl<T, const N: usize> SPSCQueue<T, N> {
    pub const fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([(); N].map(|_| None)),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
    
    #[inline(always)]
    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % N;
        
        // Check if full
        if next_head == self.tail.load(Ordering::Acquire) {
            return Err(value);
        }
        
        unsafe {
            (*self.buffer.get())[head] = Some(value);
        }
        
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }
    
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        
        // Check if empty
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        
        let value = unsafe {
            (*self.buffer.get())[tail].take()
        };
        
        self.tail.store((tail + 1) % N, Ordering::Release);
        value
    }
    
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Relaxed) == self.head.load(Ordering::Relaxed)
    }
}
```

**5.3.2 Lock-Free Order Book**

```rust
use crossbeam_skiplist::SkipMap;

pub struct LockFreeOrderBook {
    bids: SkipMap<Reverse<u64>, OrderBookLevel>,  // Descending
    asks: SkipMap<u64, OrderBookLevel>,           // Ascending
}

impl LockFreeOrderBook {
    pub fn new() -> Self {
        Self {
            bids: SkipMap::new(),
            asks: SkipMap::new(),
        }
    }
    
    #[inline(always)]
    pub fn update_bid(&self, price: u64, quantity: u64) {
        if quantity == 0 {
            self.bids.remove(&Reverse(price));
        } else {
            self.bids.insert(
                Reverse(price),
                OrderBookLevel { price, quantity, ..Default::default() }
            );
        }
    }
    
    #[inline(always)]
    pub fn best_bid(&self) -> Option<(u64, u64)> {
        self.bids.front().map(|e| (e.key().0, e.value().quantity))
    }
    
    #[inline(always)]
    pub fn best_ask(&self) -> Option<(u64, u64)> {
        self.asks.front().map(|e| (*e.key(), e.value().quantity))
    }
    
    #[inline(always)]
    pub fn mid_price(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2),
            _ => None,
        }
    }
}
```

**5.3.3 Seqlock для Read-Heavy Data**

```rust
use std::sync::atomic::{AtomicU64, Ordering};

pub struct SeqLock<T> {
    sequence: AtomicU64,
    data: UnsafeCell<T>,
}

impl<T: Copy> SeqLock<T> {
    pub fn new(data: T) -> Self {
        Self {
            sequence: AtomicU64::new(0),
            data: UnsafeCell::new(data),
        }
    }
    
    /// Writer: exclusive access required
    pub fn write(&self, value: T) {
        let seq = self.sequence.load(Ordering::Relaxed);
        self.sequence.store(seq + 1, Ordering::Release);  // Odd = writing
        
        unsafe { *self.data.get() = value; }
        
        self.sequence.store(seq + 2, Ordering::Release);  // Even = done
    }
    
    /// Reader: lock-free, retry on conflict
    #[inline(always)]
    pub fn read(&self) -> T {
        loop {
            let seq1 = self.sequence.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                // Writer in progress, spin
                std::hint::spin_loop();
                continue;
            }
            
            let value = unsafe { *self.data.get() };
            
            std::sync::atomic::fence(Ordering::Acquire);
            let seq2 = self.sequence.load(Ordering::Relaxed);
            
            if seq1 == seq2 {
                return value;
            }
            // Sequence changed, retry
        }
    }
}
```

---

### 5.4 Network I/O Optimization

**5.4.1 TCP Tuning для Low Latency**

```rust
use socket2::{Socket, Domain, Type, Protocol};
use std::net::TcpStream;

pub fn create_low_latency_socket(addr: &str) -> std::io::Result<TcpStream> {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP))?;
    
    // Disable Nagle's algorithm
    socket.set_nodelay(true)?;
    
    // Set socket buffer sizes
    socket.set_recv_buffer_size(4 * 1024 * 1024)?;  // 4MB
    socket.set_send_buffer_size(4 * 1024 * 1024)?;
    
    // Enable TCP quickack (Linux)
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::IPPROTO_TCP,
                libc::TCP_QUICKACK,
                &1i32 as *const _ as *const libc::c_void,
                std::mem::size_of::<i32>() as libc::socklen_t,
            );
        }
    }
    
    // Connect
    socket.connect(&addr.parse::<std::net::SocketAddr>()?.into())?;
    
    Ok(socket.into())
}
```

**5.4.2 io_uring для Async I/O**

```rust
use io_uring::{opcode, types, IoUring};

pub struct UringNetworkHandler {
    ring: IoUring,
    buffers: Vec<Vec<u8>>,
}

impl UringNetworkHandler {
    pub fn new(queue_depth: u32) -> std::io::Result<Self> {
        let ring = IoUring::builder()
            .setup_sqpoll(1000)  // Kernel polling thread
            .build(queue_depth)?;
        
        let buffers = (0..queue_depth)
            .map(|_| vec![0u8; 4096])
            .collect();
        
        Ok(Self { ring, buffers })
    }
    
    pub fn submit_read(&mut self, fd: i32, buffer_idx: usize) -> std::io::Result<()> {
        let read_op = opcode::Read::new(
            types::Fd(fd),
            self.buffers[buffer_idx].as_mut_ptr(),
            self.buffers[buffer_idx].len() as u32,
        )
        .build()
        .user_data(buffer_idx as u64);
        
        unsafe {
            self.ring.submission().push(&read_op)?;
        }
        self.ring.submit()?;
        
        Ok(())
    }
    
    pub fn poll_completions(&mut self) -> impl Iterator<Item = (usize, i32)> + '_ {
        self.ring.completion().map(|cqe| {
            (cqe.user_data() as usize, cqe.result())
        })
    }
}
```

**5.4.3 WebSocket Client для Crypto Exchanges**

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};

pub struct ExchangeConnector {
    url: String,
    message_handler: Box<dyn Fn(&[u8]) + Send>,
}

impl ExchangeConnector {
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();
        
        // Subscribe to channels
        let subscribe_msg = serde_json::json!({
            "method": "SUBSCRIBE",
            "params": ["btcusdt@depth@100ms", "btcusdt@trade"],
            "id": 1
        });
        write.send(Message::Text(subscribe_msg.to_string())).await?;
        
        // Process messages
        while let Some(msg) = read.next().await {
            match msg? {
                Message::Binary(data) => (self.message_handler)(&data),
                Message::Text(text) => (self.message_handler)(text.as_bytes()),
                Message::Ping(data) => write.send(Message::Pong(data)).await?,
                _ => {}
            }
        }
        
        Ok(())
    }
}
```

---

### 5.5 CPU Affinity и Scheduling

**5.5.1 CPU Pinning**

```rust
use core_affinity::CoreId;

pub fn pin_to_core(core_id: usize) -> bool {
    let core_ids = core_affinity::get_core_ids().unwrap();
    if core_id < core_ids.len() {
        core_affinity::set_for_current(core_ids[core_id])
    } else {
        false
    }
}

pub fn setup_trading_threads() {
    // Pin critical path to isolated core
    std::thread::Builder::new()
        .name("strategy".to_string())
        .spawn(move || {
            pin_to_core(2);  // Isolated core
            set_realtime_priority();
            run_strategy_loop();
        })
        .unwrap();
    
    // Pin market data handler to another core
    std::thread::Builder::new()
        .name("market_data".to_string())
        .spawn(move || {
            pin_to_core(3);
            set_realtime_priority();
            run_market_data_loop();
        })
        .unwrap();
}
```

**5.5.2 Real-Time Priority (Linux)**

```rust
#[cfg(target_os = "linux")]
pub fn set_realtime_priority() {
    use libc::{sched_param, sched_setscheduler, SCHED_FIFO};
    
    let param = sched_param { sched_priority: 99 };
    unsafe {
        sched_setscheduler(0, SCHED_FIFO, &param);
    }
}

#[cfg(target_os = "linux")]
pub fn disable_cpu_frequency_scaling() {
    // Set performance governor
    std::fs::write(
        "/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor",
        "performance"
    ).ok();
}
```

**5.5.3 NUMA Awareness**

```rust
#[cfg(target_os = "linux")]
pub fn allocate_on_node(node: i32, size: usize) -> *mut u8 {
    use libc::{numa_alloc_onnode, numa_available};
    
    unsafe {
        if numa_available() < 0 {
            return std::alloc::alloc(
                std::alloc::Layout::from_size_align(size, 64).unwrap()
            );
        }
        
        numa_alloc_onnode(size, node) as *mut u8
    }
}
```

---

### 5.6 Message Parsing и Serialization

**5.6.1 Zero-Copy Parsing**

```rust
use zerocopy::{AsBytes, FromBytes, FromZeroes};

#[derive(AsBytes, FromBytes, FromZeroes, Clone, Copy)]
#[repr(C, packed)]
pub struct RawMarketDataMessage {
    pub msg_type: u8,
    pub symbol_id: u32,
    pub price: i64,      // Fixed-point
    pub quantity: u64,
    pub timestamp_ns: u64,
}

impl RawMarketDataMessage {
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() >= std::mem::size_of::<Self>() {
            Some(unsafe { &*(bytes.as_ptr() as *const Self) })
        } else {
            None
        }
    }
    
    #[inline(always)]
    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 100_000_000.0  // 8 decimal places
    }
}
```

**5.6.2 FIX Protocol Parser**

```rust
pub struct FixMessage<'a> {
    raw: &'a [u8],
    fields: Vec<(u32, &'a [u8])>,  // Tag, Value
}

impl<'a> FixMessage<'a> {
    #[inline(always)]
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        let mut fields = Vec::with_capacity(32);
        let mut pos = 0;
        
        while pos < data.len() {
            // Parse tag
            let tag_end = memchr::memchr(b'=', &data[pos..])?;
            let tag: u32 = parse_int(&data[pos..pos+tag_end])?;
            pos += tag_end + 1;
            
            // Parse value
            let value_end = memchr::memchr(0x01, &data[pos..])?;  // SOH delimiter
            let value = &data[pos..pos+value_end];
            pos += value_end + 1;
            
            fields.push((tag, value));
        }
        
        Some(Self { raw: data, fields })
    }
    
    #[inline(always)]
    pub fn get(&self, tag: u32) -> Option<&'a [u8]> {
        self.fields.iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, v)| *v)
    }
}

#[inline(always)]
fn parse_int(bytes: &[u8]) -> Option<u32> {
    let mut result = 0u32;
    for &b in bytes {
        if b < b'0' || b > b'9' { return None; }
        result = result * 10 + (b - b'0') as u32;
    }
    Some(result)
}
```

**5.6.3 JSON Parsing for REST APIs**

```rust
use simd_json::prelude::*;

pub fn parse_binance_depth(data: &mut [u8]) -> Result<OrderBookSnapshot, simd_json::Error> {
    let value: simd_json::BorrowedValue = simd_json::to_borrowed_value(data)?;
    
    let bids = value["bids"].as_array().unwrap();
    let asks = value["asks"].as_array().unwrap();
    
    // Parse with SIMD acceleration
    // ...
    
    Ok(OrderBookSnapshot { /* ... */ })
}
```

---

### 5.7 Profiling и Benchmarking

**5.7.1 Latency Measurement**

```rust
use std::time::Instant;
use hdrhistogram::Histogram;

pub struct LatencyProfiler {
    histogram: Histogram<u64>,
    start_times: Vec<Instant>,
}

impl LatencyProfiler {
    pub fn new() -> Self {
        Self {
            histogram: Histogram::new_with_bounds(1, 1_000_000_000, 3).unwrap(),
            start_times: Vec::with_capacity(10000),
        }
    }
    
    #[inline(always)]
    pub fn start(&mut self) -> usize {
        let id = self.start_times.len();
        self.start_times.push(Instant::now());
        id
    }
    
    #[inline(always)]
    pub fn stop(&mut self, id: usize) {
        let elapsed = self.start_times[id].elapsed().as_nanos() as u64;
        self.histogram.record(elapsed).ok();
    }
    
    pub fn report(&self) {
        println!("Latency Statistics:");
        println!("  Min:    {:>10} ns", self.histogram.min());
        println!("  p50:    {:>10} ns", self.histogram.value_at_quantile(0.50));
        println!("  p99:    {:>10} ns", self.histogram.value_at_quantile(0.99));
        println!("  p99.9:  {:>10} ns", self.histogram.value_at_quantile(0.999));
        println!("  Max:    {:>10} ns", self.histogram.max());
    }
}
```

**5.7.2 CPU Cycle Counting**

```rust
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn rdtsc() -> u64 {
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn rdtscp() -> (u64, u32) {
    let mut aux: u32 = 0;
    let tsc = unsafe {
        core::arch::x86_64::__rdtscp(&mut aux)
    };
    (tsc, aux)
}

pub fn cycles_to_ns(cycles: u64, cpu_freq_ghz: f64) -> f64 {
    cycles as f64 / cpu_freq_ghz
}
```

**5.7.3 Criterion Benchmarks**

```rust
use criterion::{criterion_group, criterion_main, Criterion, black_box};

fn benchmark_order_book(c: &mut Criterion) {
    let mut book = OrderBook::new();
    
    c.bench_function("order_book_update", |b| {
        b.iter(|| {
            black_box(book.update_bid(100_00000000, 1000));
        })
    });
    
    c.bench_function("order_book_best_bid", |b| {
        b.iter(|| {
            black_box(book.best_bid())
        })
    });
}

fn benchmark_message_parsing(c: &mut Criterion) {
    let raw_message = include_bytes!("../testdata/sample_message.bin");
    
    c.bench_function("fix_parse", |b| {
        b.iter(|| {
            black_box(FixMessage::parse(raw_message))
        })
    });
}

criterion_group!(benches, benchmark_order_book, benchmark_message_parsing);
criterion_main!(benches);
```

**5.7.4 perf и flamegraph**

```bash
# Record performance data
perf record -g -F 99 ./trading_system

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg

# Or use cargo-flamegraph
cargo flamegraph --bin trading_system
```

---

### 5.8 Production Architecture

**5.8.1 Error Handling**

```rust
#[derive(Debug)]
pub enum TradingError {
    NetworkError(std::io::Error),
    ParseError(String),
    RiskLimitExceeded { limit: f64, attempted: f64 },
    OrderRejected { reason: String },
    SystemOverload,
}

// Use Result everywhere, never panic in production
pub type TradingResult<T> = Result<T, TradingError>;

// Fast path: use error codes instead of full errors
#[repr(u8)]
pub enum FastError {
    Ok = 0,
    NetworkTimeout = 1,
    ParseFailed = 2,
    RiskLimit = 3,
    QueueFull = 4,
}
```

**5.8.2 Graceful Degradation**

```rust
pub struct CircuitBreaker {
    failure_count: AtomicU32,
    last_failure: AtomicU64,
    state: AtomicU8,  // 0=Closed, 1=Open, 2=HalfOpen
}

impl CircuitBreaker {
    const FAILURE_THRESHOLD: u32 = 5;
    const RECOVERY_TIME_MS: u64 = 5000;
    
    pub fn allow(&self) -> bool {
        match self.state.load(Ordering::Relaxed) {
            0 => true,  // Closed
            1 => {      // Open
                let now = current_time_ms();
                if now - self.last_failure.load(Ordering::Relaxed) > Self::RECOVERY_TIME_MS {
                    self.state.store(2, Ordering::Release);  // HalfOpen
                    true
                } else {
                    false
                }
            }
            2 => true,  // HalfOpen - allow one attempt
            _ => false,
        }
    }
    
    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.state.store(0, Ordering::Release);
    }
    
    pub fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed);
        if count >= Self::FAILURE_THRESHOLD {
            self.state.store(1, Ordering::Release);
            self.last_failure.store(current_time_ms(), Ordering::Relaxed);
        }
    }
}
```

**5.8.3 Monitoring и Metrics**

```rust
use prometheus::{Counter, Histogram, Registry};

pub struct TradingMetrics {
    pub orders_sent: Counter,
    pub orders_filled: Counter,
    pub tick_to_trade_latency: Histogram,
    pub pnl: prometheus::Gauge,
}

impl TradingMetrics {
    pub fn new(registry: &Registry) -> Self {
        let orders_sent = Counter::new("orders_sent_total", "Total orders sent").unwrap();
        registry.register(Box::new(orders_sent.clone())).unwrap();
        
        let latency = Histogram::with_opts(
            prometheus::HistogramOpts::new("tick_to_trade_ns", "Tick to trade latency")
                .buckets(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0])
        ).unwrap();
        registry.register(Box::new(latency.clone())).unwrap();
        
        // ...
        
        Self { /* ... */ }
    }
}
```

---

## Инструментарий

### Rust crates
```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
crossbeam = "0.8"
parking_lot = "0.12"
dashmap = "5.5"
socket2 = "0.5"
io-uring = "0.6"
zerocopy = "0.7"
simd-json = "0.13"
hdrhistogram = "7.5"
criterion = "0.5"
core_affinity = "0.8"
prometheus = "0.13"
tracing = "0.1"
```

### System tools
```bash
# Linux kernel tuning
sysctl -w net.core.rmem_max=26214400
sysctl -w net.core.wmem_max=26214400
sysctl -w net.ipv4.tcp_rmem="4096 87380 26214400"
sysctl -w net.ipv4.tcp_wmem="4096 65536 26214400"
sysctl -w net.ipv4.tcp_low_latency=1

# CPU isolation
isolcpus=2,3,4,5  # in kernel cmdline

# IRQ affinity
echo 1 > /proc/irq/XX/smp_affinity  # Bind NIC IRQs
```

---

## Практические задания

### Задание 5.1: SPSC Queue Implementation
**Цель:** Реализовать lock-free SPSC queue
- Benchmark: throughput и latency
- Compare с crossbeam-channel
- Target: <50ns per operation

### Задание 5.2: Order Book Engine
**Цель:** Построить high-performance order book
- Operations: add, modify, cancel, match
- Target: <100ns per operation
- Support: 1M+ orders

### Задание 5.3: Market Data Handler
**Цель:** Real-time market data processing
- Connect to Binance WebSocket
- Parse and update order book
- Measure tick-to-update latency

### Задание 5.4: Complete Trading System
**Цель:** Интегрировать все компоненты
- Market data → Strategy → Order management
- End-to-end latency profiling
- Paper trading на testnet

### Задание 5.5: Production Hardening
**Цель:** Подготовить систему к production
- Error handling
- Monitoring и alerting
- Graceful shutdown
- Configuration management

---

## Критерии оценки

1. **Latency**: tick-to-trade < 20μs на commodity hardware
2. **Throughput**: 1M+ messages/sec
3. **Reliability**: no crashes, graceful degradation
4. **Observability**: comprehensive metrics и logging

---

## Связи с другими главами

| Глава | Связь |
|-------|-------|
| 01-stochastic-calculus | Efficient Monte Carlo implementation |
| 02-market-microstructure | Order book data structures |
| 03-portfolio-optimization | Real-time rebalancing |
| 04-ml-time-series | Model inference pipeline |

---

## Рекомендуемая литература

### Книги
1. "Systems Performance" — Brendan Gregg
2. "The Art of Multiprocessor Programming" — Herlihy, Shavit
3. "C++ High Performance" — Andrist, Sehr

### Онлайн ресурсы
1. LMAX Disruptor pattern
2. Aeron messaging system documentation
3. Jane Street tech blog
4. Two Sigma engineering blog

---

## Заметки по написанию

- Много working code примеров с benchmarks
- Показать реальные latency numbers
- Включить debugging техники (perf, flamegraph)
- Секция "Common Mistakes" для каждой темы
- Production war stories и lessons learned
