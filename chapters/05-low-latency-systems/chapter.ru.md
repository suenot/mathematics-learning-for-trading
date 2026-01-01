# Глава 5: Системы низкой задержки и практическая реализация

## Введение

В мире алгоритмической торговли **скорость решает всё**. Разница в несколько микросекунд может означать разницу между прибылью и убытком. В этой главе мы погрузимся в архитектуру ultra-low-latency торговых систем и научимся писать код, который работает на грани возможностей современного железа.

**Что такое low-latency система?**

Latency (задержка) — это время от получения рыночных данных до отправки ордера на биржу. В современных HFT-системах это время измеряется в микросекундах (μs) или даже наносекундах (ns).

```
1 секунда     = 1,000 миллисекунд (ms)
1 миллисекунда = 1,000 микросекунд (μs)
1 микросекунда = 1,000 наносекунд (ns)
```

**Целевые показатели:**
- Retail trading: 100-500 ms
- Institutional: 1-10 ms
- HFT: 1-100 μs
- Ultra-HFT: <1 μs

---

## 5.1 Архитектура Low-Latency Trading System

### 5.1.1 Компоненты системы

Типичная торговая система состоит из следующих компонентов:

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

**Описание компонентов:**

1. **Market Data Handler** — принимает данные с биржи (цены, объёмы, сделки)
2. **Order Book** — хранит текущее состояние книги заявок
3. **Strategy Engine** — принимает торговые решения
4. **Risk Manager** — проверяет лимиты и риски
5. **Order Router** — направляет ордера на биржу
6. **Exchange Connector** — сетевое соединение с биржей

### 5.1.2 Latency Budget (бюджет задержки)

Каждый компонент имеет свой "бюджет" времени:

| Компонент | Целевая задержка | Критический путь? |
|-----------|------------------|-------------------|
| Сетевой I/O | <10 μs | Да |
| Парсинг market data | <1 μs | Да |
| Обновление order book | <100 ns | Да |
| Решение стратегии | <1 μs | Да |
| Проверка рисков | <100 ns | Да |
| Сериализация ордера | <500 ns | Да |
| Логирование | N/A | Нет (async) |
| Мониторинг | N/A | Нет (async) |

**Общая цель tick-to-trade: <15 μs**

### 5.1.3 Принципы проектирования

```rust
// Принципы low-latency дизайна в коде

// 1. Однопоточный hot path — никаких блокировок
fn process_market_data(data: &MarketData) -> Option<Order> {
    // Весь критический путь в одном потоке
    // Нет mutex, нет contention
    update_order_book(data);
    let signal = calculate_signal();
    if signal.is_strong() {
        Some(create_order(signal))
    } else {
        None
    }
}

// 2. Pre-allocation — никаких аллокаций в критическом пути
struct PreallocatedBuffers {
    orders: Vec<Order>,      // Заранее выделено 1000 ордеров
    messages: Vec<Message>,  // Заранее выделено 10000 сообщений
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

// 3. Cache optimization — данные рядом в памяти
#[repr(C, align(64))]  // Выравнивание по cache line
struct HotData {
    best_bid: u64,
    best_ask: u64,
    mid_price: u64,
    spread: u64,
    // Всё помещается в одну cache line (64 байта)
}
```

---

## 5.2 Memory Management и Cache Optimization

### 5.2.1 Почему память важна?

Доступ к памяти — одна из главных причин задержек:

```
Тип доступа          | Задержка
---------------------|----------
L1 cache             | ~1 ns
L2 cache             | ~3 ns
L3 cache             | ~10 ns
RAM                  | ~100 ns
SSD                  | ~100 μs
Network (localhost)  | ~500 μs
```

**Вывод:** Разница между L1 cache и RAM — 100 раз! Поэтому правильная организация данных критически важна.

### 5.2.2 Cache-Aligned Structures

```rust
use std::sync::atomic::{AtomicU32, AtomicU64};

/// Структура, выровненная по cache line (64 байта на x86)
/// Это предотвращает "false sharing" между ядрами CPU
#[repr(C, align(64))]
pub struct OrderBookLevel {
    pub price: u64,          // 8 байт - цена в fixed-point формате
    pub quantity: u64,       // 8 байт - количество
    pub order_count: u32,    // 4 байта - число ордеров на уровне
    pub update_seq: u32,     // 4 байта - sequence number
    _padding: [u8; 40],      // Дополнение до 64 байт
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

    /// Обновление уровня — inline для скорости
    #[inline(always)]
    pub fn update(&mut self, quantity: u64, order_count: u32) {
        self.quantity = quantity;
        self.order_count = order_count;
        self.update_seq += 1;
    }
}

/// Состояние для каждого ядра CPU
/// Выравнивание предотвращает false sharing
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
        // Проверяем, что структуры занимают ровно 64 байта
        assert_eq!(std::mem::size_of::<OrderBookLevel>(), 64);
        assert_eq!(std::mem::size_of::<PerCoreState>(), 64);
    }
}
```

### 5.2.3 Object Pool — переиспользование объектов

```rust
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Пул объектов для избежания аллокаций в hot path
///
/// # Пример использования
/// ```
/// let pool: ObjectPool<Order, 1000> = ObjectPool::new();
///
/// // Получаем объект из пула (без аллокации!)
/// let order = pool.acquire().unwrap();
/// order.price = 100;
///
/// // Возвращаем в пул
/// pool.release(order);
/// ```
pub struct ObjectPool<T: Default, const N: usize> {
    storage: UnsafeCell<[Option<T>; N]>,
    free_indices: UnsafeCell<Vec<usize>>,
    count: AtomicUsize,
}

// Safety: мы контролируем доступ через atomic операции
unsafe impl<T: Default + Send, const N: usize> Send for ObjectPool<T, N> {}
unsafe impl<T: Default + Send, const N: usize> Sync for ObjectPool<T, N> {}

impl<T: Default + Clone, const N: usize> ObjectPool<T, N> {
    /// Создание нового пула с предварительной аллокацией всех объектов
    pub fn new() -> Self {
        let storage: [Option<T>; N] = std::array::from_fn(|_| Some(T::default()));
        let free_indices: Vec<usize> = (0..N).collect();

        Self {
            storage: UnsafeCell::new(storage),
            free_indices: UnsafeCell::new(free_indices),
            count: AtomicUsize::new(N),
        }
    }

    /// Получить объект из пула
    /// Возвращает None если пул пуст
    #[inline(always)]
    pub fn acquire(&self) -> Option<PooledObject<T, N>> {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return None;
        }

        // Безопасно, так как мы единственный writer в этом потоке
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

    /// Вернуть объект в пул (вызывается автоматически через Drop)
    fn release(&self, index: usize, obj: T) {
        unsafe {
            let storage = &mut *self.storage.get();
            storage[index] = Some(obj);
            let indices = &mut *self.free_indices.get();
            indices.push(index);
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Количество доступных объектов
    pub fn available(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// Объект из пула — автоматически возвращается при drop
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

### 5.2.4 Arena Allocator для сообщений

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// Arena allocator — сверхбыстрая аллокация для временных данных
///
/// Идея: выделяем большой блок памяти и "нарезаем" его по мере необходимости.
/// В конце цикла просто сбрасываем указатель — освобождение мгновенное!
pub struct MessageArena {
    buffer: Box<[u8]>,
    offset: AtomicUsize,
    capacity: usize,
}

impl MessageArena {
    /// Создание арены заданного размера
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity].into_boxed_slice(),
            offset: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Аллокация блока памяти
    /// Возвращает None если места нет
    #[inline(always)]
    pub fn alloc(&self, size: usize) -> Option<&mut [u8]> {
        // Выравниваем по 8 байт для производительности
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
            // Откатываем, если не хватило места
            self.offset.fetch_sub(aligned_size, Ordering::Relaxed);
            None
        }
    }

    /// Аллокация типизированного объекта
    #[inline(always)]
    pub fn alloc_obj<T: Sized>(&self) -> Option<&mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        // Учитываем выравнивание
        let current = self.offset.load(Ordering::Relaxed);
        let aligned_start = (current + align - 1) & !(align - 1);
        let needed = aligned_start - current + size;

        let slice = self.alloc(needed)?;
        let ptr = slice.as_mut_ptr().add(aligned_start - current);

        unsafe { Some(&mut *(ptr as *mut T)) }
    }

    /// Сброс арены — мгновенное освобождение всей памяти
    #[inline(always)]
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
    }

    /// Использовано байт
    pub fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Доступно байт
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

        // Аллоцируем несколько блоков
        let block1 = arena.alloc(100).unwrap();
        assert_eq!(block1.len(), 100);

        let block2 = arena.alloc(200).unwrap();
        assert_eq!(block2.len(), 200);

        // Сбрасываем — мгновенно!
        arena.reset();
        assert_eq!(arena.used(), 0);
    }
}
```

---

## 5.3 Lock-Free Data Structures

### 5.3.1 Зачем нужны lock-free структуры?

Обычные mutex/lock имеют серьёзные проблемы для low-latency:

1. **Contention** — потоки ждут друг друга
2. **Context switch** — переключение занимает ~1-10 μs
3. **Priority inversion** — низкоприоритетный поток блокирует высокоприоритетный
4. **Непредсказуемость** — задержка может варьироваться в широких пределах

Lock-free структуры используют atomic операции вместо блокировок.

### 5.3.2 SPSC Queue (Single Producer Single Consumer)

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

/// Lock-free очередь для одного producer и одного consumer
///
/// Это самая быстрая возможная очередь для сценария,
/// когда один поток пишет, другой читает.
///
/// # Пример
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
    head: AtomicUsize,  // Позиция записи (producer)
    tail: AtomicUsize,  // Позиция чтения (consumer)
}

// Safety: разные потоки работают с разными позициями
unsafe impl<T: Send, const N: usize> Send for SPSCQueue<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for SPSCQueue<T, N> {}

impl<T, const N: usize> SPSCQueue<T, N> {
    /// Создание новой очереди
    pub fn new() -> Self {
        Self {
            buffer: UnsafeCell::new(std::array::from_fn(|_| None)),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Добавить элемент в очередь
    /// Возвращает Err(value) если очередь полна
    #[inline(always)]
    pub fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % N;

        // Проверяем, не полна ли очередь
        if next_head == self.tail.load(Ordering::Acquire) {
            return Err(value);
        }

        // Записываем значение
        unsafe {
            (*self.buffer.get())[head] = Some(value);
        }

        // Публикуем новую позицию head
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Извлечь элемент из очереди
    /// Возвращает None если очередь пуста
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);

        // Проверяем, не пуста ли очередь
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }

        // Читаем значение
        let value = unsafe {
            (*self.buffer.get())[tail].take()
        };

        // Публикуем новую позицию tail
        self.tail.store((tail + 1) % N, Ordering::Release);
        value
    }

    /// Проверка на пустоту
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Relaxed) == self.head.load(Ordering::Relaxed)
    }

    /// Количество элементов в очереди
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

    /// Очередь полна?
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

/// Уровень цены в order book
#[derive(Debug, Clone, Default)]
pub struct PriceLevel {
    pub price: u64,      // Цена в fixed-point (умножена на 10^8)
    pub quantity: u64,   // Суммарный объём
    pub order_count: u32, // Количество ордеров
}

/// High-performance Order Book
///
/// Использует BTreeMap для O(log n) операций
/// с дополнительными оптимизациями для частых операций
pub struct OrderBook {
    bids: BTreeMap<std::cmp::Reverse<u64>, PriceLevel>,  // Убывающий порядок
    asks: BTreeMap<u64, PriceLevel>,                      // Возрастающий порядок
    sequence: AtomicU64,

    // Кэшированные значения для быстрого доступа
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

    /// Обновить уровень bid
    #[inline(always)]
    pub fn update_bid(&mut self, price: u64, quantity: u64) {
        if quantity == 0 {
            // Удаляем уровень
            self.bids.remove(&std::cmp::Reverse(price));
        } else {
            // Добавляем/обновляем уровень
            self.bids.insert(
                std::cmp::Reverse(price),
                PriceLevel {
                    price,
                    quantity,
                    order_count: 1,
                }
            );
        }

        // Обновляем кэш лучшего bid
        if let Some((key, _)) = self.bids.first_key_value() {
            self.cached_best_bid.store(key.0, Ordering::Relaxed);
        } else {
            self.cached_best_bid.store(0, Ordering::Relaxed);
        }

        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Обновить уровень ask
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

        // Обновляем кэш лучшего ask
        if let Some((key, _)) = self.asks.first_key_value() {
            self.cached_best_ask.store(*key, Ordering::Relaxed);
        } else {
            self.cached_best_ask.store(u64::MAX, Ordering::Relaxed);
        }

        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Лучший bid (самая высокая цена покупки)
    #[inline(always)]
    pub fn best_bid(&self) -> Option<(u64, u64)> {
        self.bids.first_key_value()
            .map(|(k, v)| (k.0, v.quantity))
    }

    /// Лучший ask (самая низкая цена продажи)
    #[inline(always)]
    pub fn best_ask(&self) -> Option<(u64, u64)> {
        self.asks.first_key_value()
            .map(|(k, v)| (*k, v.quantity))
    }

    /// Быстрый доступ к лучшему bid через кэш
    #[inline(always)]
    pub fn best_bid_cached(&self) -> u64 {
        self.cached_best_bid.load(Ordering::Relaxed)
    }

    /// Быстрый доступ к лучшему ask через кэш
    #[inline(always)]
    pub fn best_ask_cached(&self) -> u64 {
        self.cached_best_ask.load(Ordering::Relaxed)
    }

    /// Mid price — средняя цена между лучшим bid и ask
    #[inline(always)]
    pub fn mid_price(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2),
            _ => None,
        }
    }

    /// Spread — разница между лучшим ask и bid
    #[inline(always)]
    pub fn spread(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }

    /// Глубина рынка (сумма объёмов на N лучших уровнях)
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

    /// Sequence number для отслеживания обновлений
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

        // Добавляем уровни
        book.update_bid(100_00000000, 1000);  // 100.00 @ 1000
        book.update_bid(99_50000000, 2000);   // 99.50 @ 2000
        book.update_ask(100_50000000, 500);   // 100.50 @ 500
        book.update_ask(101_00000000, 1500);  // 101.00 @ 1500

        // Проверяем best bid/ask
        assert_eq!(book.best_bid(), Some((100_00000000, 1000)));
        assert_eq!(book.best_ask(), Some((100_50000000, 500)));

        // Проверяем spread
        assert_eq!(book.spread(), Some(50000000)); // 0.50
    }
}
```

### 5.3.4 SeqLock для read-heavy данных

```rust
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::cell::UnsafeCell;

/// SeqLock — оптимизирован для частого чтения, редкой записи
///
/// Идеально подходит для данных, которые:
/// - Читаются очень часто (миллионы раз в секунду)
/// - Записываются редко (один writer)
/// - Нужен snapshot consistency
///
/// # Как это работает
/// 1. Writer увеличивает sequence на 1 (нечётное = запись идёт)
/// 2. Writer записывает данные
/// 3. Writer увеличивает sequence на 1 (чётное = запись завершена)
/// 4. Reader проверяет sequence до и после чтения
/// 5. Если sequence изменился или нечётный — повторяем чтение
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

    /// Запись данных (только один writer!)
    pub fn write(&self, value: T) {
        let seq = self.sequence.load(Ordering::Relaxed);

        // Помечаем начало записи (нечётное число)
        self.sequence.store(seq + 1, Ordering::Release);

        // Записываем данные
        unsafe { *self.data.get() = value; }

        // Помечаем конец записи (чётное число)
        self.sequence.store(seq + 2, Ordering::Release);
    }

    /// Чтение данных (lock-free, может быть много readers)
    #[inline(always)]
    pub fn read(&self) -> T {
        loop {
            // Читаем sequence
            let seq1 = self.sequence.load(Ordering::Acquire);

            // Если нечётное — writer пишет, ждём
            if seq1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }

            // Читаем данные
            let value = unsafe { *self.data.get() };

            // Memory fence для корректности
            fence(Ordering::Acquire);

            // Проверяем, не изменился ли sequence
            let seq2 = self.sequence.load(Ordering::Relaxed);

            if seq1 == seq2 {
                return value;
            }
            // Sequence изменился — повторяем
        }
    }

    /// Попытка чтения (возвращает None если writer активен)
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

/// Пример: shared market state
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

        // Читаем
        let state = lock.read();
        assert_eq!(state.best_bid, 100);

        // Пишем
        lock.write(MarketState {
            best_bid: 102,
            best_ask: 103,
            last_trade_price: 102,
            volume_24h: 1100,
        });

        // Читаем обновлённые данные
        let state = lock.read();
        assert_eq!(state.best_bid, 102);
    }
}
```

---

## 5.4 Network I/O Optimization

### 5.4.1 TCP Tuning для низкой задержки

```rust
use std::net::TcpStream;
use std::os::unix::io::AsRawFd;

/// Настройки TCP сокета для low-latency
pub struct TcpTuning;

impl TcpTuning {
    /// Применить low-latency настройки к сокету
    pub fn apply(stream: &TcpStream) -> std::io::Result<()> {
        use socket2::{Socket, TcpKeepalive};
        use std::time::Duration;

        // Конвертируем в socket2::Socket для расширенных настроек
        let socket = Socket::from(stream.try_clone()?);

        // 1. Отключаем алгоритм Nagle
        // Nagle буферизует маленькие пакеты — это увеличивает latency!
        socket.set_nodelay(true)?;

        // 2. Увеличиваем буферы сокета
        socket.set_recv_buffer_size(4 * 1024 * 1024)?;  // 4MB
        socket.set_send_buffer_size(4 * 1024 * 1024)?;  // 4MB

        // 3. Настраиваем keepalive
        let keepalive = TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10));
        socket.set_tcp_keepalive(&keepalive)?;

        Ok(())
    }

    /// Linux-специфичные оптимизации
    #[cfg(target_os = "linux")]
    pub fn apply_linux_optimizations(stream: &TcpStream) -> std::io::Result<()> {
        let fd = stream.as_raw_fd();

        unsafe {
            // TCP_QUICKACK — отключаем отложенные ACK
            let quickack: libc::c_int = 1;
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_QUICKACK,
                &quickack as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );

            // TCP_NODELAY ещё раз для надёжности
            let nodelay: libc::c_int = 1;
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_NODELAY,
                &nodelay as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );

            // SO_BUSY_POLL — активный polling вместо прерываний
            let busy_poll: libc::c_int = 50;  // микросекунды
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

/// Создание low-latency TCP соединения
pub fn create_low_latency_connection(addr: &str) -> std::io::Result<TcpStream> {
    let stream = TcpStream::connect(addr)?;

    TcpTuning::apply(&stream)?;

    #[cfg(target_os = "linux")]
    TcpTuning::apply_linux_optimizations(&stream)?;

    Ok(stream)
}
```

### 5.4.2 WebSocket Client для криптобирж

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};

/// Сообщение рыночных данных от Binance
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

/// Сообщение о сделке
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

/// Подписка на каналы Binance
#[derive(Serialize)]
struct SubscribeMessage {
    method: String,
    params: Vec<String>,
    id: u64,
}

/// Low-latency WebSocket клиент для Binance
pub struct BinanceWebSocket {
    url: String,
}

impl BinanceWebSocket {
    pub fn new() -> Self {
        Self {
            url: "wss://stream.binance.com:9443/ws".to_string(),
        }
    }

    /// Подключение и обработка сообщений
    pub async fn connect_and_subscribe<F>(
        &self,
        symbols: &[&str],
        mut handler: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(MarketEvent) + Send,
    {
        // Подключаемся
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Формируем подписки
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

        // Отправляем подписку
        let msg = serde_json::to_string(&subscribe)?;
        write.send(Message::Text(msg)).await?;

        // Обрабатываем сообщения
        while let Some(msg) = read.next().await {
            match msg? {
                Message::Text(text) => {
                    // Пытаемся распарсить разные типы сообщений
                    if let Ok(depth) = serde_json::from_str::<BinanceDepthUpdate>(&text) {
                        handler(MarketEvent::Depth(depth));
                    } else if let Ok(trade) = serde_json::from_str::<BinanceTrade>(&text) {
                        handler(MarketEvent::Trade(trade));
                    }
                }
                Message::Ping(data) => {
                    // Отвечаем на ping мгновенно
                    write.send(Message::Pong(data)).await?;
                }
                Message::Close(_) => break,
                _ => {}
            }
        }

        Ok(())
    }
}

/// События рынка
pub enum MarketEvent {
    Depth(BinanceDepthUpdate),
    Trade(BinanceTrade),
}

impl Default for BinanceWebSocket {
    fn default() -> Self {
        Self::new()
    }
}

/// Пример использования
#[cfg(test)]
mod tests {
    use super::*;

    // #[tokio::test]  // Раскомментируйте для реального теста
    async fn test_binance_connection() {
        let client = BinanceWebSocket::new();

        client.connect_and_subscribe(
            &["BTCUSDT", "ETHUSDT"],
            |event| {
                match event {
                    MarketEvent::Depth(d) => {
                        println!("Depth update: {} bids, {} asks",
                            d.bids.len(), d.asks.len());
                    }
                    MarketEvent::Trade(t) => {
                        println!("Trade: {} @ {}", t.quantity, t.price);
                    }
                }
            }
        ).await.unwrap();
    }
}
```

---

## 5.5 CPU Affinity и Scheduling

### 5.5.1 Привязка потоков к ядрам CPU

```rust
use std::thread;

/// CPU Affinity — привязка потока к конкретному ядру
///
/// Зачем это нужно:
/// 1. Избегаем миграции потока между ядрами (дорогая операция)
/// 2. Улучшаем cache locality
/// 3. Предсказуемая latency
pub struct CpuAffinity;

impl CpuAffinity {
    /// Привязать текущий поток к ядру
    #[cfg(target_os = "linux")]
    pub fn pin_to_core(core_id: usize) -> bool {
        unsafe {
            let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut cpuset);
            libc::CPU_SET(core_id, &mut cpuset);

            let result = libc::sched_setaffinity(
                0,  // текущий поток
                std::mem::size_of::<libc::cpu_set_t>(),
                &cpuset,
            );

            result == 0
        }
    }

    /// Получить количество ядер CPU
    pub fn num_cores() -> usize {
        num_cpus::get()
    }

    /// Получить количество физических ядер (без hyperthreading)
    pub fn num_physical_cores() -> usize {
        num_cpus::get_physical()
    }
}

/// Настройка приоритета потока
pub struct ThreadPriority;

impl ThreadPriority {
    /// Установить real-time приоритет (требует root)
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

    /// Установить высокий приоритет (не требует root)
    #[cfg(target_os = "linux")]
    pub fn set_high() -> bool {
        unsafe {
            libc::setpriority(libc::PRIO_PROCESS, 0, -20) == 0
        }
    }
}

/// Настройка торговых потоков
pub fn setup_trading_threads() {
    // Критический путь — отдельное изолированное ядро
    thread::Builder::new()
        .name("strategy".to_string())
        .spawn(move || {
            // Привязываем к ядру 2 (обычно изолированное)
            CpuAffinity::pin_to_core(2);
            // Максимальный приоритет
            ThreadPriority::set_realtime(99);

            // Основной цикл стратегии
            run_strategy_loop();
        })
        .expect("Failed to spawn strategy thread");

    // Market data handler — другое ядро
    thread::Builder::new()
        .name("market_data".to_string())
        .spawn(move || {
            CpuAffinity::pin_to_core(3);
            ThreadPriority::set_realtime(98);

            run_market_data_loop();
        })
        .expect("Failed to spawn market data thread");

    // Logging — низкий приоритет, любое ядро
    thread::Builder::new()
        .name("logger".to_string())
        .spawn(move || {
            // Не привязываем — пусть OS решает
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

### 5.5.2 Системные настройки Linux

```bash
#!/bin/bash
# Скрипт настройки системы для low-latency trading

# 1. Изоляция ядер CPU от планировщика
# Добавить в /etc/default/grub: GRUB_CMDLINE_LINUX="isolcpus=2,3,4,5"

# 2. Отключение энергосбережения CPU
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > $cpu
done

# 3. Настройки сети
sysctl -w net.core.rmem_max=26214400
sysctl -w net.core.wmem_max=26214400
sysctl -w net.ipv4.tcp_rmem="4096 87380 26214400"
sysctl -w net.ipv4.tcp_wmem="4096 65536 26214400"
sysctl -w net.ipv4.tcp_low_latency=1
sysctl -w net.ipv4.tcp_timestamps=0
sysctl -w net.ipv4.tcp_sack=0

# 4. Huge Pages для уменьшения TLB miss
echo 1024 > /proc/sys/vm/nr_hugepages

# 5. Отключение swap
swapoff -a

# 6. IRQ affinity — привязка прерываний NIC к ядру 0
# (чтобы не мешать изолированным ядрам)
for irq in $(cat /proc/interrupts | grep eth0 | awk '{print $1}' | tr -d ':'); do
    echo 1 > /proc/irq/$irq/smp_affinity
done

echo "Low-latency tuning applied!"
```

---

## 5.6 Message Parsing и Serialization

### 5.6.1 Zero-Copy Parsing

```rust
/// Zero-copy парсинг бинарных сообщений
///
/// Идея: вместо копирования данных — просто "смотрим" на них
/// через правильно выровненный указатель

/// Сырое сообщение рыночных данных (бинарный протокол)
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct RawMarketDataMessage {
    pub msg_type: u8,
    pub symbol_id: u32,
    pub price: i64,       // Fixed-point, 8 десятичных знаков
    pub quantity: u64,
    pub timestamp_ns: u64,
}

impl RawMarketDataMessage {
    /// Парсинг без копирования — O(1)!
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() >= std::mem::size_of::<Self>() {
            // Safety: проверили размер, структура packed
            Some(unsafe { &*(bytes.as_ptr() as *const Self) })
        } else {
            None
        }
    }

    /// Конвертация цены в f64
    #[inline(always)]
    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 100_000_000.0
    }

    /// Проверка типа сообщения
    #[inline(always)]
    pub fn is_trade(&self) -> bool {
        self.msg_type == 1
    }

    #[inline(always)]
    pub fn is_quote(&self) -> bool {
        self.msg_type == 2
    }
}

/// Сериализация без аллокаций
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
        // Создаём сообщение
        let msg = RawMarketDataMessage {
            msg_type: 1,
            symbol_id: 12345,
            price: 50000_00000000,  // 50000.00
            quantity: 1000,
            timestamp_ns: 1234567890,
        };

        // Сериализуем
        let bytes = msg.to_bytes();

        // Парсим без копирования
        let parsed = RawMarketDataMessage::from_bytes(bytes).unwrap();

        assert_eq!(parsed.symbol_id, 12345);
        assert_eq!(parsed.price_f64(), 50000.0);
    }
}
```

### 5.6.2 FIX Protocol Parser

```rust
/// FIX Protocol — стандарт для обмена финансовыми сообщениями
///
/// Формат: Tag=Value|Tag=Value|...
/// Разделитель: SOH (0x01)

/// Распарсенное FIX сообщение
pub struct FixMessage<'a> {
    raw: &'a [u8],
    fields: Vec<(u32, &'a [u8])>,
}

/// Стандартные FIX теги
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
    /// Парсинг FIX сообщения
    #[inline]
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        let mut fields = Vec::with_capacity(32);
        let mut pos = 0;

        while pos < data.len() {
            // Ищем '='
            let eq_pos = find_byte(&data[pos..], b'=')?;

            // Парсим тег
            let tag = parse_u32(&data[pos..pos + eq_pos])?;
            pos += eq_pos + 1;

            // Ищем разделитель SOH
            let soh_pos = find_byte(&data[pos..], 0x01)?;

            // Значение
            let value = &data[pos..pos + soh_pos];
            pos += soh_pos + 1;

            fields.push((tag, value));
        }

        Some(Self { raw: data, fields })
    }

    /// Получить значение поля по тегу
    #[inline(always)]
    pub fn get(&self, tag: u32) -> Option<&'a [u8]> {
        self.fields.iter()
            .find(|(t, _)| *t == tag)
            .map(|(_, v)| *v)
    }

    /// Получить значение как строку
    #[inline(always)]
    pub fn get_str(&self, tag: u32) -> Option<&'a str> {
        self.get(tag).and_then(|v| std::str::from_utf8(v).ok())
    }

    /// Получить значение как число
    #[inline(always)]
    pub fn get_u64(&self, tag: u32) -> Option<u64> {
        self.get(tag).and_then(|v| parse_u64(v))
    }

    /// Получить значение как f64
    #[inline(always)]
    pub fn get_f64(&self, tag: u32) -> Option<f64> {
        self.get_str(tag).and_then(|s| s.parse().ok())
    }

    /// Тип сообщения
    pub fn msg_type(&self) -> Option<&'a str> {
        self.get_str(fix_tags::MSG_TYPE)
    }

    /// Символ
    pub fn symbol(&self) -> Option<&'a str> {
        self.get_str(fix_tags::SYMBOL)
    }
}

/// Быстрый поиск байта
#[inline(always)]
fn find_byte(data: &[u8], byte: u8) -> Option<usize> {
    memchr::memchr(byte, data)
}

/// Быстрый парсинг u32
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

/// Быстрый парсинг u64
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

/// Построитель FIX сообщений
pub struct FixMessageBuilder {
    buffer: Vec<u8>,
}

impl FixMessageBuilder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(512),
        }
    }

    /// Добавить поле
    pub fn field(mut self, tag: u32, value: &str) -> Self {
        use std::fmt::Write;
        write!(&mut self.buffer, "{}={}\x01", tag, value).unwrap();
        self
    }

    /// Добавить числовое поле
    pub fn field_num(mut self, tag: u32, value: u64) -> Self {
        use std::fmt::Write;
        write!(&mut self.buffer, "{}={}\x01", tag, value).unwrap();
        self
    }

    /// Добавить поле с ценой
    pub fn field_price(mut self, tag: u32, value: f64) -> Self {
        use std::fmt::Write;
        write!(&mut self.buffer, "{}={:.8}\x01", tag, value).unwrap();
        self
    }

    /// Построить сообщение
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
        // Пример FIX сообщения (SOH = 0x01)
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

## 5.7 Profiling и Benchmarking

### 5.7.1 Измерение задержки

```rust
use std::time::Instant;

/// Профайлер задержки с HDR гистограммой
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

    /// Записать измерение (в наносекундах)
    #[inline(always)]
    pub fn record(&mut self, latency_ns: u64) {
        if self.samples.len() < self.capacity {
            self.samples.push(latency_ns);
        }
    }

    /// Измерить время выполнения функции
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

    /// Статистика
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

    /// Очистить данные
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

/// Макрос для удобного измерения
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
/// Счётчик тактов CPU для ультра-точных измерений
pub struct CpuCycleCounter;

impl CpuCycleCounter {
    /// Читаем TSC (Time Stamp Counter)
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn rdtsc() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    /// RDTSCP — более точная версия с барьером
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn rdtscp() -> u64 {
        let mut aux: u32 = 0;
        unsafe { core::arch::x86_64::__rdtscp(&mut aux) }
    }

    /// Конвертация тактов в наносекунды
    /// cpu_freq_ghz — частота процессора в GHz
    #[inline(always)]
    pub fn cycles_to_ns(cycles: u64, cpu_freq_ghz: f64) -> f64 {
        cycles as f64 / cpu_freq_ghz
    }

    /// Измерить количество тактов для выполнения функции
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

/// Определение частоты CPU
#[cfg(target_os = "linux")]
pub fn get_cpu_freq_ghz() -> Option<f64> {
    use std::fs;

    // Читаем из /proc/cpuinfo
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

### 5.7.3 Benchmarks с Criterion

```rust
// benches/order_book_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, black_box};

// Предполагаем, что OrderBook определён в нашем crate
// use trading_system::OrderBook;

fn benchmark_order_book(c: &mut Criterion) {
    let mut book = OrderBook::new();

    // Заполняем книгу
    for i in 0..100 {
        book.update_bid(100_00000000 - i * 10000000, 1000);
        book.update_ask(100_00000000 + i * 10000000, 1000);
    }

    c.bench_function("order_book_update_bid", |b| {
        b.iter(|| {
            black_box(book.update_bid(99_00000000, 500));
        })
    });

    c.bench_function("order_book_best_bid", |b| {
        b.iter(|| {
            black_box(book.best_bid())
        })
    });

    c.bench_function("order_book_mid_price", |b| {
        b.iter(|| {
            black_box(book.mid_price())
        })
    });

    c.bench_function("order_book_spread", |b| {
        b.iter(|| {
            black_box(book.spread())
        })
    });
}

fn benchmark_spsc_queue(c: &mut Criterion) {
    let queue: SPSCQueue<u64, 1024> = SPSCQueue::new();

    c.bench_function("spsc_push", |b| {
        b.iter(|| {
            let _ = queue.push(black_box(42));
            queue.pop();
        })
    });

    c.bench_function("spsc_pop", |b| {
        queue.push(42).unwrap();
        b.iter(|| {
            black_box(queue.pop())
        })
    });
}

fn benchmark_fix_parsing(c: &mut Criterion) {
    let msg = b"35=D\x0149=CLIENT\x0156=EXCHANGE\x0155=BTCUSD\x0154=1\x0138=100\x0144=50000.00\x01";

    c.bench_function("fix_parse", |b| {
        b.iter(|| {
            black_box(FixMessage::parse(msg))
        })
    });
}

criterion_group!(
    benches,
    benchmark_order_book,
    benchmark_spsc_queue,
    benchmark_fix_parsing,
);
criterion_main!(benches);
```

---

## 5.8 Production Architecture

### 5.8.1 Обработка ошибок

```rust
use std::fmt;

/// Типы ошибок торговой системы
#[derive(Debug)]
pub enum TradingError {
    /// Сетевая ошибка
    Network(std::io::Error),
    /// Ошибка парсинга
    Parse(String),
    /// Превышен лимит риска
    RiskLimit { limit: f64, attempted: f64 },
    /// Ордер отклонён
    OrderRejected { reason: String },
    /// Система перегружена
    SystemOverload,
    /// Отключение от биржи
    Disconnected,
    /// Таймаут
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

/// Result type для торговой системы
pub type TradingResult<T> = Result<T, TradingError>;

/// Быстрые error codes для hot path (без аллокаций)
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

/// Circuit Breaker — защита от каскадных сбоев
///
/// Состояния:
/// - Closed: нормальная работа
/// - Open: блокируем запросы (после N ошибок)
/// - HalfOpen: пробуем восстановить (после таймаута)
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

    /// Проверить, разрешено ли выполнение
    #[inline(always)]
    pub fn allow(&self) -> bool {
        match self.state.load(Ordering::Relaxed) {
            0 => true,  // Closed — разрешено
            1 => {      // Open — проверяем таймаут
                let now = current_time_ms();
                let last = self.last_failure_ms.load(Ordering::Relaxed);

                if now - last > self.recovery_time_ms {
                    // Переходим в HalfOpen
                    self.state.store(2, Ordering::Release);
                    true
                } else {
                    false
                }
            }
            2 => true,  // HalfOpen — разрешаем одну попытку
            _ => false,
        }
    }

    /// Записать успешное выполнение
    #[inline(always)]
    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.state.store(0, Ordering::Release);  // -> Closed
    }

    /// Записать ошибку
    #[inline(always)]
    pub fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        if count >= self.failure_threshold {
            self.state.store(1, Ordering::Release);  // -> Open
            self.last_failure_ms.store(current_time_ms(), Ordering::Relaxed);
        }
    }

    /// Текущее состояние
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

/// Управление корректным завершением системы
pub struct ShutdownController {
    shutdown_requested: Arc<AtomicBool>,
}

impl ShutdownController {
    pub fn new() -> Self {
        Self {
            shutdown_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Получить токен для проверки shutdown
    pub fn token(&self) -> ShutdownToken {
        ShutdownToken {
            shutdown_requested: Arc::clone(&self.shutdown_requested),
        }
    }

    /// Инициировать shutdown
    pub fn shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
    }

    /// Установить обработчик Ctrl+C
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

/// Токен для проверки запроса на shutdown
#[derive(Clone)]
pub struct ShutdownToken {
    shutdown_requested: Arc<AtomicBool>,
}

impl ShutdownToken {
    /// Проверить, запрошен ли shutdown
    #[inline(always)]
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Acquire)
    }
}

/// Пример использования в main loop
pub fn run_trading_loop(token: ShutdownToken) {
    println!("Trading loop started. Press Ctrl+C to stop.");

    while !token.is_shutdown_requested() {
        // Обработка рыночных данных
        // Выполнение стратегии
        // Отправка ордеров

        std::thread::sleep(std::time::Duration::from_micros(100));
    }

    println!("Trading loop stopped gracefully.");
}
```

---

## 5.9 Полный пример: Trading System

```rust
//! Полный пример торговой системы с низкой задержкой

use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Конфигурация торговой системы
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

/// Торговая система
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

    /// Запуск системы
    pub fn run(&self) {
        // Устанавливаем обработчик сигналов
        self.shutdown.install_signal_handler();

        let token = self.shutdown.token();
        let order_book = Arc::clone(&self.order_book);
        let circuit_breaker = Arc::clone(&self.circuit_breaker);
        let strategy_core = self.config.strategy_core;

        // Поток стратегии
        let strategy_handle = thread::Builder::new()
            .name("strategy".to_string())
            .spawn(move || {
                #[cfg(target_os = "linux")]
                CpuAffinity::pin_to_core(strategy_core);

                Self::strategy_loop(token, order_book, circuit_breaker);
            })
            .expect("Failed to spawn strategy thread");

        // Ждём завершения
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
            // Проверяем circuit breaker
            if !circuit_breaker.allow() {
                thread::sleep(Duration::from_millis(100));
                continue;
            }

            // Измеряем latency одной итерации
            profiler.measure(|| {
                // 1. Читаем order book
                let book = order_book.read().unwrap();

                // 2. Вычисляем сигнал
                if let (Some((bid, _)), Some((ask, _))) = (book.best_bid(), book.best_ask()) {
                    let mid = (bid + ask) / 2;
                    let spread = ask - bid;

                    // Простая стратегия: если spread большой — можно заработать
                    if spread > 100_000 {  // > 0.001
                        // Здесь была бы отправка ордера
                        let _order_price = mid;
                    }
                }
            });

            iteration += 1;

            // Каждые 10000 итераций выводим статистику
            if iteration % 10000 == 0 {
                let stats = profiler.stats();
                println!("Iteration {}: p50={} ns, p99={} ns",
                    iteration, stats.p50, stats.p99);
                profiler.reset();
            }

            // Небольшая пауза чтобы не перегружать CPU
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

## 5.10 Практические задания

### Задание 5.1: Реализация SPSC Queue

**Цель:** Реализовать lock-free SPSC очередь и сравнить с `crossbeam-channel`.

**Требования:**
- Поддержка generic типов
- Batch операции (push_batch, pop_batch)
- Benchmark: target <50 ns per operation

```rust
// Шаблон для реализации
pub struct BatchSPSCQueue<T, const N: usize> {
    // TODO: реализовать
}

impl<T, const N: usize> BatchSPSCQueue<T, N> {
    pub fn push_batch(&self, items: &[T]) -> usize {
        // TODO: вернуть количество успешно добавленных
        todo!()
    }

    pub fn pop_batch(&self, buffer: &mut [T]) -> usize {
        // TODO: вернуть количество извлечённых
        todo!()
    }
}
```

### Задание 5.2: High-Performance Order Book

**Цель:** Построить order book с поддержкой 1M+ ордеров.

**Требования:**
- O(1) для best bid/ask
- O(log n) для add/remove
- Поддержка L2 и L3 данных
- Target: <100 ns per operation

### Задание 5.3: Market Data Handler

**Цель:** Подключиться к Binance WebSocket и обновлять order book в реальном времени.

**Требования:**
- Подключение к testnet
- Парсинг depth и trade сообщений
- Измерение tick-to-update latency
- Обработка reconnect

### Задание 5.4: Полная торговая система

**Цель:** Интегрировать все компоненты в работающую систему.

**Требования:**
- Market data → Order book → Strategy → Order management
- End-to-end latency profiling
- Paper trading на testnet
- Graceful shutdown

### Задание 5.5: Production Hardening

**Цель:** Подготовить систему к production.

**Требования:**
- Comprehensive error handling
- Monitoring с Prometheus metrics
- Alerting при критических событиях
- Configuration management (TOML/YAML)

---

## Заключение

В этой главе мы изучили:

1. **Архитектуру** low-latency торговых систем
2. **Memory management** — cache alignment, object pools, arena allocators
3. **Lock-free структуры** — SPSC queues, SeqLock
4. **Network optimization** — TCP tuning, WebSocket
5. **CPU optimization** — affinity, scheduling, NUMA
6. **Profiling** — latency measurement, benchmarking
7. **Production patterns** — error handling, circuit breaker, graceful shutdown

**Ключевые принципы:**
- Измеряй перед оптимизацией
- Избегай аллокаций в hot path
- Используй lock-free где возможно
- Привязывай критические потоки к ядрам
- Всегда имей план graceful degradation

---

## Рекомендуемая литература

### Книги
1. "Systems Performance" — Brendan Gregg
2. "The Art of Multiprocessor Programming" — Herlihy, Shavit
3. "Algorithmic and High-Frequency Trading" — Cartea, Jaimungal, Penalva

### Онлайн ресурсы
1. LMAX Disruptor pattern
2. Aeron messaging documentation
3. Jane Street tech blog
4. Two Sigma engineering blog

---

*Следующая глава: Теория информации и Kelly Criterion*
