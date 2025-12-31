# Глава 4: Машинное обучение для финансовых временных рядов

## Метаданные
- **Уровень сложности**: Продвинутый
- **Предварительные требования**: Основы ML, глубокое обучение, главы 1-3
- **Языки реализации**: Python/PyTorch (основной для ML), Rust (inference, feature engineering)
- **Расчётный объём**: 120-150 страниц

---

## Цели главы

1. Освоить современные deep learning архитектуры для финансовых данных
2. Понять специфику ML для трейдинга: non-stationarity, regime changes, lookahead bias
3. Реализовать production-ready pipelines для feature engineering и inference
4. Изучить reinforcement learning для trading agents
5. Построить ensemble систему с proper backtesting

---

## Научная база

### Фундаментальные работы
1. **Hochreiter S., Schmidhuber J.** (1997) "Long Short-Term Memory" — основа LSTM
2. **Vaswani A. et al.** (2017) "Attention Is All You Need" — Transformer architecture
3. **López de Prado M.** (2018) "Advances in Financial Machine Learning" — best practices

### Современные исследования (2023-2025)
4. **Time series forecasting in financial markets** (2025) — WJAETS, LSTM vs GRU vs Transformer comparison
5. **LSTM-mTrans-MLP hybrid model** (2025) — MDPI Forecasting, robust volatility prediction
6. **Stock Market Volatility Forecasting: Deep Learning** (2025) — TiDE, DeepAR, TCN comparison
7. **Zhang Z., Zohren S.** (2021) "Deep Learning for Market by Order Data" — Oxford Man Institute
8. **Multi-Horizon Forecasting for Limit Order Books** — Novel deep learning, IPU acceleration
9. **Deep RL for Quantitative Trading** (2023) — ACM TIST comprehensive survey
10. **Physics-informed Transformer for volatility surface** (2024) — Quantitative Finance

### Time Series Specific
11. **Temporal Fusion Transformers** (TFT) — Google Research
12. **Informer** (2021) — efficient long-sequence forecasting
13. **Autoformer** (2021) — decomposition-based transformer
14. **PatchTST** (2023) — patching for time series
15. **TimesFM** (2024) — Google's foundation model for time series

---

## Структура главы

### 4.1 Особенности ML для финансовых данных

**4.1.1 Challenges**

| Challenge | Описание | Mitigation |
|-----------|----------|------------|
| Non-stationarity | Распределение меняется со временем | Walk-forward validation, regime detection |
| Low signal-to-noise | Большинство движений — noise | Careful feature engineering, ensembles |
| Lookahead bias | Использование будущей информации | Strict time-based splits |
| Survivorship bias | Данные только живых компаний | Include delisted securities |
| Data snooping | Multiple hypothesis testing | Out-of-sample validation, Bonferroni |
| Regime changes | Разные market conditions | Regime-aware models |

**4.1.2 Proper Cross-Validation**

```python
# WRONG: Random split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# CORRECT: Time-based split with embargo
def walk_forward_validation(data, n_splits=5, embargo_days=5):
    """
    Purged K-fold with embargo to prevent lookahead
    """
    splits = []
    n = len(data)
    fold_size = n // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end + embargo_days  # Embargo period
        test_end = test_start + fold_size
        
        train_idx = range(0, train_end)
        test_idx = range(test_start, min(test_end, n))
        splits.append((train_idx, test_idx))
    
    return splits
```

**4.1.3 Feature Engineering Best Practices**

```python
# Triple barrier labeling (López de Prado)
def triple_barrier_labels(prices, horizon, upper=0.02, lower=-0.02):
    """
    Labels based on first barrier hit:
    +1: upper barrier (profit)
    -1: lower barrier (loss)
    0: horizontal barrier (time limit)
    """
    labels = []
    for i in range(len(prices) - horizon):
        future = prices[i+1:i+horizon+1] / prices[i] - 1
        
        if any(future >= upper):
            first_up = np.argmax(future >= upper)
            first_down = np.argmax(future <= lower) if any(future <= lower) else float('inf')
            labels.append(1 if first_up < first_down else -1)
        elif any(future <= lower):
            labels.append(-1)
        else:
            labels.append(0)
    
    return labels
```

---

### 4.2 Feature Engineering на Rust

**4.2.1 Почему Rust для features?**
- Real-time computation requirements
- Memory efficiency для больших datasets
- Zero-copy integration с streaming data

**4.2.2 Technical Indicators**

```rust
pub struct TechnicalIndicators {
    window_sizes: Vec<usize>,
}

impl TechnicalIndicators {
    /// Exponential Moving Average
    pub fn ema(&self, prices: &[f64], span: usize) -> Vec<f64> {
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut ema = Vec::with_capacity(prices.len());
        ema.push(prices[0]);
        
        for i in 1..prices.len() {
            ema.push(alpha * prices[i] + (1.0 - alpha) * ema[i-1]);
        }
        ema
    }
    
    /// Relative Strength Index
    pub fn rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        let mut gains = vec![0.0; returns.len()];
        let mut losses = vec![0.0; returns.len()];
        
        for (i, &r) in returns.iter().enumerate() {
            if r > 0.0 { gains[i] = r; }
            else { losses[i] = -r; }
        }
        
        // EMA of gains and losses
        let avg_gain = self.ema(&gains, period);
        let avg_loss = self.ema(&losses, period);
        
        avg_gain.iter().zip(avg_loss.iter())
            .map(|(&g, &l)| {
                if l == 0.0 { 100.0 }
                else { 100.0 - 100.0 / (1.0 + g / l) }
            })
            .collect()
    }
    
    /// Bollinger Bands
    pub fn bollinger_bands(&self, prices: &[f64], period: usize, num_std: f64) 
        -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = self.sma(prices, period);
        let std = self.rolling_std(prices, period);
        
        let upper: Vec<f64> = sma.iter().zip(std.iter())
            .map(|(&m, &s)| m + num_std * s)
            .collect();
        let lower: Vec<f64> = sma.iter().zip(std.iter())
            .map(|(&m, &s)| m - num_std * s)
            .collect();
        
        (lower, sma, upper)
    }
}
```

**4.2.3 Microstructure Features**

```rust
pub struct MicrostructureFeatures;

impl MicrostructureFeatures {
    /// Order book imbalance
    pub fn order_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
        (bid_vol - ask_vol) / (bid_vol + ask_vol)
    }
    
    /// Microprice
    pub fn microprice(bid: f64, ask: f64, bid_vol: f64, ask_vol: f64) -> f64 {
        (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
    }
    
    /// Volume-weighted price deviation
    pub fn vwap_deviation(prices: &[f64], volumes: &[f64], current_price: f64) -> f64 {
        let vwap: f64 = prices.iter().zip(volumes.iter())
            .map(|(&p, &v)| p * v)
            .sum::<f64>() / volumes.iter().sum::<f64>();
        
        (current_price - vwap) / vwap
    }
    
    /// Kyle's Lambda (price impact estimate)
    pub fn kyle_lambda(returns: &[f64], signed_volume: &[f64]) -> f64 {
        // Linear regression: returns ~ signed_volume
        let n = returns.len() as f64;
        let sum_xy: f64 = returns.iter().zip(signed_volume.iter())
            .map(|(&r, &v)| r * v)
            .sum();
        let sum_x: f64 = signed_volume.iter().sum();
        let sum_y: f64 = returns.iter().sum();
        let sum_x2: f64 = signed_volume.iter().map(|&v| v * v).sum();
        
        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    }
}
```

---

### 4.3 Deep Learning Architectures

**4.3.1 LSTM для Time Series**

```python
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # Causal for trading!
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )
    
    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Self-attention over sequence
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            need_weights=True,
        )
        
        # Use last timestep
        out = self.fc(attn_out[:, -1, :])
        
        if return_attention:
            return out, attn_weights
        return out
```

**4.3.2 Transformer for Financial Data**

```python
class FinancialTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder (causal mask for trading)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(d_model, 1)
    
    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Causal mask (prevent looking ahead)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer
        x = self.transformer(x, mask=mask)
        
        # Output from last position
        return self.output_head(x[:, -1, :])
```

**4.3.3 Temporal Fusion Transformer (TFT)**

```python
# Using pytorch-forecasting library
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

def create_tft_model(training_data: TimeSeriesDataSet):
    model = TemporalFusionTransformer.from_dataset(
        training_data,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,  # 7 quantiles
        loss=QuantileLoss(),
        log_interval=10,
    )
    return model
```

**4.3.4 Modern Architectures Comparison**

| Model | Strengths | Weaknesses | Best For |
|-------|-----------|------------|----------|
| LSTM | Long-term memory, stable | Sequential training | Single series |
| GRU | Faster than LSTM | Less capacity | Quick experiments |
| Transformer | Parallel, attention viz | Quadratic complexity | Rich features |
| TFT | Interpretable, multi-horizon | Complex setup | Production |
| PatchTST | Channel independence | New, less tested | Multivariate |
| TimesFM | Zero-shot capability | Large model | Quick baseline |

---

### 4.4 Volatility Forecasting

**4.4.1 GARCH Family + Deep Learning**

```python
class GARCH_LSTM(nn.Module):
    """
    Hybrid: GARCH for baseline + LSTM for residual dynamics
    """
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        # GARCH(1,1) parameters (learnable)
        self.omega = nn.Parameter(torch.tensor(0.0001))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))
        
        # LSTM for residuals
        self.lstm = nn.LSTM(input_size + 1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x, returns):
        batch, seq_len, _ = x.shape
        
        # GARCH volatility
        sigma2 = torch.zeros(batch, seq_len)
        sigma2[:, 0] = self.omega / (1 - self.alpha - self.beta)  # Unconditional
        
        for t in range(1, seq_len):
            sigma2[:, t] = (self.omega + 
                           self.alpha * returns[:, t-1]**2 + 
                           self.beta * sigma2[:, t-1])
        
        # Concatenate GARCH vol with features
        garch_vol = sigma2.unsqueeze(-1).sqrt()
        x_combined = torch.cat([x, garch_vol], dim=-1)
        
        # LSTM refinement
        lstm_out, _ = self.lstm(x_combined)
        vol_adjustment = self.output(lstm_out[:, -1, :])
        
        return garch_vol[:, -1, :] + vol_adjustment
```

**4.4.2 Realized Volatility Prediction**

HAR-RV model + Deep Learning:
```
RV_{t+h} = β₀ + β_d·RV_t + β_w·RV_t^{(week)} + β_m·RV_t^{(month)} + ε
```

Deep HAR:
```python
class DeepHAR(nn.Module):
    def __init__(self, exog_size=0):
        super().__init__()
        
        # HAR features: daily, weekly, monthly RV
        self.har_weights = nn.Linear(3 + exog_size, 1, bias=True)
        
        # Nonlinear component
        self.nonlinear = nn.Sequential(
            nn.Linear(3 + exog_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        self.mix = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, rv_daily, rv_weekly, rv_monthly, exog=None):
        har_features = torch.stack([rv_daily, rv_weekly, rv_monthly], dim=-1)
        if exog is not None:
            har_features = torch.cat([har_features, exog], dim=-1)
        
        linear = self.har_weights(har_features)
        nonlinear = self.nonlinear(har_features)
        
        return torch.sigmoid(self.mix) * linear + (1 - torch.sigmoid(self.mix)) * nonlinear
```

---

### 4.5 Reinforcement Learning for Trading

**4.5.1 Environment Design**

```python
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        commission: float = 0.001,
        max_position: float = 1.0,
    ):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position = max_position
        
        # Action: position target [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: price features + portfolio state
        n_features = len(data.columns) - 1  # Exclude close
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features + 3,),  # features + [balance, position, unrealized_pnl]
            dtype=np.float32
        )
        
        self.reset()
    
    def step(self, action):
        target_position = action[0] * self.max_position
        
        # Execute trade
        trade_size = target_position - self.position
        trade_cost = abs(trade_size) * self.current_price * self.commission
        
        # Update state
        self.position = target_position
        self.balance -= trade_cost
        
        # Move to next timestep
        self.current_step += 1
        old_price = self.current_price
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate reward
        price_change = (self.current_price - old_price) / old_price
        reward = self.position * price_change - trade_cost / self.initial_balance
        
        # Risk-adjusted reward (optional)
        # reward = reward - 0.5 * self.gamma * reward**2
        
        done = self.current_step >= len(self.data) - 1
        
        return self._get_obs(), reward, done, False, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_price = self.data.iloc[0]['close']
        return self._get_obs(), {}
    
    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        features = row.drop('close').values.astype(np.float32)
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.position,
            self.position * (self.current_price - self.data.iloc[0]['close']) / self.initial_balance
        ], dtype=np.float32)
        return np.concatenate([features, portfolio_state])
```

**4.5.2 PPO для Trading**

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

def train_trading_agent(env, eval_env, total_timesteps=1_000_000):
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./ppo_trading/",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )
    
    return model
```

**4.5.3 DQN для Market Making**

```python
class DQNMarketMaker(nn.Module):
    """
    Action space: discrete spread levels
    """
    def __init__(self, state_dim, n_spread_levels=21):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_spread_levels),  # Q-value for each spread
        )
    
    def forward(self, state):
        return self.net(state)
    
    def select_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_spread_levels)
        
        with torch.no_grad():
            q_values = self(state)
            return q_values.argmax().item()
```

---

### 4.6 Model Deployment and Inference

**4.6.1 ONNX Export for Rust Inference**

```python
# Export trained model to ONNX
import torch.onnx

def export_to_onnx(model, sample_input, output_path):
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
```

**4.6.2 Rust Inference with ONNX Runtime**

```rust
use ort::{Environment, Session, Value};
use ndarray::Array2;

pub struct ModelInference {
    session: Session,
}

impl ModelInference {
    pub fn new(model_path: &str) -> Result<Self, ort::Error> {
        let environment = Environment::builder()
            .with_name("trading_inference")
            .build()?;
        
        let session = Session::builder(&environment)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?  // Single thread for deterministic latency
            .with_model_from_file(model_path)?;
        
        Ok(Self { session })
    }
    
    pub fn predict(&self, features: &Array2<f32>) -> Result<f32, ort::Error> {
        let input = Value::from_array(&self.session.allocator(), features)?;
        let outputs = self.session.run(vec![input])?;
        
        let output: Vec<f32> = outputs[0].try_extract()?.view().to_owned().into_raw_vec();
        Ok(output[0])
    }
}
```

**4.6.3 Feature Pipeline in Rust**

```rust
pub struct FeaturePipeline {
    technical: TechnicalIndicators,
    microstructure: MicrostructureFeatures,
    lookback: usize,
    feature_buffer: VecDeque<Vec<f64>>,
}

impl FeaturePipeline {
    pub fn update(&mut self, tick: &MarketTick) -> Option<Vec<f64>> {
        // Compute features
        let features = vec![
            tick.price,
            tick.volume,
            self.technical.rsi_last(),
            self.microstructure.order_imbalance(tick.bid_vol, tick.ask_vol),
            // ... more features
        ];
        
        self.feature_buffer.push_back(features);
        if self.feature_buffer.len() > self.lookback {
            self.feature_buffer.pop_front();
        }
        
        if self.feature_buffer.len() == self.lookback {
            Some(self.flatten_buffer())
        } else {
            None
        }
    }
}
```

---

## Инструментарий

### Python
```python
# requirements.txt
torch>=2.0
pytorch-lightning>=2.0
pytorch-forecasting>=1.0
transformers>=4.30
stable-baselines3>=2.0
gymnasium>=0.29
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
optuna>=3.3  # Hyperparameter tuning
wandb>=0.15  # Experiment tracking
onnx>=1.14
onnxruntime>=1.15
```

### Rust
```toml
[dependencies]
ort = "1.16"              # ONNX Runtime
ndarray = "0.15"
tokio = "1.0"
serde = { version = "1.0", features = ["derive"] }
```

---

## Практические задания

### Задание 4.1: Walk-Forward Backtesting Framework
**Цель:** Построить правильный backtesting pipeline
- Purged K-fold cross-validation
- Embargo periods
- Transaction cost modeling
- Performance metrics: Sharpe, Sortino, Calmar

### Задание 4.2: LSTM vs Transformer Comparison
**Цель:** Сравнить архитектуры на crypto data
- Dataset: BTC-USDT 1-minute data, 2 years
- Target: next 5-minute return direction
- Metrics: accuracy, precision, recall, F1, profit factor

### Задание 4.3: Volatility Forecasting Pipeline
**Цель:** Production-ready volatility prediction
- HAR baseline
- Deep HAR
- Integration with options pricing (implied vs realized)

### Задание 4.4: RL Trading Agent
**Цель:** Train and deploy RL agent
- Environment with realistic transaction costs
- PPO vs SAC comparison
- Paper trading evaluation

### Задание 4.5: End-to-End ML System
**Цель:** Full pipeline from data to trading
- Feature engineering in Rust
- Model training in Python
- ONNX export and Rust inference
- Latency benchmarks

---

## Критерии оценки

1. **No lookahead bias**: строгое разделение train/test по времени
2. **Realistic assumptions**: transaction costs, slippage, latency
3. **Statistical significance**: proper hypothesis testing
4. **Reproducibility**: seeded experiments, version control

---

## Связи с другими главами

| Глава | Связь |
|-------|-------|
| 01-stochastic-calculus | Feature engineering из SDE |
| 02-market-microstructure | LOB features, order flow prediction |
| 03-portfolio-optimization | ML-based return/cov prediction |
| 05-low-latency-systems | Real-time inference architecture |

---

## Рекомендуемая литература

### Книги
1. López de Prado M. "Advances in Financial Machine Learning" (2018)
2. López de Prado M. "Machine Learning for Asset Managers" (2020)
3. Jansen S. "Machine Learning for Algorithmic Trading" (2020)

### Курсы и ресурсы
1. Coursera: "Machine Learning for Trading" — Georgia Tech
2. Oxford Man Institute publications
3. arXiv q-fin.CP (Computational Finance)

---

## Заметки по написанию

- Акцент на pitfalls: lookahead, overfitting, data snooping
- Показать failure cases: когда ML не работает лучше простых базелайнов
- Production considerations: monitoring, retraining, drift detection
- Включить interview-style вопросы для self-assessment
- Real code, real data, real results (не toy examples)
