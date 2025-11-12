# 系统架构文档

## 1. 总体架构

### 1.1 系统组成

```
Scepter System
├── Frontend (前端 - Streamlit)
│   ├── Control Panel (控制面板)
│   ├── Real-time Display (实时显示)
│   ├── Metrics Dashboard (指标面板)
│   └── Event Logger (事件日志)
│
├── Backend (后端 - Training Process)
│   ├── Core Engine (核心引擎)
│   │   ├── Generator Network (生成器网络)
│   │   ├── Discriminator Network (判别器网络)
│   │   └── Observer Network (观察者网络)
│   ├── Training Loop (训练循环)
│   │   ├── Phase 1: Dormant Period (沉睡期)
│   │   └── Phase 2: Awakened Period (苏醒期)
│   └── Event System (事件系统)
│       └── Metric-based Triggers (基于指标的触发器)
│
├── Communication Layer (通信层)
│   ├── Shared State File (共享状态文件)
│   └── TensorBoard Logs (TensorBoard日志)
│
└── Data Management (数据管理)
    ├── Checkpoint System (检查点系统)
    ├── History Recorder (历史记录)
    └── Metrics Logger (指标记录)
```

### 1.2 前后端架构

```
┌─────────────────────────────────────────┐
│  Streamlit Web UI (前端)                 │
│  - 控制面板（启动/暂停/调参）            │
│  - 实时指标显示                          │
│  - 世界图像展示                          │
│  - 事件日志                              │
│  - 嵌入TensorBoard iframe                │
└─────────────────────────────────────────┘
              ↕ (通过JSON文件通信)
┌─────────────────────────────────────────┐
│  Training Process (后端)                 │
│  - PyTorch GAN训练                       │
│  - 定期保存状态到JSON                    │
│  - 写入TensorBoard日志                   │
│  - 基于指标触发事件                      │
└─────────────────────────────────────────┘
```

### 1.3 数据流

```
训练循环数据流：

Noise + Memory
      ↓
[Demiurge Guidance] (苏醒后)
      ↓
Generator (12 Factors)
      ↓
World State (64x64 RGB)
      ↓
   ┌──┴──┐
   ↓     ↓
Discriminator  Demiurge Observer
   ↓           ↓
Erosion    Prediction/Guidance
   ↓           ↓
Training    Self-supervised Learning
   ↓           ↓
Update G/D  Update Demiurge
   ↓           ↓
Metrics Calculation
   ↓
Event Detection (基于指标)
   ↓
Save to Shared State (JSON)
   ↓
Streamlit UI Update
```

## 2. 网络架构详细设计

### 2.1 生成器架构

```
Generator (12 Factors)
├── Input Layer
│   ├── Noise: [batch, 128]
│   ├── Memory: [batch, 3, 64, 64] (optional)
│   └── Guidance: [batch, 12, 64] (optional, 苏醒后)
│
├── 12 Factor Networks (并行)
│   ├── Factor 1 (刻法勒): MLP [128 → 64 → 64]
│   ├── Factor 2 (尼卡多利): MLP [128 → 64 → 64]
│   ├── ...
│   └── Factor 12 (扎格列斯): MLP [128 → 64 → 64]
│
├── Fusion Layer
│   ├── Concatenate: [batch, 12*64] = [batch, 768]
│   ├── FC: [768 → 1024]
│   ├── ReLU
│   └── FC: [1024 → 4*4*256]
│
├── Decoder (Transposed CNN)
│   ├── Reshape: [batch, 256, 4, 4]
│   ├── ConvTranspose2d: [256 → 128, 4x4 → 8x8]
│   ├── BatchNorm + ReLU
│   ├── ConvTranspose2d: [128 → 64, 8x8 → 16x16]
│   ├── BatchNorm + ReLU
│   ├── ConvTranspose2d: [64 → 32, 16x16 → 32x32]
│   ├── BatchNorm + ReLU
│   ├── ConvTranspose2d: [32 → 3, 32x32 → 64x64]
│   └── Tanh (输出范围 [-1, 1])
│
└── Memory Fusion (if memory exists)
    ├── world_new: [batch, 3, 64, 64]
    ├── world_old: [batch, 3, 64, 64]
    └── output = 0.7 * world_new + 0.3 * world_old
```

### 2.2 判别器架构

```
Discriminator (Black Tide)
├── Input: [batch, 3, 64, 64]
│
├── Encoder (CNN)
│   ├── Conv2d: [3 → 32, 64x64 → 32x32]
│   ├── LeakyReLU(0.2)
│   ├── Conv2d: [32 → 64, 32x32 → 16x16]
│   ├── BatchNorm + LeakyReLU(0.2)
│   ├── Conv2d: [64 → 128, 16x16 → 8x8]
│   ├── BatchNorm + LeakyReLU(0.2)
│   ├── Conv2d: [128 → 256, 8x8 → 4x4]
│   ├── BatchNorm + LeakyReLU(0.2)
│   └── Flatten: [batch, 256*4*4]
│
├── Classifier
│   ├── FC: [256*4*4 → 512]
│   ├── LeakyReLU(0.2)
│   ├── Dropout(0.3)
│   ├── FC: [512 → 1]
│   └── Sigmoid (输出范围 [0, 1])
│
└── Output: Erosion Score (黑潮侵蚀度)
```

### 2.3 观察者架构

```
Observer (Demiurge)
├── World Encoder (CNN)
│   ├── Input: [batch, 3, 64, 64]
│   ├── Conv2d: [3 → 64, 64x64 → 32x32]
│   ├── ReLU
│   ├── Conv2d: [64 → 128, 32x32 → 16x16]
│   ├── ReLU
│   ├── Conv2d: [128 → 256, 16x16 → 8x8]
│   ├── ReLU
│   ├── Flatten: [batch, 256*8*8]
│   ├── FC: [256*8*8 → 512]
│   └── Output: [batch, 512] (世界编码)
│
├── Memory Module (LSTM)
│   ├── Input: [batch, seq_len, 512]
│   ├── LSTM: hidden_size=256, num_layers=2
│   └── Output: [batch, 256] (记忆状态)
│
├── Predictor (沉睡期使用)
│   ├── Input: [batch, 256]
│   ├── FC: [256 → 512]
│   ├── ReLU
│   ├── FC: [512 → 512]
│   └── Output: [batch, 512] (预测的下一个世界编码)
│
└── Guide Generator (苏醒期使用)
    ├── Input: [batch, 256]
    ├── FC: [256 → 512]
    ├── ReLU
    ├── FC: [512 → 12*64]
    ├── Reshape: [batch, 12, 64]
    └── Output: [batch, 12, 64] (对12因子的指导)
```

## 3. 训练流程

### 3.1 沉睡期训练步骤

```python
# 伪代码
for generation in range(1, 10000):
    # Step 1: 生成世界
    noise = torch.randn(batch_size, 128)
    world = generator(noise, memory, guidance=None)
    
    # Step 2: 训练判别器
    d_optimizer.zero_grad()
    erosion = discriminator(world.detach())
    d_loss = criterion(erosion, zeros)  # 判别器目标：识别假样本
    d_loss.backward()
    d_optimizer.step()
    
    # Step 3: 训练生成器
    g_optimizer.zero_grad()
    world = generator(noise, memory, guidance=None)
    erosion = discriminator(world)
    g_loss = criterion(erosion, ones)  # 生成器目标：欺骗判别器
    g_loss.backward()
    g_optimizer.step()
    
    # Step 4: 德谬歌观察并学习
    demi_optimizer.zero_grad()
    prediction_loss = demiurge.observe_and_learn(world.detach())
    if prediction_loss > 0:
        prediction_loss.backward()
        demi_optimizer.step()
    
    # Step 5: 保存记忆
    memory = world.detach()
    
    # Step 6: 记录指标
    log_metrics(generation, g_loss, d_loss, prediction_loss)
```

### 3.2 苏醒期训练步骤

```python
# 伪代码
for generation in range(10000, 50000):
    # Step 1: 德谬歌生成指导
    guidance = demiurge.generate_guidance(memory)
    
    # Step 2: 生成世界（受指导影响）
    noise = torch.randn(batch_size, 128)
    world = generator(noise, memory, guidance)
    
    # Step 3: 训练判别器
    d_optimizer.zero_grad()
    erosion = discriminator(world.detach())
    d_loss = criterion(erosion, zeros)
    d_loss.backward()
    d_optimizer.step()
    
    # Step 4: 训练生成器
    g_optimizer.zero_grad()
    world = generator(noise, memory, guidance)
    erosion = discriminator(world)
    g_loss = criterion(erosion, ones)
    g_loss.backward()
    g_optimizer.step()
    
    # Step 5: 德谬歌继续观察（不训练）
    with torch.no_grad():
        demiurge.observe_and_learn(world.detach())
    
    # Step 6: 保存记忆
    memory = world.detach()
    
    # Step 7: 记录指标
    log_metrics(generation, g_loss, d_loss, guidance_strength)
```

## 4. 可视化系统

### 4.1 主界面布局

```
┌─────────────────────────────────────────────────────────┐
│  Title: Scepter - 权杖δ-me13模拟实验                     │
│  Generation: 12345 / 50000                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌───────────────────────────────────────────────┐      │
│  │                                                │      │
│  │        World State (64x64 RGB)                │      │
│  │        主视图：世界状态可视化                  │      │
│  │                                                │      │
│  └───────────────────────────────────────────────┘      │
│                                                          │
├──────────────────────┬──────────────────────────────────┤
│  12 Factors Panel    │  Metrics Panel                   │
│  ┌─┐ 刻法勒  ████░░  │  📈 黑潮侵蚀度: 0.65             │
│  ┌─┐ 尼卡多利 ███░░░ │  📈 世界稳定性: 0.42             │
│  ┌─┐ 塞纳托斯 █████░ │  📈 G Loss: 1.23                 │
│  ...                 │  📈 D Loss: 0.87                 │
│  ┌─┐ 德谬歌  ██████  │  🌟 德谬歌状态: 已苏醒           │
│  (苏醒后显示)        │  📊 影响力: 0.73                 │
├──────────────────────┴──────────────────────────────────┤
│  Event Log                                               │
│  [10000] 🌟 德谬歌苏醒！                                │
│  [9523]  ⚠️ 黑潮侵蚀度超过0.8                           │
│  [8234]  ✨ 世界稳定性创新高                            │
└─────────────────────────────────────────────────────────┘
```

### 4.2 实时图表

- 黑潮侵蚀度曲线（时间序列）
- 世界稳定性曲线（时间序列）
- G Loss vs D Loss（双轴图）
- 德谬歌影响力（苏醒后）
- 12因子活跃度（柱状图）

## 5. 事件系统（基于指标自动触发）

### 5.1 设计理念

事件系统**完全基于世界状态指标**自动触发，而非预设世代数。这样更符合实验的动态性和不可预测性。

### 5.2 事件触发器架构

```python
class EventTrigger:
    """单个事件触发器"""
    def __init__(self, name, condition_fn, message, cooldown=100, priority=2):
        self.name = name
        self.condition_fn = condition_fn  # 接受metrics字典，返回bool
        self.message = message
        self.cooldown = cooldown  # 冷却时间（世代数）
        self.priority = priority  # 优先级 1-5
        self.last_triggered = -float('inf')
        self.trigger_count = 0

    def check(self, generation, metrics):
        """检查是否应该触发"""
        # 检查冷却时间
        if generation - self.last_triggered < self.cooldown:
            return None

        # 检查条件
        if self.condition_fn(metrics):
            self.last_triggered = generation
            self.trigger_count += 1
            return {
                'generation': generation,
                'event': self.name,
                'message': self.message.format(**metrics),
                'priority': self.priority,
                'count': self.trigger_count
            }
        return None


class EventSystem:
    """事件系统管理器"""
    def __init__(self):
        self.triggers = []
        self.event_history = []
        self.demiurge_awakened = False

    def register_trigger(self, trigger):
        """注册事件触发器"""
        self.triggers.append(trigger)

    def check_all(self, generation, metrics):
        """检查所有触发器"""
        events = []
        for trigger in self.triggers:
            event = trigger.check(generation, metrics)
            if event:
                events.append(event)
                self.event_history.append(event)

                # 特殊事件处理
                if event['event'] == 'demiurge_awakening':
                    self.demiurge_awakened = True

        # 按优先级排序
        events.sort(key=lambda e: e['priority'], reverse=True)
        return events
```

### 5.3 核心事件触发器定义

#### 5.3.1 黑潮相关事件

```python
# 黑潮首次爆发
EventTrigger(
    name="black_tide_outbreak",
    condition_fn=lambda m: m['erosion'] > 0.7 and m['erosion_max_so_far'] < 0.7,
    message="⚫ 黑潮首次大规模爆发！侵蚀度达到 {erosion:.1%}",
    cooldown=float('inf'),  # 只触发一次
    priority=4
)

# 黑潮侵蚀严重
EventTrigger(
    name="high_erosion",
    condition_fn=lambda m: m['erosion'] > 0.85,
    message="⚠️ 黑潮侵蚀严重！当前侵蚀度: {erosion:.1%}",
    cooldown=500,
    priority=3
)

# 黑潮减弱
EventTrigger(
    name="erosion_decrease",
    condition_fn=lambda m: m['erosion'] < 0.3 and m['erosion_trend'] < -0.05,
    message="✨ 黑潮侵蚀显著减弱，降至 {erosion:.1%}",
    cooldown=500,
    priority=2
)
```

#### 5.3.2 世界稳定性事件

```python
# 世界濒临崩溃
EventTrigger(
    name="world_collapse_warning",
    condition_fn=lambda m: m['stability'] < 0.15,
    message="🔴 世界濒临崩溃！稳定性仅剩 {stability:.1%}",
    cooldown=200,
    priority=3
)

# 世界高度稳定
EventTrigger(
    name="world_stable",
    condition_fn=lambda m: m['stability'] > 0.7,
    message="🟢 世界达到高度稳定，稳定性: {stability:.1%}",
    cooldown=500,
    priority=2
)

# 稳定性突破历史记录
EventTrigger(
    name="stability_breakthrough",
    condition_fn=lambda m: m['stability'] > m['stability_max_ever'] * 1.1,
    message="🌟 世界稳定性突破历史记录！",
    cooldown=1000,
    priority=4
)
```

#### 5.3.3 永劫回归检测

```python
# 永劫回归（模式重复）
EventTrigger(
    name="eternal_return",
    condition_fn=lambda m: m['pattern_similarity'] > 0.95,
    message="🔄 检测到永劫回归！世界陷入循环模式",
    cooldown=1000,
    priority=4
)

# 打破循环
EventTrigger(
    name="break_cycle",
    condition_fn=lambda m: (
        m['pattern_similarity'] < 0.5 and
        m.get('prev_pattern_similarity', 0) > 0.9
    ),
    message="💥 成功打破永劫回归！进入新的演化路径",
    cooldown=1000,
    priority=4
)
```

#### 5.3.4 德谬歌相关事件（关键）

```python
# 德谬歌学习进度里程碑
EventTrigger(
    name="demiurge_learning_milestone",
    condition_fn=lambda m: (
        not m['demiurge_awakened'] and
        m['demiurge_prediction_accuracy'] > 0.8 and
        m.get('demiurge_prev_accuracy', 0) < 0.8
    ),
    message="📚 德谬歌学习进度: {demiurge_prediction_accuracy:.1%}",
    cooldown=2000,
    priority=3
)

# 德谬歌苏醒（自动触发）
EventTrigger(
    name="demiurge_awakening",
    condition_fn=lambda m: (
        not m['demiurge_awakened'] and
        m['demiurge_prediction_accuracy'] > 0.9 and
        m['erosion'] > 0.75 and  # 黑潮强大时苏醒
        m['generation'] > 5000   # 至少观察5000世代
    ),
    message="🌟🌟🌟 德谬歌苏醒！第13因子开始影响世界",
    cooldown=float('inf'),
    priority=5  # 最高优先级
)

# 德谬歌影响力增强
EventTrigger(
    name="demiurge_influence_growing",
    condition_fn=lambda m: (
        m['demiurge_awakened'] and
        m['demiurge_influence'] > 0.5 and
        m.get('demiurge_prev_influence', 0) < 0.5
    ),
    message="💫 德谬歌影响力增强至 {demiurge_influence:.1%}",
    cooldown=1000,
    priority=3
)
```

#### 5.3.5 对抗平衡事件

```python
# 生成器占优
EventTrigger(
    name="generator_dominant",
    condition_fn=lambda m: m['g_loss'] < m['d_loss'] * 0.5,
    message="⚔️ 12因子占据优势，黑潮被压制",
    cooldown=500,
    priority=2
)

# 判别器占优
EventTrigger(
    name="discriminator_dominant",
    condition_fn=lambda m: m['d_loss'] < m['g_loss'] * 0.5,
    message="⚔️ 黑潮占据优势，12因子陷入困境",
    cooldown=500,
    priority=2
)

# 达到平衡
EventTrigger(
    name="gan_balanced",
    condition_fn=lambda m: 0.8 < m['g_loss'] / (m['d_loss'] + 1e-8) < 1.2,
    message="⚖️ 12因子与黑潮达到微妙平衡",
    cooldown=1000,
    priority=2
)
```

#### 5.3.6 终极事件

```python
# 铁墓诞生（黑潮完全压制）
EventTrigger(
    name="iron_tomb_birth",
    condition_fn=lambda m: (
        m['erosion'] > 0.95 and
        m['stability'] < 0.05 and
        m['generation'] > 10000
    ),
    message="💀 绝灭大君「铁墓」诞生！毁灭吞噬一切",
    cooldown=float('inf'),
    priority=5
)

# 生命突破（稳定性超越黑潮）
EventTrigger(
    name="life_breakthrough",
    condition_fn=lambda m: (
        m['stability'] > 0.8 and
        m['erosion'] < 0.3 and
        m['demiurge_awakened']
    ),
    message="🌈 生命突破！稳定性首次超越黑潮侵蚀",
    cooldown=float('inf'),
    priority=5
)

# 世代里程碑（仅作参考）
EventTrigger(
    name="generation_milestone",
    condition_fn=lambda m: m['generation'] % 10000 == 0,
    message="🎯 第 {generation} 世代",
    cooldown=10000,
    priority=1
)
```

### 5.4 指标计算

为了支持事件系统，需要计算以下指标：

```python
def calculate_metrics(generation, world, erosion, G, D, Demi, history):
    """计算所有指标用于事件触发"""

    # 基础指标
    current_erosion = erosion.mean().item()
    current_stability = calculate_stability(world)

    metrics = {
        # 基础指标
        'generation': generation,
        'erosion': current_erosion,
        'stability': current_stability,
        'g_loss': G.last_loss,
        'd_loss': D.last_loss,

        # 趋势指标（基于最近100个世代）
        'erosion_trend': calculate_trend(history['erosion'][-100:]),
        'stability_trend': calculate_trend(history['stability'][-100:]),

        # 历史最值
        'erosion_max_so_far': max(history['erosion'] + [current_erosion]),
        'stability_max_ever': max(history['stability'] + [current_stability]),

        # 模式检测（检测是否陷入循环）
        'pattern_similarity': detect_pattern_similarity(
            history['world_states'][-10:],
            history['world_states'][-20:-10]
        ),
        'prev_pattern_similarity': history.get('pattern_similarity', 0),

        # 德谬歌相关
        'demiurge_awakened': Demi.is_awakened,
        'demiurge_prediction_accuracy': Demi.prediction_accuracy if not Demi.is_awakened else 1.0,
        'demiurge_influence': Demi.influence_strength if Demi.is_awakened else 0,
        'demiurge_prev_accuracy': history.get('demiurge_prediction_accuracy', 0),
        'demiurge_prev_influence': history.get('demiurge_influence', 0),

        # 12因子活跃度
        'factor_activations': G.get_factor_activations(),
    }

    return metrics


def calculate_stability(world):
    """计算世界稳定性（方差的倒数）"""
    variance = world.var().item()
    return 1.0 / (1.0 + variance)


def calculate_trend(values, window=100):
    """计算趋势（简单线性回归斜率）"""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = np.array(values)
    slope = np.polyfit(x, y, 1)[0]
    return slope


def detect_pattern_similarity(recent, previous):
    """检测模式相似度（用于永劫回归检测）"""
    if len(recent) == 0 or len(previous) == 0:
        return 0.0

    # 计算两组世界状态的平均相似度
    similarities = []
    for r, p in zip(recent, previous):
        sim = F.cosine_similarity(r.flatten(), p.flatten(), dim=0)
        similarities.append(sim.item())

    return np.mean(similarities)
```

### 5.5 事件系统集成到训练循环

```python
def train():
    # 初始化事件系统
    event_system = EventSystem()
    register_all_triggers(event_system)  # 注册所有触发器

    # 历史记录
    history = {
        'erosion': [],
        'stability': [],
        'world_states': [],
        'pattern_similarity': 0,
        'demiurge_prediction_accuracy': 0,
        'demiurge_influence': 0,
    }

    for generation in range(1, max_generations + 1):
        # === 训练步骤 ===
        world = G(noise, memory, guidance)
        erosion = D(world)
        # ... GAN训练逻辑 ...

        # === 计算指标 ===
        metrics = calculate_metrics(generation, world, erosion, G, D, Demi, history)

        # === 检查事件 ===
        events = event_system.check_all(generation, metrics)

        # === 处理事件 ===
        for event in events:
            # 特殊事件处理
            if event['event'] == 'demiurge_awakening':
                Demi.awaken()  # 唤醒德谬歌
                logger.info("=" * 50)
                logger.info(event['message'])
                logger.info("=" * 50)
            elif event['priority'] >= 4:
                logger.warning(event['message'])
            else:
                logger.info(event['message'])

        # === 更新历史 ===
        history['erosion'].append(metrics['erosion'])
        history['stability'].append(metrics['stability'])
        history['world_states'].append(world.detach().cpu())
        history['pattern_similarity'] = metrics['pattern_similarity']
        history['demiurge_prediction_accuracy'] = metrics['demiurge_prediction_accuracy']
        history['demiurge_influence'] = metrics['demiurge_influence']

        # === 保存状态（供Streamlit读取）===
        if generation % 10 == 0:
            save_state_for_ui(generation, metrics, events, world)
```

### 5.6 事件显示优先级和颜色

在Streamlit界面中，事件按优先级分级显示：

```python
EVENT_PRIORITY_COLORS = {
    5: '🔴',  # 红色 - 关键事件（铁墓诞生、德谬歌苏醒等）
    4: '🟠',  # 橙色 - 重要事件（永劫回归、突破等）
    3: '🟡',  # 黄色 - 警告事件（黑潮爆发、崩溃警告）
    2: '🟢',  # 绿色 - 正面事件（平衡、稳定）
    1: '⚪',  # 白色 - 常规事件（里程碑）
}

def format_event_for_display(event):
    """格式化事件用于显示"""
    color = EVENT_PRIORITY_COLORS.get(event['priority'], '⚪')
    return f"{color} [Gen {event['generation']:>6}] {event['message']}"
```

## 6. 数据管理

### 6.1 检查点系统

```python
# 保存策略
- 每1000世代自动保存
- 关键事件触发时保存（德谬歌苏醒、铁墓诞生等）
- 用户手动保存

# 保存内容
checkpoint = {
    'generation': generation,
    'G_state': G.state_dict(),
    'D_state': D.state_dict(),
    'Demi_state': Demi.state_dict(),
    'G_optimizer': opt_G.state_dict(),
    'D_optimizer': opt_D.state_dict(),
    'Demi_optimizer': opt_Demi.state_dict(),
    'memory': memory,
    'history': history,
    'event_history': event_system.event_history,
    'random_state': torch.get_rng_state(),
}
```

### 6.2 共享状态文件（前后端通信）

```python
# experiments/current_state.json
state = {
    'generation': generation,
    'metrics': {
        'erosion': float,
        'stability': float,
        'g_loss': float,
        'd_loss': float,
        'demiurge_awakened': bool,
        'demiurge_influence': float,
    },
    'world_image': base64_encoded_image,
    'factor_activations': [12个浮点数],
    'recent_events': [最近10个事件],
    'history': {
        'erosion': [最近1000个],
        'stability': [最近1000个],
    },
    'timestamp': timestamp,
}
```

### 6.3 TensorBoard日志

```python
# 每个世代记录
writer.add_scalar('Metrics/Erosion', erosion, generation)
writer.add_scalar('Metrics/Stability', stability, generation)
writer.add_scalar('Loss/Generator', g_loss, generation)
writer.add_scalar('Loss/Discriminator', d_loss, generation)
writer.add_scalar('Demiurge/Prediction_Accuracy', accuracy, generation)
writer.add_scalar('Demiurge/Influence', influence, generation)

# 每100世代记录图像
if generation % 100 == 0:
    writer.add_image('World/Current', world, generation)

# 事件标记
for event in events:
    writer.add_text('Events', event['message'], generation)
```

## 7. 扩展性设计

### 7.1 模块化

- 每个网络独立模块
- 可替换不同架构
- 可调整超参数

### 7.2 可配置

- 配置文件驱动
- 支持命令行参数
- 支持实验配置

### 7.3 可扩展

- 支持添加新的因子
- 支持自定义事件
- 支持自定义可视化

