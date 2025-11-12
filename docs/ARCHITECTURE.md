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

#### 整体结构

```
Generator (12 Factors)
├── Input Layer
│   ├── Noise: [batch, 128]
│   ├── Memory: [batch, 3, 64, 64] (optional)
│   └── Guidance: [batch, 12, 64] (optional, 苏醒后)
│
├── 12 Factor Networks (并行，各有差异)
│   ├── Factor 1-12: 各自独立的小型网络
│   └── 输出: 12 × [batch, 1, 64, 64]
│
├── Fusion Layer (三层架构)
│   ├── Layer 1: 因子分组融合
│   ├── Layer 2: 空间竞争 (Attention)
│   └── Layer 3: RGB解码
│
└── Post-processing (后处理)
    ├── 黑潮侵蚀效果叠加
    └── 德谬歌影响效果叠加 (苏醒后)
```

#### 12因子网络详细设计

**基础结构**（所有因子共享）：
```
Input: [batch, 128] (noise)
    ↓
FC: [128 → 256]
    ↓
ReLU
    ↓
FC: [256 → 512]
    ↓
ReLU
    ↓
FC: [512 → 64*64]
    ↓
Reshape: [batch, 1, 64, 64]
    ↓
因子特定激活函数
    ↓
Output: [batch, 1, 64, 64]
```

**12因子的差异化设计**：

| 因子 | 激活函数 | 正则化 | 特殊机制 |
|------|----------|--------|----------|
| 雅努斯 | Sigmoid | L2 | 平滑约束 |
| 塔兰顿 | ReLU | L1 | 稀疏约束 |
| 欧洛尼斯 | Tanh | - | 强记忆权重 |
| 吉奥里亚 | ReLU6 | 最小化方差 | - |
| 法吉娜 | LeakyReLU | L1 | - |
| 艾格勒 | Softplus | - | 边界清晰约束 |
| 刻法勒 | ReLU6 | - | 均匀分布约束 |
| 瑟希斯 | GELU | - | 高频特征约束 |
| 墨涅塔 | Tanh | 负L2（过拟合） | - |
| 尼卡多利 | Hardtanh | - | 二值化约束 |
| 塞纳托斯 | Swish | - | 周期性约束 |
| 扎格列斯 | 双峰激活 | - | 双峰分布约束 |

**自定义激活函数**：

```python
# 双峰激活（扎格列斯专用）
class BimodalActivation(nn.Module):
    def forward(self, x):
        # 将输出推向 -1 或 +1
        return torch.tanh(3 * x)  # 放大后tanh，接近双峰
```

#### 融合层详细设计

**Layer 1: 因子分组融合**

```
12个因子 [batch, 12, 64, 64]
    ↓
按泰坦分组：
├── 命运组 (雅努斯、塔兰顿、欧洛尼斯) [batch, 3, 64, 64]
├── 支柱组 (吉奥里亚、法吉娜、艾格勒) [batch, 3, 64, 64]
├── 创生组 (刻法勒、瑟希斯、墨涅塔) [batch, 3, 64, 64]
└── 灾厄组 (尼卡多利、塞纳托斯、扎格列斯) [batch, 3, 64, 64]
    ↓
每组通过小型卷积网络融合：
    Conv2d: [3 → 64, kernel=3, padding=1]
    BatchNorm + ReLU
    Conv2d: [64 → 64, kernel=3, padding=1]
    BatchNorm
    ↓
输出: 4 × [batch, 64, 64, 64]
```

**Layer 2: 空间竞争（Attention）**

```
4个特征图 [batch, 64, 64, 64] × 4
    ↓
为每组计算竞争力分数：
    Conv2d: [64 → 32, kernel=1]
    ReLU
    Conv2d: [32 → 1, kernel=1]
    ↓
得到: 4 × [batch, 1, 64, 64]
    ↓
Softmax归一化（dim=1）
    ↓
竞争权重: [batch, 4, 64, 64]
    ↓
加权融合:
    weighted_sum = Σ(weight_i × feature_i)
    ↓
输出: [batch, 64, 64, 64]
```

**Layer 3: RGB解码**

```
融合特征 [batch, 64, 64, 64]
    ↓
Conv2d: [64 → 32, kernel=3, padding=1]
    ↓
ReLU
    ↓
Conv2d: [32 → 3, kernel=1]
    ↓
Tanh (值域 [-1, 1])
    ↓
输出: [batch, 3, 64, 64] (RGB世界图像)
```

#### 后处理效果叠加

**黑潮侵蚀效果**：

```python
# 1. 判别器计算侵蚀图
erosion_map = Discriminator(world_rgb)  # [batch, 1, 64, 64]

# 2. 叠加黑潮效果
world_rgb = world_rgb * (1 - erosion_map)
```

**德谬歌影响效果**（苏醒后）：

```python
if demiurge_awakened:
    # 1. 德谬歌生成影响力图
    guidance_strength = Demiurge.get_influence_map(...)  # [batch, 1, 64, 64]

    # 2. 叠加金色光晕
    golden_color = torch.tensor([1.0, 0.84, 0.0])  # RGB
    world_rgb = world_rgb + guidance_strength * golden_color.view(1, 3, 1, 1)

    # 3. 裁剪
    world_rgb = torch.clamp(world_rgb, -1, 1)
```

#### 记忆融合

```python
if memory_exists:
    world_new = generated_world  # [batch, 3, 64, 64]
    world_old = previous_world   # [batch, 3, 64, 64]

    # 加权融合
    output = 0.7 * world_new + 0.3 * world_old
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

### 2.3 观察者架构（德谬歌）

#### 设计理念

**德谬歌的本质**：
- 原动力：爱（游戏设定）
- 作用：通过"记忆"的力量影响世界
- 学习内容：记忆12因子的协作模式，理解因果关系

**学习目标**：
1. 记忆哪些因子组合能产生稳定的世界
2. 预测世界的演化（理解因果）
3. 识别永劫回归（循环模式）
4. 苏醒后提醒12因子调整

#### 网络结构

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
│   ├── FC: [256*8*8 → 256]
│   └── Output: [batch, 256] (世界特征)
│
├── Factor Encoder
│   ├── Input: [batch, 12] (12因子活跃度)
│   ├── FC: [12 → 64]
│   └── Output: [batch, 64] (因子特征)
│
├── Memory Module (LSTM)
│   ├── Input: [batch, 320] (256世界 + 64因子)
│   ├── LSTM: hidden_size=512, num_layers=2
│   └── Output: [batch, 512] (记忆状态)
│
├── Predictor (沉睡期训练)
│   ├── Input: [batch, 512]
│   ├── FC: [512 → 256]
│   ├── ReLU
│   ├── FC: [256 → 3*64*64]
│   ├── Tanh
│   └── Output: [batch, 3, 64, 64] (预测的下一个世界)
│
└── Advisor (苏醒期使用)
    ├── Input: [batch, 512]
    ├── FC: [512 → 128]
    ├── ReLU
    ├── FC: [128 → 12]
    ├── Tanh
    └── Output: [batch, 12] (对12因子的调整建议)
```

#### 指导机制

**指导信号形式**：
- 输出：`[batch, 12]` 全局调整向量
- 每个因子一个标量
- 正值：建议增强该因子
- 负值：建议抑制该因子
- 值域：[-1, 1]

**指导信号注入**：乘法调制

```python
# 1. 德谬歌生成调整建议
guidance = Demiurge.generate_guidance(world, factor_activities)  # [batch, 12]

# 2. 计算调制系数
guidance_strength = 0.5  # 可配置
adjustment = 1.0 + guidance_strength * guidance  # 范围 [0.5, 1.5]

# 3. 调制因子输出
for i in range(12):
    factor_outputs[i] = factor_outputs[i] * adjustment[:, i].view(-1, 1, 1, 1)
```

**效果**：
- `guidance[i] = +1.0` → `adjustment[i] = 1.5` → 因子增强50%
- `guidance[i] = -1.0` → `adjustment[i] = 0.5` → 因子抑制50%
- `guidance[i] = 0.0` → `adjustment[i] = 1.0` → 因子不变

**指导强度控制**：渐进式增强

```python
# 刚苏醒时影响较小，逐渐增强
progress = (generation - awakening_generation) / 5000
guidance_strength = 0.1 + 0.4 * min(progress, 1.0)  # 从0.1增强到0.5
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

#### 指标计算总览

为了支持事件系统，需要计算以下指标：

| 指标 | 计算方法 | 值域 | 用途 |
|------|---------|------|------|
| 黑潮侵蚀度 | 判别器输出平均值 | [0, 1] | 事件触发、可视化 |
| 世界稳定性 | 1 - 归一化方差 | [0, 1] | 事件触发、可视化 |
| 12因子活跃度 | L1范数归一化 | [0, 1]×12 | 德谬歌输入、可视化 |
| 德谬歌预测准确度 | SSIM + 滚动平均 | [0, 1] | 苏醒条件判定 |
| 德谬歌影响力 | 归一化L1 × 强度 | [0, 1] | 影响力监控 |
| 模式相似度 | SSIM（窗口20） | [0, 1] | 永劫回归检测 |
| 趋势 | EWMA差分 | R | 趋势分析 |

#### 详细实现

```python
from torchmetrics.image import StructuralSimilarityIndexMeasure
import numpy as np

# ===== 1. 黑潮侵蚀度 =====
def calculate_erosion(discriminator_output):
    """
    Args:
        discriminator_output: [batch, 1, 64, 64]
    Returns:
        erosion_score: [0, 1]
        erosion_map: [batch, 1, 64, 64]
    """
    erosion_score = discriminator_output.mean()
    erosion_map = discriminator_output
    return erosion_score.item(), erosion_map


# ===== 2. 世界稳定性 =====
def calculate_stability(world):
    """
    Args:
        world: [batch, 3, 64, 64]，值域 [-1, 1]
    Returns:
        stability: [0, 1]
    """
    # RGB三通道方差
    var_r = world[:, 0, :, :].var()
    var_g = world[:, 1, :, :].var()
    var_b = world[:, 2, :, :].var()
    variance = (var_r + var_g + var_b) / 3

    # 归一化（假设方差范围 [0, 2]）
    normalized_var = torch.clamp(variance, 0, 2) / 2
    stability = 1 - normalized_var

    return stability.item()


# ===== 3. 12因子活跃度 =====
def calculate_factor_activities(factor_outputs):
    """
    Args:
        factor_outputs: List of [batch, 1, 64, 64] × 12
    Returns:
        activities: [12]，值域 [0, 1]
    """
    activities = []
    for factor_output in factor_outputs:
        activity = torch.abs(factor_output).mean()
        activities.append(activity.item())

    activities = torch.tensor(activities)
    max_activity = activities.max()
    if max_activity > 0:
        activities = activities / max_activity

    return activities


# ===== 4. 德谬歌预测准确度 =====
class DemiurgeAccuracyTracker:
    def __init__(self, window_size=100):
        self.ssim = StructuralSimilarityIndexMeasure()
        self.accuracy_history = []
        self.window_size = window_size

    def update(self, predicted_world, actual_world):
        ssim_score = self.ssim(predicted_world, actual_world)
        accuracy = (ssim_score + 1) / 2  # 归一化到 [0, 1]

        self.accuracy_history.append(accuracy.item())
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)

        return accuracy.item()

    def get_smoothed_accuracy(self):
        if len(self.accuracy_history) == 0:
            return 0.0
        return np.mean(self.accuracy_history)


# ===== 5. 德谬歌影响力 =====
def calculate_demiurge_influence(guidance, guidance_strength):
    """
    Args:
        guidance: [batch, 12]，值域 [-1, 1]
        guidance_strength: [0.1, 0.5]
    Returns:
        influence: [0, 1]
    """
    avg_guidance = torch.abs(guidance).mean()
    actual_influence = avg_guidance * guidance_strength
    normalized_influence = actual_influence / 0.5  # 最大值归一化

    return normalized_influence.item()


# ===== 6. 永劫回归检测 =====
def detect_eternal_return(world_history, window_size=20, threshold=0.95):
    """
    Args:
        world_history: 历史世界图像列表
        window_size: 比较窗口（默认20）
        threshold: 相似度阈值（默认0.95）
    Returns:
        is_eternal_return: bool
        avg_similarity: float
    """
    if len(world_history) < 2 * window_size:
        return False, 0.0

    ssim = StructuralSimilarityIndexMeasure()
    recent_worlds = world_history[-window_size:]
    previous_worlds = world_history[-2*window_size:-window_size]

    similarities = []
    for recent in recent_worlds:
        for previous in previous_worlds:
            sim = ssim(recent, previous)
            similarities.append(sim.item())

    avg_similarity = np.mean(similarities)
    is_eternal_return = avg_similarity > threshold

    return is_eternal_return, avg_similarity


# ===== 7. 趋势计算 =====
def calculate_trend(metric_history, window=100, alpha=0.1):
    """
    使用EWMA计算趋势

    Args:
        metric_history: 历史指标值
        window: 窗口大小（默认100）
        alpha: 平滑系数（默认0.1）
    Returns:
        trend: 趋势值（正=上升，负=下降）
    """
    if len(metric_history) < window:
        return 0.0

    recent = metric_history[-window:]

    # 计算EWMA
    ewma = [recent[0]]
    for value in recent[1:]:
        ewma.append(alpha * value + (1 - alpha) * ewma[-1])

    # 趋势 = (最近值 - 窗口开始值) / 窗口大小
    trend = (ewma[-1] - ewma[0]) / window

    return trend


# ===== 8. 综合指标计算 =====
def calculate_metrics(generation, world, erosion, factor_outputs,
                     guidance, guidance_strength,
                     demiurge_accuracy_tracker, history):
    """计算所有指标用于事件触发"""

    # 基础指标
    current_erosion, erosion_map = calculate_erosion(erosion)
    current_stability = calculate_stability(world)
    factor_activities = calculate_factor_activities(factor_outputs)

    # 德谬歌指标
    demiurge_influence = 0.0
    demiurge_accuracy = 0.0
    if guidance is not None:
        demiurge_influence = calculate_demiurge_influence(guidance, guidance_strength)
    if demiurge_accuracy_tracker is not None:
        demiurge_accuracy = demiurge_accuracy_tracker.get_smoothed_accuracy()

    # 永劫回归检测
    is_eternal_return, pattern_similarity = detect_eternal_return(
        history.get('world_states', [])
    )

    metrics = {
        # 基础指标
        'generation': generation,
        'erosion': current_erosion,
        'stability': current_stability,
        'factor_activities': factor_activities.tolist(),

        # 德谬歌指标
        'demiurge_accuracy': demiurge_accuracy,
        'demiurge_influence': demiurge_influence,

        # 模式检测
        'pattern_similarity': pattern_similarity,
        'is_eternal_return': is_eternal_return,

        # 趋势指标（基于最近100个世代）
        'erosion_trend': calculate_trend(history.get('erosion', [])[-100:]),
        'stability_trend': calculate_trend(history.get('stability', [])[-100:]),

        # 历史最值
        'erosion_max_so_far': max(history.get('erosion', []) + [current_erosion]),
        'stability_max_ever': max(history.get('stability', []) + [current_stability]),

        # 历史对比
        'prev_pattern_similarity': history.get('pattern_similarity', 0),
        'demiurge_prev_accuracy': history.get('demiurge_accuracy', 0),
        'demiurge_prev_influence': history.get('demiurge_influence', 0),
    }

    return metrics
```

#### 使用示例

```python
# 在训练循环中
history = {
    'erosion': [],
    'stability': [],
    'world_states': [],
    'pattern_similarity': 0,
    'demiurge_accuracy': 0,
    'demiurge_influence': 0,
}

demiurge_accuracy_tracker = DemiurgeAccuracyTracker(window_size=100)

for generation in range(1, max_generations):
    # ... 训练步骤 ...

    # 计算指标
    metrics = calculate_metrics(
        generation=generation,
        world=world,
        erosion=erosion_output,
        factor_outputs=factor_outputs,
        guidance=guidance if awakened else None,
        guidance_strength=guidance_strength if awakened else 0,
        demiurge_accuracy_tracker=demiurge_accuracy_tracker,
        history=history
    )

    # 更新历史
    history['erosion'].append(metrics['erosion'])
    history['stability'].append(metrics['stability'])
    history['world_states'].append(world.detach().cpu())
    history['pattern_similarity'] = metrics['pattern_similarity']
    history['demiurge_accuracy'] = metrics['demiurge_accuracy']
    history['demiurge_influence'] = metrics['demiurge_influence']

    # 检查事件
    events = event_system.check_all(generation, metrics)
```
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

