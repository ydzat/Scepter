# Scepter 项目设计文档

## 1. 核心概念

### 1.1 游戏设定理解

**翁法罗斯的真相**：
- 翁法罗斯是博识尊的天体神经元——权杖δ-me13中运行的模拟世界
- 权杖原本负责计算"生命的第一因"
- 使用12个计算因子代表12种生命原动力对"熵减"进行证伪

**来古士的修正**：
- 天才俱乐部成员来古士发现权杖后进行修正
- 引入"黑潮"作为对抗机制
- 通过"再创世"机制进行世代更迭
- 目标：培育绝灭大君"铁墓"

**演算过程**：
- 已经历3千多万次轮回
- 12个因子在对抗中不断进化
- 第13因子"德谬歌"被来古士剔除，但在沉睡中载入了12因子的数据

### 1.2 技术映射

我们将这个设定映射为一个**三网络GAN系统**：

```
生成器（Generator）= 12个因子的联合体
├─ 目标：生成"稳定的生命形态/文明"
├─ 输入：随机噪声 + 上一世代的记忆
└─ 输出：世界状态（64x64 RGB图像）

判别器（Discriminator）= 黑潮
├─ 目标：判断并摧毁"不够强韧"的生命形态
├─ 输入：生成器产生的世界状态
└─ 输出：侵蚀强度（0-1，越高越接近崩溃）

观察者（Observer）= 德谬歌
├─ 沉睡期：观察并学习世界演化规律（自监督学习）
├─ 学习目标：预测下一个世界状态
└─ 苏醒期：利用学到的知识指导12因子
```

## 2. 网络架构设计

### 2.1 生成器：12因子联合体

**整体结构**：
```
输入噪声 + 记忆
    ↓
12个因子网络（并行）
    ↓
融合层（三层架构）
    ↓
64x64 RGB世界图像
```

**12个因子设计原则**：
- 相同的基础网络结构
- 通过**不同的激活函数**和**正则化偏好**体现差异
- 因子之间**不设计显式交互**，关系通过行为偏好在训练中自然涌现

**12个因子详细设计**：

#### 命运三泰坦（Fate Titans）

| 因子 | 原动力 | 命途 | 激活函数 | 行为偏好 | 输出特征 |
|------|--------|------|----------|----------|----------|
| 雅努斯（门径） | 同理 | 同谐 | Sigmoid | 平滑过渡 | 连通性 |
| 塔兰顿（律法） | 支配 | 秩序 | ReLU | 结构化 | 规则图案 |
| 欧洛尼斯（岁月） | 哀怜 | 记忆 | Tanh | 历史依赖 | 延续性 |

#### 支柱三泰坦（Pillar Titans）

| 因子 | 原动力 | 命途 | 激活函数 | 行为偏好 | 输出特征 |
|------|--------|------|----------|----------|----------|
| 吉奥里亚（大地） | 求存 | 不朽 | ReLU6 | 稳定性 | 持久不变 |
| 法吉娜（海洋） | 自否 | 虚无 | LeakyReLU | 抑制 | 低值区域 |
| 艾格勒（天空） | 奉献 | 存护 | Softplus | 保护性 | 防护结构 |

#### 创生三泰坦（Creation Titans）

| 因子 | 原动力 | 命途 | 激活函数 | 行为偏好 | 输出特征 |
|------|--------|------|----------|----------|----------|
| 刻法勒（负世） | 憎恨 | 毁灭 | ReLU6 | 持续供能 | 光明庇护 |
| 瑟希斯（理性） | 批判 | 智识 | GELU | 复杂性 | 细节丰富 |
| 墨涅塔（浪漫） | 节制 | 纯美 | Tanh | 编织+过拟合 | 金丝网络 |

#### 灾厄三泰坦（Calamity Titans）

| 因子 | 原动力 | 命途 | 激活函数 | 行为偏好 | 输出特征 |
|------|--------|------|----------|----------|----------|
| 尼卡多利（纷争） | 约束 | 巡猎 | Hardtanh | 边界明确 | 锐利边界 |
| 塞纳托斯（死亡） | 平和 | 均衡 | Swish | 循环重生 | 消解新生 |
| 扎格列斯（诡计） | 渴望 | 欢愉 | 双峰激活 | 极端化 | 大好大坏 |

**设计理念说明**：

1. **刻法勒（创生之神）**：虽然原动力是"憎恨"，但体现为对黑暗的憎恨，转化为持续的光明。ReLU6确保能量充沛但不失控。

2. **塞纳托斯（死亡与新生）**：Swish函数在负值时归零（死亡），正值时平滑上升（新生），体现轮回。

3. **墨涅塔（浪漫的过拟合）**：负L2正则化鼓励记住训练数据的每个细节，体现"编织金丝"般将所有记忆连接成网络。浪漫就是不理性地记住一切美好。

4. **扎格列斯（双面神）**：双峰激活函数输出要么极好（+1）要么极坏（-1），惩罚中间值，体现"诡计与机运"的极端性。

**因子之间的自然关系**：
- 刻法勒（光明）vs 法吉娜（虚无）→ 自然对抗
- 塔兰顿（秩序）vs 扎格列斯（混乱）→ 自然冲突
- 墨涅塔（过拟合）vs 瑟希斯（复杂）→ 互补
- 塞纳托斯（循环）vs 吉奥里亚（稳定）→ 动态平衡

**输出**：每个因子输出 [batch, 1, 64, 64] 的特征图

### 2.2 判别器：黑潮

**结构**：
- 标准CNN判别器
- 输入：64x64 RGB图像
- 输出：空间侵蚀图 `[batch, 1, 64, 64]`，值域 [0, 1]

**输出说明**：
- 判别器输出空间侵蚀图，每个像素表示该位置的侵蚀强度
- 全局侵蚀度 = 空间侵蚀图的平均值
- 保留空间信息用于可视化黑潮效果

**训练目标**：
- 识别生成的世界（标签为0）
- 不断进化，变得更强

### 2.3 观察者：德谬歌

#### 核心理念

**德谬歌的本质**：
- **原动力**：爱（游戏设定）
- **作用**：通过"记忆"的力量影响世界
- **学习内容**：不是学习"如何生成"，而是学习"如何记忆和回望"

**德谬歌学习的是**：
1. **记忆12因子的协作模式**：哪些因子组合能产生稳定的世界
2. **预测世界的演化**：理解因果关系（给定当前状态 → 预测下一状态）
3. **识别永劫回归**：通过记忆识别循环模式
4. **苏醒后的作用**：通过"记忆"提醒12因子调整
   - 不是"教12因子怎么做"
   - 而是"这个模式曾经失败/成功过，调整/保持"

#### 网络结构

```python
class Demiurge(nn.Module):
    def __init__(self):
        # 世界编码器：将世界图像编码为特征
        self.world_encoder = CNN_Encoder(
            input_channels=3,
            output_dim=256
        )

        # 因子活跃度编码器
        self.factor_encoder = nn.Linear(12, 64)

        # 记忆模块：LSTM存储历史
        self.memory = nn.LSTM(
            input_size=320,  # 256(世界) + 64(因子)
            hidden_size=512,
            num_layers=2
        )

        # 预测器（沉睡期训练）
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 64 * 64),
            nn.Tanh()
        )

        # 建议器（苏醒期使用）
        self.advisor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 12),  # 输出12个因子的调整建议
            nn.Tanh()  # 限制到 [-1, 1]
        )
```

#### 指导机制设计

**指导信号形式**：全局调整向量 `[batch, 12]`
- 每个因子一个标量
- 正值：建议增强该因子
- 负值：建议抑制该因子
- 值域：[-1, 1]

**指导信号注入方式**：乘法调制
```python
# 1. 德谬歌生成调整建议
guidance = Demiurge.generate_guidance(world, factor_activities)  # [batch, 12]
guidance = torch.tanh(guidance)  # 限制到 [-1, 1]

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

#### 训练机制

**沉睡期（1-9999世代）**：
- 观察每个世代的世界状态和12因子活跃度
- 通过自监督学习预测下一个世界
- 损失函数：`MSE(predicted_world, next_world)`
- **不影响**生成器和判别器的训练（使用 `.detach()`）

**苏醒期（10000+世代）**：
- 停止预测器的训练
- 利用学到的记忆生成指导信号
- 指导信号通过乘法调制影响12因子
- 继续观察但不再训练（只更新记忆状态）

#### 可视化

**德谬歌状态面板**：
```
┌────────────────────────────────────────────────┐
│  德谬歌状态                                     │
├────────────────────────────────────────────────┤
│  状态: 🌟 已苏醒 (第10523世代)                 │
│  影响强度: ████████░░ 0.35 (目标: 0.5)        │
├────────────────────────────────────────────────┤
│  指导建议:                                      │
│  刻法勒   ████████ +0.8 (增强)                │
│  尼卡多利 ██ -0.2 (抑制)                       │
│  塞纳托斯 ███ +0.3 (增强)                      │
│  ...                                           │
└────────────────────────────────────────────────┘
```

## 3. 训练流程

### 3.1 沉睡期训练循环

```python
for generation in range(1, 10000):
    # 1. 生成噪声
    noise = random_noise()  # [batch, noise_dim]

    # 2. 12因子各自生成特征图
    factor_outputs = [factor_i(noise) for i in range(12)]  # List of [batch, 1, 64, 64]

    # 3. 融合层生成世界
    world_new = FusionLayer(factor_outputs)  # [batch, 3, 64, 64]

    # 4. 记忆融合（与上一世代混合）
    if previous_world is not None:
        world = 0.7 * world_new + 0.3 * previous_world
    else:
        world = world_new

    # 5. 计算12因子活跃度
    factor_activities = torch.tensor([
        factor.abs().mean().item() for factor in factor_outputs
    ])  # [12]

    # 6. 判别器判断
    erosion = Discriminator(world)  # [batch, 1, 64, 64]

    # 7. 德谬歌观察并学习（自编码器式重构）
    demiurge_loss, demiurge_metrics = demiurge_loss(
        Demiurge,
        world.detach(),
        factor_activities
    )

    # 8. 训练德谬歌（不影响G/D）
    demiurge_optimizer.zero_grad()
    demiurge_loss.backward()
    demiurge_optimizer.step()

    # 10. 标准GAN训练
    D_loss = discriminator_loss_wgan_gp(D, real_world, world)
    G_loss_adv = generator_loss_wgan(D, world)
    G_loss_reg = factor_regularization_loss(factor_outputs, factor_configs)
    G_loss = G_loss_adv + 0.01 * G_loss_reg

    # 11. 优化器步骤
    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    # 12. 保存记忆
    previous_world = world.detach()
```

### 3.2 苏醒期训练循环

```python
for generation in range(10000, 50000):
    # 1. 德谬歌生成指导（基于上一世代）
    guidance = Demiurge.generate_guidance(
        previous_world,
        previous_factor_activities
    )  # [batch, 12]

    # 2. 计算指导强度（渐进式）
    progress = (generation - 10000) / 5000
    guidance_strength = 0.1 + 0.4 * min(progress, 1.0)  # 0.1 → 0.5

    # 3. 生成噪声
    noise = random_noise()

    # 4. 12因子各自生成
    factor_outputs = [factor_i(noise) for i in range(12)]

    # 5. 应用德谬歌的指导（乘法调制）
    adjustment = 1.0 + guidance_strength * guidance  # [batch, 12]
    for i in range(12):
        factor_outputs[i] = factor_outputs[i] * adjustment[:, i].view(-1, 1, 1, 1)

    # 6. 融合层生成世界
    world_new = FusionLayer(factor_outputs)  # [batch, 3, 64, 64]

    # 7. 记忆融合
    if previous_world is not None:
        world = 0.7 * world_new + 0.3 * previous_world
    else:
        world = world_new

    # 8. 计算12因子活跃度
    factor_activities = torch.tensor([
        factor.abs().mean().item() for factor in factor_outputs
    ])  # [12]

    # 9. 判别器判断
    erosion = Discriminator(world)  # [batch, 1, 64, 64]

    # 10. 继续GAN训练
    D_loss = discriminator_loss_wgan_gp(D, real_world, world)
    G_loss_adv = generator_loss_wgan(D, world)
    G_loss_reg = factor_regularization_loss(factor_outputs, factor_configs)
    G_loss = G_loss_adv + 0.01 * G_loss_reg

    # 11. 优化器步骤
    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    # 12. 德谬歌继续观察（不训练）
    with torch.no_grad():
        Demiurge.observe_and_learn(world.detach(), factor_activities)

    # 13. 保存记忆
    previous_world = world.detach()
    previous_factor_activities = factor_activities
```

### 3.3 损失函数详细定义

#### 判别器损失

**标准GAN损失**（Binary Cross Entropy）：

```python
def discriminator_loss(D, real_world, fake_world):
    """
    Args:
        D: 判别器
        real_world: 真实世界图像（如果有）或目标分布
        fake_world: 生成器生成的世界

    Returns:
        loss: 判别器损失
    """
    # 真实世界的侵蚀图（目标：全0，表示无侵蚀）
    real_erosion = D(real_world)
    real_target = torch.zeros_like(real_erosion)

    # 生成世界的侵蚀图（目标：全1，表示完全侵蚀）
    fake_erosion = D(fake_world.detach())
    fake_target = torch.ones_like(fake_erosion)

    # BCE损失
    loss_real = F.binary_cross_entropy(real_erosion, real_target)
    loss_fake = F.binary_cross_entropy(fake_erosion, fake_target)

    loss = (loss_real + loss_fake) / 2

    return loss
```

**注意**：
- 在本项目中，我们没有"真实世界"数据集
- 使用**自适应真实数据生成策略**自动解决这个问题

#### 自适应真实数据生成策略

**核心思路**：使用生成器自己生成的"高质量世界"作为真实数据

```python
class AdaptiveRealDataBuffer:
    """自适应真实数据缓冲区"""

    def __init__(self, buffer_size=1000, quality_threshold=0.7):
        """
        Args:
            buffer_size: 缓冲区大小
            quality_threshold: 质量阈值（稳定性 > threshold 才加入缓冲区）
        """
        self.buffer = []
        self.buffer_size = buffer_size
        self.quality_threshold = quality_threshold

    def add(self, world, stability):
        """添加高质量世界到缓冲区"""
        if stability > self.quality_threshold:
            self.buffer.append(world.detach().cpu())

            # 保持缓冲区大小
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)  # 移除最旧的

    def sample(self, batch_size, device):
        """从缓冲区采样"""
        if len(self.buffer) == 0:
            # 缓冲区为空时，返回None（使用初始策略）
            return None

        # 随机采样
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        samples = [self.buffer[i] for i in indices]
        return torch.stack(samples).to(device)

    def is_ready(self):
        """缓冲区是否准备好"""
        return len(self.buffer) >= self.buffer_size // 2  # 至少有一半满
```

**自适应判别器损失**：

```python
def adaptive_discriminator_loss(D, fake_world, stability, real_buffer,
                                use_wgan=True, lambda_gp=10):
    """
    自适应判别器损失

    Args:
        D: 判别器
        fake_world: 生成的世界
        stability: 当前世界的稳定性
        real_buffer: 真实数据缓冲区
        use_wgan: 是否使用WGAN-GP（默认True）
        lambda_gp: 梯度惩罚系数

    Returns:
        loss: 判别器损失
    """
    batch_size = fake_world.size(0)
    device = fake_world.device

    # 1. 尝试从缓冲区获取真实数据
    real_world = real_buffer.sample(batch_size, device)

    # 2. 如果缓冲区为空，使用初始策略
    if real_world is None:
        # 初始阶段：使用简单的目标分布
        # 创建"理想世界"：均匀分布的彩色图案
        real_world = create_ideal_world_template(batch_size, device)

    # 3. 计算损失
    if use_wgan:
        loss = discriminator_loss_wgan_gp(D, real_world, fake_world, lambda_gp)
    else:
        loss = discriminator_loss_bce(D, real_world, fake_world)

    # 4. 将高质量的生成世界加入缓冲区
    real_buffer.add(fake_world, stability)

    return loss


def create_ideal_world_template(batch_size, device):
    """
    创建理想世界模板（初始阶段使用）

    Returns:
        ideal_world: [batch, 3, 64, 64]
    """
    # 策略1：均匀分布的彩色图案
    ideal_world = torch.rand(batch_size, 3, 64, 64, device=device) * 2 - 1

    # 策略2：添加一些结构（渐变）
    x = torch.linspace(-1, 1, 64, device=device)
    y = torch.linspace(-1, 1, 64, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # 创建渐变模式
    gradient = torch.stack([xx, yy, (xx + yy) / 2], dim=0)  # [3, 64, 64]
    gradient = gradient.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch, 3, 64, 64]

    # 混合随机和渐变
    ideal_world = 0.7 * ideal_world + 0.3 * gradient

    return ideal_world
```

**使用方式**：

```python
# 初始化缓冲区
real_buffer = AdaptiveRealDataBuffer(
    buffer_size=1000,
    quality_threshold=0.7
)

# 训练循环中
for generation in range(1, max_generations):
    # ... 生成世界 ...

    # 计算稳定性
    stability = calculate_stability(world)

    # 自适应判别器训练
    D_loss = adaptive_discriminator_loss(
        D, world, stability, real_buffer,
        use_wgan=True, lambda_gp=10
    )

    # ... 其他训练步骤 ...
```

**优势**：
- ✅ 自动解决真实数据缺失问题
- ✅ 初期使用理想模板，后期使用自己生成的高质量数据
- ✅ 形成正反馈：生成器越好 → 缓冲区质量越高 → 判别器越强 → 生成器更好
- ✅ 无需手工设计真实数据

**推荐：WGAN-GP损失（保持不变）**

```python
def discriminator_loss_wgan_gp(D, real_world, fake_world, lambda_gp=10):
    """
    WGAN-GP损失（Wasserstein GAN with Gradient Penalty）

    Args:
        D: 判别器
        real_world: 真实世界（或理想模板）
        fake_world: 生成世界
        lambda_gp: 梯度惩罚系数

    Returns:
        loss: 判别器损失
    """
    # Wasserstein距离
    real_erosion = D(real_world).mean()
    fake_erosion = D(fake_world.detach()).mean()

    # 梯度惩罚
    alpha = torch.rand(real_world.size(0), 1, 1, 1).to(real_world.device)
    interpolates = alpha * real_world + (1 - alpha) * fake_world
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    loss = fake_erosion - real_erosion + lambda_gp * gradient_penalty

    return loss
```

#### 生成器损失

```python
def generator_loss(D, fake_world):
    """
    生成器损失：欺骗判别器

    Args:
        D: 判别器
        fake_world: 生成的世界

    Returns:
        loss: 生成器损失
    """
    # 生成器希望判别器输出低侵蚀度（接近0）
    fake_erosion = D(fake_world)
    target = torch.zeros_like(fake_erosion)

    loss = F.binary_cross_entropy(fake_erosion, target)

    return loss
```

**WGAN-GP版本**：

```python
def generator_loss_wgan(D, fake_world):
    """
    WGAN生成器损失
    """
    fake_erosion = D(fake_world).mean()
    loss = -fake_erosion  # 最大化判别器输出（负号转为最小化）

    return loss
```

#### 德谬歌损失（沉睡期）

**核心理念**：德谬歌学习"记忆和重构"，而非"预测未来"

```python
def demiurge_loss(Demiurge, current_world, factor_activities):
    """
    德谬歌的自监督学习损失（自编码器式）

    核心思路：
    - 德谬歌学习如何通过"记忆"重构当前世界
    - 而不是预测下一个世界（那是在学习生成器的随机性）
    - 这更符合"记忆"的设定

    Args:
        Demiurge: 德谬歌网络
        current_world: 当前世界状态 [batch, 3, 64, 64]
        factor_activities: 12因子活跃度 [batch, 12]

    Returns:
        loss: 重构损失
    """
    # 1. 编码：世界 + 因子活跃度 → 记忆特征
    world_features = Demiurge.world_encoder(current_world.detach())  # [batch, 256]
    factor_features = Demiurge.factor_encoder(factor_activities.detach())  # [batch, 64]

    # 2. 记忆模块：LSTM处理
    combined_features = torch.cat([world_features, factor_features], dim=1)  # [batch, 320]
    memory_state, _ = Demiurge.memory(combined_features.unsqueeze(0))  # [1, batch, 512]
    memory_state = memory_state.squeeze(0)  # [batch, 512]

    # 3. 重构：从记忆中重构世界
    reconstructed_world = Demiurge.predictor(memory_state)  # [batch, 12288]
    reconstructed_world = reconstructed_world.view(-1, 3, 64, 64)  # [batch, 3, 64, 64]

    # 4. 重构损失（MSE）
    reconstruction_loss = F.mse_loss(reconstructed_world, current_world.detach())

    # 5. 可选：添加因子活跃度预测损失（辅助任务）
    # 这帮助德谬歌理解"哪些因子导致了当前世界"
    predicted_activities = Demiurge.factor_predictor(memory_state)  # [batch, 12]
    activity_loss = F.mse_loss(predicted_activities, factor_activities.detach())

    # 总损失
    total_loss = reconstruction_loss + 0.1 * activity_loss

    return total_loss, {
        'reconstruction_loss': reconstruction_loss.item(),
        'activity_loss': activity_loss.item()
    }
```

**为什么这样设计**：

1. **重构任务 vs 预测任务**：
   - ❌ 预测下一世界：学习生成器的随机性（无意义）
   - ✅ 重构当前世界：学习世界的本质特征（有意义）

2. **符合"记忆"设定**：
   - 德谬歌通过"记忆"理解世界
   - 重构任务强迫德谬歌记住世界的关键特征
   - 苏醒后，这些记忆用于指导12因子

3. **辅助任务**：
   - 预测因子活跃度帮助德谬歌理解"因果关系"
   - 哪些因子组合产生了当前世界

**网络结构补充**：

```python
class Demiurge(nn.Module):
    def __init__(self):
        # ... 原有结构 ...

        # 新增：因子活跃度预测器（辅助任务）
        self.factor_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 12),
            nn.Sigmoid()  # 输出 [0, 1]
        )
```

#### 12因子的正则化损失

```python
def factor_regularization_loss(factor_outputs, factor_configs):
    """
    12因子的正则化损失

    Args:
        factor_outputs: List of [batch, 1, 64, 64] × 12
        factor_configs: 每个因子的正则化配置

    Returns:
        reg_loss: 正则化损失
    """
    reg_loss = 0.0

    for i, (output, config) in enumerate(zip(factor_outputs, factor_configs)):
        if config['regularization'] == 'L1':
            reg_loss += config['reg_weight'] * output.abs().mean()

        elif config['regularization'] == 'L2':
            reg_loss += config['reg_weight'] * (output ** 2).mean()

        elif config['regularization'] == 'negative_L2':
            # 负L2：鼓励大值（墨涅塔的过拟合）
            reg_loss -= config['reg_weight'] * (output ** 2).mean()

        elif config['regularization'] == 'variance_min':
            # 方差最小化：鼓励均匀（吉奥里亚的稳定）
            reg_loss += config['reg_weight'] * output.var()

        elif config['regularization'] == 'sparsity':
            # 稀疏性：鼓励大部分为0（法吉娜的虚无）
            reg_loss += config['reg_weight'] * (output.abs() > 0.1).float().mean()

    return reg_loss
```

#### 总损失

```python
# 判别器训练步骤
D_loss = discriminator_loss_wgan_gp(D, real_world, fake_world)

# 生成器训练步骤
G_loss_adversarial = generator_loss_wgan(D, fake_world)
G_loss_regularization = factor_regularization_loss(factor_outputs, factor_configs)
G_loss = G_loss_adversarial + 0.01 * G_loss_regularization  # 正则化权重

# 德谬歌训练步骤（沉睡期）
Demi_loss = demiurge_loss(Demiurge, current_world, factor_activities, next_world)
```

## 4. 世界状态表示

### 4.1 融合层架构

**采用方案：三层融合架构 + 因子活跃度可视化**

#### 第一层：因子分组融合

12个因子按照泰坦分组进行融合：

```python
12个因子 [batch, 12, 64, 64]
    ↓
命运三泰坦（雅努斯、塔兰顿、欧洛尼斯）→ 特征图A [batch, 64, 64, 64]
支柱三泰坦（吉奥里亚、法吉娜、艾格勒）→ 特征图B [batch, 64, 64, 64]
创生三泰坦（刻法勒、瑟希斯、墨涅塔）→ 特征图C [batch, 64, 64, 64]
灾厄三泰坦（尼卡多利、塞纳托斯、扎格列斯）→ 特征图D [batch, 64, 64, 64]
```

每组使用小型卷积网络融合：
- 输入：3个因子的特征图 [batch, 3, 64, 64]
- 输出：融合后的特征图 [batch, 64, 64, 64]

#### 第二层：空间竞争（Attention机制）

4个泰坦组通过Attention机制竞争每个像素位置的主导权：

```python
4个特征图 [batch, 64, 64, 64] × 4
    ↓
为每组计算竞争力分数 [batch, 1, 64, 64] × 4
    ↓
Softmax归一化 → 竞争权重 [batch, 4, 64, 64]
    ↓
加权融合 → 融合特征 [batch, 64, 64, 64]
```

**效果**：
- 每个像素位置由"最强"的泰坦组主导
- 图像的不同区域可能呈现不同泰坦组的特征
- 例如：中心区域可能是"创生"主导（刻法勒的光明），边缘是"命运"主导（秩序）

#### 第三层：RGB解码

将融合特征解码为RGB图像：

```python
融合特征 [batch, 64, 64, 64]
    ↓
卷积层 [64 → 32 → 3]
    ↓
Tanh激活 → RGB图像 [batch, 3, 64, 64]，值域[-1, 1]
```

### 4.2 黑潮和德谬歌效果叠加

**后处理方式**：在生成的RGB图像上叠加特殊效果

#### 黑潮侵蚀效果

```python
# 1. 判别器计算侵蚀强度
erosion_map = Discriminator(world_rgb)  # [batch, 1, 64, 64]，值域[0, 1]

# 2. 叠加黑潮效果（侵蚀度高的地方变黑）
world_rgb = world_rgb * (1 - erosion_map)
```

#### 德谬歌影响效果（苏醒后）

```python
if demiurge_awakened:
    # 1. 计算德谬歌的全局影响力（标量）
    demiurge_influence = calculate_demiurge_influence(guidance, guidance_strength)
    # demiurge_influence: [0, 1]

    # 2. 叠加全局金色滤镜
    golden_color = torch.tensor([1.0, 0.84, 0.0])  # 金色 RGB
    world_rgb = world_rgb + demiurge_influence * golden_color.view(1, 3, 1, 1)

    # 3. 裁剪到[-1, 1]
    world_rgb = torch.clamp(world_rgb, -1, 1)
```

**说明**：
- 德谬歌的影响是全局性的，不是空间性的
- 影响力越强，整个世界图像越偏向金色
- 这符合德谬歌"爱"的原动力——普照一切

### 4.3 可视化方案

#### 主图像：64x64 RGB世界状态

**空间分区效果**：
- 不同颜色区域代表不同泰坦组的势力范围
- 例如：
  - 蓝色调区域：命运三泰坦主导（秩序、和谐）
  - 绿色调区域：支柱三泰坦主导（稳定、存在）
  - 橙/黄色调区域：创生三泰坦主导（刻法勒的光明）
  - 红色调区域：灾厄三泰坦主导（纷争、变化）
- 黑色区域：黑潮侵蚀
- 金色光晕：德谬歌影响（苏醒后）

**向观众解释**：
> "你们看到的这个彩色图像，是12个因子共同作用的结果。不同颜色的区域代表不同泰坦组的势力范围。比如现在中心的橙色区域，是创生泰坦刻法勒的光明在主导；而边缘的蓝色，是命运泰坦在维持秩序。底部的黑色是黑潮侵蚀，正在威胁这个世界。"

#### 辅助可视化：12因子活跃度柱状图

**计算方式**：
```python
# 每个因子的全局活跃度（L1范数）
factor_activity = [factor.abs().mean().item() for factor in twelve_factors]
# 输出：12个标量值，范围[0, ∞)
```

**显示方式**：
- Streamlit侧边栏显示12个因子的活跃度柱状图
- 实时更新
- 可以看出哪些因子在当前世代更活跃

**界面布局**：
```
┌─────────────────────────────────────────────────┐
│  世界状态 (第5000世代)                           │
├──────────────────────────┬──────────────────────┤
│                          │  12因子活跃度         │
│   [64x64 RGB图像]        │                      │
│                          │  刻法勒 ████████ 0.8│
│   彩色区域代表不同       │  尼卡多利 ██ 0.2    │
│   泰坦组的势力范围       │  塞纳托斯 ███ 0.3   │
│                          │  欧洛尼斯 █████ 0.5 │
│   黑色=黑潮侵蚀          │  塔兰顿 ██████ 0.6  │
│   金色=德谬歌影响        │  雅努斯 ████ 0.4    │
│                          │  吉奥里亚 ███████ 0.7│
│                          │  法吉娜 █ 0.1       │
│                          │  艾格勒 █████ 0.5   │
│                          │  瑟希斯 ████ 0.4    │
│                          │  墨涅塔 ██████ 0.6  │
│                          │  扎格列斯 ███ 0.3   │
└──────────────────────────┴──────────────────────┘
```

### 4.4 可视化语义

**主图像的视觉含义**：
- 明亮区域：文明繁荣，能量充沛
- 暗淡区域：衰败，能量不足
- 黑色区域：黑潮侵蚀，世界崩溃
- 颜色多样性：世界复杂度
- 图案重复：永劫回归迹象
- 空间分区：不同泰坦组的势力范围

**因子活跃度的含义**：
- 高活跃度：该因子在当前世代影响力强
- 低活跃度：该因子在当前世代影响力弱
- 活跃度变化：反映12因子之间的动态平衡

## 5. 关键指标

### 5.1 实时监控指标

#### 黑潮侵蚀度

**定义**：判别器输出的平均值

**计算方法**：
```python
def calculate_erosion(discriminator_output):
    """
    Args:
        discriminator_output: [batch, 1, 64, 64] 判别器输出

    Returns:
        erosion_score: 全局侵蚀度 [0, 1]
        erosion_map: 空间侵蚀图 [batch, 1, 64, 64]
    """
    # 全局侵蚀度（用于事件触发）
    erosion_score = discriminator_output.mean()

    # 空间侵蚀图（用于可视化）
    erosion_map = discriminator_output

    return erosion_score.item(), erosion_map
```

**值域**：[0, 1]

**解释**：越高表示世界越脆弱，越接近崩溃

---

#### 世界稳定性

**定义**：世界图像方差的倒数（归一化）

**计算方法**：
```python
def calculate_stability(world):
    """
    Args:
        world: [batch, 3, 64, 64] 世界图像，值域 [-1, 1]

    Returns:
        stability: 稳定性 [0, 1]
    """
    # 1. 计算RGB三通道的方差
    var_r = world[:, 0, :, :].var()
    var_g = world[:, 1, :, :].var()
    var_b = world[:, 2, :, :].var()
    variance = (var_r + var_g + var_b) / 3

    # 2. 归一化（假设方差范围 [0, 2]）
    normalized_var = torch.clamp(variance, 0, 2) / 2  # [0, 1]

    # 3. 稳定性 = 1 - 归一化方差
    stability = 1 - normalized_var

    return stability.item()
```

**值域**：[0, 1]

**解释**：
- 接近1：世界高度稳定，图像变化小
- 接近0：世界混乱，图像变化剧烈

---

#### 12因子活跃度

**定义**：每个因子输出的L1范数（归一化）

**计算方法**：
```python
def calculate_factor_activities(factor_outputs):
    """
    Args:
        factor_outputs: List of [batch, 1, 64, 64] × 12

    Returns:
        activities: [12] 每个因子的活跃度 [0, 1]
    """
    activities = []

    for factor_output in factor_outputs:
        # 计算L1范数（平均绝对值）
        activity = torch.abs(factor_output).mean()
        activities.append(activity.item())

    # 转换为tensor并归一化
    activities = torch.tensor(activities)
    max_activity = activities.max()
    if max_activity > 0:
        activities = activities / max_activity

    return activities  # [12]
```

**值域**：[0, 1] × 12

**用途**：
1. 传给德谬歌作为输入特征
2. 在Streamlit中可视化柱状图
3. 分析哪些因子在主导世界生成

---

#### 德谬歌预测准确度

**定义**：预测状态与实际状态的结构相似度（SSIM）

**计算方法**：
```python
from torchmetrics.image import StructuralSimilarityIndexMeasure

class DemiurgeAccuracyTracker:
    def __init__(self, window_size=100):
        self.ssim = StructuralSimilarityIndexMeasure()
        self.accuracy_history = []
        self.window_size = window_size

    def update(self, predicted_world, actual_world):
        """计算单次预测准确度"""
        # 使用SSIM计算相似度
        ssim_score = self.ssim(predicted_world, actual_world)

        # SSIM值域 [-1, 1]，归一化到 [0, 1]
        accuracy = (ssim_score + 1) / 2

        # 添加到历史
        self.accuracy_history.append(accuracy.item())
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history.pop(0)

        return accuracy.item()

    def get_smoothed_accuracy(self):
        """获取平滑后的准确度（滚动平均）"""
        if len(self.accuracy_history) == 0:
            return 0.0
        return np.mean(self.accuracy_history)
```

**值域**：[0, 1]

**解释**：
- 接近1：德谬歌预测非常准确
- 接近0：德谬歌预测完全错误
- 使用滚动平均避免短期波动

---

#### 德谬歌影响力

**定义**：指导信号的归一化L1范数 × 指导强度

**计算方法**：
```python
def calculate_demiurge_influence(guidance, guidance_strength):
    """
    Args:
        guidance: [batch, 12] 指导信号，值域 [-1, 1]
        guidance_strength: 当前指导强度 [0.1, 0.5]

    Returns:
        influence: 影响力 [0, 1]
    """
    # 1. 计算指导信号的平均绝对值
    avg_guidance = torch.abs(guidance).mean()

    # 2. 考虑指导强度
    actual_influence = avg_guidance * guidance_strength

    # 3. 归一化（最大影响 = 1.0 × 0.5 = 0.5）
    normalized_influence = actual_influence / 0.5

    return normalized_influence.item()
```

**值域**：[0, 1]

**解释**：
- 刚苏醒时（guidance_strength=0.1）：影响力较小
- 完全苏醒后（guidance_strength=0.5）：影响力可达最大
- 反映德谬歌的真实影响

---

### 5.2 长期分析指标

#### 永劫回归检测

**定义**：检测世界状态是否进入循环模式

**计算方法**：
```python
from torchmetrics.image import StructuralSimilarityIndexMeasure

def detect_eternal_return(world_history, window_size=20, threshold=0.95):
    """
    Args:
        world_history: 历史世界图像列表
        window_size: 比较窗口大小（默认20）
        threshold: 相似度阈值（默认0.95）

    Returns:
        is_eternal_return: 是否检测到永劫回归
        avg_similarity: 平均相似度
    """
    if len(world_history) < 2 * window_size:
        return False, 0.0

    ssim = StructuralSimilarityIndexMeasure()

    # 最近20个 vs 之前20个
    recent_worlds = world_history[-window_size:]
    previous_worlds = world_history[-2*window_size:-window_size]

    # 计算平均相似度
    similarities = []
    for recent in recent_worlds:
        for previous in previous_worlds:
            sim = ssim(recent, previous)
            similarities.append(sim.item())

    avg_similarity = np.mean(similarities)
    is_eternal_return = avg_similarity > threshold

    return is_eternal_return, avg_similarity
```

**参数**：
- 窗口大小：20个世代
- 阈值：0.95

**解释**：
- 当最近20个世代与之前20个世代高度相似时，判定为永劫回归
- 意味着世界陷入循环，无法进化

---

#### 趋势计算

**定义**：指标的变化趋势（上升/下降）

**计算方法**：
```python
def calculate_trend(metric_history, window=100, alpha=0.1):
    """
    使用指数加权移动平均（EWMA）计算趋势

    Args:
        metric_history: 历史指标值列表
        window: 窗口大小（默认100）
        alpha: 平滑系数（默认0.1，越小越平滑）

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
```

**参数**：
- 窗口大小：100个世代
- 平滑系数：0.1

**用途**：
- 检测侵蚀度上升/下降趋势
- 检测稳定性改善/恶化趋势
- 用于事件触发（如"黑潮减弱"需要下降趋势）

---

#### 铁墓诞生检测

**定义**：黑潮侵蚀度持续极高

**检测方法**：
```python
def detect_tiemu_birth(erosion_history, threshold=0.9, duration=100):
    """
    Args:
        erosion_history: 侵蚀度历史
        threshold: 阈值（默认0.9）
        duration: 持续时间（默认100世代）

    Returns:
        is_tiemu_birth: 是否检测到铁墓诞生
    """
    if len(erosion_history) < duration:
        return False

    recent = erosion_history[-duration:]
    is_tiemu_birth = all(e > threshold for e in recent)

    return is_tiemu_birth
```

**参数**：
- 阈值：0.9
- 持续时间：100个世代

**意义**：判别器完全压制生成器，世界无法存续

## 6. 事件系统（基于指标自动触发）

### 6.1 设计理念

**完全基于世界状态指标自动触发**，而非预设世代数。

### 6.2 事件类型

**黑潮相关**：
- 黑潮首次爆发：侵蚀度 > 0.7（首次）
- 黑潮侵蚀严重：侵蚀度 > 0.85
- 黑潮减弱：侵蚀度 < 0.3 且下降趋势

**世界稳定性**：
- 世界濒临崩溃：稳定性 < 0.15
- 世界高度稳定：稳定性 > 0.7
- 稳定性突破：超越历史最高值

**永劫回归**：
- 检测到循环：模式相似度 > 0.95
- 打破循环：相似度从高降到低

**德谬歌相关**：
- 学习进度里程碑：预测准确度 > 0.8
- 德谬歌苏醒：准确度 > 0.9 且侵蚀度 > 0.75 且世代 > 5000
  - **说明**：世代 > 5000 是为了确保德谬歌有足够的训练时间
  - 这是唯一包含硬编码世代数的事件，其他事件完全基于指标
- 影响力增强：影响力 > 0.5

**终极事件**：
- 铁墓诞生：侵蚀度 > 0.95 且稳定性 < 0.05
- 生命突破：稳定性 > 0.8 且侵蚀度 < 0.3（德谬歌苏醒后）

### 6.3 事件优先级

- 优先级5（🔴）：铁墓诞生、德谬歌苏醒、生命突破
- 优先级4（🟠）：永劫回归、打破循环、稳定性突破
- 优先级3（🟡）：黑潮爆发、崩溃警告
- 优先级2（🟢）：平衡、稳定
- 优先级1（⚪）：世代里程碑

## 7. 超参数配置

### 7.1 网络架构参数

#### 生成器（12因子）

```yaml
generator:
  # 噪声输入
  noise_dim: 128

  # 12因子基础网络
  factor_base_network:
    input_dim: 128
    hidden_dims: [256, 512, 1024]
    output_shape: [1, 64, 64]

  # 融合层
  fusion_layer:
    # 第一层：分组融合
    group_fusion_channels: 64

    # 第二层：空间竞争
    attention_heads: 4

    # 第三层：RGB解码
    decoder_channels: [64, 32, 3]

  # 记忆融合
  memory_fusion:
    new_weight: 0.7
    old_weight: 0.3

  # 12因子正则化配置
  factor_regularization:
    # 命运三泰坦
    - name: "雅努斯"
      index: 0
      regularization: "L2"
      reg_weight: 0.01
      description: "平滑过渡，L2正则化"

    - name: "塔兰顿"
      index: 1
      regularization: "L1"
      reg_weight: 0.01
      description: "结构化，L1稀疏约束"

    - name: "欧洛尼斯"
      index: 2
      regularization: "none"
      reg_weight: 0.0
      description: "历史依赖，无正则化"

    # 支柱三泰坦
    - name: "吉奥里亚"
      index: 3
      regularization: "variance_min"
      reg_weight: 0.02
      description: "稳定性，最小化方差"

    - name: "法吉娜"
      index: 4
      regularization: "sparsity"
      reg_weight: 0.015
      description: "虚无，稀疏性约束"

    - name: "艾格勒"
      index: 5
      regularization: "L2"
      reg_weight: 0.01
      description: "保护性，L2正则化"

    # 创生三泰坦
    - name: "刻法勒"
      index: 6
      regularization: "none"
      reg_weight: 0.0
      description: "光明，无约束"

    - name: "瑟希斯"
      index: 7
      regularization: "L2"
      reg_weight: 0.005
      description: "复杂性，轻度L2"

    - name: "墨涅塔"
      index: 8
      regularization: "negative_L2"
      reg_weight: -0.01
      description: "过拟合，负L2鼓励大值"

    # 灾厄三泰坦
    - name: "尼卡多利"
      index: 9
      regularization: "L1"
      reg_weight: 0.02
      description: "边界明确，L1约束"

    - name: "塞纳托斯"
      index: 10
      regularization: "none"
      reg_weight: 0.0
      description: "循环重生，无约束"

    - name: "扎格列斯"
      index: 11
      regularization: "variance_max"
      reg_weight: -0.015
      description: "极端化，最大化方差"
```

#### 判别器（黑潮）

```yaml
discriminator:
  # CNN架构
  channels: [3, 64, 128, 256, 512]
  kernel_sizes: [4, 4, 4, 4]
  strides: [2, 2, 2, 2]

  # 输出层（保留空间信息）
  output_channels: 1
  output_activation: 'sigmoid'
```

#### 德谬歌（观察者）

```yaml
demiurge:
  # 世界编码器
  world_encoder:
    input_channels: 3
    output_dim: 256

  # 因子编码器
  factor_encoder:
    input_dim: 12
    output_dim: 64

  # 记忆模块（LSTM）
  memory:
    input_size: 320  # 256 + 64
    hidden_size: 512
    num_layers: 2
    dropout: 0.1

  # 预测器（沉睡期）
  predictor:
    hidden_dims: [512, 256]
    output_dim: 12288  # 3 * 64 * 64

  # 建议器（苏醒期）
  advisor:
    hidden_dims: [512, 128]
    output_dim: 12

  # 指导强度
  guidance_strength:
    initial: 0.1
    final: 0.5
    ramp_up_generations: 5000
```

### 7.2 训练参数

```yaml
training:
  # 基础参数
  batch_size: 16
  max_generations: 50000

  # 学习率
  learning_rates:
    generator: 0.0002
    discriminator: 0.0002
    demiurge: 0.0001

  # 优化器
  optimizer: 'Adam'
  adam_betas: [0.5, 0.999]

  # 损失函数
  loss_type: 'wgan-gp'  # 或 'bce'
  gradient_penalty_lambda: 10

  # 正则化
  factor_regularization_weight: 0.01

  # 训练阶段
  dormant_period: [1, 9999]
  awakening_generation: 10000  # 或基于指标自动触发
```

### 7.3 指标计算参数

```yaml
metrics:
  # 世界稳定性
  stability:
    variance_max: 2.0  # 归一化范围

  # 永劫回归检测
  eternal_return:
    window_size: 20
    threshold: 0.95

  # 趋势计算
  trend:
    window_size: 100
    ewma_alpha: 0.1

  # 德谬歌准确度
  demiurge_accuracy:
    rolling_window: 100

  # 铁墓诞生检测
  tiemu_birth:
    erosion_threshold: 0.9
    duration: 100
```

### 7.4 事件触发阈值

```yaml
events:
  # 黑潮相关
  black_tide:
    first_outbreak: 0.7
    severe_erosion: 0.85
    weakening: 0.3

  # 世界稳定性
  stability:
    collapse_warning: 0.15
    highly_stable: 0.7

  # 永劫回归
  eternal_return:
    detection_threshold: 0.95

  # 德谬歌相关
  demiurge:
    learning_milestone: 0.8
    awakening_accuracy: 0.9
    awakening_erosion: 0.75
    awakening_min_generation: 5000
    influence_strong: 0.5

  # 终极事件
  ultimate:
    tiemu_birth_erosion: 0.95
    tiemu_birth_stability: 0.05
    life_breakthrough_stability: 0.8
    life_breakthrough_erosion: 0.3
```

### 7.5 可视化参数

```yaml
visualization:
  # 世界图像
  world_image:
    size: [64, 64]
    value_range: [-1, 1]
    display_size: [512, 512]  # 放大显示

  # 黑潮效果
  black_tide_color: [0, 0, 0]  # 黑色

  # 德谬歌效果
  demiurge_color: [1.0, 0.84, 0.0]  # 金色 RGB

  # 更新频率
  update_frequency: 10  # 每10个世代更新一次

  # TensorBoard
  tensorboard_log_frequency: 1
```

### 7.6 检查点和保存

```yaml
checkpointing:
  # 保存频率
  save_frequency: 1000  # 每1000世代保存一次

  # 保存内容
  save_components:
    - generator
    - discriminator
    - demiurge
    - optimizers
    - metrics_history
    - event_log

  # 保存路径
  checkpoint_dir: 'experiments/checkpoints'

  # 最大保存数量
  max_checkpoints: 10  # 只保留最近10个检查点
```

## 8. 技术栈

### 7.1 核心框架
- PyTorch：深度学习框架
- NumPy：数值计算

### 7.2 可视化
- Matplotlib：静态图表
- Pygame：2D实时渲染
- Plotly：交互式图表（可选）

### 7.3 工具
- TensorBoard：训练监控
- OpenCV：图像处理
- tqdm：进度条

## 8. 硬件需求

### 8.1 最低配置
- CPU：4核心
- RAM：8GB
- GPU：无（CPU可运行）

### 8.2 推荐配置
- CPU：8核心
- RAM：16GB
- GPU：RTX 2060或更高

### 8.3 理想配置
- CPU：16核心
- RAM：32GB
- GPU：RTX 4070或A100

## 9. 前后端架构设计

### 9.1 架构选择：Streamlit + PyTorch

**为什么选择Streamlit？**
- ✅ 开发速度快（几行代码搭建界面）
- ✅ 直播友好（界面美观，自动刷新）
- ✅ 易于调参（滑块、按钮等交互组件）
- ✅ 成熟稳定（大量示例可参考）

**架构图**：
```
┌─────────────────────────────────────────┐
│  Streamlit Web UI (前端)                 │
│  - 控制面板（启动/暂停/调参）            │
│  - 实时指标显示                          │
│  - 世界图像展示                          │
│  - 事件日志                              │
│  - 嵌入TensorBoard                       │
└─────────────────────────────────────────┘
              ↕ (JSON文件通信)
┌─────────────────────────────────────────┐
│  Training Process (后端)                 │
│  - PyTorch GAN训练                       │
│  - 定期保存状态到JSON                    │
│  - 写入TensorBoard日志                   │
│  - 基于指标触发事件                      │
└─────────────────────────────────────────┘
```

### 9.2 前后端通信

**共享状态文件**: `experiments/current_state.json`
- 包含：当前世代、指标、世界图像路径、事件列表、历史数据
- 更新频率：每10个世代

**控制文件**: `experiments/control.json`
- 包含：命令（启动/暂停/保存）、参数调整
- 前端写入，后端读取

### 9.3 Streamlit界面布局

```
┌────────────────────────────────────────────────────────┐
│ 🌟 Scepter - 权杖δ-me13模拟实验                        │
├─────────────┬──────────────────────────────────────────┤
│ ⚙️ 控制面板 │  🌍 世界状态 (64x64图像)                 │
│             │  [实时更新的世界图像]                    │
│ ▶️ 开始训练 │                                          │
│ ⏸️ 暂停训练 ├──────────────┬───────────────────────────┤
│ 💾 保存     │ 📊 关键指标  │ 📜 事件日志              │
│             │ 侵蚀: 0.65   │ 🌟 德谬歌苏醒            │
│ 学习率调整  │ 稳定: 0.42   │ ⚫ 黑潮爆发              │
│ G: [滑块]   │ 世代: 10523  │                          │
│ D: [滑块]   │              │                          │
│             │              │                          │
│ 🌟 手动唤醒 │              │                          │
│   德谬歌    │              │                          │
├─────────────┴──────────────┴───────────────────────────┤
│ 📈 训练曲线（黑潮侵蚀度、世界稳定性、德谬歌影响力）    │
├────────────────────────────────────────────────────────┤
│ ⚡ 12因子活跃度（柱状图）                              │
└────────────────────────────────────────────────────────┘
```

### 9.4 项目结构（简化版）

```
Scepter/
├── app.py                      # Streamlit主界面
├── train.py                    # 训练脚本（独立运行）
│
├── models/
│   ├── __init__.py
│   ├── generator.py           # 生成器（12因子）
│   ├── discriminator.py       # 判别器（黑潮）
│   └── demiurge.py            # 观察者（德谬歌）
│
├── utils/
│   ├── __init__.py
│   ├── config.py              # 配置加载
│   ├── events.py              # 事件系统
│   ├── metrics.py             # 指标计算
│   └── shared_state.py        # 前后端状态共享
│
├── configs/
│   └── config.yaml
│
└── experiments/
    ├── current_state.json     # 当前状态（前后端共享）
    ├── control.json           # 控制命令
    ├── checkpoints/           # 检查点
    ├── outputs/               # 世界图像
    └── tensorboard/           # TensorBoard日志
```

## 10. 下一步工作

### 10.1 设计阶段（剩余）
- [ ] 超参数详细设计
- [ ] Streamlit界面原型图
- [ ] 实验方案设计

### 10.2 实现阶段
- [ ] 最小原型（1天）：基础界面 + 简单GAN
- [ ] 完整功能（2-3天）：三网络 + 事件系统 + 完整UI
- [ ] 优化测试（1天）：性能优化 + 直播测试

