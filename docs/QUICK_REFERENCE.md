# 快速参考

> 本文档提供项目的关键信息快速查阅

## 核心概念速查

### 三网络架构

```
生成器（12因子）
├─ 输入：噪声 + 记忆 + 指导（苏醒后）
├─ 输出：64x64 RGB世界图像
└─ 目标：创造能抵抗黑潮的世界

判别器（黑潮）
├─ 输入：64x64 RGB世界图像
├─ 输出：侵蚀强度 [0-1]
└─ 目标：识别并摧毁脆弱的世界

观察者（德谬歌）
├─ 沉睡期：观察并预测世界演化
├─ 苏醒期：生成指导信号
└─ 目标：帮助12因子对抗黑潮
```

### 训练阶段

**沉睡期（1-9999世代）**
- 12因子 vs 黑潮标准对抗
- 德谬歌观察并自监督学习
- 德谬歌不影响对抗

**苏醒期（10000+世代）**
- 德谬歌生成指导信号
- 12因子接受指导调整策略
- 观察世界稳定性变化

## 12因子速查表

| 序号 | 名称 | 原动力 | 命途 | 类别 |
|-----|------|--------|------|------|
| 1 | 刻法勒 | 憎恨 | 毁灭 | 创生 |
| 2 | 尼卡多利 | 约束 | 巡猎 | 灾厄 |
| 3 | 塞纳托斯 | 平和 | 均衡 | 灾厄 |
| 4 | 欧洛尼斯 | 哀怜 | 记忆 | 命运 |
| 5 | 塔兰顿 | 支配 | 秩序 | 命运 |
| 6 | 雅努斯 | 同理 | 同谐 | 命运 |
| 7 | 吉奥里亚 | 求存 | 不朽 | 支柱 |
| 8 | 法吉娜 | 自否 | 虚无 | 支柱 |
| 9 | 艾格勒 | 奉献 | 存护 | 支柱 |
| 10 | 瑟希斯 | 批判 | 智识 | 创生 |
| 11 | 墨涅塔 | 节制 | 纯美 | 创生 |
| 12 | 扎格列斯 | 渴望 | 欢愉 | 灾厄 |

## 关键指标

**黑潮侵蚀度**
- 定义：判别器输出的平均值
- 范围：0-1
- 解释：越高越危险

**世界稳定性**
- 定义：世界图像方差的倒数
- 范围：0-∞
- 解释：越高越稳定

**德谬歌影响力**
- 定义：指导信号的L1范数
- 范围：0-∞
- 解释：德谬歌的作用强度

**预测准确度**（沉睡期）
- 定义：预测与实际的相似度
- 范围：0-1
- 解释：德谬歌学习效果

## 预设事件时间线

```
第1世代      → 实验开始
第1000世代   → 黑潮首次爆发
第5000世代   → 永劫回归迹象
第10000世代  → 🌟 德谬歌苏醒
第15000世代  → 德谬歌影响力50%
第20000世代  → 稳定性超过黑潮
第33550336世代 → 对应游戏剧情
第50000世代  → 实验结束
```

## 网络参数速查

### 生成器
- 噪声维度：128
- 因子隐藏维度：64
- 输出尺寸：64x64x3
- 记忆权重：0.3
- 指导权重：0.3

### 判别器
- 输入尺寸：64x64x3
- Dropout：0.3
- 输出：单一标量

### 德谬歌
- 世界编码维度：512
- 记忆隐藏维度：256
- LSTM层数：2
- 指导维度：64
- 历史长度：10

## 训练参数速查

```yaml
batch_size: 16
learning_rate:
  generator: 0.0002
  discriminator: 0.0002
  demiurge: 0.0001
optimizer: Adam
betas: [0.5, 0.999]
```

## 常用命令

### 训练
```bash
python scripts/train.py --config configs/config.yaml
```

### 可视化
```bash
python scripts/visualize.py --checkpoint path/to/checkpoint.pth
```

### 分析
```bash
python scripts/analyze.py --metrics path/to/metrics.csv
```

### 测试
```bash
pytest tests/
```

## 文件路径速查

```
配置文件：configs/config.yaml
检查点：experiments/checkpoints/
日志：experiments/logs/
指标：experiments/metrics/
TensorBoard：experiments/tensorboard/
```

## 颜色编码（可视化）

**世界图像**：
- 明亮区域：文明繁荣
- 暗淡区域：衰败
- 黑色区域：黑潮侵蚀
- 金色光晕：德谬歌影响（苏醒后）

**12因子**：
- 每个因子有独特颜色
- 活跃度用柱状图表示

## 调试技巧

### 检查网络输出形状
```python
print(generator(noise).shape)  # 应该是 [batch, 3, 64, 64]
print(discriminator(world).shape)  # 应该是 [batch, 1]
```

### 检查梯度
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### 监控损失
```python
if g_loss > 10 or d_loss > 10:
    print("⚠️ 损失异常，可能需要调整学习率")
```

## 常见问题

**Q: 训练不收敛怎么办？**
A: 降低学习率，增加batch size，或使用梯度裁剪

**Q: 内存不足怎么办？**
A: 减小batch size，降低图像分辨率，或使用梯度累积

**Q: 德谬歌学习效果不好？**
A: 增加历史长度，调整预测器网络大小，或延后苏醒时间

**Q: 可视化卡顿？**
A: 增加更新间隔，降低FPS，或使用更简单的可视化

## 性能优化

### GPU优化
```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
```

### 数据加载优化
```python
# 使用多进程
num_workers = 4
pin_memory = True
```

### 内存优化
```python
# 及时清理
del intermediate_results
torch.cuda.empty_cache()
```

## 实验建议

### 初步实验（1000世代）
- 目标：验证网络能正常训练
- 时间：约10-30分钟
- 观察：损失是否下降

### 中期实验（10000世代）
- 目标：验证德谬歌苏醒机制
- 时间：约1-3小时
- 观察：苏醒前后的变化

### 完整实验（50000世代）
- 目标：完整运行并收集数据
- 时间：约5-15小时
- 观察：长期趋势和现象

## 直播建议

### 技术准备
- OBS设置：1920x1080, 30fps
- 场景：游戏捕获 + 摄像头（可选）
- 音频：背景音乐 + 麦克风

### 内容准备
- 准备PPT介绍设定
- 准备FAQ文档
- 设置聊天机器人

### 互动建议
- 定期解释当前状态
- 回答观众问题
- 投票决定某些参数

## 资源链接

- 游戏Wiki：https://zh.moegirl.org.cn/翁法罗斯
- PyTorch文档：https://pytorch.org/docs/
- GAN教程：https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

