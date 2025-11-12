# 项目结构说明

## 目录结构

```
Scepter/
├── README.md                   # 项目概述
├── .gitignore                  # Git忽略文件
├── requirements.txt            # Python依赖（待创建）
├── setup.py                    # 安装脚本（待创建）
│
├── configs/                    # 配置文件目录
│   ├── config_template.yaml   # 配置模板
│   └── config.yaml            # 实际配置（待创建，不提交到git）
│
├── docs/                       # 文档目录
│   ├── DESIGN.md              # 设计文档
│   ├── GAME_LORE.md           # 游戏设定参考
│   ├── ARCHITECTURE.md        # 系统架构文档
│   ├── TODO.md                # 待办事项
│   ├── PROGRESS.md            # 进度跟踪
│   ├── PROJECT_STRUCTURE.md   # 项目结构说明（本文件）
│   └── images/                # 文档图片（待创建）
│
├── src/                        # 源代码目录（待创建）
│   ├── __init__.py
│   ├── models/                # 网络模型
│   │   ├── __init__.py
│   │   ├── generator.py       # 生成器（12因子）
│   │   ├── discriminator.py   # 判别器（黑潮）
│   │   └── demiurge.py        # 观察者（德谬歌）
│   │
│   ├── training/              # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py         # 训练器
│   │   ├── losses.py          # 损失函数
│   │   └── metrics.py         # 指标计算
│   │
│   ├── visualization/         # 可视化
│   │   ├── __init__.py
│   │   ├── display.py         # 主显示界面
│   │   ├── plots.py           # 图表绘制
│   │   └── colors.py          # 颜色方案
│   │
│   ├── events/                # 事件系统
│   │   ├── __init__.py
│   │   ├── event_manager.py   # 事件管理器
│   │   └── event_handlers.py  # 事件处理器
│   │
│   ├── utils/                 # 工具函数
│   │   ├── __init__.py
│   │   ├── config.py          # 配置加载
│   │   ├── checkpoint.py      # 检查点管理
│   │   ├── logger.py          # 日志工具
│   │   └── data.py            # 数据处理
│   │
│   └── main.py                # 主程序入口
│
├── experiments/               # 实验数据目录（不提交到git）
│   ├── checkpoints/          # 检查点
│   ├── logs/                 # 日志
│   ├── tensorboard/          # TensorBoard日志
│   ├── metrics/              # 指标数据
│   └── outputs/              # 输出结果
│
├── tests/                     # 测试目录（待创建）
│   ├── __init__.py
│   ├── test_models.py        # 模型测试
│   ├── test_training.py      # 训练测试
│   └── test_utils.py         # 工具测试
│
└── scripts/                   # 脚本目录（待创建）
    ├── train.py              # 训练脚本
    ├── visualize.py          # 可视化脚本
    └── analyze.py            # 分析脚本
```

## 文件说明

### 根目录文件

**README.md**
- 项目概述
- 快速开始指南
- 项目背景和目标

**.gitignore**
- Git忽略规则
- 排除临时文件、日志、检查点等

**requirements.txt**（待创建）
- Python依赖包列表
- 用于 `pip install -r requirements.txt`

**setup.py**（待创建）
- 项目安装脚本
- 用于 `pip install -e .`

### configs/ 配置文件

**config_template.yaml**
- 配置模板
- 包含所有可配置项的说明
- 用户复制为 config.yaml 后修改

**config.yaml**（待创建）
- 实际使用的配置
- 不提交到git（在.gitignore中）
- 每个用户根据自己的环境配置

### docs/ 文档

**DESIGN.md**
- 核心概念设计
- 网络架构设计
- 训练流程设计

**GAME_LORE.md**
- 游戏设定整理
- 12因子详细信息
- 德谬歌设定

**ARCHITECTURE.md**
- 系统架构详细说明
- 数据流设计
- 模块划分

**TODO.md**
- 待办事项列表
- 按阶段组织
- 优先级标记

**PROGRESS.md**
- 项目进度跟踪
- 已完成工作记录
- 下一步计划

**PROJECT_STRUCTURE.md**（本文件）
- 项目结构说明
- 文件和目录用途

### src/ 源代码

#### models/ 网络模型

**generator.py**
- 生成器网络实现
- 12个因子子网络
- 融合层和解码器

**discriminator.py**
- 判别器网络实现
- CNN编码器
- 分类器

**demiurge.py**
- 观察者网络实现
- 编码器、LSTM、预测器、指导器

#### training/ 训练

**trainer.py**
- 主训练循环
- 沉睡期和苏醒期逻辑
- 检查点保存/加载

**losses.py**
- 损失函数定义
- GAN损失
- 预测损失

**metrics.py**
- 指标计算
- 黑潮侵蚀度
- 世界稳定性
- 德谬歌影响力

#### visualization/ 可视化

**display.py**
- 主显示界面
- Pygame或Matplotlib实现
- 实时更新

**plots.py**
- 图表绘制
- 曲线图、柱状图
- TensorBoard集成

**colors.py**
- 颜色方案定义
- 12因子颜色映射
- 黑潮、德谬歌颜色

#### events/ 事件系统

**event_manager.py**
- 事件管理器
- 预设事件和动态事件
- 事件触发逻辑

**event_handlers.py**
- 事件处理函数
- 日志记录
- 界面更新

#### utils/ 工具

**config.py**
- 配置文件加载
- YAML解析
- 配置验证

**checkpoint.py**
- 检查点保存/加载
- 模型状态管理
- 优化器状态管理

**logger.py**
- 日志工具
- 文件日志
- 控制台日志

**data.py**
- 数据处理工具
- 图像转换
- 指标记录

### experiments/ 实验数据

**checkpoints/**
- 保存的模型检查点
- 格式：checkpoint_gen_{generation}.pth

**logs/**
- 文本日志文件
- 格式：experiment_{name}_{timestamp}.log

**tensorboard/**
- TensorBoard日志
- 用于可视化训练过程

**metrics/**
- CSV格式的指标数据
- 用于后续分析

**outputs/**
- 生成的世界图像
- 分析结果图表

### tests/ 测试

**test_models.py**
- 网络模型单元测试
- 输入输出形状测试
- 前向传播测试

**test_training.py**
- 训练循环测试
- 损失计算测试
- 梯度更新测试

**test_utils.py**
- 工具函数测试
- 配置加载测试
- 检查点测试

### scripts/ 脚本

**train.py**
- 训练脚本
- 命令行参数解析
- 调用训练器

**visualize.py**
- 可视化脚本
- 加载检查点
- 生成可视化

**analyze.py**
- 分析脚本
- 读取指标数据
- 生成分析报告

## 使用流程

### 1. 安装依赖
```bash
pip install -r requirements.txt
# 或
pip install -e .
```

### 2. 配置
```bash
cp configs/config_template.yaml configs/config.yaml
# 编辑 config.yaml
```

### 3. 训练
```bash
python scripts/train.py --config configs/config.yaml
# 或
python -m src.main --config configs/config.yaml
```

### 4. 可视化
```bash
python scripts/visualize.py --checkpoint experiments/checkpoints/checkpoint_gen_10000.pth
```

### 5. 分析
```bash
python scripts/analyze.py --metrics experiments/metrics/metrics.csv
```

## 开发规范

### 代码风格
- 遵循 PEP 8
- 使用类型提示
- 添加文档字符串

### 提交规范
- 使用有意义的提交信息
- 一次提交只做一件事
- 提交前运行测试

### 分支策略
- main: 稳定版本
- dev: 开发版本
- feature/*: 功能分支
- fix/*: 修复分支

## 备注

- 所有路径使用相对路径
- 配置文件使用YAML格式
- 日志使用Python logging模块
- 检查点包含完整的训练状态

