# Scepter 项目文档索引

> 本文档提供所有项目文档的导航和概览

## 📚 文档导航

### 入门文档

**[README.md](../README.md)** - 从这里开始！
- 项目概述和背景
- 核心目标
- 快速开始指南
- 项目状态

### 设计文档

**[DESIGN.md](DESIGN.md)** - 核心设计
- 游戏设定理解
- 技术映射方案
- 网络架构设计
- 训练流程设计
- 世界状态表示
- 关键指标定义

**[ARCHITECTURE.md](ARCHITECTURE.md)** - 系统架构
- 总体架构
- 网络架构详细设计
- 训练流程详细说明
- 可视化系统设计
- 事件系统设计
- 数据管理设计

### 参考文档

**[GAME_LORE.md](GAME_LORE.md)** - 游戏设定参考
- 翁法罗斯的真相
- 权杖δ-me13详解
- 12个因子/泰坦详细信息
- 第13因子德谬歌
- 黑潮机制
- 再创世机制
- 关键事件时间线

**[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速参考
- 核心概念速查
- 12因子速查表
- 关键指标
- 网络参数
- 常用命令
- 调试技巧
- 常见问题

### 项目管理文档

**[TODO.md](TODO.md)** - 待办事项
- 按阶段组织的任务列表
- 优先级标记
- 时间估算

**[PROGRESS.md](PROGRESS.md)** - 进度跟踪
- 已完成工作
- 进行中工作
- 下一步计划
- 关键里程碑
- 风险和挑战

**[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 项目结构
- 目录结构说明
- 文件用途说明
- 使用流程
- 开发规范

### 配置文档

**[config_template.yaml](../configs/config_template.yaml)** - 配置模板
- 实验配置
- 训练配置
- 网络架构配置
- 12因子配置
- 事件系统配置
- 可视化配置

## 📖 阅读顺序建议

### 对于项目新手

1. **[README.md](../README.md)** - 了解项目是什么
2. **[GAME_LORE.md](GAME_LORE.md)** - 理解游戏设定
3. **[DESIGN.md](DESIGN.md)** - 理解技术方案
4. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速查阅关键信息

### 对于开发者

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - 理解系统架构
2. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 了解代码组织
3. **[TODO.md](TODO.md)** - 查看待办任务
4. **[config_template.yaml](../configs/config_template.yaml)** - 了解配置选项

### 对于项目管理

1. **[PROGRESS.md](PROGRESS.md)** - 查看当前进度
2. **[TODO.md](TODO.md)** - 查看任务列表
3. **[DESIGN.md](DESIGN.md)** - 了解技术方案

## 🎯 文档用途速查

| 需求 | 推荐文档 |
|------|---------|
| 了解项目背景 | README.md, GAME_LORE.md |
| 理解技术方案 | DESIGN.md, ARCHITECTURE.md |
| 开始开发 | PROJECT_STRUCTURE.md, config_template.yaml |
| 查找信息 | QUICK_REFERENCE.md |
| 跟踪进度 | PROGRESS.md, TODO.md |
| 调试问题 | QUICK_REFERENCE.md (常见问题部分) |
| 配置系统 | config_template.yaml |

## 📝 文档更新记录

### 2025-11-12
- ✅ 创建所有初始文档
- ✅ 完成项目结构设计
- ✅ 完成核心架构设计
- ✅ 创建配置模板

### 待更新
- [ ] 添加网络架构图
- [ ] 添加训练流程图
- [ ] 添加可视化界面原型图
- [ ] 添加实验结果示例

## 🔗 外部资源

### 游戏相关
- [萌娘百科 - 翁法罗斯](https://zh.moegirl.org.cn/翁法罗斯)
- [萌娘百科 - 崩坏：星穹铁道](https://zh.moegirl.org.cn/崩坏：星穹铁道)

### 技术相关
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch GAN 教程](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [GAN 训练技巧](https://github.com/soumith/ganhacks)

### 可视化相关
- [Pygame 文档](https://www.pygame.org/docs/)
- [Matplotlib 文档](https://matplotlib.org/stable/contents.html)
- [TensorBoard 文档](https://www.tensorflow.org/tensorboard)

## 💡 文档贡献指南

### 添加新文档
1. 在 `docs/` 目录下创建新文件
2. 使用 Markdown 格式
3. 在本文件中添加索引
4. 更新相关文档的链接

### 更新现有文档
1. 保持文档结构清晰
2. 添加更新日期
3. 如有重大变更，更新相关文档

### 文档规范
- 使用清晰的标题层级
- 添加代码示例
- 使用表格和列表提高可读性
- 添加必要的图表（待实现）

## 🎨 文档风格

### Markdown 规范
- 使用 ATX 风格标题（#）
- 代码块指定语言
- 使用表格对齐
- 使用 emoji 增强可读性（适度）

### 术语一致性
- 生成器 = 12因子
- 判别器 = 黑潮
- 观察者 = 德谬歌
- 世代 = generation
- 轮回 = 再创世

## 📊 文档统计

- 总文档数：9个
- Markdown文档：8个
- YAML配置：1个
- 总字数：约15000字
- 最后更新：2025-11-12

## 🚀 下一步

- [ ] 添加架构图（使用 Mermaid 或图片）
- [ ] 添加流程图
- [ ] 添加可视化界面设计图
- [ ] 创建 FAQ 文档
- [ ] 创建实验报告模板

