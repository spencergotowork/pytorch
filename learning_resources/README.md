# PyTorch 源码学习资源

## 📖 开始学习

**从这里开始**: `docs/学习指南.md`

## 📁 目录结构

```
learning_resources/
├── docs/                      # 学习文档
│   ├── 学习指南.md             # ⭐ 入口:学习计划和路线图
│   ├── 阶段1-快速入门.md       # 架构、DispatchKey、调用链
│   ├── 阶段2-核心机制.md       # Dispatcher、Autograd引擎
│   ├── 阶段3-实践项目.md       # 编译、自定义算子、测试
│   ├── 阶段4-专项深入.md       # 分布式/编译器/CUDA
│   └── 阶段5-贡献代码.md       # 提交PR、code review
│
├── demo/                      # 演示脚本
│   ├── trace_simple.py        # 追踪操作流程
│   ├── visualize_autograd.py  # 可视化计算图
│   └── custom_autograd.py     # 自定义autograd函数
│
└── tools/                     # 工具
    └── explore_pytorch.sh     # 源码探索命令行工具
```

## 🚀 快速开始

```bash
# 1. 阅读学习计划
cat docs/学习指南.md

# 2. 阅读技术文档
cat docs/阶段1-快速入门.md

# 3. 运行demo
cd /opt/learn/ai_infra/pytorch
python learning_resources/demo/trace_simple.py

# 4. 使用探索工具
./learning_resources/tools/explore_pytorch.sh show-arch
```

## 📚 文档说明

### 学习指南.md
- **唯一入口文档**
- 包含完整学习计划
- 时间安排和进度检查

### 阶段1-5文档
- **纯技术内容**
- 按学习阶段组织
- 不包含学习计划

## 🎯 学习路径

```
Week 1-2:  阶段1 (快速入门)
Week 3-5:  阶段2 (核心机制)
Week 6-8:  阶段3 (实践项目)
Week 9-12: 阶段4 (专项深入)
Week 13+:  阶段5 (贡献代码)
```

## 🔧 工具使用

```bash
# 源码探索工具
./tools/explore_pytorch.sh help
./tools/explore_pytorch.sh find-op relu
./tools/explore_pytorch.sh show-arch
```

---

**立即开始**: `cat docs/学习指南.md`
