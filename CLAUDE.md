# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

你的回答尽量简洁，只回答我的问题，不需要拓展
始终用中文与我交流
如需使用运行环境，首先运行source /opt/learn/ai_infra/env/bin/activate

## 构建命令

### 开发环境设置
```bash
# 使用预编译二进制设置开发环境
make setup-env
# 或带 CUDA 支持
make setup-env-cuda
# 或带 ROCm 支持
make setup-env-rocm
source venv/bin/activate
```

### 从源码构建
```bash
# 安装依赖
pip install -r requirements.txt

# 构建 PyTorch（可编辑安装）
python -m pip install --no-build-isolation -v -e .

# 控制构建的环境变量：
# USE_CUDA=0         - 禁用 CUDA
# USE_CUDNN=0        - 禁用 cuDNN
# USE_ROCM=1         - 启用 ROCm
# USE_XPU=1          - 启用 Intel XPU
# DEBUG=1            - 包含调试符号构建
# MAX_JOBS=N         - 限制并行构建任务数
# CMAKE_FRESH=1      - 强制刷新 CMake 配置

# 清理构建
rm -rf build/
python setup.py clean
```

### 修改后快速重建
```bash
# 仅限 Python 修改：不需要重建（可编辑安装自动生效）
# C++/CUDA 修改：
cd build
ninja torch_cpu  # 或 torch_cuda 等
```

## 测试

### Python 测试
```bash
# 运行完整测试套件
python test/run_test.py

# 运行特定测试文件
python test/test_torch.py

# 运行特定测试类或方法
python test/test_jit.py TestJit.test_Sequential

# 使用 pytest（推荐用于开发）
pytest test/test_nn.py -k Loss -v  # 仅运行名称匹配 "Loss" 的测试
```

### C++ 测试
```bash
# 构建后，C++ 测试二进制文件在 build/bin/
./build/bin/test_jit --gtest_filter=ContainerAliasingTest.MayContainAlias
```

### 代码检查
```bash
make lint        # 运行所有检查工具
make quicklint   # 仅检查修改的文件
make quickfix    # 自动修复检查问题
```

## 高层架构

PyTorch 采用分层架构设计，从下至上分为 4 个主要层级：

### 1. C10 - 核心张量库（`c10/`）
- **调度系统**：基于 `DispatchKey` 的动态调度（CPU、CUDA、HIP、XPU 等）
- **核心类型**：`TensorOptions`、`DeviceType`、`Storage`、执行上下文
- **工具函数**：注册表模式、智能指针、设备管理

### 2. ATen - 张量库（`aten/src/ATen/`）
- **调度器**：管理所有内核的中央算子注册表
- **原生算子**：1000+ 算子，支持 CPU/CUDA/HIP/XPU 实现
- **算子模式**：算子签名的类型系统（`native_functions.yaml`）
- **张量迭代器**：维度迭代、广播的抽象

### 3. Python 集成（`torch/csrc/`）
- **Autograd**：自动微分引擎（`autograd/`）
  - Variables 追踪计算图
  - Engine 通过拓扑排序执行反向传播
- **JIT 编译器**：TorchScript 编译（`jit/`）
  - SSA 形式的 Graph IR（类似 LLVM）
  - 优化通道、代码生成
- **分布式**：C10D 集合操作、RPC、分布式 autograd（`distributed/`）
- **Python 绑定**：C++ ↔ Python 桥接、异常处理

### 4. Python 前端（`torch/`）
- **核心 API**：`torch.nn`、autograd、张量操作
- **编译器**：
  - `_dynamo/` - 字节码捕获
  - `_inductor/` - 代码生成
  - `fx/` - 符号追踪
- **分布式**：高层 API（DDP、FSDP、RPC）

## 关键设计模式

### 1. 基于调度的多态性
```
单个 Dispatcher 单例路由所有操作
    ↓
内核按 (operator_name, DispatchKey) 对注册
    ↓
运行时根据张量的 DispatchKeySet 调度
    ↓
DispatchKey = 后端(CPU/CUDA/etc.) + 功能(Dense/Sparse/Quantized/etc.)
```

### 2. 算子注册流程
```
native_functions.yaml  →  TorchGen（代码生成）
    ↓
后端特定内核（CPU、CUDA 等）
    ↓
Dispatcher 注册（自动或手动）
    ↓
运行时通过 DispatchKeySet 调度
```

### 3. 双层实现模式
许多系统都有 C++ 核心 + Python 包装：
- **Autograd**：`Variable`（C++）+ `THPVariable`（Python）
- **JIT**：`Graph`（C++）+ Module 绑定（Python）
- **调度**：`Dispatcher`（C++）+ `torch.ops`（Python）

### 4. 回退内核链
如果未找到 `(op, DispatchKey)` 的内核：
1. 尝试 Autograd 变体（记录操作）
2. 尝试 Meta 张量（形状推断）
3. 尝试 Python 回退（`__torch_function__`）
4. 尝试 Functionalize（转换原地 → 函数式）

## 执行流程示例

### 前向操作
```
torch.add(x, z)
  ↓
Dispatcher.call(add_op)
  ↓
DispatchKeySet 查找 → AutogradCPU/CUDA 内核
  ↓
保存输入用于反向传播 + 调用原生内核
  ↓
ATen native::add() → C++/CUDA 内核
```

### 反向传播
```
loss.backward()
  ↓
Engine.execute()
  ↓
计算图的拓扑排序
  ↓
对每个 Node：计算 w.r.t. 输入的梯度
  ↓
在叶子变量中累积梯度
```

### JIT 编译
```
@torch.jit.script
  ↓
解析 TorchScript → AST → Graph（SSA 形式）
  ↓
优化通道（DCE、CSE、融合）
  ↓
GraphExecutor 针对输入形状/类型特殊化
  ↓
代码生成 → LLVM IR / CUDA 代码
  ↓
缓存编译函数
```

## 算子注册与调度机制

### 注册流程图
```
native_functions.yaml（算子定义）
    ↓
TorchGen（代码生成）
    ↓
后端特定内核（CPU、CUDA 等）
    ↓
Dispatcher::singleton().def() 和 .impl()
    ↓
运行时通过 DispatchKeySet 查找
```

### DispatchKey 分解
**DispatchKey** 编码：
1. 后端组件（CPU、CUDA 等）
2. 功能（Dense、Quantized、Sparse、Autograd）

调用算子时：
1. 输入张量确定 DispatchKeySet
2. Dispatcher 查找该键的注册内核
3. 如果未找到，尝试回退层级
4. Autograd 内核包装并在图中记录操作

### 自定义算子注册
**位置**：`aten/src/ATen/core/op_registration/`

两条路径：
1. **native_functions.yaml** - 用于公共 API 算子
   - 自动生成 C++ 绑定、Python 绑定、autograd 支持
   - 推荐用于内置算子

2. **自定义算子 API** - 用于扩展算子
   ```cpp
   static auto registry = torch::RegisterOperators()
       .op("namespace::op_name", ...)
       .kernel<KernelFunc>(CPU());
   ```
   - 无自动生成的 C++ API（通过 `torch.ops.namespace.op_name` 访问）
   - 需要手动实现 autograd

## Python 集成（torch.csrc）

### Python 和 C++ 之间的桥接
**位置**：`torch/csrc/`

- **Module.cpp** - 主要 Python 绑定入口点
- **Exceptions.h/cpp** - Python↔C++ 异常映射
- **PyInterpreter.cpp** - 从 C++ 调用 Python 函数

### 类型映射模式
对于每个主要数据类型（Variable、Function、Module）：
- **C++ 类型**（`variable.h`）- 包含核心逻辑，无状态
- **Python 类型**（`python_variable.h`、THPVariable）- 围绕 `shared_ptr<C++ 类型>` 的 Python 包装
- **C++ 实现**通过指针访问底层 C++ 对象

示例：Variable（autograd 张量）
- C++：`variable.h` 中的 `Variable`
- Python：`python_variable.h` 中的 `THPVariable` - 包装 `shared_ptr<Variable>`
- 在 `python_variable.cpp` 中的访问器通过指针访问底层 Variable

### GIL 管理
- **关键要求**：在调用 Python API 前用 `pybind11::gil_scoped_acquire` 获取 GIL
- Python 集成在 `csrc/` 中，C++ 逻辑在 `aten/`、`c10/` 中 - 清晰分离

## Autograd（自动微分）

### 架构
**位置**：`torch/csrc/autograd/`

- **Node**（`function.h`）- 反向操作的基类
- **Function**（`function.h`）- 用户定义的自定义反向
- **PyNode** - 桥接 C++ Node 与 Python Function 实现
- **Variable**（`variable.h`、`variable.cpp`）- 带梯度追踪的张量
  - 包含 `requires_grad`、`grad_fn`、反向图引用

- **图执行**（`engine.cpp`、`engine.h`）
  - 计算图的拓扑排序
  - 通过 Node::apply() 递归反向传播
  - 线程安全执行，工作队列管理

### 梯度计算流
```
前向：Tensor 输入 → 原生算子 → 用 grad_fn 记录的磁带
反向：torch.autograd.backward(loss) → Engine → 拓扑排序 →
      每个 Node::apply() 计算梯度 → 在叶子变量中累积
```

### 自定义 Autograd
```cpp
class MyFunction : public torch::autograd::Function<MyFunction> {
    static Tensor forward(ctx, input) { ... }
    static std::vector<Tensor> backward(ctx, grad_output) { ... }
};
```

## JIT 编译

### 架构概览
**位置**：`torch/csrc/jit/`

TorchScript 执行流程：

#### 1. **前端** - Python → IR 转换
- **Tracer** - 在执行期间捕获算子调用
- **Script** - 解析 Python 子集 TorchScript 代码
- **Parser** - 词法/语法分析产生 AST
- **IR Emitter** - AST → SSA Graph

#### 2. **中间表示**（`ir/ir.h`）
- **Graph** - 顶层容器（类似 LLVM::Function）
- **Block** - 顺序指令（基本块）
- **Node** - 单个算子（如 `aten::add`、`prim::Constant`）
- **Value** - Node 间的数据流（SSA 形式）
- 文本表示：
  ```
  graph(%0 : Tensor, %1 : Tensor):
    %2 : Tensor = aten::add(%0, %1, %2)
    return (%2)
  ```

#### 3. **Module 系统**（`api/module.h`）
- **Module** - 带参数、缓冲区、属性、方法的 TorchScript module
- **Method** - module 中的命名函数
- **Parameter** - 可学习张量槽位
- 镜像 Python 的 `nn.Module`

#### 4. **优化通道**（`passes/`）
- 前导数：常量折叠、死代码消除
- 导数保留：公共子表达式消除、循环优化
- 后导数：融合、目标特定优化
- 通过别名分析处理可变性

#### 5. **执行**
- **解释器** - Graph 的栈式执行
- **Graph Executor** - 为输入形状/类型特殊化图，缓存执行器
- **代码生成** - JIT 编译到 LLVM 或硬件特定代码

#### 6. **序列化** - 保存/加载 TorchScript modules

### Module 表示
```
Module {
  parameters: {weight: Tensor, bias: Tensor}
  buffers: {running_mean: Tensor, running_var: Tensor}
  sub_modules: {layer1: Module, layer2: Module}
  attributes: {config: Dict}
  methods: {forward: Method, backward: Method}
}

Method {
  schema: (Tensor x, int batch_size) -> Tensor
  member_inputs: [weight, bias]  // 访问的参数
  graph: Graph  // SSA 表示
  graph_executor: GraphExecutor
}
```

## 后端组织

### 后端结构
**位置**：`torch/backends/`、`c10/`

每个后端有：
1. **核心实现**
   - CPU 内核在 `aten/src/ATen/native/`（从 native_functions.yaml 生成）
   - CUDA 内核在 `aten/src/ATen/cuda/`
   - HIP 内核在 `aten/src/ATen/hip/`
   - XPU 内核在 `c10/xpu/`、`aten/src/ATen/xpu/`

2. **设备管理**
   - Device guards：自动设备上下文切换
   - 内存分配器：CUDA 的 CachingAllocator、其他的自定义分配器
   - Streams/Events：异步操作追踪

3. **后端特定功能**
   - CUDA：CUDAStream、CUDAGuard、CUDAGraphs 支持
   - HIP：ROCm 等效物
   - MPS：Metal Performance Shaders（Apple）
   - XPU：Intel 可扩展 GPU 支持
   - MTIA：Meta AI 张量加速器

### 支持的后端（DispatchKey）
- **CPU** - 默认 CPU 实现
- **CUDA** - NVIDIA GPU via CUDA
- **HIP** - AMD GPU via ROCm
- **XLA** - XLA JIT 编译（实验性）
- **MPS** - Apple Metal Performance Shaders
- **XPU** - Intel 数据中心 GPU
- **Lazy** - 懒张量评估
- **Meta** - 形状推断无计算
- **PrivateUse1/2/3** - 自定义后端钩子
- **Quantized** - 量化张量操作
- **Sparse** - 稀疏张量操作
- **NestedTensor** - 变长张量维度

## 分布式训练架构

### 高层组件
**位置**：`torch/distributed/`、`torch/csrc/distributed/`

#### 1. **集合通信**（`c10d`、`distributed_c10d.py`）
- **后端**：NCCL、Gloo、MPI、UCC
- **操作**：all_reduce、all_gather、reduce_scatter、broadcast、点对点
- **设备网格**：多 GPU/多节点设置的抽象拓扑
- **函数式集合**：现代异步 API，带 futures

#### 2. **分布式 Autograd**（`torch/csrc/distributed/autograd/`）
- **RRef（远程引用）** - 远程张量的引用
- **DistributedAutograd** - 记录跨进程的函数调用
- **反向引擎** - 合并本地和分布式梯度
- 启用通过 RPC 调用的无缝反向传播

#### 3. **RPC**（`torch/csrc/distributed/rpc/`、`torch/distributed/rpc/`）
- **TensorPipe** - 传输抽象（TCP、CUDA IPC 等）
- **RPC Agent** - 请求/响应管理
- **性能分析** - 远程分析器聚合
- 通过自动序列化的 RPC 函数调用

#### 4. **数据/模型并行模式**
- **DDP（DistributedDataParallel）** - 每个反向后梯度同步
- **FSDP（FullyShardedDataParallel）** - 内存高效的模型并行
- **Tensor 并行** - 层级模型分片（通过 `_shard`、`tensor/`）
- **Pipeline 并行**（`pipelining/`）

#### 5. **状态 Dict 管理**（`_state_dict_utils.py`）
- 检查点/恢复支持
- 跨进程分布式检查点

#### 6. **弹性训练**（`elastic/`）
- 动态工作进程添加/移除
- 容错的 rendezvous 机制

## 编译器栈

### Inductor（Python 代码生成）
**位置**：`torch/_inductor/`、`torch/csrc/inductor/`

用于急切模式执行的现代编译器：
1. **将计算捕获为类 TorchScript IR**
2. **融合操作**（内核融合）
3. **生成 C++/CUDA 代码**（通过代码生成）
4. **JIT 编译到原生代码**
5. **缓存编译函数**用于重复模式

### Dynamo（Python 字节码捕获）
**位置**：`torch/_dynamo/`

捕获 Python 函数执行：
1. **帧级字节码捕获**（无需修改源代码）
2. **在不安全操作时断开**（回退到急切）
3. **Guards** 用于动态行为（类型、形状变化）
4. **编译到 TorchScript 或 Inductor**

### FX（函数转换）
**位置**：`torch/fx/`

模块的符号追踪：
1. **符号执行**（通过模块操作）
2. **图捕获**（数据流图）
3. **图转换**通道（优化、降低）
4. **代码生成**（通过打印机）

### TorchScript（子集语言）
**位置**：`torch/jit/`、JIT 编译器

- Python 的静态类型子集
- 追踪或直接脚本编写
- 跨 Python 版本可移植
- 移动部署支持

## 关键模式和设计原则

### 模式 1：基于调度的多态性
不使用每个后端的虚函数，而是：
- 单个 Dispatcher 单例
- 按 (operator_name, DispatchKey) 对注册内核
- 基于张量的 DispatchKeySet 的运行时调度
- 支持新后端，无需重新编译

### 模式 2：双层实现
许多系统有：
- **C++ 核心**（高效，无 Python 依赖）
- **Python 包装**（便利性、可扩展性）

示例：
- Autograd：Variable（C++）+ THPVariable（Python）
- JIT：Graph（C++）+ Module 绑定（Python）
- Dispatch：Dispatcher（C++）+ torch.ops（Python）

### 模式 3：通过线程本地存储的上下文
执行上下文（设备、梯度模式、推断模式、调度键）存储在 TLS：
- 非入侵式参数传递
- 允许遗留代码无需重构即可工作
- Guards 管理作用域：
  ```cpp
  { DeviceGuard d(device); ... }  // 退出时恢复前一设备
  ```

### 模式 4：算子模式作为契约
模式定义：
- 输入：类型、默认值、别名注解
- 输出：类型、别名关系
- 启用：Python 绑定生成、类型检查、JIT 优化

### 模式 5：回退内核链
如果未找到 (op, DispatchKey) 的内核：
1. 尝试 Autograd 变体
2. 尝试 Meta 张量（形状推断）
3. 尝试 Python 回退（通过 __torch_function__）
4. 尝试 Functionalize（转换原地 → 函数式）

## 数据流示例

### 示例 1：前向操作
```
Python：y = torch.add(x, z)
  ↓
torch::autograd::details::MaybeWrap → Variable 检查
  ↓
Dispatcher::singleton().call(add_op)
  ↓
张量的 DispatchKeySet（CPU/CUDA + Dense + Autograd）
  ↓
Dispatcher 查找 → AutogradCPU / AutogradCUDA 内核
  ↓
AutogradXX 内核：
  - 保存输入/输出用于反向
  - 调用底层 AddBackward 函数对象
  - 调用原生内核（aten::add）
  ↓
原生内核：ATen native::add() 实现
  ↓
C++ 或 CUDA 内核执行
```

### 示例 2：反向传播
```
Python：loss.backward()
  ↓
Variable::backward() → autograd::Engine
  ↓
Engine::execute() → 图的拓扑排序
  ↓
对拓扑序中的每个 Node：
  - 调用 Node::apply(grad_inputs)
  - 计算 w.r.t. 输入的梯度
  - 在叶子变量中累积梯度
  ↓
结果：leaf.grad 填充累积梯度
```

### 示例 3：JIT 编译流
```
Python：@torch.jit.script 装饰器
  ↓
前端：解析 TorchScript，构建 AST
  ↓
IR Emitter：将 AST 转换为 Graph（SSA 形式）
  ↓
优化通道：DCE、CSE、内联
  ↓
GraphExecutor：为输入类型/形状特殊化
  ↓
代码生成：将 Graph 转换为 LLVM IR / CUDA 代码
  ↓
缓存：为重用存储编译函数
```

## 关键目录映射

```
pytorch/
├── c10/                          # 核心张量库（调度、类型、工具）
│   ├── core/                     # DispatchKey、Device、Storage、TensorOptions
│   ├── cuda/、hip/、xpu/         # 设备特定 API
│   └── util/                     # 注册表、智能指针、工具函数
│
├── aten/src/ATen/                # A 张量库
│   ├── core/
│   │   ├── dispatch/             # Dispatcher、OperatorEntry
│   │   ├── op_registration/      # 自定义算子注册
│   │   └── jit_type.h            # JIT 的类型系统
│   ├── native/                   # CPU 算子实现
│   ├── cuda/、hip/、mps/、xpu/   # 后端特定内核
│   └── Dispatch.h                # 公共调度器接口
│
├── torch/csrc/                   # Python 集成（C++ → Python）
│   ├── autograd/                 # Autograd：Variable、Function、Engine
│   ├── jit/                      # TorchScript 编译器
│   ├── distributed/              # 分布式训练（RPC、C10D、autograd）
│   ├── Module.cpp                # 主要 Python 绑定入口
│   └── Exceptions.h              # Python↔C++ 异常映射
│
├── torch/                         # Python 前端
│   ├── __init__.py               # 主要 torch API
│   ├── _dynamo/                  # Python 字节码捕获编译器
│   ├── _inductor/                # 代码生成编译器
│   ├── fx/                       # 函数转换
│   ├── jit/                      # JIT 用户 API
│   ├── nn/                       # 神经网络模块
│   ├── distributed/              # 分布式训练 API
│   ├── backends/                 # 后端特定 Python 绑定
│   └── autograd/                 # Autograd Python API
│
├── torchgen/                     # 从 native_functions.yaml 生成代码
└── functorch/                    # 可组合函数转换（vmap、grad 等）
```

## 编译与构建流程

1. **模式定义** - `native_functions.yaml`
   - 所有公共算子定义在这里
   - 类型提示、默认参数、别名信息

2. **代码生成** - `torchgen/`
   - 生成 C++ 绑定存根
   - 生成 Python 包装代码
   - 生成 FunctionSchema 注册
   - 为后端生成调度代码

3. **内核实现**
   - `native_functions.yaml` 定义接口
   - `aten/src/ATen/native/*.cpp` 实现内核
   - CUDA/HIP/等 的后端特定变体

4. **Dispatcher 注册**
   - 内核通过生成代码自动注册
   - 或通过自定义算子 API 手动注册

5. **构建与链接**
   - 编译为 libATen、libTorchDispatcher、libtorch
   - 通过 pybind11 的 Python 绑定

## 总结：组件关系

```
┌─────────────────────────────────────────┐
│        Python 用户代码（torch.*）        │
│    nn.Module、torch.add()、compile()    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   编译器层（torch._dynamo/fx/等）      │
│    捕获 & 优化 Python 代码              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  急切/JIT 执行（torch._inductor）       │
│   生成 & JIT 编译算子                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│    调度器（c10/aten/dispatch）          │
│  路由到后端特定内核                      │
└─────────────────────────────────────────┘
                    ↓
        ┌──────────┴──────────┬──────────┬──────────┐
        ↓                     ↓          ↓          ↓
  ┌──────────┐         ┌──────────┐ ┌──────────┐ ┌──────────┐
  │   CPU    │         │   CUDA   │ │   HIP    │ │   XPU    │
  │ 内核     │         │ 内核     │ │ 内核     │ │ 内核     │
  └──────────┘         └──────────┘ └──────────┘ └──────────┘
  （Native）            （CUBLAS、   （ROCm、     （Intel GPU）
  （MKL）              cuDNN、       MIOpen、
  （OpenMP）           cutlass）     rocBLAS）
```

**Autograd 和分布式**在所有层间包装和协调：
- Autograd：记录操作、计算梯度
- Distributed：跨进程同步张量/梯度、启用 RPC 通信

## 开发技巧

### 代码修改后
- **仅 Python 修改**：不需要重建（可编辑安装自动生效）
- **C++ 头文件/实现**：在 `build/` 目录中运行 `ninja torch_cpu`
- **算子模式修改**：需要完整重建

### 增量构建加速
- 使用 Ninja：`pip install ninja`
- 使用 CCache：`export CC="ccache gcc" CXX="ccache g++"`
- 限制并行任务：`export MAX_JOBS=4`（如果内存不足）

### 调试
- 带调试符号构建：`DEBUG=1 python -m pip install -e .`
- C++ 栈追踪：设置 `TORCH_SHOW_CPP_STACKTRACES=1`
- 调度追踪：用 `-DHAS_TORCH_SHOW_DISPATCH_TRACE` 构建，用 `TORCH_SHOW_DISPATCH_TRACE=1` 运行

### 测试策略
- 仅运行受影响的测试：`python test/test_nn.py TestNN.test_conv2d`
- 使用 pytest 子字符串匹配：`pytest test/test_nn.py -k conv -v`
- 算子修改后的 C++ 测试：`./build/bin/test_jit`

## 重要文件

- `native_functions.yaml` - 算子模式定义
- `setup.py` - 构建配置、环境变量选项
- `torch/csrc/Module.cpp` - 主要 Python 绑定入口点
- `aten/src/ATen/core/dispatch/Dispatcher.h` - Dispatcher 实现
- `torch/csrc/autograd/engine.cpp` - 反向传播执行
- `torch/csrc/jit/ir/ir.h` - JIT IR 数据结构

## CI 和贡献

- 主干健康仪表板：https://hud.pytorch.org/ci/pytorch/pytorch/main
- 贡献指南：https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions
- 开发基础设施办公时间：https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours

## 常见陷阱

1. **缺失子模块**：克隆后始终运行 `git submodule update --init --recursive`
2. **过时的构建缓存**：如有疑问，`rm -rf build/` 并重建
3. **Python 版本**：文档要求 Python < 3.13
4. **CUDA 版本不匹配**：确保 CUDA 工具包与 PyTorch 要求匹配
5. **GIL 管理**：从 C++ 调用 Python API 前始终获取 GIL
6. **调度键优先级**：Autograd 键的优先级高于后端键
