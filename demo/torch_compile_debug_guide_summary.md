# torch.compile 深度调试指南

基于PyTorch 2.8实际测试的torch.compile各阶段针对性调试方法和解决方案。

## 概述

torch.compile是PyTorch 2.0+的核心编译功能，包含三个主要阶段：
1. **TorchDynamo**: 捕获Python字节码并构建FX图
2. **AOTAutograd**: 处理自动微分和内存优化
3. **Inductor**: 生成优化的机器代码

每个阶段都有其特定的调试方法和常见问题。

## 1. TorchDynamo阶段调试

### 1.1 Graph Break分析

**问题描述**: Graph Break会导致编译效率下降，将单一图拆分为多个子图。

**调试工具**:
```python
import torch._dynamo

# 使用explain()分析函数结构
def problematic_function(x):
    y = torch.relu(x + 1)
    shape_len = len(x.shape)  # Python内置函数 - 导致break
    if y.sum() > 0:           # 数据依赖控制流 - 导致break
        z = y * 2
    else:
        z = y / 2
    return z.sum() + shape_len

explanation = torch._dynamo.explain(problematic_function)(torch.randn(10, 20))
print(f"图数量: {explanation.graph_count}")
print(f"Break数量: {explanation.graph_break_count}")
```

**常见Graph Break原因**:
- Python内置函数调用（`len()`, `print()`, `isinstance()`等）
- 数据依赖的控制流（`if tensor.sum() > 0`）
- 循环结构
- 异常处理（`try/except`）
- 副作用操作（打印、文件IO等）

**优化方案**:
```python
# 优化前（2个图，1个break）
def problematic_function(x):
    y = torch.relu(x + 1)
    shape_len = len(x.shape)
    if y.sum() > 0:
        z = y * 2
    else:
        z = y / 2
    return z.sum() + shape_len

# 优化后（1个图，0个break）
def optimized_function(x):
    y = torch.relu(x + 1)
    shape_len = x.dim()  # 使用张量方法
    condition = y.sum() > 0
    z = torch.where(condition, y * 2, y / 2)  # 条件运算
    return z.sum() + shape_len
```

### 1.2 Guard失败调试

**问题描述**: Guard检查失败导致重编译，影响性能。

**调试方法**:
```bash
# 环境变量配置
export TORCH_LOGS="guards,recompiles"
```

**常见重编译原因**:
- 输入形状变化
- 数据类型变化
- 设备变化
- 动态值变化

**解决方案**:
```python
# 使用动态形状编译
compiled_func = torch.compile(func, dynamic=True)

# 或者固定输入形状进行预热
shapes = [(64, 64), (128, 128), (256, 256)]
for shape in shapes:
    x = torch.randn(*shape)
    _ = compiled_func(x)  # 预热不同形状
```

### 1.3 动态行为处理

**问题描述**: 动态控制流和形状变化导致编译困难。

**调试工具**:
```python
def dynamic_function(x):
    if x.dim() == 1:
        return x.sum()
    elif x.dim() == 2:
        return x.trace()
    else:
        return x.flatten().sum()

# 静态编译 vs 动态编译对比
static_compiled = torch.compile(dynamic_function, dynamic=False)
dynamic_compiled = torch.compile(dynamic_function, dynamic=True)
```

## 2. AOTAutograd阶段调试

### 2.1 反向图生成错误

**问题描述**: 自动微分图生成失败或梯度计算错误。

**调试方法**:
```python
# 测试不同backend的梯度行为
backends = ["eager", "aot_eager", "inductor"]

for backend in backends:
    model.zero_grad()
    compiled_model = torch.compile(model, backend=backend)
    loss = compiled_model(x)
    loss.backward()
    
    # 检查梯度一致性
    param_grads = [p.grad.norm().item() for p in model.parameters()]
    print(f"{backend}: 梯度范数 = {param_grads}")
```

### 2.2 内存优化问题

**问题描述**: 内存使用异常或内存泄漏。

**调试工具**:
```python
import psutil

def memory_intensive_func(x):
    results = []
    current = x
    for i in range(3):
        current = torch.matmul(current, current.t())
        results.append(current.clone())  # 保存中间结果
    return sum(results).trace()

def memory_efficient_func(x):
    current = x
    for i in range(3):
        current = torch.matmul(current, current.t())
    return current.trace()

# 比较内存使用
process = psutil.Process()
initial_mem = process.memory_info().rss / 1024 / 1024
compiled_func = torch.compile(func, backend="aot_eager")
loss = compiled_func(x)
peak_mem = process.memory_info().rss / 1024 / 1024
print(f"内存使用: {peak_mem - initial_mem:.1f} MB")
```

### 2.3 函数式变换失败

**问题描述**: AOTAutograd无法处理某些操作模式。

**常见问题**:
- In-place操作
- 复杂的视图操作
- 非连续张量操作

**解决方案**:
```python
# 有问题的模式
problematic = lambda x: x.add_(1)  # in-place操作

# 优化后的模式  
optimized = lambda x: x + 1  # 函数式操作
```

## 3. Inductor阶段调试

### 3.1 代码生成调试

**环境配置**:
```bash
export TORCH_LOGS="inductor,output_code,kernel_code"
export TORCH_COMPILE_DEBUG=1
```

**调试方法**:
```python
def fusion_example(x, y):
    z1 = torch.relu(x + y)
    z2 = torch.tanh(z1 * 2)
    z3 = torch.sigmoid(z2 - 1)
    return z3.sum()

compiled_func = torch.compile(fusion_example, backend="inductor")
result = compiled_func(x, y)

# 检查生成的调试文件
# - output_*.py: 生成的Python代码
# - kernel_*.cpp/cu: 生成的C++/CUDA内核
```

### 3.2 内核融合分析

**融合模式测试**:
```python
fusion_patterns = {
    "简单融合": lambda x: torch.relu(torch.sigmoid(x)).sum(),
    "复杂融合链": lambda x: torch.softmax(torch.relu(x + 1), dim=-1).sum(),
    "广播融合": lambda x: (x + torch.ones_like(x[0:1])).sum(),
    "阻止融合": lambda x: torch.relu(x).contiguous().sigmoid().sum()
}

for name, pattern in fusion_patterns.items():
    compiled = torch.compile(pattern, backend="inductor")
    # 性能测试和分析
```

### 3.3 性能问题定位

**性能对比方法**:
```python
def benchmark_function(func, inputs, warmup=5, iterations=100):
    compiled_func = torch.compile(func, backend="inductor")
    
    # 预热
    for _ in range(warmup):
        _ = compiled_func(*inputs)
    
    # 原始性能
    start_time = time.time()
    for _ in range(iterations):
        original_result = func(*inputs)
    original_time = time.time() - start_time
    
    # 编译性能
    start_time = time.time()
    for _ in range(iterations):
        compiled_result = compiled_func(*inputs)
    compiled_time = time.time() - start_time
    
    speedup = original_time / compiled_time
    print(f"加速比: {speedup:.2f}x")
    
    # 数值正确性检查
    if torch.allclose(original_result, compiled_result, rtol=1e-4):
        print("✅ 数值一致")
    else:
        print("⚠️ 数值差异")
    
    return speedup
```

### 3.4 硬件兼容性

**CUDA支持检测**:
```python
if torch.cuda.is_available():
    print(f"CUDA设备: {torch.cuda.get_device_name()}")
    print(f"计算能力: {torch.cuda.get_device_capability()}")
    
    # 测试特定功能
    def test_tensor_cores():
        a = torch.randn(128, 128, dtype=torch.half, device='cuda')
        b = torch.randn(128, 128, dtype=torch.half, device='cuda')
        compiled = torch.compile(lambda x, y: torch.matmul(x, y))
        return compiled(a, b)
```

## 4. 常见错误模式和解决方案

### 4.1 编译失败的典型原因

| 错误类型 | 常见原因 | 解决方案 |
|---------|---------|---------|
| TorchDynamoException | 不支持的Python特性 | 使用PyTorch原生操作 |
| BackendCompilerFailed | 复杂的数据依赖 | 简化函数逻辑 |
| InductorError | 硬件不兼容 | 检查设备支持 |

### 4.2 性能回退分析

**判断是否适合编译**:
```python
def should_compile(func, test_input):
    # 简单函数可能不值得编译
    explanation = torch._dynamo.explain(func)(test_input)
    
    if explanation.op_count < 5:
        return False, "操作数太少"
    
    if explanation.graph_break_count > 2:
        return False, "Graph Break过多"
    
    return True, "适合编译"
```

**性能回退的常见原因**:
- 函数过于简单，编译开销大于收益
- Graph Break过多，图切分严重
- 内存访问模式不友好
- 频繁的重编译

### 4.3 内存异常处理

**内存泄漏检测**:
```python
import gc
import psutil

def detect_memory_leak(func, inputs, iterations=100):
    process = psutil.Process()
    compiled_func = torch.compile(func)
    
    initial_mem = process.memory_info().rss / 1024 / 1024
    
    for i in range(iterations):
        result = compiled_func(*inputs)
        
        if i % 10 == 0:
            gc.collect()
            current_mem = process.memory_info().rss / 1024 / 1024
            mem_growth = current_mem - initial_mem
            
            if mem_growth > 100:  # 超过100MB增长
                print(f"⚠️ 可能的内存泄漏: {mem_growth:.1f}MB")
                return True
    
    return False
```

### 4.4 数值精度问题

**精度问题检测**:
```python
def check_numerical_stability(func, test_inputs):
    compiled_func = torch.compile(func)
    
    for i, inputs in enumerate(test_inputs):
        original = func(*inputs)
        compiled = compiled_func(*inputs)
        
        # 检查数值差异
        if torch.isnan(original).any() or torch.isnan(compiled).any():
            print(f"⚠️ NaN检测到")
        
        if torch.isinf(original).any() or torch.isinf(compiled).any():
            print(f"⚠️ Inf检测到")
        
        max_diff = torch.abs(original - compiled).max()
        if max_diff > 1e-4:
            print(f"⚠️ 数值差异: {max_diff:.8f}")
```

## 5. 调试工具和最佳实践

### 5.1 核心调试工具

| 工具 | 功能 | 使用方法 |
|------|------|----------|
| `torch._dynamo.explain()` | 分析函数结构 | `explain(func)(input)` |
| `TORCH_LOGS` | 控制日志输出 | `export TORCH_LOGS="graph_breaks,recompiles"` |
| Backend测试 | 逐阶段调试 | `torch.compile(func, backend="eager/aot_eager/inductor")` |
| 动态编译 | 处理变化输入 | `torch.compile(func, dynamic=True)` |

### 5.2 环境变量配置

```bash
# 基础调试
export TORCH_LOGS="graph_breaks,recompiles"
export TORCH_COMPILE_DEBUG=1

# 详细调试
export TORCH_LOGS="graph_breaks,recompiles,guards"
export TORCHDYNAMO_VERBOSE=1

# 性能分析
export TORCH_LOGS="schedule,fusion,perf_hints"

# 代码查看
export TORCH_LOGS="output_code,kernel_code"

# 自动复现脚本
export TORCHDYNAMO_REPRO_AFTER="dynamo"
export TORCHDYNAMO_REPRO_LEVEL=4
```

### 5.3 程序化配置

```python
import torch._logging

# 基础配置
torch._logging.set_logs(
    graph_breaks=True,
    recompiles=True,
    guards=True
)

# 详细配置
torch._logging.set_logs(
    dynamo=logging.DEBUG,
    aot=logging.DEBUG,
    inductor=logging.DEBUG
)
```

### 5.4 调试上下文管理器

```python
class TorchCompileDebugContext:
    def __init__(self, logs="graph_breaks,recompiles"):
        self.logs = logs
        self.original_env = {}
        
    def __enter__(self):
        self.original_env = dict(os.environ)
        os.environ.update({
            "TORCH_LOGS": self.logs,
            "TORCH_COMPILE_DEBUG": "1"
        })
        return self
        
    def __exit__(self, *args):
        os.environ.clear()
        os.environ.update(self.original_env)

# 使用方式
with TorchCompileDebugContext("graph_breaks,output_code"):
    compiled_func = torch.compile(my_function)
    result = compiled_func(input_tensor)
```

### 5.5 推荐调试流程

1. **快速分析**: 使用`explain()`识别函数结构和问题点
2. **环境配置**: 设置适当的`TORCH_LOGS`级别
3. **分阶段测试**: 从`eager` → `aot_eager` → `inductor`逐步测试
4. **优化代码**: 根据Graph Break分析结果优化函数
5. **性能验证**: 检查编译后的性能提升和数值正确性
6. **问题解决**: 根据具体错误类型应用相应解决方案

### 5.6 性能优化要点

- ✅ 减少不必要的Graph Break
- ✅ 使用PyTorch原生操作替代Python内置函数
- ✅ 避免数据依赖的控制流语句
- ✅ 使用`torch.where`等条件运算替代`if/else`
- ✅ 对复杂函数进行编译，简单函数可能不值得
- ✅ 使用动态形状模式处理变化的输入
- ✅ 预热编译以消除首次执行的开销
- ✅ 定期检查数值精度和内存使用

## 6. 实际案例分析

### 案例1: Graph Break优化

```python
# 问题代码 (2个图，1个break)
def problematic_code(x):
    y = torch.relu(x)
    if len(x.shape) > 2:  # Python内置函数 + 数据依赖
        return y.sum()
    return y.mean()

# 优化代码 (1个图，0个break)
def optimized_code(x):
    y = torch.relu(x)
    condition = x.dim() > 2  # 张量方法
    return torch.where(condition, y.sum(), y.mean())

# 性能对比
explanation1 = torch._dynamo.explain(problematic_code)(torch.randn(10, 10))
explanation2 = torch._dynamo.explain(optimized_code)(torch.randn(10, 10))

print(f"优化前: {explanation1.graph_count}图, {explanation1.graph_break_count}个break")
print(f"优化后: {explanation2.graph_count}图, {explanation2.graph_break_count}个break")
```

### 案例2: 内存优化

```python
# 内存密集版本
def memory_heavy(x):
    results = []
    for i in range(5):
        x = x @ x.t()
        results.append(x.clone())  # 保存所有中间结果
    return sum(results).trace()

# 内存优化版本
def memory_light(x):
    for i in range(5):
        x = x @ x.t()  # 就地更新，不保存中间结果
    return x.trace()

# 内存使用对比
import psutil
process = psutil.Process()

for name, func in [("内存密集", memory_heavy), ("内存优化", memory_light)]:
    initial = process.memory_info().rss / 1024 / 1024
    compiled = torch.compile(func)
    _ = compiled(torch.randn(128, 128))
    peak = process.memory_info().rss / 1024 / 1024
    print(f"{name}: {peak - initial:.1f}MB")
```

## 总结

torch.compile的调试需要系统性的方法：

1. **从简单开始**: 使用`explain()`快速识别问题
2. **分阶段调试**: 逐步测试各个编译阶段
3. **优化代码结构**: 减少Graph Break比调试编译器更有效
4. **性能权衡**: 不是所有函数都适合编译
5. **持续监控**: 定期检查性能和数值正确性

记住：**优化代码结构比调试编译器更有效果**。大多数torch.compile问题都可以通过改进代码模式来解决。