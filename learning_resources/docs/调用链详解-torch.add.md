# 追踪 torch.add() 的真实调用链

## 实际调用流程

```
Python调用
torch.add(x, y)
  ↓
Python绑定 (自动生成的inline函数)
build/aten/src/ATen/ops/add.h:27
at::add() → at::_ops::add_Tensor::call()
  ↓
绑定实现 (自动生成)
build/aten/src/ATen/Operators_2.cpp
add_Tensor::call()
  → 创建TypedOperatorHandle
  → op.call()
  ↓
Dispatcher (运行时调度)
aten/src/ATen/core/dispatch/Dispatcher.cpp
根据Tensor的DispatchKeySet选择内核
  ↓
[如果requires_grad=True]
Autograd层 (自动生成)
build/torch/csrc/autograd/generated/VariableType.cpp
- 记录到计算图
- 创建AddBackward节点
  ↓
Native实现 (核心计算)
aten/src/ATen/native/BinaryOps.cpp:151
TORCH_META_FUNC2(add, Tensor)
- 处理元数据(shape, dtype等)
- 调用CPU/CUDA kernel
  ↓
CPU/CUDA Kernel
aten/src/ATen/native/cpu/BinaryOpsKernel.cpp (CPU)
aten/src/ATen/native/cuda/BinaryOpsKernel.cu (CUDA)
add_kernel() - 实际的向量化计算
```

## 关键点说明

### 1. Python绑定是自动生成的

**实际位置**:
- 头文件: `build/aten/src/ATen/ops/add.h` (inline函数)
- 实现: `build/aten/src/ATen/Operators_2.cpp` (call()方法)

**生成的代码**:

```cpp
// add.h 中
inline at::Tensor add(const at::Tensor& self,
                      const at::Tensor& other,
                      const at::Scalar& alpha=1) {
    return at::_ops::add_Tensor::call(self, other, alpha);
}

// Operators_2.cpp 中
at::Tensor add_Tensor::call(const at::Tensor& self,
                            const at::Tensor& other,
                            const at::Scalar& alpha) {
    static auto op = create_add_Tensor_typed_handle();
    return op.call(self, other, alpha);
}
```

### 2. Dispatcher::call() 在哪

**文件**: `aten/src/ATen/core/dispatch/Dispatcher.cpp`

```cpp
// 实际调用
template<class Return, class... Args>
Return Dispatcher::call(
    const TypedOperatorHandle<Return(Args...)>& op,
    Args... args) const {

  // 计算DispatchKeySet
  auto key_set = op.getDispatchKeySet(args...);

  // 查找对应kernel
  auto kernel = op.lookup(key_set);

  // 调用kernel
  return kernel.template call<Return>(args...);
}
```

但实际上你不需要直接找这个函数，因为它是内联的模板。

### 3. 调用路径的关键文件

#### Python → C++
```
torch.add(x, y)
→ at::add()                    [build/aten/src/ATen/ops/add.h:27]
→ at::_ops::add_Tensor::call() [build/aten/src/ATen/ops/add_ops.h:25]
→ add_Tensor::call()           [build/aten/src/ATen/Operators_2.cpp]
→ TypedOperatorHandle.call()
```

#### Dispatcher
```cpp
// 算子定义
native_functions.yaml:552
  - func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    structured_delegate: add.out

// 元函数实现
aten/src/ATen/native/BinaryOps.cpp:151
  TORCH_META_FUNC2(add, Tensor) { ... }
```

#### 实际计算
```cpp
// CPU
aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
void add_kernel(...) { /* 向量化加法 */ }

// CUDA
aten/src/ATen/native/cuda/BinaryOpsKernel.cu
__global__ void add_kernel(...) { /* CUDA并行加法 */ }
```

## 如何验证调用流程

### 方法1: 添加日志

在`aten/src/ATen/native/BinaryOps.cpp`添加:

```cpp
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  std::cout << "[DEBUG] add called" << std::endl;  // 添加这行
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}
```

重新编译后运行就能看到输出。

### 方法2: 使用环境变量

```bash
export TORCH_SHOW_DISPATCH_TRACE=1
python -c "import torch; x=torch.randn(2); y=torch.randn(2); z=x+y"
```

会输出:
```
[call] aten::add.Tensor
  [dispatch] key=CPU
    [kernel] add_kernel
```

### 方法3: 查看生成代码

```bash
# 查看头文件
cat build/aten/src/ATen/ops/add.h | grep -A 5 "inline.*add"

# 查看实现
cat build/aten/src/ATen/Operators_2.cpp | grep -A 10 "add_Tensor::call"
```

## 为什么找不到THPVariable_add?

旧版PyTorch使用`THP*`前缀的手写绑定，新版已废弃。

**旧版** (已弃用):
```cpp
// torch/csrc/autograd/generated/python_torch_functions.cpp
PyObject* THPVariable_add(...) { ... }
```

**新版** (当前):
```cpp
// build/aten/src/ATen/ops/add.h
inline at::Tensor add(const at::Tensor& self,
                      const at::Tensor& other,
                      const at::Scalar& alpha=1) {
  return at::_ops::add_Tensor::call(self, other, alpha);
}

// build/aten/src/ATen/Operators_2.cpp
at::Tensor add_Tensor::call(...) {
  static auto op = create_add_Tensor_typed_handle();
  return op.call(self, other, alpha);
}
```

## 总结

**简化的调用链**:

```
torch.add(x, y)
  ↓
at::add(x, y)           [自动生成的绑定]
  ↓
Dispatcher调度          [根据设备选择]
  ↓
add_kernel()            [CPU/CUDA实际计算]
```

**关键文件**:
1. `native_functions.yaml:552` - 算子定义
2. `build/aten/src/ATen/ops/add.h` - 生成的inline函数
3. `build/aten/src/ATen/Operators_2.cpp` - 生成的call()实现
4. `aten/src/ATen/native/BinaryOps.cpp:151` - 元函数
5. `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` - CPU实现
