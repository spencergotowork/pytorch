# PyTorch Architecture - Key Code Snippets

This document provides actual code snippets from the PyTorch codebase that illustrate key architectural concepts.

## 1. Dispatch Key Taxonomy

**File:** `c10/core/DispatchKey.h` (lines 36-63)

```cpp
// Backends - hardware targets
#define C10_FORALL_BACKEND_COMPONENTS(_, extra) \
  _(CPU, extra)           // CPU with OpenMP/MKL         \
  _(CUDA, extra)          // NVIDIA CUDA                 \
  _(HIP, extra)           // AMD ROCm                    \
  _(XLA, extra)           // Google TPU/XLA              \
  _(MPS, extra)           // Apple Metal                 \
  _(XPU, extra)           // Intel GPU                   \
  _(Lazy, extra)          // Lazy evaluation             \
  _(MTIA, extra)          // Meta AI Tensor Accelerator \
  _(PrivateUse1/2/3, extra) // Custom backends          \
  _(Meta, extra)          // Shape inference only        \

// Functionality - aspects of tensors
#define C10_FORALL_FUNCTIONALITY_KEYS(_) \
  _(Dense, )              // Regular dense tensors       \
  _(Quantized, ...)       // Quantized tensors          \
  _(Sparse, ...)          // Sparse tensors             \
  _(NestedTensor, ...)    // Variable-size dims         \
  _(AutogradFunctionality, Autograd)  // Gradient tracking
```

**Impact:** Each (Backend, Functionality) pair gets separate kernel implementations. Enables 15+ backends with 5 functionality variants without virtual functions.

---

## 2. Operator Registration Pattern

**File:** `aten/src/ATen/core/op_registration/README.md` (lines 36-57)

Two registration approaches:

### Built-in Operators (native_functions.yaml)
```cpp
// File: aten/src/ATen/native/BinaryOps.cpp
at::Tensor add_cpu(const at::Tensor& self, const at::Tensor& other, const Scalar& alpha) {
    // Implementation for CPU backend
    return result;
}

// Auto-generated registration from native_functions.yaml:
// "add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
// Creates: aten::add, CPU kernel, CUDA kernel, autograd kernel
```

### Custom Operators
```cpp
static auto registry = torch::RegisterOperators()
    .op("my_namespace::my_op(Tensor a, Tensor b) -> Tensor",
        torch::RegisterOperators::options()
            .kernel<decltype(my_kernel_cpu), &my_kernel_cpu>(CPU()))
            .kernel<decltype(my_kernel_cuda), &my_kernel_cuda>(CUDA()));

// Usage from Python:
// y = torch.ops.my_namespace.my_op(x, z)
```

**Impact:** Cleanly separates operator schema (interface) from kernels (implementation). Multiple backends can register for same operation.

---

## 3. Dispatcher Lookup Mechanism

**File:** `aten/src/ATen/core/dispatch/Dispatcher.h` (lines 66-132)

```cpp
class Dispatcher final {
 public:
  // Singleton pattern
  static Dispatcher& singleton();
  
  // Operator lookup by name
  std::optional<OperatorHandle> findSchema(const OperatorName& op_name);
  
  // Kernel dispatch - called for every operation
  KernelFunction call(const OperatorHandle& op, DispatchKeySet key) const;
  
 private:
  // Per-operator dispatch table
  std::unordered_map<OperatorName, impl::OperatorEntry> registry_;
};

// Usage in generated code:
// When torch.add(tensor_a, tensor_b) is called:
// 1. Determine DispatchKeySet from tensor's device/dtype
// 2. Dispatcher::singleton().call(add_op, dispatchKeySet)
// 3. Lookup kernel in OperatorEntry
// 4. Execute kernel with automatic fallbacks
```

**Impact:** Single global registry avoids duplicated dispatch tables. DispatchKeySet encodes all relevant properties.

---

## 4. Autograd Node Architecture

**File:** `torch/csrc/autograd/function.h` (excerpt)

```cpp
struct Function : public Node {
  virtual variable_list apply(variable_list&& inputs) override;
  
  // Backward implementation provided by subclass
  virtual variable_list backward(
      AutogradContext* ctx, 
      variable_list grad_outputs) = 0;
};

// Custom autograd example:
struct MulBackward : public Function {
  static Tensor forward(AutogradContext* ctx, const Tensor& a, const Tensor& b) {
    ctx->save_for_backward({a, b});  // Save for backward
    return a * b;
  }
  
  static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
    auto grad_output = grad_outputs[0];
    auto [a, b] = ctx->saved_tensors();
    
    // Chain rule: dy/da = grad_output * b, dy/db = grad_output * a
    return {grad_output * b, grad_output * a};
  }
};
```

**Impact:** Separates forward computation (kernels) from backward (gradient rules). Enables modular autograd definition.

---

## 5. JIT Intermediate Representation

**File:** `torch/csrc/jit/ir/ir.h` (concepts from OVERVIEW.md)

Graph representation (SSA form):
```
// TorchScript code:
def f(a, b):
    c = a + b
    d = c * c
    e = torch.tanh(d)
    return e + d

// Converted to Graph IR:
graph(%0 : Tensor,      // Parameter 'a'
      %1 : Tensor):     // Parameter 'b'
  %2 : Tensor = aten::add(%0, %1)    // c = a + b
  %3 : Tensor = aten::mul(%2, %2)    // d = c * c
  %4 : Tensor = aten::tanh(%3)       // e = tanh(d)
  %5 : Tensor = aten::add(%4, %3)    // return e + d
  return (%5)

// Key concepts:
// - %0, %1, ... are Values (SSA variables)
// - Each Value has exactly one defining Node
// - aten::add, aten::mul, aten::tanh are Nodes
// - Graph contains Blocks containing Nodes
// - Control flow: If/Loops are Blocks within a Node
```

**Impact:** SSA form enables optimization passes (DCE, CSE, fusion). Clean IR for analysis and transformation.

---

## 6. Python-C++ Type Bridge

**File:** `torch/csrc/autograd/README.md` (lines 8-34)

```cpp
// C++ Implementation (efficient)
class Variable : public c10::TensorBase {
 private:
  bool requires_grad_;
  std::shared_ptr<Node> grad_fn_;  // Backward graph node
};

// Python Object (convenience)
class THPVariable {
 private:
  PyObject_HEAD
  std::shared_ptr<Variable> cdata;  // Shared pointer to C++ object
};

// Python-facing methods delegate to C++ implementation:
static PyObject* THPVariable_get_grad(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  pybind11::gil_scoped_acquire gil;  // Take GIL before Python call
  
  auto grad = self->cdata->grad();   // Call C++ method
  return THPVariable_Wrap(grad);     // Wrap result in Python object
  
  END_HANDLE_TH_ERRORS
}

// Impact:
// - C++ performance for hot paths
// - Python convenience for API
// - Shared ownership via shared_ptr
// - Safe Python interaction via RAII guards
```

**Impact:** Clean separation of concerns. C++ stays fast, Python stays convenient. No duplication of logic.

---

## 7. Distributed Communication Pattern

**File:** `torch/distributed/distributed_c10d.py` (conceptual, from exploration)

```python
# Collective Communication Abstraction
class ProcessGroup:
    """Abstract backend interface"""
    def all_reduce(self, tensor):
        """Sum tensor across all ranks"""
        pass
    
    def all_gather(self, output_tensors, input_tensor):
        """Gather from all ranks"""
        pass

# Concrete implementations
nccl_group = ProcessGroup(backend='nccl')   # NVIDIA NCCL
gloo_group = ProcessGroup(backend='gloo')   # CPU/generic
mpi_group = ProcessGroup(backend='mpi')     # HPC systems

# Usage:
dist.init_process_group('nccl')
world_rank = dist.get_rank()
world_size = dist.get_world_size()

# Synchronize gradients across processes
tensor = model_output.grad
dist.all_reduce(tensor)  # Sum gradients on all ranks
tensor /= world_size     # Average

# Impact:
# - Pluggable backends (NCCL, Gloo, MPI)
# - Device mesh abstraction for multi-GPU/multi-node
# - Functional collectives with futures for async
```

**Impact:** Enables distributed training without modifying base operators. Backends pluggable.

---

## 8. Inductor Compiler Pipeline

**File:** `torch/_inductor/` (conceptual)

```python
# Step 1: Capture computation as graph
import torch
from torch._dynamo import optimize

@optimize(torch._inductor.compile)  # Dynamo captures bytecode
def forward(model, x):
    return model(x)

# Step 2: Graph fusion
# Dynamo extracts computation graph
# Inductor identifies kernel fusion opportunities:
# Original: x = relu(y); z = mul(x, alpha)
# Fused: z = fused_relu_mul(y, alpha)  # Single GPU kernel

# Step 3: Code generation
# Generates CUDA C++ code or C++ code
generated_code = """
__global__ void fused_relu_mul(float* out, const float* y, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = y[idx];
        out[idx] = (val > 0 ? val : 0) * alpha;  // Fused operation
    }
}
"""

# Step 4: JIT compilation
# Compile to native code, cache for reuse
compiled_fn = torch._inductor.compile(generated_code)
result = compiled_fn(y, alpha)

# Impact:
# - Automatic kernel fusion without user code changes
# - Generates specialized code for specific operations
# - Caches compiled kernels for repeated patterns
```

**Impact:** Bridges gap between Python-level optimization and kernel-level performance.

---

## 9. Thread-Local Execution Context

**File:** `c10/core/impl/LocalDispatchKeySet.h` (concept)

```cpp
// Thread-local storage of execution context
thread_local DispatchKeySet current_keys;
thread_local c10::Device current_device;
thread_local bool grad_enabled;
thread_local bool inference_mode;

// RAII Guard Pattern
struct DeviceGuard {
    c10::Device prev_device;
    
    explicit DeviceGuard(const c10::Device& device) {
        prev_device = current_device;  // Save previous
        current_device = device;        // Set new
    }
    
    ~DeviceGuard() {
        current_device = prev_device;   // Restore on destruction
    }
};

// Usage:
{
    DeviceGuard g(c10::Device(c10::DeviceType::CUDA, 1));
    // All operations inside this scope use CUDA:1
    auto x = torch::randn({10, 10});  // Created on CUDA:1
} // Automatically restore previous device

// Similarly for gradient mode:
struct NoGradGuard {
    bool prev_grad;
    
    NoGradGuard() {
        prev_grad = grad_enabled;
        grad_enabled = false;
    }
    
    ~NoGradGuard() {
        grad_enabled = prev_grad;
    }
};

// Impact:
// - Non-invasive context management
// - No parameter threading through functions
// - RAII ensures restoration even with exceptions
// - Enables legacy code compatibility
```

**Impact:** Context doesn't require explicit parameter passing. Guards manage scope automatically.

---

## 10. Operator Schema Contract

**File:** `aten/src/ATen/core/function_schema.h` (concept)

```cpp
// Schema defines the contract - types, aliasing, semantics
struct FunctionSchema {
    std::string name;
    std::vector<Argument> arguments;
    std::vector<Argument> returns;
};

// Example schema for aten::add
// add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
// 
// Annotations (aliases):
// add(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a)
// Meaning:
//   - (a!) - self is modified in-place (alias marker 'a' with '!')
//   - Return value aliases self (same 'a')
//   - This enables alias analysis for optimizations

// Python code generation:
// def add(self: Tensor, other: Tensor, *, alpha: Scalar = 1) -> Tensor:
//     return torch._C._nn.add(self, other, alpha)

// JIT type checking:
// If a Node has kind aten::add, schema tells us:
//   - Input 0: Tensor
//   - Input 1: Tensor
//   - Input 2: Scalar (default = 1)
//   - Output: Tensor

// Dispatch resolution:
// Schema enables correct kernel selection when multiple overloads exist:
// - add(Tensor, Tensor) -> Tensor
// - add(Tensor, Scalar) -> Tensor
// - add(int, int) -> int

// Impact:
// - Type safety across Python/C++/JIT
// - Enables automatic code generation
// - Enables optimization (alias analysis)
// - Enables polymorphism without virtual functions
```

**Impact:** Single schema definition generates Python bindings, JIT types, dispatch code, autograd signatures.

---

## Summary: Architectural Patterns in Code

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Dispatch** | c10/core/DispatchKey.h | Route to backend-specific kernels |
| **RAII Guards** | c10/core/DeviceGuard.h | Manage context (device, grad mode) |
| **Singleton** | aten/core/dispatch/Dispatcher.h | Global kernel registry |
| **Two-level impl** | torch/csrc/autograd/ | C++ core + Python wrapper |
| **SSA IR** | torch/csrc/jit/ir/ | Immutable graph representation |
| **Function schema** | aten/core/function_schema.h | Interface contract |
| **Kernel fallback** | aten/core/dispatch/ | Robustness chain |
| **Thread-local TLS** | c10/core/impl/ | Execution context |
| **Custom ops** | aten/core/op_registration/ | Extensibility |
| **Distributed ops** | torch/distributed/c10d/ | Multi-process communication |

