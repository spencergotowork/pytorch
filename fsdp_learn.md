# PyTorch FSDP 完全分片数据并行深度解析

## 目录
- [核心实现原理与代码解读](#核心实现原理与代码解读)
  - [FSDP 核心设计](#fsdp-核心设计)
  - [FSDP2 架构改进](#fsdp2-架构改进)
  - [共有核心组件](#共有核心组件)
- [极简使用指南](#极简使用指南)
  - [FSDP 基础使用](#fsdp-基础使用)
  - [FSDP2 简化使用](#fsdp2-简化使用)
  - [共性注意事项](#共性注意事项)

---

## 核心实现原理与代码解读

### FSDP 核心设计

#### 1. 参数全分片机制

**核心文件：**`torch/distributed/fsdp/_flat_param.py`

```python
# FlatParamHandle 类：参数分片的核心管理器
class FlatParamHandle:
    def __init__(self, params, module, ...):
        # 将多个参数展平为单个张量
        self.flat_param = _flatten_tensors(params)
        # 存储分片元数据
        self._shard_param_infos = _get_shard_metadata(...)
        
    def shard(self):
        """将参数分片到当前进程"""
        # 获取当前进程应该持有的参数片段
        self._local_shard = _get_shard(
            self.flat_param, 
            rank=self._rank, 
            world_size=self._world_size
        )
        
    def unshard(self):
        """使用all-gather恢复完整参数"""
        # 通过all-gather收集所有进程的参数片段
        self.flat_param = dist.all_gather(self._local_shard)
```

**参数、梯度、优化器状态分片原理：**
- **参数分片：**每个GPU只存储模型参数的1/N部分
- **梯度分片：**反向传播后使用reduce-scatter同步梯度
- **优化器状态分片：**优化器状态与参数分片保持一致

#### 2. 通信策略时机

**核心文件：**`torch/distributed/fsdp/_runtime_utils.py`

```python
# 前向传播前的all-gather钩子
def _pre_forward_unshard(state, handle):
    """前向传播前恢复参数"""
    if handle and not handle._prefetched:
        # 使用all-gather恢复完整参数用于计算
        _unshard(state, handle, unshard_stream, pre_unshard_stream)

# 前向传播后的重新分片钩子  
def _post_forward_reshard(state, handle):
    """前向传播后重新分片参数"""
    if handle and handle._training_state == HandleTrainingState.FORWARD:
        handle.reshard()  # 释放内存，重新分片

# 反向传播的reduce-scatter逻辑
def _post_backward_hook(state, handle):
    """反向传播后同步梯度"""
    # 使用reduce-scatter聚合并分片梯度
    handle._reduce_scatter_gradients()
```

#### 3. 核心类与函数标注

**主要入口：**`torch/distributed/fsdp/fully_sharded_data_parallel.py`
```python
class FullyShardedDataParallel(nn.Module):
    def __init__(self, module, sharding_strategy=ShardingStrategy.FULL_SHARD, ...):
        # 递归包装子模块
        self._fsdp_wrapped_module = _wrap_module_cls_individually(module, ...)
        # 创建参数句柄
        self._handles = _init_param_handles(...)
```

### FSDP2 架构改进

#### 1. 内存效率改进

**核心文件：**`torch/distributed/fsdp/_fully_shard/_fully_shard.py`

```python
@contract(state_cls=FSDPState)
def fully_shard(module, *, 
                mesh: Optional[DeviceMesh] = None,
                reshard_after_forward: Optional[Union[bool, int]] = None,
                mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy()):
    """
    FSDP2的组合式API，支持更细粒度的分片控制
    """
    # 支持DeviceMesh进行多维并行
    if mesh is not None:
        _apply_device_mesh_policy(module, mesh)
    
    # 更精细的重分片策略
    if isinstance(reshard_after_forward, int):
        # 支持延迟N步后重分片，平衡内存与计算
        _configure_delayed_reshard(module, reshard_after_forward)
```

#### 2. 灵活性提升 - DeviceMesh支持

**核心文件：**`torch/distributed/fsdp/_fully_shard/_fsdp_state.py`

```python
class FSDPState:
    def __init__(self, mesh: DeviceMesh, ...):
        # 支持2D/3D设备网格，实现混合并行
        self._mesh = mesh
        self._dp_mesh = mesh["dp"] if "dp" in mesh.mesh_dim_names else mesh
        self._tp_mesh = mesh["tp"] if "tp" in mesh.mesh_dim_names else None
        
    def _configure_mixed_parallelism(self):
        """配置数据并行+张量并行混合策略"""
        if self._tp_mesh is not None:
            # 张量并行维度内的参数复制
            # 数据并行维度间的参数分片
            return MixedParallelismStrategy(...)
```

#### 3. 性能优化 - 通信原语改进

**核心文件：**`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`

```python
class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.Event]  # 异步事件支持
    all_gather_work: Optional[dist.Work]     # 异步工作句柄
    param_all_gather_input_dtypes: list[list[torch.dtype]]
    param_all_gather_input_numels: list[list[int]]

def foreach_all_gather(fsdp_params, group, async_op=True):
    """
    批量all-gather，减少通信开销
    支持异步操作和事件同步
    """
    # 批量处理多个参数的all-gather
    # 使用torch.ops.c10d_functional进行优化通信
    return AllGatherResult(...)
```

#### 4. API变化对比

| 特性 | FSDP1 | FSDP2 |
|------|-------|-------|
| **初始化方式** | `FSDP(module, ...)` | `fully_shard(module, ...)` |
| **配置类** | `MixedPrecision`, `CPUOffload` | `MixedPrecisionPolicy`, `OffloadPolicy` |
| **设备管理** | 手动设置 | `DeviceMesh` 自动管理 |
| **分片粒度** | 模块级 | 函数级，更灵活 |
| **组合性** | 包装器模式 | 函数式，可组合 |

### 共有核心组件

#### 1. ShardingStrategy 枚举类

**文件：**`torch/distributed/fsdp/api.py`
```python
class ShardingStrategy(Enum):
    FULL_SHARD = auto()        # 参数+梯度+优化器状态全分片
    SHARD_GRAD_OP = auto()     # 仅梯度+优化器状态分片(类似DDP)
    NO_SHARD = auto()          # 不分片，纯复制
    HYBRID_SHARD = auto()      # 节点内FULL_SHARD，节点间复制
    _HYBRID_SHARD_ZERO2 = auto() # 节点内SHARD_GRAD_OP，节点间复制
```

#### 2. _wrap_module 递归包装逻辑

**文件：**`torch/distributed/fsdp/_wrap_utils.py`
```python
def _recursive_wrap(module, auto_wrap_policy, ...):
    """
    递归包装模块树，自底向上应用FSDP
    """
    # 后序遍历确保依赖关系正确
    for name, child in module.named_children():
        _recursive_wrap(child, auto_wrap_policy, ...)
    
    # 根据策略决定是否包装当前模块
    if auto_wrap_policy.should_wrap(module):
        return _wrap_module(module, ...)
    return module
```

#### 3. _handles 参数管理句柄

**文件：**`torch/distributed/fsdp/_flat_param.py`
```python
class FlatParamHandle:
    """参数句柄，管理一组参数的分片、通信、状态"""
    def __init__(self, params, module_fqn, ...):
        self.flat_param = FlatParameter(...)  # 展平参数
        self._training_state = HandleTrainingState.IDLE
        self._prefetched = False  # 预取状态
        
    def prefetch(self):
        """预取下一层参数，重叠通信与计算"""
        if not self._prefetched:
            self._prefetch_handle = _start_async_all_gather(...)
```

---

## 极简使用指南

### FSDP 基础使用

#### 环境初始化 + 模型包装 + 训练循环（20行核心代码）

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

# 1. 环境初始化（必选）
dist.init_process_group("nccl")  
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# 2. 模型包装（必选参数：无，推荐配置如下）
model = FSDP(
    your_model.cuda(),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    mixed_precision=MixedPrecision(param_dtype=torch.float16)  # 混合精度
)

# 3. 训练循环（标准流程）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch['input']).loss
    loss.backward()  # 自动触发reduce-scatter
    optimizer.step()  # 在分片参数上更新

# 4. 保存/加载（分布式checkpoint）
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    torch.save(model.state_dict(), "model.pth")
```

**启动命令：**
```bash
torchrun --nproc_per_node=4 train_fsdp.py
```

### FSDP2 简化使用

#### API变化 + 迁移成本（15行核心代码）

```python
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

# 1. 环境初始化（新增DeviceMesh）
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))

# 2. 模型包装（函数式API，更简洁）
model = fully_shard(
    your_model.cuda(),
    mesh=mesh,  # 设备网格管理
    mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16)
)

# 3. 训练循环（完全相同）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch['input']).loss
    loss.backward()
    optimizer.step()
```

**迁移成本评估：**
- **API变化：**`FSDP(model)` → `fully_shard(model)`
- **配置类：**`MixedPrecision` → `MixedPrecisionPolicy`
- **新增特性：**`DeviceMesh`支持，可选使用
- **兼容性：**训练循环无需修改

**选择FSDP2的场景：**
- 需要更细粒度的模块控制
- 使用多维并行（DP+TP）
- 追求更高的内存效率

### 共性注意事项

#### 1. requires_grad 设置
```python
# 确保需要训练的参数设置了requires_grad
for param in model.parameters():
    param.requires_grad = True  # FSDP会检查此标志
```

#### 2. 混合精度训练兼容配置
```python
# FSDP1配置
mixed_precision = MixedPrecision(
    param_dtype=torch.float16,    # 参数存储精度
    reduce_dtype=torch.float16,   # 梯度聚合精度  
    buffer_dtype=torch.float16    # 缓冲区精度
)

# FSDP2配置（简化）
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16
)
```

#### 3. 多卡环境最低要求
- **硬件：**至少2张GPU（单卡无分片意义）
- **通信后端：**NCCL（推荐）或Gloo
- **启动方式：**必须使用`torchrun`或`torch.distributed.launch`
- **环境变量：**自动设置`RANK`、`LOCAL_RANK`、`WORLD_SIZE`

#### 4. 常见陷阱避免
```python
# ❌ 错误：在FSDP包装前移动到GPU
model = FSDP(model.cuda())  # 参数已在GPU上，分片可能失败

# ✅ 正确：FSDP包装后移动，或在包装时指定设备
model = FSDP(model).cuda()  # 推荐
# 或
model = FSDP(model, device_id=torch.cuda.current_device())
```

#### 5. 调试技巧
```python
# 检查分片状态
print(f"参数总数: {sum(p.numel() for p in model.parameters())}")
print(f"本地参数数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 监控内存使用
torch.cuda.reset_peak_memory_stats()
# ... 训练代码 ...
print(f"峰值内存: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

## 总结

### FSDP vs FSDP2 选择指南

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **简单模型，快速上手** | FSDP | API稳定，文档丰富 |
| **大模型训练，内存紧张** | FSDP2 | 更精细的内存控制 |
| **多维并行需求** | FSDP2 | DeviceMesh原生支持 |
| **生产环境，稳定性优先** | FSDP | 经过更多验证 |
| **研究实验，追求性能** | FSDP2 | 最新优化特性 |

### 核心优势
- **内存效率：**相比DDP，内存使用降低N倍（N=GPU数量）
- **扩展性：**支持千亿参数模型训练
- **易用性：**最小代码改动，从单卡迁移到多卡
- **灵活性：**多种分片策略，适应不同场景

FSDP通过参数分片、智能通信调度和自动内存管理，为大规模模型训练提供了高效且易用的解决方案。FSDP2进一步提升了灵活性和性能，是未来发展的主要方向。