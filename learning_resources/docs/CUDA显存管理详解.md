# PyTorch CUDA 显存管理详解

本文档详细说明 PyTorch 的 CUDA 显存管理机制,从 Python 接口层到底层 C++ 实现的完整调用链路。

## 一、概览

PyTorch 使用 **CUDACachingAllocator**（缓存分配器）来管理 GPU 显存,避免频繁调用 `cudaMalloc`/`cudaFree` 带来的性能开销。

**核心思想:**
- 从 CUDA 分配大块内存,自己管理分割和复用
- 释放的内存不立即归还给 CUDA,而是缓存起来供后续分配使用
- 流感知(stream-aware):同一个 stream 上的分配可以安全复用

**代码位置:**
- Python 层: `torch/cuda/memory.py`
- Python 绑定: `torch/csrc/cuda/Module.cpp`
- C++ 核心实现: `c10/cuda/CUDACachingAllocator.cpp`
- 头文件: `c10/cuda/CUDACachingAllocator.h`

---

## 二、完整调用链路

### 2.1 Python 层接口

#### 示例代码
```python
import torch

# 方式1: 直接分配张量(最常用)
x = torch.empty(1000, 1000, device='cuda')

# 方式2: 显式调用缓存分配器
from torch.cuda import caching_allocator_alloc
ptr = caching_allocator_alloc(size=4096, device=0, stream=None)
```

#### Python 接口实现
**文件:** `torch/cuda/memory.py:110-143`

```python
def caching_allocator_alloc(size, device: "Device" = None, stream=None):
    """从 CUDA 缓存分配器分配内存"""
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)

    if stream is None:
        stream = torch.cuda.current_stream(device)
    if isinstance(stream, torch.cuda.streams.Stream):
        stream = stream.cuda_stream  # 获取底层 cudaStream_t

    with torch.cuda.device(device):
        # 调用 C++ 绑定
        return torch._C._cuda_cudaCachingAllocator_raw_alloc(size, stream)
```

**关键点:**
- `torch._C._cuda_cudaCachingAllocator_raw_alloc` 是 C++ 绑定的入口
- 传入参数: `size` (字节数), `stream` (CUDA stream 指针)

---

### 2.2 Python 绑定层

**文件:** `torch/csrc/cuda/Module.cpp:272-296`

```cpp
PyObject* THCPModule_cudaCachingAllocator_raw_alloc(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS

  // 解析 Python 参数
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(...);
    return nullptr;
  }

  auto size = PyLong_AsSsize_t(size_o);  // Python int -> C ssize_t
  cudaStream_t stream = static_cast<cudaStream_t>(PyLong_AsVoidPtr(stream_o));

  void* mem = nullptr;
  {
    pybind11::gil_scoped_release no_gil;  // 释放 GIL
    // 调用 C++ 核心实现
    mem = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream);
  }

  return PyLong_FromVoidPtr(mem);  // C 指针 -> Python int
  END_HANDLE_TH_ERRORS
}
```

**关键点:**
- 解析 Python 参数转换为 C 类型
- **释放 GIL** 避免阻塞其他 Python 线程
- 调用 `raw_alloc_with_stream` 进入 C++ 实现

**注册到 Python:**
在同文件的 2058 行将函数注册为 Python 可调用:
```cpp
{"_cuda_cudaCachingAllocator_raw_alloc",
 THCPModule_cudaCachingAllocator_raw_alloc,
 METH_VARARGS, nullptr}
```

---

### 2.3 C++ 分配器接口

**文件:** `c10/cuda/CUDACachingAllocator.cpp:4237-4250`

```cpp
void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
  if (nbytes == 0) {
    return nullptr;
  }

  void* r = nullptr;

  // 检查是否强制使用非缓存分配器(调试模式)
  if (forceUncachedAllocator() || !isEnabled()) {
    r = uncached_allocate(nbytes);  // 直接 cudaMalloc
  } else {
    c10::DeviceIndex device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));  // 获取当前设备
    malloc(&r, device, nbytes, stream);  // 进入缓存分配逻辑
  }

  return r;
}
```

**关键点:**
- 环境变量 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 可强制绕过缓存
- 正常情况下调用 `malloc` 进入缓存分配逻辑

---

### 2.4 核心分配逻辑 (NativeCachingAllocator::malloc)

**文件:** `c10/cuda/CUDACachingAllocator.cpp:3820-3838`

```cpp
void malloc(
    void** devPtr,
    c10::DeviceIndex device,
    size_t size,
    cudaStream_t stream) {

  TORCH_INTERNAL_ASSERT(
      0 <= device && static_cast<size_t>(device) < device_allocator.size(),
      "Allocator not initialized for device ", device);

  // 调用每个设备专属的分配器
  Block* block = device_allocator[device]->malloc(size, stream);

  add_allocated_block(block);  // 登记到全局 hash map
  *devPtr = block->ptr;  // 返回内存指针

  // 通知 Python 解释器(用于性能分析)
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_allocation(c10::kCUDA,
                                           reinterpret_cast<uintptr_t>(*devPtr));
  }
}
```

**架构设计:**
- `NativeCachingAllocator`: 全局单例,管理所有 GPU 设备
- `device_allocator[]`: 每个 GPU 设备有独立的 `DeviceCachingAllocator`
- `allocated_blocks`: 分片 hash map (67个分片),记录所有活跃分配,用于 `free` 时查找

---

### 2.5 设备级分配器 (DeviceCachingAllocator::malloc)

**文件:** `c10/cuda/CUDACachingAllocator.cpp:1360-1509`

```cpp
Block* malloc(size_t orig_size, cudaStream_t stream) {
  // 1. 收集上下文信息(用于历史追踪)
  auto context = maybeGatherContext(RecordContext::STATE);

  std::unique_lock<std::recursive_mutex> lock(mutex);  // 加锁

  // 2. 处理已完成的跨流事件,回收可复用内存
  if (C10_LIKELY(captures_underway.empty())) {
    process_events(context);  // 检查 cudaEvent 是否完成
  } else {
    // CUDA Graph 捕获中,尝试复用安全的块
    free_safe_blocks_in_capture(context, stream);
  }

  // 3. 内存对齐和选择内存池
  size_t size = round_size(orig_size);  // 对齐到 512 字节
  auto& pool = get_pool(size, stream);  // 选择 small 或 large 池
  const size_t alloc_size = get_allocation_size(size);  // 计算实际分配大小

  AllocParams params(device_id, size, stream, &pool, alloc_size);
  params.stat_types = get_stat_types_for_pool(pool);

  // 4. 尝试从缓存池获取
  bool block_found =
      get_free_block(params)  // 从空闲块中查找
      || (trigger_free_memory_callbacks(params) && get_free_block(params));

  // 5. 缓存池没有,尝试分配新块
  if (!block_found) {
    // 5.1 垃圾回收
    if (set_fraction &&
        AcceleratorAllocatorConfig::garbage_collection_threshold() > 0.0) {
      garbage_collect_cached_blocks(context);
    }

    // 5.2 分配新块
    block_found = alloc_block(params, false, context, lock)
        // 5.3 释放部分缓存块后重试
        || (release_available_cached_blocks(params, context) &&
            alloc_block(params, false, context, lock))
        // 5.4 释放所有缓存块后重试
        || (C10_LIKELY(captures_underway.empty()) &&
            release_cached_blocks(context, {0, 0}) &&
            alloc_block(params, true, context, lock));
  }

  // 6. 仍然失败,尝试从私有池(CUDA Graph/MemPool)借用
  if (!block_found && params.err == cudaErrorMemoryAllocation) {
    bool active_pool = params.pool->owner_PrivatePool;
    if (!active_pool) {
      for (MempoolId_t mempool_id : use_on_oom_pools) {
        // ... 尝试从其他私有池分配 ...
      }
    }
  }

  // 7. 彻底失败,抛出 OOM
  if (!block_found) {
    size_t device_free = 0;
    size_t device_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));

    // 记录 OOM 追踪
    record_trace(TraceEntry::OOM, device_free, params.size(), ...);
    stats.num_ooms += 1;

    // 调用 OOM 观察者
    for (auto& observer : oom_observers_) {
      observer(device_id, allocated_bytes, device_total, device_free);
    }

    // 抛出详细错误信息
    TORCH_CHECK_WITH(OutOfMemoryError, false,
        "CUDA out of memory. Tried to allocate ", format_size(size), ". "
        "GPU ", device_id, " has a total capacity of ",
        format_size(device_total), " of which ", format_size(device_free),
        " is free. ...");
  }

  // 8. 成功分配,拆分块(如果过大)
  if (should_split(params.block, params.size())) {
    size_t remaining = params.block->size - params.size();
    Block* remaining_block = new Block(
        params.device(), params.stream(), remaining, params.pool,
        static_cast<char*>(params.block->ptr) + params.size());

    remaining_block->splice(params.block, params.block->next);
    params.pool->insert_into_blocks(remaining_block);

    params.block->size = params.size();
  }

  // 9. 标记为已分配,更新统计
  params.block->allocated = true;
  params.block->requested_size = orig_size;
  params.block->context_when_allocated = std::move(context);

  active_blocks.insert(params.block);

  // 更新统计信息
  for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
    stats.allocation[stat_type].increase(1);
    stats.allocated_bytes[stat_type].increase(params.size());
    stats.active_bytes[stat_type].increase(params.size());
  });

  // 记录追踪
  record_trace(TraceEntry::ALLOC, int64_t(params.block->ptr),
               orig_size, stream, device_id, ...);

  return params.block;
}
```

---

## 三、关键数据结构

### 3.1 Block (内存块)

**文件:** `c10/cuda/CUDACachingAllocator.cpp:185-249`

```cpp
struct Block {
  c10::DeviceIndex device;       // GPU 设备 ID
  cudaStream_t stream;            // 分配时所在的 stream
  stream_set stream_uses;         // 曾经使用过的其他 stream

  size_t size;                    // 块大小(字节)
  size_t requested_size;          // 用户请求的大小

  BlockPool* pool;                // 所属内存池(small/large)
  void* ptr;                      // 实际内存地址

  bool allocated;                 // 是否正在使用
  bool mapped;                    // 是否映射了物理内存(可扩展段)

  Block* prev;                    // 前一个块(同一段内存分割)
  Block* next;                    // 后一个块

  int event_count;                // 未完成的 CUDA 事件数
  int64_t gc_count_base;          // GC 计数基准

  std::shared_ptr<GatheredContext> context_when_allocated;  // 分配时堆栈
  std::shared_ptr<GatheredContext> context_when_segment_allocated;  // 段分配时堆栈

  ExpandableSegment* expandable_segment_;  // 可扩展段指针

  // ... 方法 ...
  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};
```

**说明:**
- 每个 `Block` 代表一段连续的 GPU 内存
- `prev/next` 形成双向链表,用于合并和分割
- `allocated=true` 表示用户正在使用,`false` 表示在缓存池中

---

### 3.2 BlockPool (内存池)

**文件:** `c10/cuda/CUDACachingAllocator.cpp:160-181`

```cpp
struct BlockPool {
  std::set<Block*, Comparison> blocks;     // 空闲块,按大小排序
  std::set<Block*, Comparison> unmapped;   // 未映射的块(可扩展段)
  const bool is_small;                     // 是否小块池
  PrivatePool* owner_PrivatePool;          // 所属私有池(CUDA Graph)
  int64_t get_free_blocks_call_count;      // GC 计数器

  // 向池中添加块,更新 GC 计数
  std::pair<std::set<Block*>::iterator, bool>
  insert_into_blocks(Block* block) {
    block->gc_count_base = get_free_blocks_call_count;
    return blocks.insert(block);
  }

  MempoolId_t owner_MempoolId() const;
};
```

**说明:**
- `blocks` 使用 `BlockComparatorSize` 排序,查找最小满足的块
- `unmapped` 用于可扩展段,存储未映射物理内存的虚拟地址块

---

### 3.3 DeviceCachingAllocator (设备分配器)

**文件:** `c10/cuda/CUDACachingAllocator.cpp:1162-1270`

```cpp
class DeviceCachingAllocator {
 private:
  std::recursive_mutex mutex;              // 线程安全锁
  DeviceStats stats;                       // 统计信息
  c10::DeviceIndex device_id;              // 设备 ID

  BlockPool large_blocks;                  // 大块池 (>1MB)
  BlockPool small_blocks;                  // 小块池 (≤1MB)

  ska::flat_hash_set<Block*> active_blocks;  // 活跃分配

  // CUDA Graph 支持
  std::vector<std::pair<MempoolId_t,
      std::function<bool(cudaStream_t)>>> captures_underway;
  ska::flat_hash_map<MempoolId_t,
      std::unique_ptr<PrivatePool>> graph_pools;  // 私有池
  ska::flat_hash_map<Block*,
      std::vector<cudaGraphNode_t>> deferred_blocks;  // 延迟释放

  // 事件管理
  ska::flat_hash_map<cuda::CUDAStream,
      std::deque<std::pair<EventPool::Event, Block*>>> cuda_events;

  size_t total_allocated_memory;           // 总分配量
  size_t allowed_memory_maximum;           // 允许的最大值

  std::vector<ExpandableSegment*> expandable_segments_;  // 可扩展段

  bool set_fraction;                       // 是否设置了内存限制
  bool record_history;                     // 是否记录历史

  RingBuffer<TraceEntry> alloc_buffer;     // 追踪环形缓冲

  std::vector<OutOfMemoryObserver> oom_observers_;  // OOM 回调

 public:
  explicit DeviceCachingAllocator(c10::DeviceIndex id);
  Block* malloc(size_t orig_size, cudaStream_t stream);
  void free(Block* block);
  // ... 其他方法 ...
};
```

---

## 四、内存分配策略

### 4.1 池选择策略

**文件:** `c10/cuda/CUDACachingAllocator.cpp:2956-2979`

```cpp
BlockPool& get_pool(size_t size, cudaStream_t stream) {
  // 常量定义
  // kSmallSize = 1MB
  // kSmallBuffer = 2MB
  // kLargeBuffer = 20MB
  // kMinLargeAlloc = 10MB

  // 1. 检查是否在 CUDA Graph 捕获中
  if (C10_UNLIKELY(!captures_underway.empty())) {
    for (auto& entry : captures_underway) {
      if (entry.second(stream)) {  // 匹配 stream filter
        auto it = graph_pools.find(entry.first);
        if (size <= kSmallSize) {
          return it->second->small_blocks;  // Graph 私有小池
        } else {
          return it->second->large_blocks;  // Graph 私有大池
        }
      }
    }
  }

  // 2. 正常情况,根据大小选择池
  if (size <= kSmallSize) {  // ≤ 1MB
    return small_blocks;
  } else {
    return large_blocks;
  }
}
```

**分池原因:**
- 小块频繁分配/释放,单独管理减少碎片
- 大块分配策略不同,避免浪费

---

### 4.2 大小对齐和实际分配大小

**文件:** `c10/cuda/CUDACachingAllocator.cpp:2999-3007`

```cpp
static size_t get_allocation_size(size_t size) {
  if (size <= kSmallSize) {  // ≤ 1MB
    return kSmallBuffer;     // 分配 2MB
  } else if (size < kMinLargeAlloc) {  // 1MB < size < 10MB
    return kLargeBuffer;     // 分配 20MB
  } else {  // ≥ 10MB
    // 向上对齐到 kRoundLarge (2MB)
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
  }
}
```

**策略说明:**
- **小请求** (≤1MB): 统一分配 2MB,打包多个小分配
- **中等请求** (1-10MB): 分配 20MB,预留空间减少碎片
- **大请求** (≥10MB): 向上对齐到 2MB 边界

---

### 4.3 从缓存池查找

**文件:** `c10/cuda/CUDACachingAllocator.cpp:3009-3067`

```cpp
bool get_free_block(AllocParams& p) {
  BlockPool& pool = *p.pool;

  // 更新 GC 计数
  if (C10_UNLIKELY(set_fraction &&
      AcceleratorAllocatorConfig::garbage_collection_threshold() > 0.0)) {
    ++pool.get_free_blocks_call_count;
  }

  // lower_bound: 查找 >= size 的最小块
  auto it = pool.blocks.lower_bound(&p.search_key);

  // 检查 stream 是否匹配
  if (it == pool.blocks.end() || (*it)->stream != p.stream())
    return false;

  // 可扩展段特殊处理(见 4.6 节)
  if ((*it)->expandable_segment_) {
    // ... 特殊逻辑 ...
  }

  // 不要为大请求返回超大块
  if ((p.size() < AcceleratorAllocatorConfig::max_split_size()) &&
      ((*it)->size >= AcceleratorAllocatorConfig::max_split_size()))
    return false;

  // 允许超大块,但限制浪费范围
  if ((p.size() >= AcceleratorAllocatorConfig::max_split_size()) &&
      ((*it)->size >= p.size() +
       AcceleratorAllocatorConfig::max_non_split_rounding_size()))
    return false;

  p.block = *it;
  pool.blocks.erase(it);
  return true;
}
```

**查找策略:**
1. 使用 `lower_bound` 找到最小满足的块 (Best Fit)
2. **必须** stream 匹配,确保内存安全
3. 避免浪费:大请求不使用超大缓存块
4. `max_split_size` (默认无限): 超过此大小的块不拆分

---

### 4.4 分配新块

**文件:** `c10/cuda/CUDACachingAllocator.cpp:3142-3255`

```cpp
bool alloc_block(
    AllocParams& p,
    bool isRetry,
    const std::shared_ptr<GatheredContext>& ctx,
    std::unique_lock<std::recursive_mutex>& lock) {

  C10_CUDA_CHECK(cudaGetLastError());  // 检查之前的错误

  size_t size = p.alloc_size;
  void* ptr = nullptr;

  if (isRetry) {
    stats.num_alloc_retries += 1;
  }

  // 1. 检查是否超过限制
  if (set_fraction &&
      total_allocated_memory + size > allowed_memory_maximum) {
    p.err = cudaErrorMemoryAllocation;
    return false;
  }

  // 2. 选择分配方式
  if (CUDAAllocatorConfig::expandable_segments() && ...) {
    // 2.1 可扩展段方式 (见第五节)
    p.block = try_allocate_expandable_block(
        p.device(), p.stream(), p.pool, p.size(), ctx);
    if (p.block) {
      p.err = cudaSuccess;
    } else {
      p.err = cudaErrorMemoryAllocation;
    }
  } else {
    // 2.2 传统 cudaMalloc
    if (CUDAAllocatorConfig::release_lock_on_cudamalloc()) {
      auto sg = c10::make_scope_exit([&]() { lock.lock(); });
      lock.unlock();  // 临时释放锁,避免阻塞其他线程
      p.err = cudaMallocMaybeCapturing(&ptr, size, p);
    } else {
      p.err = cudaMallocMaybeCapturing(&ptr, size, p);
    }

    if (p.err != cudaSuccess) {
      if (p.err == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();  // 清除错误状态
      } else {
        C10_CUDA_CHECK(p.err);  // 其他错误立即抛出
      }
      return false;
    }
  }

  // 3. 更新统计
  if (p.pool->owner_PrivatePool) {
    p.pool->owner_PrivatePool->cudaMalloc_count++;
  }

  total_allocated_memory += size;

  // 4. 创建 Block
  p.block = new Block(
      p.device(), p.stream(), size, p.pool, static_cast<char*>(ptr));

  for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
    stats.segment[stat_type].increase(1);
    stats.reserved_bytes[stat_type].increase(size);
  });

  if (size >= AcceleratorAllocatorConfig::max_split_size())
    stats.oversize_segments.increase(1);

  stats.num_device_alloc++;

  // 5. 记录追踪
  record_trace(
      TraceEntry::SEGMENT_ALLOC,
      int64_t(p.block->ptr),
      p.block->size,
      p.stream(),
      p.device(),
      p.pool->owner_MempoolId(),
      ctx);

  p.block->context_when_segment_allocated = ctx;

  return true;
}
```

**关键点:**
- `release_lock_on_cudamalloc`: 调用 `cudaMalloc` 时释放锁
- `cudaMallocMaybeCapturing`: 支持 CUDA Graph 捕获
- 失败时清除错误状态,允许重试

---

### 4.5 垃圾回收

**文件:** `c10/cuda/CUDACachingAllocator.cpp:3078-3135`

```cpp
void garbage_collect_cached_blocks(
    const std::shared_ptr<GatheredContext>& context) {

  // 1. 计算 GC 阈值
  size_t gc_threshold = static_cast<size_t>(
      AcceleratorAllocatorConfig::garbage_collection_threshold() *
      static_cast<double>(allowed_memory_maximum));

  // 未达到阈值,不触发 GC
  if (total_allocated_memory <= gc_threshold) {
    return;
  }

  const auto target_size = total_allocated_memory - gc_threshold;
  size_t gc_reclaimed = 0;

  // 2. 计算平均年龄
  size_t total_age = 0;
  int freeable_block_count = 0;
  for (auto& b : large_blocks.blocks) {
    if (!b->is_split()) {  // 只回收未拆分的块
      total_age += b->gc_count();  // 年龄 = 距离上次使用的时间
      ++freeable_block_count;
    }
  }

  if (freeable_block_count == 0) {
    return;
  }

  // 3. 迭代回收
  bool block_freed = true;
  while (gc_reclaimed < target_size && block_freed &&
         freeable_block_count > 0) {

    double age_threshold =
        static_cast<double>(total_age) / freeable_block_count;
    block_freed = false;

    // 回收年龄超过平均值的块
    auto it = large_blocks.blocks.begin();
    while (it != large_blocks.blocks.end()) {
      Block* block = *it;
      ++it;

      if (!block->is_split() && !block->expandable_segment_ &&
          static_cast<double>(block->gc_count()) >= age_threshold) {
        block_freed = true;
        gc_reclaimed += block->size;
        total_age -= block->gc_count();
        freeable_block_count--;
        release_block(block, context);  // 归还给 CUDA
      }
    }
  }
}
```

**GC 策略:**
- **触发条件**: 总内存超过 `gc_threshold`
- **回收对象**: 未拆分、非可扩展段、年龄超过平均值的块
- **年龄计算**: 自上次插入池后,`get_free_blocks_call_count` 的增量
- **目标**: 回收到低于阈值

---

## 五、可扩展段 (Expandable Segments)

### 5.1 问题背景

**文件:** `c10/cuda/CUDACachingAllocator.cpp:265-299`

传统分配方式的问题:

```
场景: 批次大小动态变化

初始批次 N:
  分配 N*A, N*B, N*A*B ... (恰好大小)

批次 N+1:
  需要 (N+1)*A, (N+1)*B, (N+1)*A*B

问题:
  - 旧的 N*A 块太小,无法复用
  - 新分配 (N+1)*A,旧块留下尾部碎片
  - 50 层模型 → 50+ 个碎片段
  - 无法回收这些碎片(被占用)

结果: OOM,即使总碎片足够
```

**图示:**
```
传统方式:
Segment1: [N*A 使用] [尾部碎片]
Segment2: [N*B 使用] [尾部碎片]
Segment3: [(N+1)*A*B 新分配]  ← OOM!

可扩展段:
ExpandableSeg: [N*A][N*B][扩展→][N+1)*A*B]  ← 动态增长
```

---

### 5.2 核心思想

使用 **CUDA Driver API** 的虚拟内存管理:
- `cuMemAddressReserve`: 预留虚拟地址空间
- `cuMemCreate`: 创建物理内存句柄
- `cuMemMap`: 映射物理内存到虚拟地址
- `cuMemUnmap`: 解除映射

**流程:**
1. 创建时预留大块虚拟地址 (如 2GB)
2. 初始只映射小部分物理内存 (如 2MB)
3. 需要更多内存时,动态映射更多物理页
4. 释放时解除映射,归还物理内存

---

### 5.3 数据结构

**文件:** `c10/cuda/CUDACachingAllocator.cpp:321-758`

```cpp
struct ExpandableSegment {
  c10::DeviceIndex device_;              // 设备 ID
  std::optional<cudaStream_t> stream_;   // 所属 stream

  CUdeviceptr ptr_;                      // 虚拟地址起点
  size_t segment_size_;                  // 每个段的大小 (如 2MB)
  size_t mapped_size_;                   // 已映射物理内存大小
  size_t max_handles_;                   // 最大句柄数 (虚拟地址/段大小)

  struct Handle {
    CUmemGenericAllocationHandle handle;  // 物理内存句柄
    std::optional<std::variant<int, CUmemFabricHandle>> shareable_handle;
  };

  std::vector<std::optional<Handle>> handles_;  // 每个段的句柄
  std::vector<c10::DeviceIndex> peers_;         // P2P 设备

  // 构造函数: 预留虚拟地址
  ExpandableSegment(
      c10::DeviceIndex device,
      std::optional<cudaStream_t> stream,
      size_t segment_size,
      std::vector<c10::DeviceIndex> peers);

  // 映射物理内存
  SegmentRange map(SegmentRange range);

  // 解除映射
  SegmentRange unmap(SegmentRange range);

  // IPC 共享
  SegmentRange share(SegmentRange range, std::ostream& buf);
  static std::unique_ptr<ExpandableSegment> fromShared(...);

  char* ptr() const { return reinterpret_cast<char*>(ptr_); }
  size_t size() const { return max_handles_ * segment_size_; }
  size_t getMappedSize() const { return mapped_size_; }
};
```

---

### 5.4 构造函数: 预留虚拟地址

**文件:** `c10/cuda/CUDACachingAllocator.cpp:325-391`

```cpp
ExpandableSegment(
    c10::DeviceIndex device,
    std::optional<cudaStream_t> stream,
    size_t segment_size,
    std::vector<c10::DeviceIndex> peers)
    : device_(device),
      stream_(stream),
      segment_size_(segment_size),
      mapped_size_(0),
      peers_(std::move(peers)) {

  // 1. 查询可用虚拟地址
  size_t free = 0, total = 0;
  C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemGetInfo_(&free, &total));

  size_t expected_avail = free + CUDAAllocatorConfig::pinned_max_register_threads() *
                          CUDAAllocatorConfig::pinned_num_register_threads();

  // 2. 计算最大句柄数 (虚拟地址范围)
  // 最多预留 2 倍可用内存,分成 segment_size_ 大小的段
  max_handles_ = expected_avail * 2 / segment_size_;

  size_t size = segment_size_ * max_handles_;

  // 3. 预留虚拟地址
  CUmemAddressReserveHandle reserve_handle;
  C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressReserve_(
      &ptr_, size, 0ULL, 0, 0, &reserve_handle));

  // 预留成功,虚拟地址范围: [ptr_, ptr_ + size)
  // 此时未映射任何物理内存,mapped_size_ = 0
}
```

**关键点:**
- 虚拟地址预留不消耗物理内存
- `max_handles_` 限制最大可扩展大小
- `segment_size_` 通常为 2MB (CUDA 最小页大小)

---

### 5.5 映射物理内存

**文件:** `c10/cuda/CUDACachingAllocator.cpp:392-472`

```cpp
SegmentRange map(SegmentRange range) {
  // 1. 计算需要映射的段范围
  auto begin = segmentLeft(range.ptr);    // 向下对齐
  auto end = segmentRight(range.ptr + range.size);  // 向上对齐

  // 2. 检查是否已全部映射
  if (begin + 1 >= end) {
    return SegmentRange(nullptr, 0);  // 已映射,无需操作
  }

  // 检查是否映射完整
  bool all_mapped = true;
  for (auto i : c10::irange(begin, end)) {
    if (!handles_.at(i)) {
      all_mapped = false;
      break;
    }
  }
  if (all_mapped) {
    return rangeFromHandles(begin, end);
  }

  // 3. 扩展 handles_ 数组
  while (end > handles_.size()) {
    handles_.emplace_back(std::nullopt);
  }

  // 4. 为每个段创建物理内存
  for (auto i : c10::irange(begin, end)) {
    if (handles_.at(i)) continue;  // 已映射,跳过

    CUmemGenericAllocationHandle handle = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  // 固定内存
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<int>(device_);

    // 支持 IPC 共享
    if (CUDAAllocatorConfig::expandable_segments_handle_type() != ...) {
      prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    } else {
      prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    }

    // 分配物理内存
    auto status = DriverAPI::get()->cuMemCreate_(&handle, segment_size_, &prop, 0);

    if (status != CUDA_SUCCESS) {
      if (status == CUDA_ERROR_OUT_OF_MEMORY) {
        // OOM,回滚已分配的句柄
        for (auto j : c10::irange(begin, i)) {
          auto h = handles_.at(j).value();
          handles_.at(j) = std::nullopt;
          C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h.handle));
        }
        trimHandles();
        return rangeFromHandles(begin, begin);  // 返回空范围
      } else {
        C10_CUDA_DRIVER_CHECK(status);
      }
    }

    handles_.at(i) = Handle{handle, std::nullopt};
  }

  // 5. 映射到虚拟地址
  mapAndSetAccess(begin, end);

  return rangeFromHandles(begin, end);
}

void mapAndSetAccess(size_t begin, size_t end) {
  for (auto i : c10::irange(begin, end)) {
    auto h = handles_.at(i).value();

    // 映射物理内存到虚拟地址
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemMap_(
        ptr_ + segment_size_ * i,  // 虚拟地址
        segment_size_,              // 大小
        0,
        h.handle,                   // 物理内存句柄
        0));

    // 设置访问权限
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = static_cast<int>(device_);
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
        ptr_ + segment_size_ * i, segment_size_, &desc, 1));

    // 为 P2P 设备设置访问权限
    for (auto peer : peers_) {
      desc.location.id = static_cast<int>(peer);
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
          ptr_ + segment_size_ * i, segment_size_, &desc, 1));
    }
  }

  mapped_size_ += (end - begin) * segment_size_;
}
```

**映射流程:**
1. 计算需要的段范围 `[begin, end)`
2. 为每个段调用 `cuMemCreate_` 分配物理内存
3. 调用 `cuMemMap_` 映射到虚拟地址
4. 调用 `cuMemSetAccess_` 设置访问权限
5. 更新 `mapped_size_`

---

### 5.6 分配器集成

**文件:** `c10/cuda/CUDACachingAllocator.cpp:2697-2842`

```cpp
Block* try_allocate_expandable_block(
    c10::DeviceIndex device,
    cudaStream_t stream,
    BlockPool* pool,
    size_t size,
    const std::shared_ptr<GatheredContext>& ctx) {

  // 1. 查找或创建可扩展段
  Block* candidate = find_expandable_block(device, stream, pool, size);

  // 2. 如果是未映射的块,映射物理内存
  if (!candidate->mapped &&
      !map_block(candidate, std::min(candidate->size, size), ctx)) {
    return nullptr;  // 映射失败 (OOM)
  }

  TORCH_INTERNAL_ASSERT(candidate->mapped);

  // 3. 如果块不够大,继续映射后续块
  while (candidate->size < size) {
    auto remaining = size - candidate->size;
    auto new_candidate = candidate->next;

    if (!map_block(new_candidate,
                   std::min(remaining, candidate->next->size), ctx)) {
      return nullptr;
    }

    candidate = new_candidate;  // 合并后的块
  }

  // 4. 从空闲池中移除
  pool->blocks.erase(candidate);

  return candidate;
}

bool map_block(
    Block* to_map,
    size_t size,
    const std::shared_ptr<GatheredContext>& ctx) {

  TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);

  // 1. 调用 ExpandableSegment::map
  auto mapped_range =
      to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});

  // 2. 映射失败
  if (mapped_range.size == 0) {
    return false;
  }

  TORCH_INTERNAL_ASSERT(
      mapped_range.ptr == to_map->ptr && mapped_range.size >= size);

  BlockPool& pool = *to_map->pool;
  pool.unmapped.erase(to_map);
  to_map->mapped = true;

  // 3. 如果映射的大小小于块大小,拆分
  if (mapped_range.size < to_map->size) {
    Block* remaining = new Block(
        to_map->device,
        to_map->stream,
        to_map->size - mapped_range.size,
        &pool,
        static_cast<char*>(to_map->ptr) + mapped_range.size);

    remaining->mapped = false;
    remaining->expandable_segment_ = to_map->expandable_segment_;
    remaining->splice(to_map, to_map->next);

    pool.unmapped.insert(remaining);  // 未映射部分放回 unmapped
    to_map->size = mapped_range.size;
  }

  // 4. 尝试与相邻块合并
  try_merge_blocks(to_map, to_map->prev, pool);
  try_merge_blocks(to_map, to_map->next, pool);

  pool.insert_into_blocks(to_map);

  // 5. 更新统计
  total_allocated_memory += mapped_range.size;
  for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
    stats.reserved_bytes[stat_type].increase(mapped_range.size);
  });

  stats.num_device_alloc++;
  record_trace(TraceEntry::SEGMENT_MAP, ...);

  return true;
}
```

**可扩展段的优势:**
1. **减少碎片**: 所有块在同一虚拟地址段内,连续增长
2. **按需映射**: 只映射实际使用的物理内存
3. **动态扩展**: 批次大小变化时,直接扩展现有段

---

### 5.7 示例场景

```
初始状态:
ExpandableSegment:
  虚拟地址: [0x7f00_0000, 0x7f00_0000 + 2GB)  (预留)
  物理映射: [0x7f00_0000, 0x7f00_0000 + 2MB)  (初始映射)

分配 1MB:
  从已映射的 2MB 中分割,无需 cuMemMap

分配 5MB:
  当前已映射 2MB,不足
  调用 map(5MB):
    - 计算需要 3 个段 (5MB / 2MB = 2.5 → 3)
    - cuMemCreate_ 分配 3 个 2MB 物理内存
    - cuMemMap_ 映射到 [0x7f00_0200, 0x7f00_0200 + 6MB)
    - 总映射: 2MB + 6MB = 8MB

释放 1MB 块:
  标记为空闲,放回 blocks 池
  物理内存仍然映射,可立即复用

GC 时:
  调用 unmap([1MB 块的地址, 1MB]):
    - 解除映射对应的段
    - cuMemRelease_ 归还物理内存
    - 虚拟地址保留,可再次映射
```

---

## 六、跨流使用与事件同步

### 6.1 问题

```python
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    x = torch.empty(1000, 1000, device='cuda')  # 在 stream1 分配

with torch.cuda.stream(stream2):
    y = x + 1  # 在 stream2 使用 x 的内存
```

**风险:**
- `x` 在 stream1 分配
- 如果 `x` 释放,内存可能被 stream1 复用
- 但 stream2 可能还在使用这块内存
- **竞争条件 (Race Condition)**

---

### 6.2 解决方案: recordStream

**文件:** `c10/cuda/CUDACachingAllocator.cpp:1545-1585`

```python
# 用户代码
x.record_stream(stream2)  # 告诉分配器: stream2 也在使用 x
```

```cpp
void recordStream(Block* block, cuda::CUDAStream stream) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  // 1. 检查是否在 CUDA Graph 捕获中
  if (stream.stream() == block->stream ||
      stream.stream() == get_default_stream()) {
    return;  // 同一 stream 或默认 stream,无需记录
  }

  // 2. 记录使用的 stream
  if (in_cudagraph_capture(stream.stream())) {
    // CUDA Graph 捕获中,延迟处理
    block_to_cudagraph_stream_uses[block].insert(stream);
  } else {
    block->stream_uses.insert(stream);
  }
}
```

**释放时的处理:**

**文件:** `c10/cuda/CUDACachingAllocator.cpp:1595-1650`

```cpp
void free(Block* block) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  block->allocated = false;

  // 1. 获取上下文
  auto context = maybeGatherContext(RecordContext::ALL);

  // 2. 记录 FREE_REQUESTED 事件
  record_trace(TraceEntry::FREE_REQUESTED, ...);

  StatTypes stat_types = get_stat_types_for_pool(*block->pool);
  for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
    stats.allocation[stat_type].decrease(1);
    stats.allocated_bytes[stat_type].decrease(block->size);
  });

  // 3. 检查是否有跨流使用
  if (!block->stream_uses.empty()) {
    // 有其他 stream 使用过,插入事件
    insert_events(block);
  }

  // 4. 检查事件是否完成
  if (block->event_count > 0) {
    // 事件未完成,暂不归还缓存池
    // 等待 process_events() 检查
    return;
  }

  // 5. 立即归还缓存池
  free_block(block, context);
}
```

**事件插入:**

**文件:** `c10/cuda/CUDACachingAllocator.cpp:3581-3598`

```cpp
void insert_events(Block* block) {
  c10::DeviceIndex prev_device = 0;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&prev_device));

  stream_set streams(std::move(block->stream_uses));
  AT_ASSERT(block->stream_uses.empty());

  // 为每个使用的 stream 插入事件
  for (auto& stream : streams) {
    C10_CUDA_CHECK(c10::cuda::SetDevice(stream.device_index()));

    // 创建 CUDA 事件
    EventPool::Event event = create_event_internal(stream.device_index());
    C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));

    block->event_count++;
    cuda_events[stream].emplace_back(std::move(event), block);
  }

  C10_CUDA_CHECK(c10::cuda::MaybeSetDevice(prev_device));
}
```

**事件检查:**

**文件:** `c10/cuda/CUDACachingAllocator.cpp:3619-3670`

```cpp
void process_events(const std::shared_ptr<GatheredContext>& context) {
  insert_events_deferred_until_no_capture(context);

  // 遍历每个 stream 的事件队列
  for (auto it = cuda_events.begin(); it != cuda_events.end();) {
    while (!it->second.empty()) {
      auto& e = it->second.front();
      EventPool::Event event = std::move(e.first);
      Block* block = e.second;

      // 查询事件是否完成
      cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));

      if (err == cudaErrorNotReady) {
        // 未完成,保留事件
        (void)cudaGetLastError();
        e.first = std::move(event);
        break;  // 停止处理此 stream 的后续事件
      } else if (err != cudaSuccess) {
        C10_CUDA_CHECK(err);
      }

      // 事件完成,减少计数
      block->event_count--;
      if (block->event_count == 0) {
        // 所有事件完成,归还缓存池
        free_block(block, context);
      }

      // 移除已完成的事件
      it->second.pop_front();
    }

    // 队列为空,移除此 stream
    if (it->second.empty()) {
      it = cuda_events.erase(it);
    } else {
      ++it;
    }
  }
}
```

---

### 6.3 流程图

```
分配 (stream1):
  Block.stream = stream1
  Block.stream_uses = {}

跨流使用 (stream2):
  record_stream(block, stream2)
  Block.stream_uses = {stream2}

释放:
  free(block)
    → allocated = false
    → 发现 stream_uses 非空
    → insert_events(block):
        创建 cudaEvent
        cudaEventRecord(event, stream2)
        cuda_events[stream2].push({event, block})
        block.event_count = 1
    → 暂不归还缓存池

下次分配:
  malloc(...)
    → process_events():
        cudaEventQuery(event)
        → 完成: block.event_count--
        → event_count == 0: free_block(block)
            归还缓存池
```

---

## 七、CUDA Graph 支持

### 7.1 问题

CUDA Graph 将一系列操作"冻结"成图,重放时使用相同的内存地址。

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    x = torch.empty(1000, 1000, device='cuda')  # 地址 0x7f00_1000
    y = x + 1

g.replay()  # x 必须仍在 0x7f00_1000
```

**挑战:**
- 图记录的地址必须在重放时有效
- 缓存分配器会复用内存,地址可能变化

---

### 7.2 解决方案: 私有池

**文件:** `c10/cuda/CUDACachingAllocator.cpp:1182-1244`

```cpp
class DeviceCachingAllocator {
  // 捕获中的图
  std::vector<std::pair<MempoolId_t,
      std::function<bool(cudaStream_t)>>> captures_underway;

  // 图的私有池
  ska::flat_hash_map<MempoolId_t,
      std::unique_ptr<PrivatePool>, MempoolIdHash> graph_pools;

  // 可释放的私有池
  ska::flat_hash_map<MempoolId_t,
      PrivatePool*, MempoolIdHash> graph_pools_freeable;
};
```

**私有池结构:**

```cpp
struct PrivatePool {
  MempoolId_t id;
  BlockPool small_blocks;
  BlockPool large_blocks;
  int use_count;  // 引用计数
  CUDAAllocator* allocator;  // 自定义分配器
  int cudaMalloc_count;

  PrivatePool(MempoolId_t id, CUDAAllocator* allocator = nullptr)
      : id(id),
        small_blocks(true, this),
        large_blocks(false, this),
        use_count(1),
        allocator(allocator),
        cudaMalloc_count(0) {}
};
```

---

### 7.3 捕获流程

**文件:** `c10/cuda/CUDACachingAllocator.cpp:2425-2485`

```cpp
void beginAllocateToPool(
    MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  // 将 mempool_id 和 filter 加入 captures_underway
  captures_underway.emplace_back(mempool_id, std::move(filter));

  auto context = maybeGatherContext(RecordContext::STATE);
  insert_events_deferred_until_no_capture(context);
}

void endAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  // 从 captures_underway 移除
  captures_underway.erase(
      std::remove_if(captures_underway.begin(), captures_underway.end(),
          [&](const auto& entry) { return entry.first == mempool_id; }),
      captures_underway.end());
}
```

**分配时的池选择 (见 4.1 节):**

```cpp
BlockPool& get_pool(size_t size, cudaStream_t stream) {
  // 检查是否在捕获中
  if (C10_UNLIKELY(!captures_underway.empty())) {
    for (auto& entry : captures_underway) {
      if (entry.second(stream)) {  // filter 匹配
        auto it = graph_pools.find(entry.first);
        TORCH_INTERNAL_ASSERT(it != graph_pools.end());

        if (size <= kSmallSize) {
          return it->second->small_blocks;  // 私有小池
        } else {
          return it->second->large_blocks;  // 私有大池
        }
      }
    }
  }

  // 正常池
  // ...
}
```

---

### 7.4 私有池的生命周期

```python
# 1. 开始捕获
torch.cuda.graph._begin_capture(mempool_id)
  → beginAllocateToPool(mempool_id, filter)
  → captures_underway.push(mempool_id, filter)
  → graph_pools[mempool_id] = new PrivatePool(...)

# 2. 捕获期间的分配
x = torch.empty(..., device='cuda')
  → malloc(..., stream)
  → get_pool(size, stream)
      → 匹配 filter → 返回 graph_pools[mempool_id]->small_blocks
  → 从私有池分配

# 3. 结束捕获
torch.cuda.graph._end_capture(mempool_id)
  → endAllocateToPool(mempool_id)
  → captures_underway.erase(mempool_id)
  → graph_pools[mempool_id] 保留

# 4. 重放
g.replay()
  → 使用相同的内存地址 (私有池保留)

# 5. 销毁图
del g
  → releasePool(mempool_id)
  → graph_pools[mempool_id]->use_count--
  → use_count == 0:
      → graph_pools_freeable[mempool_id] = pool
      → 后续可被 free_cached_blocks 释放
```

---

## 八、配置选项

### 8.1 环境变量

```bash
# 禁用缓存分配器 (调试用)
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# 启用可扩展段
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置最大分割大小 (默认无限)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 垃圾回收阈值 (0.0-1.0)
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8

# 释放锁模式 (cudaMalloc 时释放全局锁)
export PYTORCH_CUDA_ALLOC_CONF=release_lock_on_malloc:True
```

---

### 8.2 Python API

```python
# 设置内存限制
torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # 最多用 50%

# 清空缓存
torch.cuda.empty_cache()

# 内存统计
print(torch.cuda.memory_allocated())  # 已分配
print(torch.cuda.memory_reserved())   # 已保留(包括缓存)
print(torch.cuda.max_memory_allocated())  # 峰值

# 内存快照
snapshot = torch.cuda.memory_snapshot()

# 内存总结
print(torch.cuda.memory_summary())
```

---

## 九、调试技巧

### 9.1 启用历史追踪

```python
torch.cuda.memory._record_memory_history(
    enabled=True,
    context='state',  # 'state', 'alloc', 'all'
    stacks='all',
    max_entries=100000
)

# ... 运行代码 ...

snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

分析快照:
```bash
python -m torch.cuda._memory_viz trace_plot memory_snapshot.pickle -o trace.html
```

---

### 9.2 查看分配堆栈

```python
torch.cuda.memory._record_memory_history(enabled=True, stacks='all')

x = torch.empty(1000, 1000, device='cuda')

snapshot = torch.cuda.memory._snapshot()
for trace in snapshot['device_traces'][0]:
    if trace['action'] == 'alloc':
        print(f"Address: {trace['addr']}")
        print(f"Size: {trace['size']}")
        if 'frames' in trace:
            for frame in trace['frames']:
                print(f"  {frame['filename']}:{frame['line']} in {frame['name']}")
```

---

### 9.3 检测内存泄漏

```python
import gc

torch.cuda.empty_cache()
gc.collect()

before = torch.cuda.memory_allocated()

# ... 运行可疑代码 ...

torch.cuda.empty_cache()
gc.collect()

after = torch.cuda.memory_allocated()

if after > before:
    print(f"Leaked: {after - before} bytes")
```

---

## 十、总结

### 调用链路回顾

```
Python:
  torch.cuda.caching_allocator_alloc(size, stream)
    ↓
  torch._C._cuda_cudaCachingAllocator_raw_alloc(size, stream)

Python 绑定 (torch/csrc/cuda/Module.cpp):
  THCPModule_cudaCachingAllocator_raw_alloc
    ↓ (释放 GIL)
  c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream)

C++ 分配器 (c10/cuda/CUDACachingAllocator.cpp):
  NativeCachingAllocator::raw_alloc_with_stream
    ↓
  NativeCachingAllocator::malloc(&ptr, device, size, stream)
    ↓
  device_allocator[device]->malloc(size, stream)
    ↓
  DeviceCachingAllocator::malloc(size, stream)
    ↓
  ┌─ process_events()              # 回收跨流内存
  ├─ get_pool(size, stream)        # 选择池
  ├─ get_free_block(params)        # 查找缓存
  ├─ alloc_block(params, ...)      # 分配新块
  │   ├─ cudaMallocMaybeCapturing  # 传统方式
  │   └─ try_allocate_expandable_block  # 可扩展段
  │       └─ ExpandableSegment::map
  │           └─ cuMemCreate + cuMemMap
  └─ 返回 Block*
```

### 关键机制

1. **缓存复用**: 避免频繁 `cudaMalloc`/`cudaFree`
2. **流感知**: 同一 stream 的分配可安全复用
3. **双池设计**: small (≤1MB) / large (>1MB) 分别管理
4. **可扩展段**: 动态映射物理内存,减少碎片
5. **跨流同步**: 通过 `cudaEvent` 确保内存安全
6. **CUDA Graph**: 私有池保证地址稳定性
7. **垃圾回收**: 基于年龄的 LRU 策略

### 性能优化建议

1. **减少跨流使用**: 尽量在同一 stream 分配和使用
2. **及时释放**: 不再使用的张量立即 `del`
3. **定期清空缓存**: `torch.cuda.empty_cache()`
4. **启用可扩展段**: 批次大小变化时减少碎片
5. **设置内存限制**: 避免 OOM 时系统卡死
