# PyTorch `torch.utils.data.DataLoader` 调用链全解

本文面向需要从 Python API 一直理解到底层 worker 的开发者，梳理 `DataLoader` 的主要入口、状态机、线程/进程模型以及与 C++ 模块的交互。文中所有行号均来自当前仓库。

## 1. Python API 入口 (`torch/utils/data/dataloader.py`)

1. **构造函数**（`DataLoader.__init__`, `torch/utils/data/dataloader.py:141-425`）  
   - 校验 `num_workers`、`prefetch_factor`、`persistent_workers` 等参数，并把 IterableDataPipe/MapDataPipe 包装成可序列化对象。  
   - 根据 `Dataset` 类型决定 `_dataset_kind`（Map/Iterable），并在 Iterable 模式下强制使用 `_InfiniteConstantSampler`（`lines 93-101`）。  
   - 依据 `shuffle`、`sampler`、`batch_sampler`、`batch_size` 推导出最终的 `_index_sampler`（batch 模式使用 `BatchSampler`，见 `lines 387-399`）。  
   - 记录 `collate_fn`（默认 `default_collate` 或 `default_convert`）、`pin_memory`、`timeout` 等配置。

2. **迭代器生成**：  
   - `DataLoader.__iter__`（`lines 486-500`）在每次迭代时调用 `_get_iterator`，若 `persistent_workers=True` 则复用已有迭代器并调用 `_reset`。  
   - `_get_iterator`（`lines 427-432`）根据 `num_workers` 返回 `_SingleProcessDataLoaderIter` 或 `_MultiProcessingDataLoaderIter`。这两个类都继承自 `_BaseDataLoaderIter`。

3. **`_BaseDataLoaderIter` 初始化**（`lines 641-775`）  
   - 复制 `dataset` 引用，如果是 DataPipe 且处于分布式环境会通过 `_share_dist_seed` 把种子广播给所有 rank（`lines 641-655`）。  
   - 计算 `_index_sampler` 的迭代器、`_base_seed`，处理 `pin_memory`/`pin_memory_device` 的可用性，设置 `_profile_name` 供 profiler 使用。  
   - `_reset`（`lines 717-727`）会重新建立 `sampler` 迭代器，并且在 DataPipe 场景下重新下发共享随机种子。

4. **公共迭代逻辑**：`__next__`（`lines 735-757`）调用子类 `_next_data`，并维护 IterableDataset 的 `__len__` 误差告警。

## 2. 单进程数据流 (`_SingleProcessDataLoaderIter`)

- 位置：`torch/utils/data/dataloader.py:775-818`。  
- 在初始化时创建 `_DatasetFetcher`（`_DatasetKind.create_fetcher`，Map/Iterable 选择不同 fetcher）。  
- `_next_data`（`lines 806-814`）对 sampler 取下一个 index，调用 fetcher 从 `dataset` 中抓取数据，并在 `pin_memory=True` 时调用 `_utils.pin_memory.pin_memory`（`torch/utils/data/_utils/pin_memory.py:1-78`）把结果拷到 pinned memory，再返回给上层。

## 3. 多进程数据流 (`_MultiProcessingDataLoaderIter`)

### 3.1 初始化阶段

- 入口：`torch/utils/data/dataloader.py:848-1205`。关键步骤：
  1. 解析 `prefetch_factor`、`in_order` 等参数，决定使用的 multiprocessing context。  
  2. 如果 `dataset` 是 DataPipe，给 `worker_init_fn` 加上 `_sharding_worker_init_fn` 包装（`lines 875-892`），用于在 worker 内完成分布式/多进程切分。  
  3. 构建 `self._worker_result_queue`、每个 worker 的 `index_queue`，然后启动 worker 进程：  
     ```python
     w = multiprocessing_context.Process(
         target=_utils.worker._worker_loop,
         args=(dataset_kind, dataset, index_queue, self._worker_result_queue, ...)
     )
     w.daemon = True
     w.start()
     ```  
     —— 代码在 `torch/utils/data/dataloader.py:908-943`。  
  4. 若开启 `pin_memory`，再额外启动守护线程 `_utils.pin_memory._pin_memory_loop`（`lines 945-970`），它负责把 worker 再回来的 tensor 置于 pinned memory。  
  5. 注册 worker pid 到 C++ 侧的全局表：`_utils.signal_handling._set_worker_pids(...)`（`lines 1183-1189`）。该函数调用的是 `torch._C._set_worker_pids`，实现位于 `torch/csrc/DataLoader.cpp:72-132`。

### 3.2 运行期调度

- `_reset`（`lines 1217-1271`）会为 persistent workers 发送 `_ResumeIteration`，并重新预取 `prefetch_factor * num_workers` 个任务。  
- `_try_put_index`（`lines 1391-1411`）负责把新的 `(send_idx, sample_index)` 下发到活跃 worker 的 `index_queue` 中，同时维护 `_task_info` 与各 worker 的 outstanding task 数。  
- `_get_data` 与 `_try_get_data`（`lines 1538-1654`）从 `self._data_queue`（可能是 `worker_result_queue` 或 pin memory 的输出队列）里取结果，并在超时时通过 `w.is_alive()` 监控 worker。  
- `_next_data`（`lines 1656-1734`）则组合了上面所有状态：它会检查 out-of-order/乱序模式、`_IterableDatasetStopIteration`（worker 耗尽 IterableDataset 时返回的标记）以及异常包装 `ExceptionWrapper`。如果 `in_order=False`，一旦发现数据已返回会立刻处理而不是缓存。  
- `_process_data`（`lines 1736-1743`）在成功拿到数据后调用 `_try_put_index` 继续补充任务，并在遇到 `ExceptionWrapper` 时 `reraise()`，这样 Python 端能得到真实异常栈。

### 3.3 退出与清理

- `_shutdown_workers`（`lines 1749-1807`）实现了 NOTE 中的“先停 pin_memory，再停 worker”协议：  
  1. 设置 `self._pin_memory_thread_done_event`，往 `worker_result_queue` 推 `(None, None)` 让线程醒来，然后 join。  
  2. 设置 `self._workers_done_event` 并给每个 `index_queue` 发送 `None`，等待 worker `join`。  
  3. 调用 `_utils.signal_handling._remove_worker_pids` 移除 pid 注册（`torch/csrc/DataLoader.cpp:134-185`）。  
  4. 若 join 超时则 `w.terminate()`。  
- `__del__`（`line 1813`）只调用 `_shutdown_workers()`，因为 worker/thread 都是 daemon，Python 退出时无法保证执行顺序，所以 `_utils.python_exit_status`（`torch/utils/data/_utils/__init__.py:12-48`）会提前在 `atexit` 中被置位，防止清理到一半卡住。

## 4. Worker 进程 (`torch/utils/data/_utils/worker.py`)

### 4.1 `ManagerWatchdog`

- 定义在文件开头（`lines 12-74`），用于检测 manager（主进程）是否还存活。Linux 通过比较 `os.getppid()`，Windows 则调用 `WaitForSingleObject` 检测句柄状态。

### 4.2 `_worker_loop`

- 入口：`torch/utils/data/_utils/worker.py:228-348`。流程如下：
  1. 注册 C 侧 SIGBUS/SIGSEGV/SIGFPE/SIGTERM 处理器：`signal_handling._set_worker_signal_handlers()`（对应 `torch/csrc/DataLoader.cpp:5-70`）。  
  2. 设定线程名、把 `torch.set_num_threads(1)`，并用 `base_seed + worker_id` 初始化 Python `random`、`torch.manual_seed` 以及可选的 NumPy 随机数（`lines 249-270`）。  
  3. 若 dataset 是 DataPipe，则通过 `shared_seed` 和 `apply_random_seed`（`lines 272-284`）保证每个 worker 的 sharding/随机性一致。  
  4. 填充 `_worker_info`（`lines 286-295`），让用户可以在 `get_worker_info()` 中取到 `id/num_workers/seed/dataset`。  
  5. 运行用户自定义的 `worker_init_fn` 并创建 fetcher。若 init 失败则记录在 `init_exception`，下一次发送给主进程（`lines 300-315`）。  
  6. 主循环：  
     - `index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)`（`lines 326-333`），根据消息类型分支：  
       * `_ResumeIteration`：用于 persistent workers，重建 fetcher 并把 `(resume_token, None)` 送回主线程。  
       * `None`：最终退出信号，必须在 `done_event` 或 `iteration_end` 被置位之后才会出现，否则断言失败。  
       * `(task_idx, sample_idx)`：调用 `fetcher.fetch` 取数据。IterableDataset 若抛 `StopIteration` 会返回 `_IterableDatasetStopIteration(worker_id)`（`lines 355-371`）。  
     - 成功的数据或异常统一 `data_queue.put((idx, data))`。  
  7. 若 `done_event` 已 set，在退出前对 `data_queue.cancel_join_thread()` 并关闭其底层句柄（`lines 379-383`），防止主进程卡在队列的 join 上。

## 5. Fetcher 逻辑 (`torch/utils/data/_utils/fetch.py`)

- `_MapDatasetFetcher.fetch`：  
  - auto-collation 时会尝试调用 `dataset.__getitems__`（批量切片）或退化成列表推导；否则直接 `dataset[possibly_batched_index]`。  
- `_IterableDatasetFetcher.fetch`：  
  - 初始化时就保存 `dataset_iter = iter(dataset)`；auto-collation 会循环 `possibly_batched_index` 并把每次 `next()` 的结果 append 到 `data`。  
  - 如果批次长度不足且 `drop_last=True`，会 `raise StopIteration`。  
- 两者最后都交给 `collate_fn` 生成最终 batch。实现位于 `torch/utils/data/_utils/fetch.py:1-55`。

## 6. Pin-Memory 线程 (`torch/utils/data/_utils/pin_memory.py`)

- `_pin_memory_loop`（`lines 15-53`）是单独的守护线程：  
  1. 调用 `torch.accelerator.set_device_index(device_id)` 绑定当前加速器。  
  2. `in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)` 从 worker 结果队列中取 `(idx, data)`。  
  3. 如果没有 pending 的 `ExceptionWrapper` 并且 `done_event` 未触发，就递归调用 `pin_memory(data, device)`，该函数能处理 `Tensor`、`Mapping`、`Sequence` 以及实现了 `pin_memory` 方法的自定义类型（`lines 55-115`）。  
  4. 最终把 `(idx, pinned_data)` 使用阻塞 `out_queue.put` 送回主线程。`done_event` 被置位或线程收到 `(None, None)` 时退出。

## 7. 信号与异常管道 (`torch/utils/data/_utils/signal_handling.py` & `torch/csrc/DataLoader.cpp`)

1. **Python 端**：  
   - `_set_worker_pids(id(iter), (pid0, pid1, ...))` 和 `_remove_worker_pids` 用于在 C++ 侧维护 `worker_pids` map。  
   - `_set_SIGCHLD_handler`（`lines 38-64`）在主线程安装 handler：一旦收到 SIGCHLD，就调用 `torch._C._error_if_any_worker_fails()` 检查每个 pid 的状态。

2. **C++ 端**（`torch/csrc/DataLoader.cpp`）：  
   - `THPModule_setWorkerSignalHandlers` 注册 SIGBUS/SIGSEGV/SIGTERM/SIGFPE 的 handler（`lines 23-69`），worker 进程会在 `_worker_loop` 入口调用。  
   - `THPModule_errorIfAnyWorkerFails`（`lines 83-131`）遍历 `worker_pids`，通过 `waitid(..., WNOHANG | WNOWAIT)` 查询退出状态，针对非 0 exit code、被信号杀死等情况构造详细错误信息抛给 Python。  
   - `THPModule_setWorkerPIDs`/`_remove_worker_pids` 则负责在 `worker_pids` map 中增删记录。

## 8. 整体调用链概览

```
DataLoader.__iter__
 └─ _get_iterator()
    ├─ _SingleProcessDataLoaderIter (pin_memory? → _utils.pin_memory.pin_memory)
    │   └─ _DatasetKind.create_fetcher → _MapDatasetFetcher / _IterableDatasetFetcher
    │       └─ Dataset / DataPipe.__getitem__/__iter__
    └─ _MultiProcessingDataLoaderIter
        ├─ spawn workers running _utils.worker._worker_loop
        │   └─ _DatasetKind.create_fetcher → fetch data → data_queue.put
        ├─ optional _utils.pin_memory._pin_memory_loop
        ├─ main thread _next_data() → _get_data() → _try_put_index()
        └─ signal handling via torch._C._set_worker_pids / _error_if_any_worker_fails
```

## 9. 参考/定位索引

| 组件 | 位置 |
| --- | --- |
| `DataLoader` API & iterators | `torch/utils/data/dataloader.py:1-1813` |
| Worker loop & helpers | `torch/utils/data/_utils/worker.py:1-383` |
| Fetchers | `torch/utils/data/_utils/fetch.py:1-55` |
| Pin memory | `torch/utils/data/_utils/pin_memory.py:1-115` |
| Signal handling (Python) | `torch/utils/data/_utils/signal_handling.py:1-64` |
| Signal handling (C++) | `torch/csrc/DataLoader.cpp:1-185` |
| Python exit flag | `torch/utils/data/_utils/__init__.py:1-48` |

借助上述文件和行号即可沿着调用链逐层追踪 `DataLoader` 的行为，从 Python 参数解析一直到 worker 进程、信号处理与资源清理的底层实现。
