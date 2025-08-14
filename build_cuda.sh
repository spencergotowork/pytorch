#!/bin/bash
# PyTorch CUDA 编译脚本 (适用于 16GB 内存 + CUDA 12.4)

set -e

echo "=== PyTorch CUDA 编译配置 ==="

# 1. 激活虚拟环境
source /opt/learn/ai_infra/env/bin/activate

# 2. 设置 CUDA 环境变量(必需!)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# 3. 启用 CUDA 相关组件
export USE_CUDA=1
export USE_CUDNN=1              # 如果安装了 cuDNN
export USE_NCCL=1               # 多 GPU 通信(可选)

# 4. 指定 GPU 架构(重要!节省编译时间和内存)
# RTX 4060 是 Ada 架构,计算能力 8.9
export TORCH_CUDA_ARCH_LIST="8.9"

# 如果有多种 GPU 或想兼容其他卡,可以用:
# export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"  # 支持多种架构

# 5. 限制并行编译任务(避免内存溢出)
export MAX_JOBS=2

# 6. 其他优化选项
export USE_NINJA=1              # 使用 Ninja 构建系统
export BUILD_TEST=0             # 不编译测试,节省内存
export CMAKE_BUILD_TYPE=Release # Release 模式

# 7. 显示配置信息
echo ""
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc version:"
nvcc --version | grep "release"
echo ""
echo "GPU 信息:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
echo ""
echo "MAX_JOBS: $MAX_JOBS"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo ""

# 8. 清理旧构建(可选,首次编译跳过)
# read -p "是否清理旧的构建目录? (y/N) " -n 1 -r
# echo
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     echo "清理 build/ 目录..."
#     rm -rf build/
# fi

# 9. 安装依赖
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 10. 开始编译
echo ""
echo "=== 开始编译 PyTorch (预计 2-4 小时) ==="
echo "可以在另一个终端运行 'watch -n 1 free -h' 监控内存"
echo ""

pip install --no-build-isolation -v -e .

echo ""
echo "=== 编译完成! ==="
echo ""
echo "验证 CUDA 支持:"
echo "python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")'"
