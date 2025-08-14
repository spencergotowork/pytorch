#!/bin/bash
# PyTorch CUDA 编译脚本 (修复 NCCL 错误)

set -e

echo "=== PyTorch CUDA 编译配置 (修复版) ==="

# 1. 清理可能的 ccache 环境变量
unset CC
unset CXX
echo "✓ 已清理 CC/CXX 环境变量"

# 2. 激活虚拟环境
source /opt/learn/ai_infra/env/bin/activate
echo "✓ 已激活虚拟环境"

# 3. 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# 4. 启用 CUDA，禁用 NCCL 和 flash-attention
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0              # 禁用 NCCL (单GPU不需要)
export USE_SYSTEM_NCCL=0       # 不使用系统 NCCL
export USE_FLASH_ATTENTION=0   # 禁用 flash-attention (避免编译问题)

# 5. GPU 架构设置
export TORCH_CUDA_ARCH_LIST="8.9"

# 6. 限制并行任务 (避免目录冲突)
export MAX_JOBS=4

# 7. 其他优化
export USE_NINJA=1
export BUILD_TEST=0
export CMAKE_BUILD_TYPE=Debug

# 8. 显示配置
echo ""
echo "配置信息:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  USE_CUDA: $USE_CUDA"
echo "  USE_NCCL: $USE_NCCL"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS: $MAX_JOBS"
echo ""

# 9. 清理旧的构建(重要!)
echo "清理旧的构建文件..."
rm -rf build/
python setup.py clean
echo "✓ 清理完成"
echo ""

# 10. 安装依赖
pip install -r requirements.txt

# 11. 开始编译
echo "=== 开始编译 ==="
echo ""

pip install --no-build-isolation -v -e .

echo ""
echo "=== 编译完成! ==="
