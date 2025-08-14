#!/bin/bash

# PyTorch 源码探索工具
# 帮助快速定位和查看PyTorch源码

set -e

PYTORCH_ROOT="/opt/learn/ai_infra/pytorch"
cd "$PYTORCH_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示使用帮助
show_help() {
    cat << EOF
PyTorch 源码探索工具

用法: $0 <命令> [参数]

命令:
  find-op <op_name>          查找算子定义和实现
  find-func <func_name>      查找函数定义
  find-class <class_name>    查找类定义
  show-op <op_name>          显示算子的完整信息
  list-ops                   列出所有算子
  explore-dir <dir>          探索目录结构
  show-arch                  显示架构概览
  help                       显示此帮助信息

示例:
  $0 find-op add             # 查找add算子
  $0 find-func backward      # 查找backward函数
  $0 find-class Dispatcher   # 查找Dispatcher类
  $0 show-op mul             # 显示mul算子的详细信息
  $0 list-ops | grep conv    # 列出所有卷积相关算子
  $0 explore-dir torch/csrc/autograd  # 查看autograd目录

EOF
}

# 查找算子定义
find_operator() {
    local op_name="$1"
    print_header "查找算子: $op_name"

    print_info "1. 在 native_functions.yaml 中查找定义..."
    if grep -n "^- func: $op_name" aten/src/ATen/native/native_functions.yaml 2>/dev/null; then
        echo ""
    else
        print_warning "未在 native_functions.yaml 中找到精确匹配"
        print_info "尝试模糊搜索..."
        grep -n "func:.*$op_name" aten/src/ATen/native/native_functions.yaml 2>/dev/null | head -5 || echo "无结果"
    fi

    echo ""
    print_info "2. 在 ATen native 中查找实现..."
    find aten/src/ATen/native -name "*.cpp" -o -name "*.cu" 2>/dev/null | \
        xargs grep -l "\\b${op_name}\\b" 2>/dev/null | head -10 || echo "无结果"

    echo ""
    print_info "3. 在 Python 层查找..."
    find torch -name "*.py" 2>/dev/null | \
        xargs grep -l "def ${op_name}" 2>/dev/null | head -10 || echo "无结果"
}

# 查找函数定义
find_function() {
    local func_name="$1"
    print_header "查找函数: $func_name"

    print_info "C++ 实现:"
    find . -name "*.cpp" -o -name "*.h" 2>/dev/null | \
        xargs grep -n "\\b${func_name}\\(" 2>/dev/null | head -15 || echo "无结果"

    echo ""
    print_info "Python 实现:"
    find torch -name "*.py" 2>/dev/null | \
        xargs grep -n "def ${func_name}" 2>/dev/null | head -15 || echo "无结果"
}

# 查找类定义
find_class() {
    local class_name="$1"
    print_header "查找类: $class_name"

    print_info "C++ 类定义:"
    find . -name "*.h" -o -name "*.cpp" 2>/dev/null | \
        xargs grep -n "class.*\\b${class_name}\\b" 2>/dev/null | head -10 || echo "无结果"

    echo ""
    print_info "Python 类定义:"
    find torch -name "*.py" 2>/dev/null | \
        xargs grep -n "class ${class_name}" 2>/dev/null | head -10 || echo "无结果"
}

# 显示算子的详细信息
show_operator_detail() {
    local op_name="$1"
    print_header "算子详细信息: $op_name"

    print_info "算子定义 (native_functions.yaml):"
    echo "---"
    grep -A 20 "^- func: $op_name" aten/src/ATen/native/native_functions.yaml 2>/dev/null | head -25 || \
        echo "未找到定义"
    echo ""

    print_info "相关实现文件:"
    local impl_files=$(find aten/src/ATen/native -name "*.cpp" -o -name "*.cu" 2>/dev/null | \
        xargs grep -l "\\b${op_name}\\b" 2>/dev/null | head -5)

    if [ -n "$impl_files" ]; then
        echo "$impl_files"
        echo ""
        print_info "第一个实现文件的相关代码:"
        local first_file=$(echo "$impl_files" | head -1)
        echo "文件: $first_file"
        echo "---"
        grep -A 10 "\\b${op_name}\\b" "$first_file" 2>/dev/null | head -20
    else
        echo "未找到实现文件"
    fi
}

# 列出所有算子
list_operators() {
    print_header "所有算子列表"
    print_info "从 native_functions.yaml 提取..."

    grep "^- func:" aten/src/ATen/native/native_functions.yaml 2>/dev/null | \
        sed 's/^- func: //' | \
        cut -d'(' -f1 | \
        sort | \
        uniq | \
        head -50

    echo ""
    print_info "显示前50个算子,使用 grep 可以过滤"
}

# 探索目录结构
explore_directory() {
    local dir="$1"
    print_header "探索目录: $dir"

    if [ ! -d "$dir" ]; then
        print_error "目录不存在: $dir"
        return 1
    fi

    print_info "目录结构:"
    tree -L 2 "$dir" 2>/dev/null || ls -la "$dir"

    echo ""
    print_info "关键文件:"
    find "$dir" -maxdepth 2 -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.py" \) 2>/dev/null | \
        head -20

    echo ""
    print_info "代码统计:"
    find "$dir" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" \) 2>/dev/null | \
        xargs wc -l 2>/dev/null | tail -1 || echo "无法统计"
}

# 显示架构概览
show_architecture() {
    print_header "PyTorch 架构概览"

    cat << 'EOF'

PyTorch 分层架构:

┌─────────────────────────────────────────┐
│  Python 前端 (torch/)                   │
│  - nn.Module, optim, autograd          │
│  - 编译器: _dynamo, _inductor, fx       │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Python/C++ 绑定 (torch/csrc/)         │
│  - autograd 引擎                        │
│  - JIT 编译器                           │
│  - 分布式训练                           │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  ATen 张量库 (aten/)                    │
│  - Dispatcher 调度系统                  │
│  - 算子实现 (CPU/CUDA/etc)             │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  C10 核心库 (c10/)                      │
│  - DispatchKey, Device, TensorOptions  │
│  - 核心类型和工具                       │
└─────────────────────────────────────────┘

核心目录:
  c10/              核心类型和调度系统
  aten/             张量运算库
  torch/csrc/       Python绑定和核心引擎
  torch/            Python前端API
  torchgen/         代码生成工具

关键文件:
  native_functions.yaml           算子定义
  c10/core/DispatchKey.h         调度键定义
  torch/csrc/autograd/engine.h   反向传播引擎
  aten/src/ATen/core/dispatch/Dispatcher.h  调度器

EOF

    echo ""
    print_info "更多详细信息请查看:"
    echo "  - CLAUDE.md (完整架构说明)"
    echo "  - QUICK_START.md (快速上手)"
    echo "  - BEGINNER_GUIDE.md (学习指南)"
}

# 主程序
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    local command="$1"
    shift

    case "$command" in
        find-op)
            if [ $# -eq 0 ]; then
                print_error "请提供算子名称"
                echo "用法: $0 find-op <op_name>"
                exit 1
            fi
            find_operator "$1"
            ;;
        find-func)
            if [ $# -eq 0 ]; then
                print_error "请提供函数名称"
                echo "用法: $0 find-func <func_name>"
                exit 1
            fi
            find_function "$1"
            ;;
        find-class)
            if [ $# -eq 0 ]; then
                print_error "请提供类名称"
                echo "用法: $0 find-class <class_name>"
                exit 1
            fi
            find_class "$1"
            ;;
        show-op)
            if [ $# -eq 0 ]; then
                print_error "请提供算子名称"
                echo "用法: $0 show-op <op_name>"
                exit 1
            fi
            show_operator_detail "$1"
            ;;
        list-ops)
            list_operators
            ;;
        explore-dir)
            if [ $# -eq 0 ]; then
                print_error "请提供目录路径"
                echo "用法: $0 explore-dir <dir>"
                exit 1
            fi
            explore_directory "$1"
            ;;
        show-arch)
            show_architecture
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
