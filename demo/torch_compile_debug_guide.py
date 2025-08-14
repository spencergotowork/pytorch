#!/usr/bin/env python3
"""
torch.compile 调试方法实用指南

基于PyTorch 2.5源码研究和实际测试的完整总结
"""

import torch
print(f"PyTorch版本: {torch.__version__}")

# ============================================================================
# 1. 环境变量配置（复制粘贴使用）
# ============================================================================

ENVIRONMENT_SETUP = '''
# === 基础调试设置 ===
export TORCH_LOGS="graph_breaks,recompiles,guards"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=/tmp/torch_debug

# === 详细调试设置 ===  
export TORCH_LOGS="+dynamo,+aot,+inductor,graph_breaks,recompiles,guards,output_code,kernel_code"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR=/tmp/torch_debug
export TORCHDYNAMO_VERBOSE=1

# === 包含追踪的完整设置 ===
export TORCH_LOGS="+all,graph_breaks,recompiles,guards,dynamic,output_code,schedule,fusion"
export TORCH_COMPILE_DEBUG=1 
export TORCH_COMPILE_DEBUG_DIR=/tmp/torch_debug
export TORCH_TRACE=/tmp/torch_trace
export TORCHDYNAMO_VERBOSE=1

# === 错误复现设置 ===
export TORCHDYNAMO_REPRO_AFTER=dynamo
export TORCHDYNAMO_REPRO_LEVEL=2
'''

print("=" * 80)
print("torch.compile 调试方法实用指南")
print("=" * 80)

print("\n🔧 环境变量配置（复制使用）:")
print(ENVIRONMENT_SETUP)

# ============================================================================
# 2. 程序化配置模板
# ============================================================================

PROGRAMMATIC_CONFIG = '''
import torch._logging

# 基础日志配置
torch._logging.set_logs(
    graph_breaks=True,      # 图中断信息
    recompiles=True,        # 重编译原因
    guards=True,            # 守卫条件
    output_code=True        # 生成的代码
)

# 全面日志配置
torch._logging.set_logs(
    dynamo=torch._logging.logging.DEBUG,
    aot=torch._logging.logging.DEBUG,
    inductor=torch._logging.logging.DEBUG,
    graph_breaks=True,
    guards=True,
    recompiles=True,
    dynamic=True,
    output_code=True,
    kernel_code=True,
    schedule=True,
    fusion=True,
    perf_hints=True
)

# Dynamo配置
import torch._dynamo.config as dynamo_config
dynamo_config.verbose = True
dynamo_config.log_file_name = "dynamo.log"

# Inductor配置  
import torch._inductor.config as inductor_config
inductor_config.debug = True
inductor_config.verbose_progress = True
inductor_config.comment_origin = True
'''

print("⚙️ 程序化配置模板:")
print(PROGRAMMATIC_CONFIG)

# ============================================================================
# 3. 实用API示例
# ============================================================================

API_EXAMPLES = '''
# === 使用explain分析函数 ===
import torch._dynamo

def my_function(x):
    y = torch.relu(x + 1)
    if y.sum() > 0:  # 这会导致graph break
        return y * 2
    return y / 2

# 分析函数（注意新的调用方式）
explanation = torch._dynamo.explain(my_function)(torch.randn(10))

print(f"图数量: {explanation.graph_count}")
print(f"中断数量: {explanation.graph_break_count}")
print(f"中断原因: {explanation.break_reasons}")

# === 测试不同backend ===
def test_func(x, y):
    return torch.matmul(x, y)

x, y = torch.randn(100, 100), torch.randn(100, 100)

# 不同编译阶段
for backend in ["eager", "aot_eager", "inductor"]:
    compiled = torch.compile(test_func, backend=backend)
    result = compiled(x, y)
    print(f"{backend}: {result.shape}")

# === 动态形状测试 ===
@torch.compile(dynamic=True)  # 启用动态形状
def dynamic_func(x):
    return x.sum()

# 测试不同形状
for shape in [(10,), (20,), (30,)]:
    x = torch.randn(shape)
    result = dynamic_func(x)
    print(f"Shape {shape}: {result}")
'''

print("🔍 实用API示例:")
print(API_EXAMPLES)

# ============================================================================
# 4. 调试场景速查
# ============================================================================

print("\n🎯 常见调试场景速查表:")
print("-" * 60)

scenarios = {
    "查看graph breaks": {
        "环境变量": "TORCH_LOGS=graph_breaks",
        "程序化": "torch._logging.set_logs(graph_breaks=True)",
        "用途": "找出代码中导致图中断的位置"
    },
    
    "查看重编译原因": {
        "环境变量": "TORCH_LOGS=recompiles,guards", 
        "程序化": "torch._logging.set_logs(recompiles=True, guards=True)",
        "用途": "理解函数为什么被重新编译"
    },
    
    "查看生成代码": {
        "环境变量": "TORCH_LOGS=output_code,kernel_code",
        "程序化": "torch._logging.set_logs(output_code=True, kernel_code=True)",
        "用途": "检查Inductor生成的Python/C++/CUDA代码"
    },
    
    "调试动态形状": {
        "环境变量": "TORCH_LOGS=dynamic",
        "程序化": "torch._logging.set_logs(dynamic=True)",
        "配置": "torch.compile(func, dynamic=True)",
        "用途": "追踪动态形状推理过程"
    },
    
    "性能分析": {
        "环境变量": "TORCH_LOGS=schedule,fusion,perf_hints",
        "程序化": "torch._logging.set_logs(schedule=True, fusion=True, perf_hints=True)",
        "用途": "分析调度和融合决策"
    }
}

for scenario, config in scenarios.items():
    print(f"\n{scenario}:")
    for key, value in config.items():
        print(f"  {key}: {value}")

# ============================================================================
# 5. 生成的文件说明
# ============================================================================

print(f"\n📁 调试文件说明:")
print("-" * 60)

file_types = {
    "torchdynamo/debug.log": "Dynamo调试日志",
    "torchdynamo/graph_*.py": "捕获的FX图Python代码",
    "torchdynamo/repro_*.py": "自动生成的复现脚本（错误时）",
    "torchinductor/output_*.py": "Inductor生成的最终Python代码",
    "torchinductor/kernel_*.cpp": "生成的C++ CPU内核代码",
    "torchinductor/kernel_*.cu": "生成的CUDA GPU内核代码",
    "aot_autograd/*.py": "AOTAutograd生成的前向/后向图"
}

for file_pattern, description in file_types.items():
    print(f"  {file_pattern}: {description}")

# ============================================================================
# 6. 实用工具
# ============================================================================

print(f"\n🛠️ 推荐工具:")
print("-" * 60)

tools = {
    "tlparse": {
        "安装": "pip install tlparse",
        "使用": "export TORCH_TRACE=/tmp/trace && python script.py && tlparse /tmp/trace",
        "功能": "可视化编译时间线和图中断"
    },
    
    "调试上下文管理器": {
        "代码": """
class DebugContext:
    def __init__(self, logs="graph_breaks,recompiles"):
        self.logs = logs
        self.old_env = {}
    
    def __enter__(self):
        import os, tempfile
        self.debug_dir = tempfile.mkdtemp(prefix="debug_")
        self.old_env = dict(os.environ)
        os.environ.update({
            "TORCH_LOGS": self.logs,
            "TORCH_COMPILE_DEBUG": "1",
            "TORCH_COMPILE_DEBUG_DIR": self.debug_dir
        })
        return self
    
    def __exit__(self, *args):
        import os
        os.environ.clear()
        os.environ.update(self.old_env)

# 使用方式
with DebugContext("graph_breaks,output_code") as debug:
    @torch.compile
    def my_func(x): return x * 2
    result = my_func(torch.randn(10))
    print(f"调试文件: {debug.debug_dir}")
""",
        "功能": "临时启用调试模式"
    }
}

for tool_name, tool_info in tools.items():
    print(f"\n{tool_name}:")
    for key, value in tool_info.items():
        if key == "代码":
            print(f"  {key}:\n{value}")
        else:
            print(f"  {key}: {value}")

# ============================================================================
# 7. 最佳实践
# ============================================================================

print(f"\n💡 最佳实践建议:")
print("-" * 60)

best_practices = [
    "1. 从简单的TORCH_LOGS=graph_breaks开始",
    "2. 使用torch._dynamo.explain()快速分析函数结构", 
    "3. 对于复杂问题，启用完整调试：+dynamo,+aot,+inductor",
    "4. 使用不同backend测试各编译阶段：eager -> aot_eager -> inductor",
    "5. 保存调试文件用于离线分析",
    "6. 使用tlparse分析大模型的编译过程",
    "7. 通过减少graph break优化编译效果",
    "8. 使用dynamic=True处理变化的输入形状",
    "9. 设置TORCHDYNAMO_REPRO_AFTER自动生成复现脚本",
    "10. 结合cProfile分析编译性能瓶颈"
]

for practice in best_practices:
    print(f"  {practice}")

# ============================================================================
# 8. 快速开始模板
# ============================================================================

QUICK_START = '''
#!/usr/bin/env python3
# 快速开始模板

import os
import torch
import torch._logging

# 1. 设置调试环境
os.environ["TORCH_LOGS"] = "graph_breaks,recompiles"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

# 2. 分析函数
def my_function(x):
    return torch.relu(x + 1)

explanation = torch._dynamo.explain(my_function)(torch.randn(10))
print(f"图数量: {explanation.graph_count}")

# 3. 编译和测试
@torch.compile
def compiled_func(x):
    return torch.matmul(x, x.T)

x = torch.randn(100, 100)
result = compiled_func(x)
print(f"结果: {result.shape}")

# 4. 查看调试文件
debug_dir = os.environ.get("TORCH_COMPILE_DEBUG_DIR", "/tmp")
print(f"调试文件位置: {debug_dir}")
'''

print(f"\n🚀 快速开始模板:")
print(QUICK_START)

print(f"\n{'='*80}")
print("总结：本指南提供了torch.compile调试的完整方法")
print("建议先从基础配置开始，逐步增加调试信息的详细程度")
print(f"{'='*80}")