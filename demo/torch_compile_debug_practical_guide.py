#!/usr/bin/env python3
"""
torch.compile 实用调试指南 - 针对性解决方案

深入分析torch.compile各阶段的调试方法，提供具体的工具和实际案例
"""

import os
import sys
import torch
import torch._dynamo
import torch._logging
import tempfile
import traceback
import time
import warnings
from typing import Dict, List, Any

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# ============================================================================
# 1. TorchDynamo阶段调试：Graph Break和动态行为
# ============================================================================

def setup_dynamo_debugging():
    """配置TorchDynamo调试环境"""
    
    print("\n🔍 TorchDynamo调试配置")
    print("-" * 50)
    
    # 环境变量配置
    debug_env = {
        "TORCH_LOGS": "graph_breaks,recompiles,guards,dynamic",
        "TORCH_COMPILE_DEBUG": "1",
        "TORCHDYNAMO_VERBOSE": "1"
    }
    
    for key, value in debug_env.items():
        os.environ[key] = str(value)
        print(f"✅ 设置 {key}={value}")
    
    # 程序化配置
    torch._logging.set_logs(
        graph_breaks=True,
        recompiles=True, 
        guards=True,
        dynamic=True
    )
    
    print("✅ TorchDynamo调试环境已配置")

def analyze_graph_breaks_detailed():
    """深度分析Graph Break原因"""
    
    print("\n💥 Graph Break深度分析")
    print("-" * 50)
    
    def problematic_function(x):
        """包含多种Graph Break的函数"""
        
        # 1. 正常张量操作（不会break）
        y = torch.relu(x + 1)
        
        # 2. Python内置函数（会break）
        shape_info = len(x.shape)
        
        # 3. 数据依赖的控制流（会break）
        if y.sum() > 0:
            z = y * 2
        else:
            z = y / 2
            
        # 4. 打印语句（会break） 
        print(f"中间结果: {z.mean():.4f}")
        
        # 5. 循环（可能break）
        for i in range(3):
            z = z + i * 0.1
            
        # 6. 异常处理（会break）
        try:
            result = z.sum() / shape_info
        except:
            result = torch.tensor(0.0)
            
        return result
    
    # 使用explain分析
    test_input = torch.randn(10, 20)
    
    print("使用torch._dynamo.explain()分析函数结构...")
    explanation = torch._dynamo.explain(problematic_function)(test_input)
    
    print(f"\n📊 分析结果:")
    print(f"  图数量: {explanation.graph_count}")
    print(f"  Graph Break数量: {explanation.graph_break_count}")
    print(f"  操作总数: {explanation.op_count}")
    
    print(f"\n💥 详细Break原因:")
    for i, reason in enumerate(explanation.break_reasons):
        print(f"  {i+1}. {reason}")
    
    # 分类break原因
    break_categories = {}
    for reason in explanation.break_reasons:
        category = classify_break_reason(reason)
        break_categories[category] = break_categories.get(category, 0) + 1
    
    print(f"\n📈 Break类型统计:")
    for category, count in break_categories.items():
        print(f"  {category}: {count}次")
    
    return explanation

def classify_break_reason(reason) -> str:
    """分类Graph Break原因"""
    # 处理不同类型的reason对象
    if hasattr(reason, 'reason'):
        reason_str = str(reason.reason).lower()
    else:
        reason_str = str(reason).lower()
    
    if any(word in reason_str for word in ['builtin', 'len', 'print']):
        return "Python内置操作"
    elif any(word in reason_str for word in ['control', 'if', 'branch', 'jump']):
        return "控制流"
    elif 'data-dependent' in reason_str:
        return "数据依赖"
    elif any(word in reason_str for word in ['loop', 'for', 'while']):
        return "循环结构"
    elif any(word in reason_str for word in ['exception', 'try', 'except']):
        return "异常处理"
    else:
        return "其他"

def debug_guard_failures():
    """调试Guard失败和重编译问题"""
    
    print("\n🛡️ Guard失败调试")
    print("-" * 50)
    
    def shape_dependent_function(x):
        """形状依赖的函数"""
        return torch.matmul(x, x.transpose(-2, -1)).sum()
    
    # 编译函数
    compiled_func = torch.compile(shape_dependent_function)
    
    # 测试不同形状，观察重编译
    shapes = [(10, 10), (20, 20), (10, 10), (30, 30), (10, 10)]
    
    print("测试不同输入形状的编译行为:")
    
    for i, shape in enumerate(shapes):
        print(f"\n测试 {i+1}: shape={shape}")
        
        # 重置统计
        torch._dynamo.reset()
        
        x = torch.randn(*shape)
        start_time = time.time()
        result = compiled_func(x)
        end_time = time.time()
        
        print(f"  结果: {result:.4f}")
        print(f"  执行时间: {(end_time - start_time)*1000:.2f} ms")
        
    # 显示编译统计
    try:
        from torch._dynamo.utils import compile_times
        stats = compile_times()
        print(f"\n📊 编译统计: {dict(stats)}")
    except:
        print("无法获取编译统计")

def demonstrate_dynamic_shape_debugging():
    """演示动态形状调试"""
    
    print("\n🎭 动态形状调试")
    print("-" * 50)
    
    def dynamic_shape_function(x):
        """需要动态形状处理的函数"""
        # 根据输入维度选择不同处理方式
        if x.dim() == 1:
            return x.sum()
        elif x.dim() == 2:
            return x.trace()
        else:
            return x.flatten().sum()
    
    print("测试静态编译模式:")
    static_compiled = torch.compile(dynamic_shape_function)
    
    test_inputs = [
        torch.randn(10),        # 1D
        torch.randn(5, 5),      # 2D 
        torch.randn(2, 3, 4)    # 3D
    ]
    
    for i, x in enumerate(test_inputs):
        try:
            result = static_compiled(x)
            print(f"  输入{i+1} {x.shape}: ✅ {result:.4f}")
        except Exception as e:
            print(f"  输入{i+1} {x.shape}: ❌ {e}")
    
    print("\n测试动态编译模式:")
    dynamic_compiled = torch.compile(dynamic_shape_function, dynamic=True)
    
    for i, x in enumerate(test_inputs):
        try:
            result = dynamic_compiled(x)
            print(f"  输入{i+1} {x.shape}: ✅ {result:.4f}")
        except Exception as e:
            print(f"  输入{i+1} {x.shape}: ❌ {e}")

# ============================================================================
# 2. AOTAutograd阶段调试：反向图和内存优化
# ============================================================================

def setup_aot_debugging():
    """配置AOTAutograd调试"""
    
    print("\n🔄 AOTAutograd调试配置") 
    print("-" * 50)
    
    # 设置环境变量
    os.environ["TORCH_LOGS"] = "aot,functorch,graph_breaks"
    
    torch._logging.set_logs(
        aot=torch._logging.logging.DEBUG,
        graph_breaks=True
    )
    
    print("✅ AOTAutograd调试环境已配置")

def debug_backward_graph_issues():
    """调试反向图生成问题"""
    
    print("\n🔄 反向图生成调试")
    print("-" * 50)
    
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.fc2 = torch.nn.Linear(20, 5)
            self.dropout = torch.nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x.sum()
    
    model = TestModel()
    x = torch.randn(32, 10, requires_grad=True)
    
    print("测试不同backend的梯度计算:")
    
    backends = ["eager", "aot_eager", "inductor"]
    
    for backend in backends:
        print(f"\n--- Backend: {backend} ---")
        
        try:
            # 重置梯度
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            
            # 编译并执行
            if backend == "eager":
                compiled_model = model
            else:
                compiled_model = torch.compile(model, backend=backend)
            
            loss = compiled_model(x)
            loss.backward()
            
            # 检查梯度
            param_grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            input_grad = x.grad.norm().item() if x.grad is not None else 0
            
            print(f"  Loss: {loss.item():.4f}")
            print(f"  参数梯度范数: {[f'{g:.4f}' for g in param_grads]}")
            print(f"  输入梯度范数: {input_grad:.4f}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")

def debug_memory_optimization():
    """调试内存优化问题"""
    
    print("\n💾 内存优化调试")
    print("-" * 50)
    
    def memory_test_function(x):
        """内存密集型函数"""
        # 创建多个中间结果
        results = []
        current = x
        
        for i in range(3):
            current = torch.matmul(current, current.t())
            current = torch.relu(current)
            results.append(current.clone())  # 强制保留中间结果
        
        return sum(results).trace()
    
    x = torch.randn(64, 64, requires_grad=True)
    
    print("比较不同编译模式的内存行为:")
    
    import psutil
    process = psutil.Process()
    
    test_configs = [
        ("原始函数", None),
        ("AOT Eager", "aot_eager"), 
        ("Inductor", "inductor")
    ]
    
    for name, backend in test_configs:
        print(f"\n--- {name} ---")
        
        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            if backend is None:
                func = memory_test_function
            else:
                func = torch.compile(memory_test_function, backend=backend)
            
            # 执行函数
            loss = func(x)
            loss.backward()
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_memory - initial_memory
            
            print(f"  Loss: {loss.item():.4f}")
            print(f"  内存使用: {memory_usage:.2f} MB")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
        
        # 清理梯度
        if x.grad is not None:
            x.grad.zero_()

# ============================================================================
# 3. Inductor阶段调试：代码生成和性能优化
# ============================================================================

def setup_inductor_debugging():
    """配置Inductor调试"""
    
    print("\n💻 Inductor调试配置")
    print("-" * 50)
    
    # 设置调试环境变量
    debug_env = {
        "TORCH_LOGS": "inductor,output_code,schedule,fusion",
        "TORCH_COMPILE_DEBUG": "1",
        "TORCHINDUCTOR_DEBUG": "1"
    }
    
    for key, value in debug_env.items():
        os.environ[key] = str(value)
        print(f"✅ 设置 {key}={value}")
    
    # 程序化配置
    torch._logging.set_logs(
        inductor=torch._logging.logging.DEBUG,
        output_code=True,
        schedule=True,
        fusion=True
    )

def debug_kernel_fusion():
    """调试内核融合"""
    
    print("\n🔀 内核融合调试")
    print("-" * 50)
    
    # 测试不同的融合模式
    fusion_patterns = {
        "简单融合": lambda x: torch.relu(torch.sigmoid(x)),
        "复杂融合链": lambda x: torch.softmax(torch.relu(x + 1) * torch.tanh(x - 1), dim=-1),
        "广播融合": lambda x: x + torch.randn(1, x.shape[1]),
        "归约融合": lambda x: (x * 2).sum(dim=1, keepdim=True),
        "阻止融合": lambda x: torch.relu(x).contiguous().sigmoid()  # contiguous阻止融合
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(256, 256, device=device)
    
    for name, pattern in fusion_patterns.items():
        print(f"\n测试: {name}")
        
        try:
            # 编译函数
            compiled_pattern = torch.compile(pattern, backend="inductor")
            
            # 预热
            for _ in range(3):
                _ = compiled_pattern(x)
            
            # 性能测试
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(100):
                result = compiled_pattern(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.time()
            avg_time = (end_time - start_time) / 100 * 1000  # ms
            
            print(f"  ✅ 成功: {result.shape}")
            print(f"  平均执行时间: {avg_time:.3f} ms")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")

def debug_performance_regression():
    """调试性能回退问题"""
    
    print("\n⚡ 性能回退调试")
    print("-" * 50)
    
    def create_test_functions():
        """创建不同类型的测试函数"""
        
        return {
            "矩阵乘法": lambda x, y: torch.matmul(x, y),
            "复合运算": lambda x: torch.softmax(torch.relu(x @ x.t()), dim=-1),
            "元素运算": lambda x: torch.sin(x) + torch.cos(x) * torch.tanh(x),
            "归约运算": lambda x: x.sum(dim=1, keepdim=True).expand_as(x)
        }
    
    test_functions = create_test_functions()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 准备测试输入
    test_inputs = {
        "矩阵乘法": (torch.randn(256, 256, device=device), torch.randn(256, 256, device=device)),
        "复合运算": (torch.randn(128, 128, device=device),),
        "元素运算": (torch.randn(512, 512, device=device),),
        "归约运算": (torch.randn(1000, 100, device=device),)
    }
    
    print("性能对比测试:")
    
    for name, func in test_functions.items():
        print(f"\n--- {name} ---")
        inputs = test_inputs[name]
        
        try:
            # 原始函数性能
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.time()
            for _ in range(50):
                original_result = func(*inputs)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            original_time = time.time() - start_time
            
            # 编译函数性能
            compiled_func = torch.compile(func, backend="inductor")
            
            # 预热
            for _ in range(5):
                _ = compiled_func(*inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.time()
            for _ in range(50):
                compiled_result = compiled_func(*inputs)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            compiled_time = time.time() - start_time
            
            # 计算加速比
            speedup = original_time / compiled_time if compiled_time > 0 else float('inf')
            
            print(f"  原始时间: {original_time*1000:.2f} ms")
            print(f"  编译时间: {compiled_time*1000:.2f} ms")
            print(f"  加速比: {speedup:.2f}x")
            
            # 检查数值正确性
            if torch.allclose(original_result, compiled_result, rtol=1e-4, atol=1e-4):
                print(f"  ✅ 数值一致")
            else:
                print(f"  ❌ 数值不一致")
                
            if speedup < 1.0:
                print(f"  ⚠️  性能回退，可能原因:")
                print(f"     - 函数太简单，编译开销大于收益")
                print(f"     - Graph Break过多")
                print(f"     - 内存访问模式不友好")
                
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")

# ============================================================================
# 4. 错误模式分析和解决方案
# ============================================================================

def analyze_common_error_patterns():
    """分析常见错误模式"""
    
    print("\n🔧 常见错误模式分析")
    print("=" * 60)
    
    error_examples = {
        "Graph Break过多": create_graph_break_example,
        "编译失败": create_compilation_failure_example,
        "数值不一致": create_numerical_inconsistency_example,
        "内存泄漏": create_memory_leak_example
    }
    
    for error_name, example_func in error_examples.items():
        print(f"\n--- {error_name} ---")
        
        try:
            problematic_func, optimized_func = example_func()
            
            # 测试有问题的版本
            print("有问题的版本:")
            test_function_issues(problematic_func)
            
            # 测试优化版本
            print("优化后的版本:")
            test_function_issues(optimized_func)
            
        except Exception as e:
            print(f"❌ 示例创建失败: {e}")

def create_graph_break_example():
    """创建Graph Break问题示例"""
    
    def problematic_version(x):
        # 大量graph break
        y = torch.relu(x)
        
        # Python内置函数
        shape_len = len(x.shape)
        
        # 数据依赖分支
        if y.sum() > 0:
            z = y * 2
        else:
            z = y / 2
            
        # 打印
        print(f"Sum: {z.sum()}")
        
        return z.mean() + shape_len
    
    def optimized_version(x):
        # 减少graph break
        y = torch.relu(x)
        
        # 使用张量操作而非Python内置函数
        shape_len = x.dim()
        
        # 使用条件运算
        condition = y.sum() > 0
        z = torch.where(condition, y * 2, y / 2)
        
        # 避免打印等副作用
        return z.mean() + shape_len
    
    return problematic_version, optimized_version

def create_compilation_failure_example():
    """创建编译失败示例"""
    
    def problematic_version(x):
        # 使用不支持的特性
        try:
            # 动态属性访问
            attr_name = 'sum'
            result = getattr(x, attr_name)()
        except:
            # 异常处理
            result = torch.tensor(0.0)
        
        # exec调用
        exec_code = "temp = x * 2"
        exec(exec_code)
        
        return result + locals().get('temp', torch.tensor(0.0))
    
    def optimized_version(x):
        # 使用支持的操作
        result = x.sum()
        temp = x * 2
        return result + temp.sum()
    
    return problematic_version, optimized_version

def create_numerical_inconsistency_example():
    """创建数值不一致示例"""
    
    def problematic_version(x):
        # 可能导致数值不稳定的操作
        y = torch.exp(x)  # 可能溢出
        z = torch.log(y + 1e-8)  # 数值不稳定
        return z.sum()
    
    def optimized_version(x):
        # 数值稳定的版本
        # 使用log1p和expm1
        y = torch.clamp(x, min=-10, max=10)  # 防止溢出
        z = torch.log1p(torch.exp(y))  # 数值稳定的log(1+exp(x))
        return z.sum()
    
    return problematic_version, optimized_version

def create_memory_leak_example():
    """创建内存泄漏示例"""
    
    def problematic_version(x):
        # 保存大量中间结果
        intermediates = []
        current = x
        
        for i in range(5):
            current = current @ current.t()
            intermediates.append(current.detach().clone())  # 保存所有中间结果
        
        return sum(intermediates).trace()
    
    def optimized_version(x):
        # 避免保存不必要的中间结果
        current = x
        
        for i in range(5):
            current = current @ current.t()
            # 不保存中间结果
        
        return current.trace()
    
    return problematic_version, optimized_version

def test_function_issues(func):
    """测试函数的问题"""
    
    x = torch.randn(50, 50)
    
    try:
        # 分析explain
        explanation = torch._dynamo.explain(func)(x)
        print(f"  Graph数量: {explanation.graph_count}")
        print(f"  Break数量: {explanation.graph_break_count}")
        
        # 尝试编译
        compiled = torch.compile(func)
        result = compiled(x)
        print(f"  ✅ 编译成功，结果: {result:.4f}")
        
    except Exception as e:
        print(f"  ❌ 编译失败: {e}")

# ============================================================================
# 5. 调试工具和最佳实践
# ============================================================================

def demonstrate_debugging_tools():
    """演示调试工具的使用"""
    
    print("\n🛠️ 调试工具使用指南")
    print("=" * 60)
    
    print("\n1. 环境变量配置:")
    print("   基础调试: export TORCH_LOGS='graph_breaks,recompiles'")
    print("   详细调试: export TORCH_LOGS='+dynamo,+aot,+inductor'")
    print("   性能分析: export TORCH_LOGS='schedule,fusion,perf_hints'")
    
    print("\n2. 程序化配置:")
    print("""
    import torch._logging
    
    # 基础配置
    torch._logging.set_logs(
        graph_breaks=True,
        recompiles=True,
        guards=True
    )
    
    # 详细配置
    torch._logging.set_logs(
        dynamo=torch._logging.logging.DEBUG,
        aot=torch._logging.logging.DEBUG,
        inductor=torch._logging.logging.DEBUG
    )
    """)
    
    print("\n3. 分析工具:")
    print("   - torch._dynamo.explain(): 分析函数结构")
    print("   - torch.compile(backend='eager'): 测试Dynamo阶段")  
    print("   - torch.compile(backend='aot_eager'): 测试AOT阶段")
    print("   - torch.compile(backend='inductor'): 完整编译")
    
    print("\n4. 调试流程:")
    print("   Step 1: 使用explain()快速识别问题")
    print("   Step 2: 设置适当的日志级别")
    print("   Step 3: 逐阶段测试backend")
    print("   Step 4: 分析生成的调试文件")
    print("   Step 5: 优化代码减少graph break")

def create_debug_context_manager():
    """创建调试上下文管理器"""
    
    print("\n📦 调试上下文管理器")
    print("-" * 50)
    
    class TorchCompileDebugContext:
        """torch.compile调试上下文管理器"""
        
        def __init__(self, logs="graph_breaks,recompiles", debug_dir=None):
            self.logs = logs
            self.debug_dir = debug_dir or tempfile.mkdtemp(prefix="torch_debug_")
            self.original_env = {}
            
        def __enter__(self):
            # 保存原始环境
            self.original_env = dict(os.environ)
            
            # 设置调试环境
            os.environ.update({
                "TORCH_LOGS": self.logs,
                "TORCH_COMPILE_DEBUG": "1",
                "TORCH_COMPILE_DEBUG_DIR": self.debug_dir
            })
            
            # 程序化配置
            torch._logging.set_logs(
                graph_breaks=True,
                recompiles=True,
                guards=True
            )
            
            print(f"✅ 调试环境已激活，文件保存在: {self.debug_dir}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # 恢复环境
            os.environ.clear()
            os.environ.update(self.original_env)
            
            print("✅ 调试环境已清理")
    
    # 使用示例
    print("使用示例:")
    print("""
    with TorchCompileDebugContext("graph_breaks,output_code") as debug:
        @torch.compile
        def my_func(x):
            return torch.relu(x).sum()
        
        result = my_func(torch.randn(100))
        print(f"调试文件: {debug.debug_dir}")
    """)
    
    return TorchCompileDebugContext

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数 - 全面展示torch.compile调试方法"""
    
    print("torch.compile 针对性调试方法完整指南")
    print("=" * 80)
    
    try:
        # 1. TorchDynamo阶段调试
        print("\n🔍 第一部分：TorchDynamo阶段调试")
        setup_dynamo_debugging()
        analyze_graph_breaks_detailed()
        debug_guard_failures()
        demonstrate_dynamic_shape_debugging()
        
        # 2. AOTAutograd阶段调试  
        print("\n🔄 第二部分：AOTAutograd阶段调试")
        setup_aot_debugging()
        debug_backward_graph_issues()
        debug_memory_optimization()
        
        # 3. Inductor阶段调试
        print("\n💻 第三部分：Inductor阶段调试")
        setup_inductor_debugging()
        debug_kernel_fusion()
        debug_performance_regression()
        
        # 4. 错误模式分析
        print("\n🔧 第四部分：错误模式分析")
        analyze_common_error_patterns()
        
        # 5. 调试工具
        print("\n🛠️ 第五部分：调试工具和最佳实践")
        demonstrate_debugging_tools()
        create_debug_context_manager()
        
        print(f"\n✅ 全面的torch.compile调试指南演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()