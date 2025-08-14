#!/usr/bin/env python3
"""
torch.compile 针对性调试方法总结

基于实际测试的torch.compile各阶段调试方法和解决方案
"""

import os
import torch
import torch._dynamo
import torch._logging
import time
import tempfile
import traceback

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# ============================================================================
# 1. TorchDynamo阶段调试方法
# ============================================================================

def demonstrate_dynamo_debugging():
    """TorchDynamo阶段调试演示"""
    
    print("\n" + "="*80)
    print("1. TorchDynamo阶段调试：定位Graph Break和动态行为问题")
    print("="*80)
    
    # 环境变量配置
    print("\n🔧 调试环境配置：")
    print("export TORCH_LOGS='graph_breaks,recompiles,guards,dynamic'")
    print("export TORCH_COMPILE_DEBUG=1")
    
    # 程序化配置
    torch._logging.set_logs(
        graph_breaks=True,
        recompiles=True,
        guards=True,
        dynamic=True
    )
    
    print("\n💥 Graph Break案例分析：")
    
    def problematic_function(x):
        """包含多种Graph Break的函数"""
        # 正常张量操作
        y = torch.relu(x + 1)
        
        # Python内置函数 - 会导致graph break
        shape_len = len(x.shape)
        
        # 数据依赖控制流 - 会导致graph break
        if y.sum() > 0:
            z = y * 2
        else:
            z = y / 2
            
        # 副作用操作 - 会导致graph break
        # print(f"中间结果: {z.mean():.4f}")
        
        return z.sum() + shape_len
    
    # 使用explain分析
    test_input = torch.randn(10, 20)
    explanation = torch._dynamo.explain(problematic_function)(test_input)
    
    print(f"📊 分析结果：")
    print(f"  - 图数量: {explanation.graph_count}")
    print(f"  - Graph Break数量: {explanation.graph_break_count}")
    print(f"  - 操作总数: {explanation.op_count}")
    
    print(f"\n💡 优化方案 - 减少Graph Break：")
    
    def optimized_function(x):
        """优化后的函数"""
        y = torch.relu(x + 1)
        
        # 使用张量操作而非Python内置函数
        shape_len = x.dim()
        
        # 使用条件运算替代控制流
        condition = y.sum() > 0
        z = torch.where(condition, y * 2, y / 2)
        
        # 避免副作用操作
        return z.sum() + shape_len
    
    optimized_explanation = torch._dynamo.explain(optimized_function)(test_input)
    
    print(f"优化后：")
    print(f"  - 图数量: {optimized_explanation.graph_count}")
    print(f"  - Graph Break数量: {optimized_explanation.graph_break_count}")
    print(f"  - 操作总数: {optimized_explanation.op_count}")
    
    print(f"\n🛡️ Guard失败调试：")
    
    def shape_dependent_function(x):
        return torch.matmul(x, x.transpose(-2, -1)).sum()
    
    compiled_func = torch.compile(shape_dependent_function)
    
    # 测试不同形状观察重编译
    shapes = [(10, 10), (20, 20), (10, 10)]  # 第三个重复测试缓存
    
    for i, shape in enumerate(shapes):
        x = torch.randn(*shape)
        start_time = time.time()
        result = compiled_func(x)
        end_time = time.time()
        
        print(f"  测试{i+1} {shape}: {result:.2f} ({(end_time-start_time)*1000:.1f}ms)")
    
    print(f"\n🎭 动态形状处理：")
    
    def dynamic_function(x):
        if x.dim() == 1:
            return x.sum()
        elif x.dim() == 2:
            return x.trace()
        else:
            return x.flatten().sum()
    
    # 静态编译
    static_compiled = torch.compile(dynamic_function, dynamic=False)
    # 动态编译
    dynamic_compiled = torch.compile(dynamic_function, dynamic=True)
    
    test_inputs = [torch.randn(10), torch.randn(5, 5), torch.randn(2, 3, 4)]
    
    print("  静态模式 vs 动态模式：")
    for i, x in enumerate(test_inputs):
        try:
            static_result = static_compiled(x)
            print(f"    输入{i+1} {x.shape}: 静态✅ ({static_result:.2f})")
        except Exception as e:
            print(f"    输入{i+1} {x.shape}: 静态❌")
        
        try:
            dynamic_result = dynamic_compiled(x)
            print(f"    输入{i+1} {x.shape}: 动态✅ ({dynamic_result:.2f})")
        except Exception as e:
            print(f"    输入{i+1} {x.shape}: 动态❌")
    
    return explanation, optimized_explanation

# ============================================================================
# 2. AOTAutograd阶段调试方法
# ============================================================================

def demonstrate_aot_debugging():
    """AOTAutograd阶段调试演示"""
    
    print("\n" + "="*80)
    print("2. AOTAutograd阶段调试：反向图生成和内存优化问题")
    print("="*80)
    
    # 配置AOT调试
    torch._logging.set_logs(
        aot=True,
        graph_breaks=True
    )
    
    print("\n🔄 反向图生成调试：")
    
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 20)
            self.fc2 = torch.nn.Linear(20, 5)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x.sum()
    
    model = TestModel()
    x = torch.randn(16, 10, requires_grad=True)
    
    backends = ["eager", "aot_eager", "inductor"]
    
    print("  测试不同backend的梯度计算：")
    
    for backend in backends:
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
            
            print(f"    {backend}: loss={loss:.3f}, 参数梯度={[f'{g:.3f}' for g in param_grads]}")
            
        except Exception as e:
            print(f"    {backend}: ❌ {e}")
    
    print(f"\n💾 内存优化调试：")
    
    def memory_test_function(x):
        """内存密集型函数"""
        results = []
        current = x
        
        for i in range(3):
            current = torch.matmul(current, current.t())
            current = torch.relu(current)
            results.append(current.clone())  # 保存中间结果
        
        return sum(results).trace()
    
    x = torch.randn(32, 32, requires_grad=True)
    
    import psutil
    process = psutil.Process()
    
    test_configs = [("原始函数", None), ("AOT编译", "aot_eager")]
    
    for name, backend in test_configs:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            if backend is None:
                func = memory_test_function
            else:
                func = torch.compile(memory_test_function, backend=backend)
            
            loss = func(x)
            loss.backward()
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_memory - initial_memory
            
            print(f"    {name}: loss={loss:.3f}, 内存使用={memory_usage:.1f}MB")
            
        except Exception as e:
            print(f"    {name}: ❌ {e}")
        
        # 清理梯度
        if x.grad is not None:
            x.grad.zero_()
    
    print(f"\n🔧 函数式变换问题：")
    
    # 测试有问题的模式
    problematic_patterns = {
        "In-place操作": lambda x: x.add_(1).sum(),
        "复杂视图操作": lambda x: x.view(-1)[::2].sum(),
        "非连续张量": lambda x: x.transpose(0, 1).sum()
    }
    
    x = torch.randn(8, 8, requires_grad=True)
    
    for name, pattern in problematic_patterns.items():
        try:
            compiled = torch.compile(pattern, backend="aot_eager")
            result = compiled(x.clone())
            print(f"    {name}: ✅ {result:.3f}")
        except Exception as e:
            print(f"    {name}: ❌ 变换失败")

# ============================================================================
# 3. Inductor阶段调试方法
# ============================================================================

def demonstrate_inductor_debugging():
    """Inductor阶段调试演示"""
    
    print("\n" + "="*80)
    print("3. Inductor阶段调试：代码生成、内核融合和性能优化")
    print("="*80)
    
    # 配置Inductor调试
    torch._logging.set_logs(
        inductor=True,
        schedule=True,
        fusion=True,
        output_code=True
    )
    
    print(f"\n💻 代码生成调试：")
    
    def fusion_example(x, y):
        """展示内核融合的例子"""
        z1 = torch.relu(x + y)
        z2 = torch.tanh(z1 * 2)
        z3 = torch.sigmoid(z2 - 1)
        return z3.sum()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(256, 256, device=device)
    y = torch.randn(256, 256, device=device)
    
    try:
        compiled_func = torch.compile(fusion_example, backend="inductor")
        result = compiled_func(x, y)
        print(f"    融合示例: ✅ 结果={result:.4f}")
    except Exception as e:
        print(f"    融合示例: ❌ {e}")
    
    print(f"\n🔀 内核融合分析：")
    
    fusion_patterns = {
        "简单融合": lambda x: torch.relu(torch.sigmoid(x)).sum(),
        "复杂融合": lambda x: torch.softmax(torch.relu(x + 1), dim=-1).sum(),
        "广播融合": lambda x: (x + torch.randn(1, x.shape[1], device=x.device)).sum(),
        "阻止融合": lambda x: torch.relu(x).contiguous().sigmoid().sum()
    }
    
    x = torch.randn(128, 128, device=device)
    
    for name, pattern in fusion_patterns.items():
        try:
            compiled = torch.compile(pattern, backend="inductor")
            
            # 预热
            for _ in range(3):
                _ = compiled(x)
            
            # 性能测试
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.time()
            for _ in range(50):
                result = compiled(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.time()
            avg_time = (end_time - start_time) / 50 * 1000  # ms
            
            print(f"    {name}: ✅ {result:.3f} ({avg_time:.2f}ms)")
            
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    print(f"\n⚡ 性能分析：")
    
    def create_perf_test_functions():
        return {
            "矩阵乘法": lambda x, y: torch.matmul(x, y).sum(),
            "复合运算": lambda x: torch.softmax(torch.relu(x), dim=-1).sum(),
            "元素运算": lambda x: (torch.sin(x) + torch.cos(x)).sum()
        }
    
    perf_functions = create_perf_test_functions()
    
    # 准备测试输入
    test_inputs = {
        "矩阵乘法": (torch.randn(128, 128, device=device), torch.randn(128, 128, device=device)),
        "复合运算": (torch.randn(256, 256, device=device),),
        "元素运算": (torch.randn(512, 512, device=device),)
    }
    
    for name, func in perf_functions.items():
        inputs = test_inputs[name]
        
        try:
            # 原始性能
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.time()
            for _ in range(20):
                original_result = func(*inputs)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            original_time = time.time() - start_time
            
            # 编译性能
            compiled_func = torch.compile(func, backend="inductor")
            
            # 预热
            for _ in range(5):
                _ = compiled_func(*inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.time()
            for _ in range(20):
                compiled_result = compiled_func(*inputs)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            compiled_time = time.time() - start_time
            
            speedup = original_time / compiled_time if compiled_time > 0 else float('inf')
            
            print(f"    {name}:")
            print(f"      原始: {original_time*1000:.1f}ms, 编译: {compiled_time*1000:.1f}ms")
            print(f"      加速比: {speedup:.2f}x")
            
            # 检查正确性
            if torch.allclose(original_result, compiled_result, rtol=1e-4):
                print(f"      ✅ 数值一致")
            else:
                print(f"      ⚠️  数值差异")
                
        except Exception as e:
            print(f"    {name}: ❌ {e}")

# ============================================================================
# 4. 常见错误模式和解决方案
# ============================================================================

def demonstrate_error_patterns():
    """演示常见错误模式和解决方案"""
    
    print("\n" + "="*80)
    print("4. 常见错误模式和解决方案")
    print("="*80)
    
    print(f"\n❌ 编译失败典型原因：")
    
    # 1. Graph Break过多
    def graph_break_heavy(x):
        y = torch.relu(x)
        
        # 多个break点
        shape_len = len(x.shape)  # Python内置函数
        
        if y.sum() > 0:  # 数据依赖分支
            z = y * 2
        else:
            z = y / 2
            
        return z.sum() + shape_len
    
    def graph_break_optimized(x):
        y = torch.relu(x)
        shape_len = x.dim()  # 张量方法
        condition = y.sum() > 0
        z = torch.where(condition, y * 2, y / 2)  # 条件运算
        return z.sum() + shape_len
    
    x = torch.randn(50, 50)
    
    print("  Graph Break优化前后对比：")
    
    for name, func in [("优化前", graph_break_heavy), ("优化后", graph_break_optimized)]:
        try:
            explanation = torch._dynamo.explain(func)(x)
            compiled = torch.compile(func)
            result = compiled(x)
            
            print(f"    {name}: 图数={explanation.graph_count}, Break数={explanation.graph_break_count}, 结果={result:.3f}")
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    print(f"\n⚠️ 性能回退分析：")
    
    # 简单函数可能不适合编译
    def simple_function(x):
        return x.sum()
    
    def complex_function(x):
        return torch.softmax(torch.relu(x @ x.t()) + 1, dim=-1).trace()
    
    x = torch.randn(100, 100)
    
    for name, func in [("简单函数", simple_function), ("复杂函数", complex_function)]:
        try:
            # 原始性能
            start_time = time.time()
            for _ in range(100):
                orig_result = func(x)
            orig_time = time.time() - start_time
            
            # 编译性能
            compiled = torch.compile(func)
            for _ in range(3):  # 预热
                _ = compiled(x)
            
            start_time = time.time()
            for _ in range(100):
                comp_result = compiled(x)
            comp_time = time.time() - start_time
            
            speedup = orig_time / comp_time
            
            print(f"    {name}: 加速比={speedup:.2f}x", end="")
            if speedup < 1.0:
                print(" ⚠️ 性能回退")
            else:
                print(" ✅ 性能提升")
                
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    print(f"\n💾 内存问题处理：")
    
    def memory_hungry(x):
        # 保存大量中间结果
        results = []
        current = x
        for i in range(3):
            current = current @ current.t()
            results.append(current.clone())
        return sum(results).trace()
    
    def memory_efficient(x):
        # 避免保存不必要的中间结果
        current = x
        for i in range(3):
            current = current @ current.t()
        return current.trace()
    
    x = torch.randn(64, 64)
    
    import psutil
    process = psutil.Process()
    
    for name, func in [("内存密集", memory_hungry), ("内存优化", memory_efficient)]:
        initial_mem = process.memory_info().rss / 1024 / 1024
        
        try:
            compiled = torch.compile(func)
            result = compiled(x)
            
            peak_mem = process.memory_info().rss / 1024 / 1024
            mem_usage = peak_mem - initial_mem
            
            print(f"    {name}: 结果={result:.3f}, 内存={mem_usage:.1f}MB")
            
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    print(f"\n🔢 数值精度问题：")
    
    def numerically_unstable(x):
        # 可能数值不稳定
        return torch.log(torch.exp(x) + 1e-8).sum()
    
    def numerically_stable(x):
        # 数值稳定版本
        return torch.nn.functional.softplus(x).sum()
    
    x = torch.randn(100, 100) * 10  # 大数值测试
    
    for name, func in [("不稳定版本", numerically_unstable), ("稳定版本", numerically_stable)]:
        try:
            original = func(x)
            compiled = torch.compile(func)(x)
            
            diff = torch.abs(original - compiled)
            
            print(f"    {name}: 原始={original:.3f}, 编译={compiled:.3f}, 差异={diff:.6f}")
            
        except Exception as e:
            print(f"    {name}: ❌ {e}")

# ============================================================================
# 5. 调试工具和最佳实践总结
# ============================================================================

def summarize_debugging_tools():
    """总结调试工具和最佳实践"""
    
    print("\n" + "="*80)
    print("5. 调试工具和最佳实践总结")
    print("="*80)
    
    print(f"\n🛠️ 核心调试工具：")
    
    tools = {
        "torch._dynamo.explain()": "快速分析函数结构和graph break",
        "TORCH_LOGS环境变量": "控制详细的日志输出",
        "不同backend测试": "eager -> aot_eager -> inductor 逐阶段调试",
        "动态编译模式": "dynamic=True处理变化的输入形状",
        "编译统计": "分析重编译原因和性能瓶颈"
    }
    
    for tool, desc in tools.items():
        print(f"  • {tool}: {desc}")
    
    print(f"\n📋 推荐调试流程：")
    
    workflow = [
        "1. 使用explain()快速识别问题点",
        "2. 设置适当的TORCH_LOGS级别",
        "3. 从eager backend开始逐阶段测试", 
        "4. 分析graph break原因并优化代码",
        "5. 检查编译后的性能和数值正确性",
        "6. 根据具体问题应用相应解决方案"
    ]
    
    for step in workflow:
        print(f"  {step}")
    
    print(f"\n⚡ 性能优化建议：")
    
    tips = [
        "• 减少不必要的graph break",
        "• 使用PyTorch原生操作替代Python内置函数",
        "• 避免数据依赖的控制流",
        "• 使用条件运算torch.where替代if/else",
        "• 预热编译消除首次执行开销",
        "• 对于简单函数考虑是否真的需要编译",
        "• 使用动态形状模式处理变化输入",
        "• 定期检查内存使用和数值稳定性"
    ]
    
    for tip in tips:
        print(f"  {tip}")
    
    print(f"\n🎯 环境变量速查：")
    
    env_vars = {
        "基础调试": "TORCH_LOGS='graph_breaks,recompiles'",
        "详细调试": "TORCH_LOGS='+dynamo,+aot,+inductor'", 
        "性能分析": "TORCH_LOGS='schedule,fusion,perf_hints'",
        "代码查看": "TORCH_LOGS='output_code,kernel_code'",
        "启用调试": "TORCH_COMPILE_DEBUG=1",
        "自动复现": "TORCHDYNAMO_REPRO_AFTER='dynamo'"
    }
    
    for purpose, setting in env_vars.items():
        print(f"  {purpose}: {setting}")

# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行完整的调试演示"""
    
    print("torch.compile 针对性调试方法完整指南")
    print("=" * 80)
    print("本指南提供torch.compile各阶段的具体调试方法和实际案例")
    
    try:
        # 1. TorchDynamo调试
        dynamo_results = demonstrate_dynamo_debugging()
        
        # 2. AOTAutograd调试
        demonstrate_aot_debugging()
        
        # 3. Inductor调试
        demonstrate_inductor_debugging()
        
        # 4. 错误模式分析
        demonstrate_error_patterns()
        
        # 5. 工具总结
        summarize_debugging_tools()
        
        print(f"\n✅ torch.compile调试指南演示完成！")
        print(f"💡 记住：从简单的explain()开始，逐步深入到具体阶段的调试")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()