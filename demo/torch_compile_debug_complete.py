#!/usr/bin/env python3
"""
torch.compile 深度调试指南 - 实用版本

针对TorchDynamo、AOTAutograd、Inductor各阶段的调试方法和解决方案
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
# torch.compile 各阶段调试方法总结
# ============================================================================

def main():
    """torch.compile 深度调试完整指南"""
    
    print("\n" + "="*80)
    print("torch.compile 深度调试完整指南")
    print("="*80)
    
    # 设置基础调试环境
    torch._logging.set_logs(
        graph_breaks=True,
        recompiles=True,
        guards=True
    )
    
    print("\n🎯 调试目标:")
    print("1. TorchDynamo阶段: Graph Break分析、Guard调试、动态行为")
    print("2. AOTAutograd阶段: 反向图生成、内存优化、函数式变换")
    print("3. Inductor阶段: 代码生成、内核融合、性能优化")
    print("4. 错误模式: 编译失败、性能回退、内存异常、数值精度")
    
    # ========================================================================
    # 第一部分: TorchDynamo阶段调试
    # ========================================================================
    
    print("\n" + "="*60)
    print("1. TorchDynamo阶段调试方法")
    print("="*60)
    
    print("\n🔍 Graph Break分析:")
    
    def problematic_function(x):
        """包含多种Graph Break的函数"""
        # 正常张量操作
        y = torch.relu(x + 1)
        
        # Python内置函数 - 会导致break
        shape_len = len(x.shape)
        
        # 数据依赖的控制流 - 会导致break
        if y.sum() > 0:
            z = y * 2
        else:
            z = y / 2
            
        return z.sum() + shape_len
    
    def optimized_function(x):
        """优化后减少Graph Break的函数"""
        y = torch.relu(x + 1)
        
        # 使用张量方法代替Python内置函数
        shape_len = x.dim()
        
        # 使用条件运算代替控制流
        condition = y.sum() > 0
        z = torch.where(condition, y * 2, y / 2)
        
        return z.sum() + shape_len
    
    # 分析Graph Break
    test_input = torch.randn(10, 20)
    
    print("  原始函数分析:")
    try:
        explanation = torch._dynamo.explain(problematic_function)(test_input)
        print(f"    图数量: {explanation.graph_count}")
        print(f"    Graph Break数量: {explanation.graph_break_count}")
        print(f"    操作总数: {explanation.op_count}")
        
        print(f"    Break原因:")
        for i, reason in enumerate(explanation.break_reasons):
            print(f"      {i+1}. {str(reason)[:100]}...")
    except Exception as e:
        print(f"    ❌ 分析失败: {e}")
    
    print("  优化后函数分析:")
    try:
        opt_explanation = torch._dynamo.explain(optimized_function)(test_input)
        print(f"    图数量: {opt_explanation.graph_count}")
        print(f"    Graph Break数量: {opt_explanation.graph_break_count}")
        print(f"    操作总数: {opt_explanation.op_count}")
    except Exception as e:
        print(f"    ❌ 分析失败: {e}")
    
    print("\n🛡️ Guard失败调试:")
    
    def shape_dependent_func(x):
        return torch.matmul(x, x.transpose(-2, -1)).sum()
    
    compiled_func = torch.compile(shape_dependent_func)
    
    # 测试不同形状观察重编译
    shapes = [(10, 10), (20, 20), (10, 10), (30, 30)]
    
    print("  测试不同输入形状的重编译行为:")
    for i, shape in enumerate(shapes):
        x = torch.randn(*shape)
        start_time = time.time()
        result = compiled_func(x)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        
        print(f"    测试{i+1} {shape}: {result:.2f} ({execution_time:.1f}ms)")
        if i > 0 and execution_time > 100:  # 首次编译通常较慢
            print(f"      ⚠️ 可能发生了重编译")
    
    print("\n🎭 动态形状处理:")
    
    def dynamic_shape_func(x):
        if x.dim() == 1:
            return x.sum()
        elif x.dim() == 2:
            return x.trace() 
        else:
            return x.flatten().sum()
    
    # 测试静态和动态编译
    static_compiled = torch.compile(dynamic_shape_func, dynamic=False)
    dynamic_compiled = torch.compile(dynamic_shape_func, dynamic=True)
    
    test_inputs = [
        torch.randn(10),
        torch.randn(5, 5),
        torch.randn(2, 3, 4)
    ]
    
    print("  静态编译 vs 动态编译对比:")
    for i, x in enumerate(test_inputs):
        print(f"    输入{i+1} {x.shape}:")
        
        # 测试静态编译
        try:
            static_result = static_compiled(x)
            print(f"      静态: ✅ {static_result:.3f}")
        except Exception as e:
            print(f"      静态: ❌ 失败")
        
        # 测试动态编译
        try:
            dynamic_result = dynamic_compiled(x)
            print(f"      动态: ✅ {dynamic_result:.3f}")
        except Exception as e:
            print(f"      动态: ❌ 失败")
    
    # ========================================================================
    # 第二部分: AOTAutograd阶段调试
    # ========================================================================
    
    print("\n" + "="*60)
    print("2. AOTAutograd阶段调试方法")
    print("="*60)
    
    print("\n🔄 反向图生成调试:")
    
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
    
    print("  不同backend的梯度计算测试:")
    
    for backend in backends:
        print(f"    测试 {backend}:")
        
        try:
            # 重置梯度
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
            
            # 编译模型
            if backend == "eager":
                compiled_model = model
            else:
                compiled_model = torch.compile(model, backend=backend)
            
            # 前向和反向传播
            loss = compiled_model(x)
            loss.backward()
            
            # 检查梯度
            param_grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            input_grad_norm = x.grad.norm().item() if x.grad is not None else 0
            
            print(f"      Loss: {loss:.4f}")
            print(f"      参数梯度范数: {[f'{g:.4f}' for g in param_grad_norms]}")
            print(f"      输入梯度范数: {input_grad_norm:.4f}")
            
        except Exception as e:
            print(f"      ❌ 失败: {e}")
    
    print("\n💾 内存优化调试:")
    
    def memory_intensive_func(x):
        """内存密集型函数"""
        results = []
        current = x
        
        for i in range(3):
            current = torch.matmul(current, current.t())
            current = torch.relu(current)
            results.append(current.clone())  # 保存中间结果
        
        return sum(results).trace()
    
    def memory_efficient_func(x):
        """内存优化版本"""
        current = x
        
        for i in range(3):
            current = torch.matmul(current, current.t()) 
            current = torch.relu(current)
        
        return current.trace()
    
    x = torch.randn(32, 32, requires_grad=True)
    
    import psutil
    process = psutil.Process()
    
    functions = [
        ("内存密集版本", memory_intensive_func),
        ("内存优化版本", memory_efficient_func)
    ]
    
    print("  内存使用对比:")
    
    for name, func in functions:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            compiled_func = torch.compile(func, backend="aot_eager")
            loss = compiled_func(x.clone())
            loss.backward()
            
            peak_mem = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = peak_mem - initial_mem
            
            print(f"    {name}: Loss={loss:.4f}, 内存={memory_usage:.1f}MB")
            
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    # ========================================================================
    # 第三部分: Inductor阶段调试
    # ========================================================================
    
    print("\n" + "="*60)
    print("3. Inductor阶段调试方法")
    print("="*60)
    
    print("\n💻 代码生成和内核融合:")
    
    def fusion_example(x, y):
        """内核融合示例"""
        z1 = torch.relu(x + y)
        z2 = torch.tanh(z1 * 2)
        z3 = torch.sigmoid(z2 - 1)
        return z3.sum()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(128, 128, device=device)
    y = torch.randn(128, 128, device=device)
    
    print(f"  在{device}设备上测试内核融合:")
    
    try:
        compiled_fusion = torch.compile(fusion_example, backend="inductor")
        result = compiled_fusion(x, y)
        print(f"    ✅ 融合示例成功: {result:.4f}")
    except Exception as e:
        print(f"    ❌ 融合失败: {e}")
    
    print("\n🔀 不同融合模式测试:")
    
    fusion_patterns = {
        "简单融合": lambda x: torch.relu(torch.sigmoid(x)).sum(),
        "复杂融合链": lambda x: torch.softmax(torch.relu(x + 1), dim=-1).sum(),
        "广播融合": lambda x: (x + torch.ones(1, x.shape[1], device=x.device)).sum(),
        "归约融合": lambda x: (x * 2).sum(dim=1).mean(),
        "阻止融合": lambda x: torch.relu(x).contiguous().sigmoid().sum()
    }
    
    x = torch.randn(64, 64, device=device)
    
    for name, pattern in fusion_patterns.items():
        try:
            compiled_pattern = torch.compile(pattern, backend="inductor")
            
            # 预热编译
            for _ in range(3):
                _ = compiled_pattern(x)
            
            # 性能测试
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(50):
                result = compiled_pattern(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 50 * 1000  # ms
            
            print(f"    {name}: ✅ {result:.4f} ({avg_time:.2f}ms)")
            
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    print("\n⚡ 性能分析:")
    
    perf_functions = {
        "矩阵乘法": lambda x, y: torch.matmul(x, y).sum(),
        "复合运算": lambda x: torch.softmax(torch.relu(x @ x.t()), dim=-1).sum(),
        "元素运算": lambda x: (torch.sin(x) + torch.cos(x) * torch.tanh(x)).sum()
    }
    
    test_inputs = {
        "矩阵乘法": (torch.randn(64, 64, device=device), torch.randn(64, 64, device=device)),
        "复合运算": (torch.randn(32, 32, device=device),),
        "元素运算": (torch.randn(128, 128, device=device),)
    }
    
    print("  编译前后性能对比:")
    
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
            
            # 计算加速比
            speedup = original_time / compiled_time if compiled_time > 0 else float('inf')
            
            print(f"    {name}:")
            print(f"      原始: {original_time*1000:.1f}ms, 编译: {compiled_time*1000:.1f}ms")
            print(f"      加速比: {speedup:.2f}x", end="")
            
            if speedup >= 1.1:
                print(" ✅ 性能提升")
            elif speedup >= 0.9:
                print(" ➖ 性能相当")
            else:
                print(" ⚠️ 性能回退")
            
            # 数值正确性检查
            if torch.allclose(original_result, compiled_result, rtol=1e-4, atol=1e-4):
                print(f"      ✅ 数值一致")
            else:
                print(f"      ⚠️ 数值差异: {torch.abs(original_result - compiled_result).max():.6f}")
                
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    # ========================================================================
    # 第四部分: 常见错误模式和解决方案
    # ========================================================================
    
    print("\n" + "="*60)
    print("4. 常见错误模式和解决方案")
    print("="*60)
    
    print("\n❌ Graph Break过多问题:")
    
    def graph_break_heavy(x):
        """Graph Break过多的函数"""
        y = torch.relu(x)
        
        # 多个break点
        shape_len = len(x.shape)  # Python内置函数
        
        if y.sum() > 0:  # 数据依赖分支
            z = y * 2
            print(f"Positive sum: {y.sum()}")  # 打印语句
        else:
            z = y / 2
        
        # 循环
        for i in range(3):
            z = z + i * 0.1
            
        return z.sum() + shape_len
    
    def graph_break_optimized(x):
        """优化后的函数"""
        y = torch.relu(x)
        
        # 使用张量方法
        shape_len = x.dim()
        
        # 使用条件运算
        condition = y.sum() > 0
        z = torch.where(condition, y * 2, y / 2)
        
        # 向量化操作
        additions = torch.arange(3, dtype=x.dtype, device=x.device) * 0.1
        z = z + additions.sum()
        
        return z.sum() + shape_len
    
    x = torch.randn(32, 32)
    
    print("  优化前后对比:")
    
    for name, func in [("优化前", graph_break_heavy), ("优化后", graph_break_optimized)]:
        try:
            explanation = torch._dynamo.explain(func)(x)
            compiled_func = torch.compile(func)
            
            start_time = time.time()
            result = compiled_func(x)
            end_time = time.time()
            
            print(f"    {name}:")
            print(f"      Graph数量: {explanation.graph_count}")
            print(f"      Break数量: {explanation.graph_break_count}")
            print(f"      执行时间: {(end_time-start_time)*1000:.1f}ms")
            print(f"      结果: {result:.4f}")
            
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    print("\n⚠️ 性能回退分析:")
    
    # 简单函数可能不适合编译
    simple_func = lambda x: x.sum()
    complex_func = lambda x: torch.softmax(torch.relu(x @ x.t()) + 1, dim=-1).trace()
    
    x = torch.randn(64, 64)
    
    for name, func in [("简单函数", simple_func), ("复杂函数", complex_func)]:
        try:
            # 测试编译开销
            start_time = time.time()
            for _ in range(100):
                orig_result = func(x)
            orig_time = time.time() - start_time
            
            compiled_func = torch.compile(func)
            
            # 预热
            for _ in range(3):
                _ = compiled_func(x)
            
            start_time = time.time()
            for _ in range(100):
                comp_result = compiled_func(x)
            comp_time = time.time() - start_time
            
            speedup = orig_time / comp_time if comp_time > 0 else float('inf')
            
            print(f"    {name}: 加速比={speedup:.2f}x", end="")
            
            if speedup < 0.9:
                print(" ⚠️ 性能回退，可能原因:")
                print("      - 函数过于简单，编译开销大于收益")
                print("      - Graph Break过多")
                print("      - 内存访问模式不友好")
            else:
                print(" ✅ 适合编译")
                
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    print("\n🔢 数值精度问题:")
    
    def numerically_unstable(x):
        """数值不稳定的函数"""
        # 大指数可能溢出
        large_x = x * 100
        return torch.log(torch.exp(large_x) + 1e-8).sum()
    
    def numerically_stable(x):
        """数值稳定的版本"""
        # 使用数值稳定的实现
        large_x = torch.clamp(x * 100, min=-50, max=50)
        return torch.nn.functional.softplus(large_x).sum()
    
    x = torch.randn(50, 50)
    
    print("  数值稳定性测试:")
    
    for name, func in [("不稳定版本", numerically_unstable), ("稳定版本", numerically_stable)]:
        try:
            original = func(x)
            compiled = torch.compile(func)(x)
            
            diff = torch.abs(original - compiled).max()
            
            print(f"    {name}:")
            print(f"      原始结果: {original:.6f}")
            print(f"      编译结果: {compiled:.6f}")
            print(f"      最大差异: {diff:.8f}")
            
            if diff < 1e-5:
                print(f"      ✅ 数值稳定")
            else:
                print(f"      ⚠️ 数值不稳定")
                
        except Exception as e:
            print(f"    {name}: ❌ {e}")
    
    # ========================================================================
    # 调试工具和最佳实践总结
    # ========================================================================
    
    print("\n" + "="*60)
    print("5. 调试工具和最佳实践总结")
    print("="*60)
    
    print("\n🛠️ 核心调试工具:")
    
    tools = {
        "torch._dynamo.explain()": "快速分析函数结构，识别Graph Break",
        "环境变量TORCH_LOGS": "控制详细日志输出",
        "不同backend测试": "eager -> aot_eager -> inductor 逐阶段调试",
        "动态形状编译": "dynamic=True 处理变化的输入",
        "编译时间分析": "测量首次编译和执行时间"
    }
    
    for tool, description in tools.items():
        print(f"  • {tool}")
        print(f"    {description}")
    
    print("\n📋 推荐调试流程:")
    
    workflow_steps = [
        "1. 使用explain()快速识别问题函数的结构",
        "2. 设置TORCH_LOGS环境变量查看详细日志", 
        "3. 从eager backend开始，逐步测试编译阶段",
        "4. 分析Graph Break原因，优化函数实现",
        "5. 检查编译后的性能提升和数值正确性",
        "6. 根据具体错误类型应用相应解决方案"
    ]
    
    for step in workflow_steps:
        print(f"  {step}")
    
    print("\n⚡ 性能优化要点:")
    
    optimization_tips = [
        "• 减少不必要的Graph Break",
        "• 使用PyTorch原生操作替代Python内置函数", 
        "• 避免数据依赖的控制流语句",
        "• 使用torch.where等条件运算替代if/else",
        "• 对复杂函数进行编译，简单函数可能不值得编译",
        "• 使用动态形状模式处理变化的输入",
        "• 预热编译以消除首次执行的开销",
        "• 定期检查数值精度和内存使用"
    ]
    
    for tip in optimization_tips:
        print(f"  {tip}")
    
    print("\n🎯 常用环境变量:")
    
    env_settings = {
        "基础调试": "export TORCH_LOGS='graph_breaks,recompiles'",
        "详细调试": "export TORCH_LOGS='graph_breaks,recompiles,guards'",
        "性能分析": "export TORCH_LOGS='schedule,fusion'", 
        "代码查看": "export TORCH_LOGS='output_code'",
        "启用调试文件": "export TORCH_COMPILE_DEBUG=1",
        "详细输出": "export TORCHDYNAMO_VERBOSE=1"
    }
    
    for purpose, setting in env_settings.items():
        print(f"  {purpose}: {setting}")
    
    print(f"\n✅ torch.compile调试指南完成！")
    print(f"💡 核心思路：从简单的explain()开始，逐步深入各阶段的具体问题")
    print(f"🚀 记住：优化代码结构比调试编译器更有效")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        traceback.print_exc()