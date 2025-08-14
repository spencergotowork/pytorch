#!/usr/bin/env python3
"""
torch.compile 调试实战演示

运行前请先激活环境: source /opt/ml/torch_env/bin/activate
然后运行: python torch_compile_debug_demo.py
"""

import os
import tempfile
import torch
import torch._logging
import torch._dynamo

def setup_debug_environment():
    """设置调试环境"""
    debug_dir = tempfile.mkdtemp(prefix="torch_debug_")
    
    # 设置环境变量
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles,guards"
    os.environ["TORCH_COMPILE_DEBUG"] = "1" 
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = debug_dir
    
    # 程序化设置日志
    torch._logging.set_logs(
        graph_breaks=True,
        recompiles=True,
        guards=True,
        output_code=True
    )
    
    print(f"调试环境已设置，输出目录: {debug_dir}")
    return debug_dir

def demo_explain_api():
    """演示explain API"""
    print("\n" + "="*60)
    print("1. torch._dynamo.explain() API 演示")
    print("="*60)
    
    def problematic_function(x):
        y = torch.relu(x + 1)
        # 这会导致graph break
        if y.sum() > 0:
            return y * 2
        else:
            return y / 2
    
    # 使用explain分析
    print("分析带有graph break的函数...")
    explanation = torch._dynamo.explain(problematic_function)(torch.randn(5))
    
    print(f"图数量: {explanation.graph_count}")
    print(f"中断数量: {explanation.graph_break_count}")
    print(f"操作数量: {explanation.op_count}")
    print(f"每图操作数: {explanation.ops_per_graph}")
    
    if explanation.break_reasons:
        print("中断原因:")
        for i, reason in enumerate(explanation.break_reasons):
            print(f"  {i+1}. {reason}")
    
    return explanation

def demo_different_backends():
    """演示不同backend的效果"""
    print("\n" + "="*60)
    print("2. 不同编译backend对比演示")
    print("="*60)
    
    def simple_matmul(x, y):
        return torch.matmul(x, y)
    
    x = torch.randn(100, 100, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(100, 100, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    backends = [
        ("eager", "仅TorchDynamo图捕获"),
        ("aot_eager", "Dynamo + AOTAutograd"),
        ("inductor", "完整编译栈")
    ]
    
    results = {}
    for backend_name, description in backends:
        print(f"\n测试 {backend_name} - {description}")
        
        try:
            compiled_func = torch.compile(simple_matmul, backend=backend_name)
            result = compiled_func(x, y)
            results[backend_name] = result
            print(f"  成功! 输出形状: {result.shape}")
        except Exception as e:
            print(f"  失败: {e}")
    
    # 验证结果一致性
    if len(results) > 1:
        base_result = next(iter(results.values()))
        print(f"\n结果一致性检查:")
        for name, result in results.items():
            is_close = torch.allclose(result, base_result, rtol=1e-4)
            print(f"  {name}: {'✓' if is_close else '✗'}")
    
    return results

def demo_graph_breaks():
    """演示graph break的查看"""
    print("\n" + "="*60) 
    print("3. Graph Break 分析演示")
    print("="*60)
    
    @torch.compile
    def function_with_breaks(x):
        # 正常操作
        y = torch.relu(x)
        
        # 打印 - 会导致graph break
        print(f"中间结果: {y.sum()}")  
        
        # 数据依赖分支 - 会导致graph break
        if y.sum() > 0:
            z = y * 2
        else:
            z = y / 2
            
        # Python内置函数 - 会导致graph break 
        length = len(x.shape)
        
        return z.sum() + length
    
    print("运行包含多个graph break的函数...")
    print("注意观察输出的graph break信息:")
    
    x = torch.randn(10, 10)
    result = function_with_breaks(x)
    print(f"最终结果: {result}")
    
    return result

def demo_dynamic_shapes():
    """演示动态形状处理"""
    print("\n" + "="*60)
    print("4. 动态形状处理演示") 
    print("="*60)
    
    # 启用动态形状日志
    torch._logging.set_logs(dynamic=True, recompiles=True)
    
    @torch.compile(dynamic=True)
    def dynamic_function(x):
        return x.sum()
    
    shapes = [(5,), (10,), (20,), (5,)]  # 最后一个重复测试缓存
    
    print("测试不同输入形状:")
    for i, shape in enumerate(shapes):
        x = torch.randn(shape)
        result = dynamic_function(x)
        print(f"  Shape {shape}: result = {result:.4f}")
        
        if i == 0:
            print("    (首次编译)")
        elif shape == (5,):
            print("    (重复形状，应使用缓存)")
    
    return shapes

def demo_code_inspection():
    """演示生成代码的查看"""
    print("\n" + "="*60)
    print("5. 生成代码检查演示")
    print("="*60)
    
    # 启用代码输出日志
    torch._logging.set_logs(output_code=True)
    
    @torch.compile(backend="inductor")
    def fusion_example(x, y):
        # 这些操作应该被融合
        z = x + y
        z = torch.relu(z) 
        z = z * 2
        return z.sum()
    
    print("编译包含融合机会的函数...")
    print("查看输出中的生成代码信息:")
    
    x = torch.randn(100, 100, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(100, 100, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    result = fusion_example(x, y)
    print(f"融合后的结果: {result}")
    
    return result

def check_debug_files(debug_dir):
    """检查生成的调试文件"""
    print("\n" + "="*60)
    print("6. 调试文件检查")
    print("="*60)
    
    import glob
    
    # 查找调试文件
    patterns = {
        "Python代码": "**/*.py",
        "日志文件": "**/*.log", 
        "文本文件": "**/*.txt"
    }
    
    for file_type, pattern in patterns.items():
        files = glob.glob(os.path.join(debug_dir, pattern), recursive=True)
        print(f"{file_type}: {len(files)}个文件")
        
        for file in files[:3]:  # 显示前3个
            rel_path = os.path.relpath(file, debug_dir)
            size = os.path.getsize(file)
            print(f"  - {rel_path} ({size} bytes)")
            
        if len(files) > 3:
            print(f"  - ... 还有 {len(files)-3} 个文件")
    
    print(f"\n完整调试文件位于: {debug_dir}")
    return debug_dir

def main():
    """主演示函数"""
    print("torch.compile 调试实战演示")
    print("PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())
    
    # 设置调试环境
    debug_dir = setup_debug_environment()
    
    try:
        # 1. explain API演示
        demo_explain_api()
        
        # 2. 不同backend对比
        demo_different_backends()
        
        # 3. graph break分析
        demo_graph_breaks()
        
        # 4. 动态形状处理
        demo_dynamic_shapes()
        
        # 5. 代码生成检查
        demo_code_inspection()
        
        # 6. 调试文件检查
        check_debug_files(debug_dir)
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("演示完成!")
    print("="*80)
    
    print(f"\n📁 调试文件保存在: {debug_dir}")
    print("📋 主要收获:")
    print("  1. torch._dynamo.explain() 可以分析函数结构")
    print("  2. 不同backend适用于不同调试需求")
    print("  3. TORCH_LOGS环境变量控制日志输出")
    print("  4. graph break会影响编译效果")
    print("  5. 动态形状需要特殊处理")
    
    print("\n🛠️  推荐后续步骤:")
    print("  1. 查看生成的调试文件")
    print("  2. 尝试减少自己代码中的graph break")
    print("  3. 根据性能提示优化模型")

if __name__ == "__main__":
    main()