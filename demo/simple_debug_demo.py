#!/usr/bin/env python3
"""
torch.compile 调试文件生成演示（简化版）
"""

import os
import tempfile
import torch
import torch._logging

def main():
    print("torch.compile 调试演示（简化版）")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 创建调试目录
    debug_dir = tempfile.mkdtemp(prefix="torch_debug_simple_")
    print(f"调试目录: {debug_dir}")
    
    # 设置基本调试环境
    os.environ["TORCH_LOGS"] = "graph_breaks,recompiles,output_code"
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    if "TORCH_COMPILE_DEBUG_DIR" not in os.environ:
        os.environ["TORCH_COMPILE_DEBUG_DIR"] = debug_dir
    
    print("\n=== 1. 基本explain演示 ===")
    
    def simple_function_with_break(x):
        y = torch.relu(x + 1)
        # 这会导致graph break
        if y.sum() > 0:
            return y * 2
        return y / 2
    
    # 使用explain
    explanation = torch._dynamo.explain(simple_function_with_break)(torch.randn(10))
    print(f"图数量: {explanation.graph_count}")
    print(f"Graph break数量: {explanation.graph_break_count}")
    print(f"中断原因: {explanation.break_reasons}")
    
    print("\n=== 2. 编译和运行 ===")
    
    @torch.compile
    def matrix_ops(x, y):
        z = torch.matmul(x, y)
        z = torch.relu(z)
        return z.sum()
    
    # 运行编译函数
    x = torch.randn(50, 50)
    y = torch.randn(50, 50)
    result = matrix_ops(x, y)
    print(f"矩阵运算结果: {result}")
    
    print("\n=== 3. 查看调试文件 ===")
    
    # 检查生成的文件
    import glob
    all_files = glob.glob(os.path.join(debug_dir, "**/*"), recursive=True)
    files = [f for f in all_files if os.path.isfile(f)]
    
    print(f"生成的调试文件数量: {len(files)}")
    for i, file in enumerate(files[:10]):  # 显示前10个
        rel_path = os.path.relpath(file, debug_dir)
        size = os.path.getsize(file)
        print(f"  {i+1}. {rel_path} ({size} bytes)")
    
    if len(files) > 10:
        print(f"  ... 还有 {len(files)-10} 个文件")
    
    # 显示部分文件内容
    print("\n=== 4. 文件内容预览 ===")
    for file in files[:3]:
        if file.endswith(('.py', '.txt', '.log')) and os.path.getsize(file) < 2000:
            print(f"\n--- {os.path.basename(file)} ---")
            try:
                with open(file, 'r') as f:
                    content = f.read().strip()
                    print(content[:500] + ("..." if len(content) > 500 else ""))
            except Exception as e:
                print(f"读取失败: {e}")
    
    print(f"\n✅ 完成！调试文件保存在: {debug_dir}")
    
    # 环境变量总结
    print(f"\n=== 关键环境变量总结 ===")
    key_vars = ["TORCH_LOGS", "TORCH_COMPILE_DEBUG", "TORCH_COMPILE_DEBUG_DIR"]
    for var in key_vars:
        value = os.environ.get(var, "未设置")
        print(f"{var} = {value}")
    
    return debug_dir

if __name__ == "__main__":
    main()