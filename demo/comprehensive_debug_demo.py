#!/usr/bin/env python3
"""
torch.compile 详细调试文件生成演示

确保生成各种调试文件的示例
"""

import os
import tempfile
import torch
import torch._dynamo
import torch._logging

def create_comprehensive_debug_example():
    """创建一个能生成丰富调试信息的示例"""
    
    # 创建调试目录
    debug_dir = tempfile.mkdtemp(prefix="torch_debug_comprehensive_")
    print(f"调试目录: {debug_dir}")
    
    # 设置全面的调试环境
    debug_env = {
        "TORCH_LOGS": "+all,graph_breaks,recompiles,guards,dynamic,output_code,kernel_code,schedule,fusion",
        "TORCH_COMPILE_DEBUG": "1",
        "TORCH_COMPILE_DEBUG_DIR": debug_dir,
        "TORCHDYNAMO_VERBOSE": "1",
        "TORCHINDUCTOR_DEBUG_FUSION": "1",
        "TORCH_TRACE": debug_dir,
    }
    
    # 应用环境变量
    for key, value in debug_env.items():
        os.environ[key] = str(value)
        print(f"设置 {key}={value}")
    
    # 配置torch._dynamo
    torch._dynamo.config.verbose = True
    torch._dynamo.config.log_file_name = os.path.join(debug_dir, "dynamo_debug.log")
    torch._dynamo.config.output_directory = debug_dir
    
    # 配置torch._inductor
    torch._inductor.config.debug = True
    torch._inductor.config.verbose_progress = True
    torch._inductor.config.comment_origin = True
    
    return debug_dir

def run_complex_model():
    """运行一个复杂的模型来生成更多调试信息"""
    
    class ComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(128, 256)
            self.linear2 = torch.nn.Linear(256, 128) 
            self.linear3 = torch.nn.Linear(128, 64)
            self.dropout = torch.nn.Dropout(0.1)
            self.batch_norm = torch.nn.BatchNorm1d(256)
            
        def forward(self, x):
            # 第一层
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.batch_norm(x)
            x = self.dropout(x)
            
            # 第二层
            x = self.linear2(x)
            x = torch.tanh(x)
            
            # 条件分支 - 会导致graph break
            if x.sum() > 0:
                x = self.linear3(x)
                x = torch.softmax(x, dim=-1)
            else:
                x = torch.zeros_like(x[:, :64])  # 形状匹配
            
            # 打印 - 也会导致graph break
            print(f"中间输出形状: {x.shape}")
            
            return x.mean()
    
    model = ComplexModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 编译模型
    print("编译复杂模型...")
    compiled_model = torch.compile(model, backend="inductor", mode="default")
    
    # 测试不同输入大小（会导致重编译）
    batch_sizes = [16, 32, 64, 16]  # 最后一个重复测试缓存
    
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n--- 测试 batch_size={batch_size} ---")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(batch_size, 128, device=device)
        
        with torch.no_grad():
            result = compiled_model(x)
            print(f"输出: {result}")
    
    return model

def inspect_generated_files(debug_dir):
    """详细检查生成的调试文件"""
    
    print(f"\n{'='*80}")
    print("调试文件详细检查")
    print(f"{'='*80}")
    
    import glob
    import os
    
    # 递归查找所有文件
    all_files = []
    for root, dirs, files in os.walk(debug_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    
    print(f"总共生成了 {len(all_files)} 个调试文件")
    
    # 按文件类型分类
    file_types = {}
    for file in all_files:
        ext = os.path.splitext(file)[1] or 'no_ext'
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(file)
    
    # 显示每种类型的文件
    for ext, files in file_types.items():
        print(f"\n{ext} 文件 ({len(files)}个):")
        for file in files[:5]:  # 最多显示5个
            rel_path = os.path.relpath(file, debug_dir)
            size = os.path.getsize(file)
            print(f"  - {rel_path} ({size} bytes)")
            
            # 显示小文件的内容
            if size < 1000 and ext in ['.py', '.txt', '.log']:
                try:
                    with open(file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            print(f"    内容: {content[:200]}...")
                except:
                    pass
                    
        if len(files) > 5:
            print(f"  ... 还有 {len(files)-5} 个文件")
    
    return all_files

def demonstrate_explain_with_files():
    """结合explain API和文件输出演示"""
    
    print(f"\n{'='*80}")
    print("explain API + 文件输出演示")
    print(f"{'='*80}")
    
    def multi_break_function(x):
        """包含多个graph break的函数"""
        # 正常张量操作
        y = torch.relu(x + 1)
        
        # Python内置函数调用 - graph break
        shape_len = len(x.shape)
        
        # 数据依赖分支 - graph break  
        if y.sum() > 0:
            z = y * 2
        else:
            z = y / 2
            
        # 打印语句 - graph break
        print(f"中间结果: {z.mean()}")
        
        # 循环 - 可能的graph break
        for i in range(3):
            z = z + i
            
        return z.sum() + shape_len
    
    # 使用explain分析
    print("使用explain分析复杂函数...")
    explanation = torch._dynamo.explain(multi_break_function)(torch.randn(10, 20))
    
    print(f"\nExplain结果:")
    print(f"  图数量: {explanation.graph_count}")
    print(f"  graph break数量: {explanation.graph_break_count}")
    print(f"  总操作数: {explanation.op_count}")
    print(f"  每图操作数: {[len(ops) for ops in explanation.ops_per_graph]}")
    
    print(f"\nGraph Break原因:")
    for i, reason in enumerate(explanation.break_reasons):
        print(f"  {i+1}. {reason}")
    
    # 现在编译并运行函数
    print(f"\n编译并运行函数...")
    compiled_func = torch.compile(multi_break_function)
    result = compiled_func(torch.randn(10, 20))
    print(f"最终结果: {result}")
    
    return explanation

def main():
    print("torch.compile 详细调试文件生成演示")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 1. 设置全面的调试环境
    debug_dir = create_comprehensive_debug_example()
    
    try:
        # 2. explain API演示
        explanation = demonstrate_explain_with_files()
        
        # 3. 运行复杂模型
        model = run_complex_model()
        
        # 4. 检查生成的文件
        files = inspect_generated_files(debug_dir)
        
        print(f"\n{'='*80}")
        print("总结")
        print(f"{'='*80}")
        print(f"✅ 生成了 {len(files)} 个调试文件")
        print(f"📁 调试目录: {debug_dir}")
        print(f"📊 分析了 {explanation.graph_count} 个图，{explanation.graph_break_count} 个中断")
        
        print(f"\n推荐查看的文件:")
        print(f"  - 图代码文件 (*.py)")
        print(f"  - 日志文件 (*.log)")
        print(f"  - 内核代码 (*.cpp, *.cu)")
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    return debug_dir

if __name__ == "__main__":
    main()