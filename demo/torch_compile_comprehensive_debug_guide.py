#!/usr/bin/env python3
"""
torch.compile 全阶段深度调试指南

详细介绍TorchDynamo、AOTAutograd、Inductor各阶段的调试方法和实际案例
"""

import os
import sys
import torch
import torch._dynamo
import torch._inductor
import torch._logging
import tempfile
import traceback
from typing import Dict, List, Any
import warnings
import json

print(f"PyTorch版本: {torch.__version__}")
print(f"Python版本: {sys.version}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# ============================================================================
# 第一部分：TorchDynamo阶段调试
# ============================================================================

class TorchDynamoDebugger:
    """TorchDynamo阶段专用调试器"""
    
    def __init__(self, debug_dir: str = None):
        self.debug_dir = debug_dir or tempfile.mkdtemp(prefix="dynamo_debug_")
        print(f"TorchDynamo调试目录: {self.debug_dir}")
        
    def setup_dynamo_debugging(self):
        """配置TorchDynamo详细调试"""
        
        # 设置环境变量
        debug_env = {
            "TORCH_LOGS": "+dynamo,graph_breaks,recompiles,guards,dynamic",
            "TORCH_COMPILE_DEBUG": "1",
            "TORCH_COMPILE_DEBUG_DIR": self.debug_dir,
            "TORCHDYNAMO_VERBOSE": "1",
            # 关键：自动生成复现脚本
            "TORCHDYNAMO_REPRO_AFTER": "dynamo",
            "TORCHDYNAMO_REPRO_LEVEL": "4"
        }
        
        for key, value in debug_env.items():
            os.environ[key] = str(value)
            
        # 程序化配置
        import torch._dynamo.config as dynamo_config
        dynamo_config.verbose = True
        dynamo_config.log_file_name = os.path.join(self.debug_dir, "dynamo_detailed.log")
        dynamo_config.output_directory = self.debug_dir
        
        # 启用所有调试选项
        dynamo_config.debug_dir_root = self.debug_dir
        dynamo_config.capture_scalar_outputs = True
        dynamo_config.capture_dynamic_output_shape_ops = True
        
        print("✅ TorchDynamo调试环境已配置")
        
    def analyze_graph_breaks(self, func, *args, **kwargs):
        """深度分析Graph Break原因"""
        
        print("\n🔍 Graph Break分析")
        print("-" * 50)
        
        # 1. 使用explain API分析
        try:
            explanation = torch._dynamo.explain(func)(*args, **kwargs)
            
            print(f"📊 统计信息:")
            print(f"  - 图数量: {explanation.graph_count}")
            print(f"  - Graph Break数量: {explanation.graph_break_count}")  
            print(f"  - 总操作数: {explanation.op_count}")
            print(f"  - 每图操作数: {[len(ops) for ops in explanation.ops_per_graph]}")
            
            print(f"\n💥 Graph Break详细原因:")
            for i, reason in enumerate(explanation.break_reasons):
                print(f"  {i+1}. {reason}")
                
            # 分析break原因的类型
            break_types = {}
            for reason in explanation.break_reasons:
                reason_type = self._classify_break_reason(reason)
                break_types[reason_type] = break_types.get(reason_type, 0) + 1
                
            print(f"\n📈 Break类型统计:")
            for break_type, count in break_types.items():
                print(f"  - {break_type}: {count}次")
                
            return explanation
            
        except Exception as e:
            print(f"❌ explain分析失败: {e}")
            return None
    
    def _classify_break_reason(self, reason: str) -> str:
        """分类Graph Break原因"""
        reason_lower = reason.lower()
        
        if "builtin" in reason_lower:
            return "Python内置函数"
        elif "control flow" in reason_lower or "if" in reason_lower:
            return "控制流"
        elif "data-dependent" in reason_lower:
            return "数据依赖"
        elif "print" in reason_lower or "output" in reason_lower:
            return "副作用操作"
        elif "unsupported" in reason_lower:
            return "不支持的操作"
        elif "dynamic" in reason_lower:
            return "动态行为"
        else:
            return "其他"
    
    def debug_guard_failures(self, func, test_inputs: List):
        """调试Guard失败问题"""
        
        print("\n🛡️ Guard失败调试")
        print("-" * 50)
        
        # 编译函数
        compiled_func = torch.compile(func, backend="eager")
        
        # 测试不同输入，观察重编译
        for i, inputs in enumerate(test_inputs):
            print(f"\n测试输入 {i+1}: {[inp.shape if hasattr(inp, 'shape') else type(inp) for inp in inputs]}")
            
            # 清理dynamo状态
            torch._dynamo.reset()
            
            try:
                result = compiled_func(*inputs)
                print(f"  ✅ 执行成功: {result.shape if hasattr(result, 'shape') else type(result)}")
            except Exception as e:
                print(f"  ❌ 执行失败: {e}")
                
        # 检查重编译统计
        stats = torch._dynamo.utils.compile_times()
        print(f"\n📊 编译统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def analyze_unsupported_features(self):
        """分析不支持的Python特性"""
        
        print("\n🚫 不支持特性测试")
        print("-" * 50)
        
        unsupported_cases = {
            "exec/eval": lambda x: exec("result = x * 2"),
            "globals()访问": lambda x: globals().get('torch', torch).relu(x),
            "动态属性访问": lambda x: getattr(x, 'sum')(),
            "复杂解包": lambda x: (lambda *args: sum(args))(*x.unbind()),
            "异常处理": lambda x: x if x.sum() > 0 else (_ for _ in ()).throw(ValueError())
        }
        
        test_input = torch.randn(10)
        
        for name, func in unsupported_cases.items():
            try:
                explanation = torch._dynamo.explain(func)(test_input)
                print(f"  {name}: {explanation.graph_break_count} breaks")
                for reason in explanation.break_reasons:
                    print(f"    原因: {reason}")
            except Exception as e:
                print(f"  {name}: 分析失败 - {e}")

    def demonstrate_dynamic_behavior_issues(self):
        """演示动态行为导致的编译问题"""
        
        print("\n🎭 动态行为问题演示")
        print("-" * 50)
        
        def dynamic_function(x, threshold=0.5):
            """包含动态行为的函数"""
            
            # 动态形状操作
            if x.dim() > 2:
                x = x.flatten(1)
            
            # 数据依赖的控制流
            mask = x.abs() > threshold
            if mask.any():
                # 动态选择操作
                result = torch.where(mask, x * 2, x / 2)
            else:
                result = x.clone()
                
            # 动态循环次数
            for _ in range(int(x.sum().item() % 5)):
                result = result + 0.1
                
            return result
        
        # 测试不同输入
        test_cases = [
            torch.randn(10),
            torch.randn(5, 4), 
            torch.randn(2, 3, 4),
            torch.ones(10) * 0.1,  # 低于阈值
            torch.ones(10) * 2.0   # 高于阈值
        ]
        
        for i, x in enumerate(test_cases):
            print(f"\n测试案例 {i+1}: shape={x.shape}, range=({x.min():.2f}, {x.max():.2f})")
            
            try:
                explanation = torch._dynamo.explain(dynamic_function)(x)
                print(f"  Graph数量: {explanation.graph_count}")
                print(f"  Break数量: {explanation.graph_break_count}")
                
                # 实际编译运行
                compiled = torch.compile(dynamic_function)
                result = compiled(x)
                print(f"  结果shape: {result.shape}")
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")

# ============================================================================
# 第二部分：AOTAutograd阶段调试  
# ============================================================================

class AOTAutogradDebugger:
    """AOTAutograd阶段专用调试器"""
    
    def __init__(self, debug_dir: str = None):
        self.debug_dir = debug_dir or tempfile.mkdtemp(prefix="aot_debug_")
        print(f"AOTAutograd调试目录: {self.debug_dir}")
        
    def setup_aot_debugging(self):
        """配置AOTAutograd详细调试"""
        
        debug_env = {
            "TORCH_LOGS": "+aot,+functorch,graph_breaks,recompiles",
            "TORCH_COMPILE_DEBUG": "1",
            "TORCH_COMPILE_DEBUG_DIR": self.debug_dir,
            "AOT_AUTOGRAD_CACHE": "0",  # 禁用缓存便于调试
        }
        
        for key, value in debug_env.items():
            os.environ[key] = str(value)
            
        # AOT配置
        import torch._functorch.config as functorch_config
        functorch_config.debug_partitioner = True
        functorch_config.use_fake_tensor = True
        
        print("✅ AOTAutograd调试环境已配置")
        
    def debug_backward_graph_generation(self):
        """调试反向图生成"""
        
        print("\n🔄 反向图生成调试")
        print("-" * 50)
        
        class GradientModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 5)
                self.dropout = torch.nn.Dropout(0.1)
                
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                return x.sum()
        
        model = GradientModel()
        x = torch.randn(32, 10, requires_grad=True)
        
        print("测试不同编译backend的梯度行为:")
        
        backends = ["eager", "aot_eager", "inductor"]
        
        for backend in backends:
            print(f"\n--- Backend: {backend} ---")
            
            try:
                # 重置模型状态
                model.zero_grad()
                if hasattr(x.grad, 'zero_'):
                    x.grad.zero_()
                
                # 编译模型
                compiled_model = torch.compile(model, backend=backend)
                
                # 前向传播
                loss = compiled_model(x)
                print(f"前向loss: {loss:.4f}")
                
                # 反向传播
                loss.backward()
                
                # 检查梯度
                param_grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                input_grad = x.grad.norm().item() if x.grad is not None else 0
                
                print(f"参数梯度范数: {param_grads}")
                print(f"输入梯度范数: {input_grad:.4f}")
                
            except Exception as e:
                print(f"❌ Backend {backend} 失败: {e}")
                traceback.print_exc()
    
    def debug_memory_optimization_issues(self):
        """调试内存优化问题"""
        
        print("\n💾 内存优化问题调试")
        print("-" * 50)
        
        def memory_intensive_function(x):
            """内存密集型函数"""
            intermediates = []
            
            for i in range(5):
                x = torch.matmul(x, x.transpose(-2, -1))
                x = torch.relu(x)
                intermediates.append(x.clone())  # 强制保留中间结果
                
            # 使用所有中间结果（防止优化掉）
            result = sum(intermediates)
            return result.sum()
        
        x = torch.randn(64, 64, requires_grad=True)
        
        print("比较不同编译模式的内存使用:")
        
        import psutil
        process = psutil.Process()
        
        for mode, config in [
            ("原始", {"compile": False}),
            ("AOT Eager", {"compile": True, "backend": "aot_eager"}),
            ("Inductor", {"compile": True, "backend": "inductor"})
        ]:
            print(f"\n--- {mode} ---")
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 记录初始内存
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                if config["compile"]:
                    func = torch.compile(memory_intensive_function, backend=config["backend"])
                else:
                    func = memory_intensive_function
                
                # 执行并计算梯度
                loss = func(x)
                loss.backward()
                
                # 记录峰值内存
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = peak_memory - initial_memory
                
                print(f"内存使用: {memory_usage:.2f} MB")
                print(f"Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"❌ 执行失败: {e}")
            
            # 清理
            if hasattr(x, 'grad') and x.grad is not None:
                x.grad.zero_()
    
    def debug_functional_transformation_failures(self):
        """调试函数式变换失败"""
        
        print("\n🔧 函数式变换失败调试")
        print("-" * 50)
        
        # 测试有问题的变换模式
        problematic_patterns = {
            "In-place操作": lambda x: x.add_(1),
            "视图操作复杂性": lambda x: x.view(-1)[::2].reshape(x.shape[0], -1),
            "条件赋值": lambda x: x.masked_fill_(x > 0, -1),
            "非连续张量": lambda x: x.transpose(0, 1).contiguous().t()
        }
        
        x = torch.randn(8, 8, requires_grad=True)
        
        for name, func in problematic_patterns.items():
            print(f"\n测试: {name}")
            
            try:
                # AOT Autograd编译
                compiled = torch.compile(func, backend="aot_eager")
                result = compiled(x.clone())
                print(f"  ✅ 成功: {result.shape}")
                
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                
                # 尝试分析原因
                try:
                    explanation = torch._dynamo.explain(func)(x.clone())
                    print(f"  Break原因: {explanation.break_reasons}")
                except:
                    pass

# ============================================================================
# 第三部分：Inductor阶段调试
# ============================================================================

class InductorDebugger:
    """Inductor阶段专用调试器"""
    
    def __init__(self, debug_dir: str = None):
        self.debug_dir = debug_dir or tempfile.mkdtemp(prefix="inductor_debug_")
        print(f"Inductor调试目录: {self.debug_dir}")
        
    def setup_inductor_debugging(self):
        """配置Inductor详细调试"""
        
        debug_env = {
            "TORCH_LOGS": "+inductor,output_code,kernel_code,schedule,fusion,perf_hints",
            "TORCH_COMPILE_DEBUG": "1", 
            "TORCH_COMPILE_DEBUG_DIR": self.debug_dir,
            "TORCHINDUCTOR_DEBUG_FUSION": "1",
            "TORCHINDUCTOR_DEBUG_SCHEDULER": "1"
        }
        
        for key, value in debug_env.items():
            os.environ[key] = str(value)
            
        # Inductor配置
        import torch._inductor.config as inductor_config
        inductor_config.debug = True
        inductor_config.verbose_progress = True
        inductor_config.comment_origin = True
        inductor_config.generate_intermediate_hooks = True
        
        print("✅ Inductor调试环境已配置")
        
    def debug_generated_code(self):
        """调试生成的代码"""
        
        print("\n💻 生成代码调试")
        print("-" * 50)
        
        def fusion_example(x, y):
            """展示内核融合的例子"""
            z1 = torch.relu(x + y)
            z2 = torch.tanh(z1 * 2)
            z3 = torch.sigmoid(z2 - 1)
            return z3.sum()
        
        if torch.cuda.is_available():
            x = torch.randn(1024, 1024, device='cuda')
            y = torch.randn(1024, 1024, device='cuda')
            device_type = "CUDA"
        else:
            x = torch.randn(1024, 1024)
            y = torch.randn(1024, 1024)
            device_type = "CPU"
        
        print(f"在{device_type}上测试内核生成:")
        
        # 编译并运行
        compiled_func = torch.compile(fusion_example, backend="inductor")
        result = compiled_func(x, y)
        
        print(f"计算结果: {result:.4f}")
        
        # 检查生成的代码文件
        self._inspect_generated_code_files()
        
    def _inspect_generated_code_files(self):
        """检查生成的代码文件"""
        
        import glob
        
        # 查找不同类型的生成文件
        file_patterns = {
            "Python代码": "**/*output*.py",
            "C++ CPU代码": "**/*.cpp", 
            "CUDA代码": "**/*.cu",
            "调试日志": "**/*.log"
        }
        
        for file_type, pattern in file_patterns.items():
            files = glob.glob(os.path.join(self.debug_dir, pattern), recursive=True)
            
            if files:
                print(f"\n{file_type} ({len(files)}个文件):")
                for file_path in files[:3]:  # 显示前3个
                    rel_path = os.path.relpath(file_path, self.debug_dir)
                    size = os.path.getsize(file_path)
                    print(f"  - {rel_path} ({size} bytes)")
                    
                    # 显示部分内容
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(500)  # 前500个字符
                            if content.strip():
                                print(f"    预览: {content[:200]}...")
                    except Exception as e:
                        print(f"    无法读取: {e}")
                        
                if len(files) > 3:
                    print(f"  ... 还有 {len(files) - 3} 个文件")
    
    def debug_kernel_fusion_issues(self):
        """调试内核融合问题"""
        
        print("\n🔀 内核融合问题调试")
        print("-" * 50)
        
        def test_fusion_patterns():
            """测试不同的融合模式"""
            
            patterns = {
                "点对点融合": lambda x: torch.relu(torch.sigmoid(x)),
                "广播融合": lambda x: x + torch.randn(1, x.shape[1]),
                "归约融合": lambda x: (x * 2).sum(dim=1),
                "复杂融合链": lambda x: torch.softmax(torch.relu(x + 1) * torch.tanh(x - 1), dim=-1),
                "阻止融合的操作": lambda x: torch.relu(x).contiguous().sigmoid()
            }
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.randn(512, 512, device=device)
            
            for name, pattern in patterns.items():
                print(f"\n测试: {name}")
                
                try:
                    # 编译并运行
                    compiled = torch.compile(pattern, backend="inductor") 
                    result = compiled(x)
                    print(f"  ✅ 成功: shape={result.shape}")
                    
                    # 性能测试
                    import time
                    
                    # 预热
                    for _ in range(5):
                        _ = compiled(x)
                    
                    # 测试性能
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    
                    for _ in range(100):
                        _ = compiled(x)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 100 * 1000  # ms
                    print(f"  平均执行时间: {avg_time:.3f} ms")
                    
                except Exception as e:
                    print(f"  ❌ 失败: {e}")
        
        test_fusion_patterns()
        
    def debug_performance_issues(self):
        """调试性能问题"""
        
        print("\n⚡ 性能问题调试") 
        print("-" * 50)
        
        def create_performance_test_cases():
            """创建性能测试用例"""
            
            test_cases = {
                "矩阵乘法": {
                    "func": lambda x, y: torch.matmul(x, y),
                    "inputs": (torch.randn(512, 512), torch.randn(512, 512))
                },
                
                "复合操作": {
                    "func": lambda x: torch.softmax(torch.relu(x @ x.t()) + 1, dim=-1),
                    "inputs": (torch.randn(256, 256),)
                },
                
                "归约操作": {
                    "func": lambda x: x.sum(dim=1, keepdim=True).expand_as(x) * x,
                    "inputs": (torch.randn(1000, 1000),)
                }
            }
            
            return test_cases
        
        test_cases = create_performance_test_cases()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 将输入移到对应设备
        for name, case in test_cases.items():
            case["inputs"] = tuple(inp.to(device) for inp in case["inputs"])
        
        print("性能对比测试:")
        
        import time
        
        for name, case in test_cases.items():
            print(f"\n--- {name} ---")
            func = case["func"]
            inputs = case["inputs"]
            
            # 测试原始函数性能
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            for _ in range(50):
                _ = func(*inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            original_time = time.time() - start_time
            
            # 测试编译后性能
            compiled_func = torch.compile(func, backend="inductor")
            
            # 预热编译
            for _ in range(3):
                _ = compiled_func(*inputs)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            for _ in range(50):
                _ = compiled_func(*inputs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            compiled_time = time.time() - start_time
            
            speedup = original_time / compiled_time if compiled_time > 0 else float('inf')
            
            print(f"原始时间: {original_time*1000:.2f} ms")
            print(f"编译时间: {compiled_time*1000:.2f} ms")  
            print(f"加速比: {speedup:.2f}x")
            
            if speedup < 1.0:
                print("⚠️  编译后性能下降，可能原因:")
                print("   - 编译开销大于收益")
                print("   - 融合策略不当")
                print("   - 内存访问模式变差")
    
    def debug_hardware_compatibility(self):
        """调试硬件兼容性问题"""
        
        print("\n🖥️ 硬件兼容性调试")
        print("-" * 50)
        
        # 检查硬件能力
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name()}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"计算能力: {torch.cuda.get_device_capability()}")
            
            # 测试CUDA特定功能
            cuda_features = {
                "Tensor Cores": self._test_tensor_cores,
                "混合精度": self._test_mixed_precision,
                "大内存": self._test_large_memory
            }
            
            for name, test_func in cuda_features.items():
                try:
                    result = test_func()
                    print(f"{name}: {'✅ 支持' if result else '❌ 不支持'}")
                except Exception as e:
                    print(f"{name}: ❌ 测试失败 - {e}")
        else:
            print("CPU模式")
            
            # 测试CPU特定功能
            cpu_features = {
                "AVX指令集": self._test_avx,
                "多核并行": self._test_multicore,
                "大矩阵运算": self._test_large_cpu_ops
            }
            
            for name, test_func in cpu_features.items():
                try:
                    result = test_func()
                    print(f"{name}: {'✅ 支持' if result else '❌ 不支持'}")
                except Exception as e:
                    print(f"{name}: ❌ 测试失败 - {e}")
    
    def _test_tensor_cores(self):
        """测试Tensor Core支持"""
        try:
            # Tensor Core需要特定的数据类型和大小
            a = torch.randn(128, 128, dtype=torch.half, device='cuda')
            b = torch.randn(128, 128, dtype=torch.half, device='cuda')
            
            @torch.compile
            def matmul_test(x, y):
                return torch.matmul(x, y)
            
            result = matmul_test(a, b)
            return result.shape == (128, 128)
        except:
            return False
    
    def _test_mixed_precision(self):
        """测试混合精度支持"""
        try:
            with torch.autocast(device_type='cuda'):
                x = torch.randn(64, 64, device='cuda')
                
                @torch.compile
                def mixed_precision_test(inp):
                    return torch.matmul(inp, inp.t())
                
                result = mixed_precision_test(x)
                return result.dtype in [torch.half, torch.float]
        except:
            return False
    
    def _test_large_memory(self):
        """测试大内存操作"""
        try:
            # 尝试分配较大内存
            x = torch.randn(2048, 2048, device='cuda')
            
            @torch.compile
            def large_mem_test(inp):
                return (inp @ inp.t()).sum()
            
            result = large_mem_test(x)
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        except:
            return False
    
    def _test_avx(self):
        """测试AVX指令集支持"""
        # 这里简化测试，实际需要更复杂的检测
        try:
            x = torch.randn(1000, 1000)
            
            @torch.compile
            def avx_test(inp):
                return torch.sum(inp * inp)
            
            result = avx_test(x)
            return True
        except:
            return False
    
    def _test_multicore(self):
        """测试多核并行"""
        try:
            original_threads = torch.get_num_threads()
            torch.set_num_threads(4)
            
            x = torch.randn(500, 500)
            
            @torch.compile
            def multicore_test(inp):
                return torch.matmul(inp, inp.t()).sum()
            
            result = multicore_test(x)
            torch.set_num_threads(original_threads)
            return True
        except:
            return False
    
    def _test_large_cpu_ops(self):
        """测试CPU大矩阵运算"""
        try:
            x = torch.randn(1000, 1000)
            
            @torch.compile
            def large_cpu_test(inp):
                return torch.matmul(inp, inp.t()).trace()
            
            result = large_cpu_test(x)
            return True
        except:
            return False

# ============================================================================
# 第四部分：常见错误模式和解决方案
# ============================================================================

class ErrorPatternAnalyzer:
    """常见错误模式分析器"""
    
    def __init__(self):
        self.error_database = self._build_error_database()
        
    def _build_error_database(self):
        """构建错误数据库"""
        
        return {
            # 编译失败
            "编译失败": {
                "模式": [
                    "TorchDynamoException",
                    "BackendCompilerFailed", 
                    "InductorError"
                ],
                "常见原因": [
                    "不支持的Python特性",
                    "动态控制流",
                    "复杂的数据依赖",
                    "第三方库调用"
                ],
                "解决方案": [
                    "使用torch._dynamo.explain()分析",
                    "简化函数逻辑",
                    "添加torch._dynamo.skip()装饰器",
                    "使用fallback机制"
                ]
            },
            
            # 性能回退
            "性能回退": {
                "模式": [
                    "编译后比原始版本慢",
                    "频繁重编译",
                    "内存使用增加"
                ],
                "常见原因": [
                    "过多的graph break",
                    "不当的内核融合",
                    "Guard检查开销",
                    "编译缓存未命中"
                ],
                "解决方案": [
                    "减少graph break",
                    "优化数据流",
                    "使用动态形状模式",
                    "预热编译"
                ]
            },
            
            # 内存异常
            "内存异常": {
                "模式": [
                    "OutOfMemoryError",
                    "内存泄漏",
                    "内存碎片"
                ],
                "常见原因": [
                    "中间张量未及时释放",
                    "循环引用",
                    "批次大小过大",
                    "梯度累积问题"
                ],
                "解决方案": [
                    "启用内存优化",
                    "使用检查点机制",
                    "调整批次大小",
                    "定期清理缓存"
                ]
            },
            
            # 数值精度
            "数值精度": {
                "模式": [
                    "结果不一致",
                    "NaN或Inf出现",
                    "梯度异常"
                ],
                "常见原因": [
                    "混合精度问题",
                    "数值稳定性",
                    "优化改变计算顺序",
                    "并行计算精度损失"
                ],
                "解决方案": [
                    "使用严格模式",
                    "检查数值范围",
                    "添加数值稳定性处理",
                    "对比参考实现"
                ]
            }
        }
    
    def analyze_compilation_failure(self, error_msg: str, func=None):
        """分析编译失败"""
        
        print("\n❌ 编译失败分析")
        print("-" * 50)
        
        # 错误分类
        error_type = self._classify_error(error_msg)
        print(f"错误类型: {error_type}")
        
        if error_type in self.error_database:
            info = self.error_database[error_type]
            
            print("\n可能原因:")
            for reason in info["常见原因"]:
                print(f"  - {reason}")
                
            print("\n建议解决方案:")
            for solution in info["解决方案"]:
                print(f"  - {solution}")
        
        # 如果提供了函数，进行详细分析
        if func is not None:
            try:
                print("\n详细分析:")
                explanation = torch._dynamo.explain(func)(torch.randn(10))
                print(f"Graph Break原因: {explanation.break_reasons}")
            except Exception as e:
                print(f"无法分析函数: {e}")
    
    def _classify_error(self, error_msg: str) -> str:
        """分类错误消息"""
        
        error_msg_lower = error_msg.lower()
        
        for error_type, info in self.error_database.items():
            for pattern in info["模式"]:
                if pattern.lower() in error_msg_lower:
                    return error_type
        
        return "未知错误"
    
    def demonstrate_common_issues(self):
        """演示常见问题和解决方案"""
        
        print("\n🔧 常见问题演示")
        print("=" * 60)
        
        issues = {
            "Graph Break过多": {
                "有问题版本": self._create_problematic_function,
                "优化版本": self._create_optimized_function,
                "说明": "减少不必要的Python调用和控制流"
            },
            
            "性能回退": {
                "有问题版本": self._create_slow_function,
                "优化版本": self._create_fast_function, 
                "说明": "优化内存访问模式和计算逻辑"
            },
            
            "内存问题": {
                "有问题版本": self._create_memory_hungry_function,
                "优化版本": self._create_memory_efficient_function,
                "说明": "避免不必要的中间张量和内存拷贝"
            }
        }
        
        for issue_name, issue_info in issues.items():
            print(f"\n--- {issue_name} ---")
            print(f"说明: {issue_info['说明']}")
            
            # 测试有问题版本
            problematic_func = issue_info["有问题版本"]()
            self._test_function_performance(f"{issue_name} (有问题)", problematic_func)
            
            # 测试优化版本
            optimized_func = issue_info["优化版本"]()
            self._test_function_performance(f"{issue_name} (优化后)", optimized_func)
    
    def _create_problematic_function(self):
        """创建有Graph Break问题的函数"""
        def problematic(x):
            # 多个graph break
            y = torch.relu(x)
            
            # Python内置函数
            shape_len = len(x.shape)
            
            # 数据依赖分支
            if y.sum() > 0:
                z = y * 2
            else:
                z = y / 2
            
            # 打印语句
            print(f"中间结果: {z.mean()}")
            
            return z.sum() + shape_len
        
        return problematic
    
    def _create_optimized_function(self):
        """创建优化后的函数"""
        def optimized(x):
            # 尽量使用PyTorch原生操作
            y = torch.relu(x)
            
            # 使用条件运算而不是分支
            condition = y.sum() > 0
            z = torch.where(condition, y * 2, y / 2)
            
            # 避免打印和其他副作用
            return z.sum() + x.dim()  # 使用x.dim()而不是len(x.shape)
        
        return optimized
    
    def _create_slow_function(self):
        """创建性能较差的函数"""
        def slow_func(x):
            # 低效的内存访问模式
            result = x.clone()
            for i in range(10):
                result = result + x[i % x.shape[0]]  # 逐个元素访问
            return result.sum()
        
        return slow_func
    
    def _create_fast_function(self):
        """创建高效的函数"""
        def fast_func(x):
            # 向量化操作
            result = x * 11  # 相当于x + 10*x
            return result.sum()
        
        return fast_func
    
    def _create_memory_hungry_function(self):
        """创建内存消耗大的函数"""
        def memory_hungry(x):
            # 创建很多中间张量
            intermediates = []
            current = x
            for i in range(5):
                current = current @ current.t()  # 每次都创建新张量
                intermediates.append(current.clone())
            
            return sum(intermediates).trace()
        
        return memory_hungry
    
    def _create_memory_efficient_function(self):
        """创建内存高效的函数"""  
        def memory_efficient(x):
            # 就地操作，减少内存分配
            result = x @ x.t()
            for i in range(4):
                result = result @ x.t()
            
            return result.trace()
        
        return memory_efficient
    
    def _test_function_performance(self, name: str, func):
        """测试函数性能"""
        
        x = torch.randn(100, 100)
        
        try:
            # 分析graph break
            explanation = torch._dynamo.explain(func)(x)
            
            # 编译和测试
            compiled = torch.compile(func)
            result = compiled(x)
            
            print(f"  {name}:")
            print(f"    Graph数量: {explanation.graph_count}")
            print(f"    Break数量: {explanation.graph_break_count}")
            print(f"    结果: {result:.4f}")
            
        except Exception as e:
            print(f"  {name}: ❌ 失败 - {e}")

# ============================================================================
# 第五部分：综合调试演示
# ============================================================================

def comprehensive_debug_demonstration():
    """综合调试演示"""
    
    print("\n" + "="*80)
    print("torch.compile 综合调试演示")
    print("="*80)
    
    # 创建调试目录
    base_debug_dir = tempfile.mkdtemp(prefix="torch_compile_full_debug_")
    print(f"主调试目录: {base_debug_dir}")
    
    # 初始化各阶段调试器
    dynamo_debugger = TorchDynamoDebugger(os.path.join(base_debug_dir, "dynamo"))
    aot_debugger = AOTAutogradDebugger(os.path.join(base_debug_dir, "aot"))
    inductor_debugger = InductorDebugger(os.path.join(base_debug_dir, "inductor"))
    error_analyzer = ErrorPatternAnalyzer()
    
    try:
        # 1. TorchDynamo阶段调试
        print("\n🔍 第一阶段：TorchDynamo调试")
        dynamo_debugger.setup_dynamo_debugging()
        
        def complex_test_function(x, y):
            """复杂测试函数"""
            # 正常张量操作
            z = torch.matmul(x, y.t())
            z = torch.relu(z)
            
            # 会导致graph break的操作
            if z.sum() > 0:
                z = z * 2
            
            # 动态行为
            for i in range(int(z.mean().item()) % 3):
                z = z + 0.1
                
            return z.sum()
        
        x = torch.randn(64, 32)
        y = torch.randn(64, 32)
        
        explanation = dynamo_debugger.analyze_graph_breaks(complex_test_function, x, y)
        dynamo_debugger.analyze_unsupported_features()
        dynamo_debugger.demonstrate_dynamic_behavior_issues()
        
        # 2. AOTAutograd阶段调试
        print("\n🔄 第二阶段：AOTAutograd调试")
        aot_debugger.setup_aot_debugging()
        aot_debugger.debug_backward_graph_generation()
        aot_debugger.debug_memory_optimization_issues()
        aot_debugger.debug_functional_transformation_failures()
        
        # 3. Inductor阶段调试
        print("\n💻 第三阶段：Inductor调试")
        inductor_debugger.setup_inductor_debugging()
        inductor_debugger.debug_generated_code()
        inductor_debugger.debug_kernel_fusion_issues()
        inductor_debugger.debug_performance_issues()
        inductor_debugger.debug_hardware_compatibility()
        
        # 4. 错误模式分析
        print("\n🔧 第四阶段：错误模式分析")
        error_analyzer.demonstrate_common_issues()
        
        print(f"\n✅ 综合调试演示完成！")
        print(f"所有调试文件保存在: {base_debug_dir}")
        
        # 生成调试报告
        generate_debug_report(base_debug_dir, explanation)
        
    except Exception as e:
        print(f"❌ 调试演示过程中出错: {e}")
        traceback.print_exc()
    
    return base_debug_dir

def generate_debug_report(debug_dir: str, explanation=None):
    """生成调试报告"""
    
    report_path = os.path.join(debug_dir, "debug_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# torch.compile 调试报告\n\n")
        f.write(f"生成时间: {torch.utils.data.get_worker_info()}\n")
        f.write(f"PyTorch版本: {torch.__version__}\n\n")
        
        if explanation:
            f.write("## Graph Break分析\n\n")
            f.write(f"- 图数量: {explanation.graph_count}\n")
            f.write(f"- Break数量: {explanation.graph_break_count}\n")
            f.write(f"- 操作总数: {explanation.op_count}\n\n")
            
            f.write("### Break原因:\n\n")
            for i, reason in enumerate(explanation.break_reasons):
                f.write(f"{i+1}. {reason}\n")
            f.write("\n")
        
        f.write("## 调试文件结构\n\n")
        
        # 递归列出所有文件
        for root, dirs, files in os.walk(debug_dir):
            level = root.replace(debug_dir, '').count(os.sep)
            indent = '  ' * level
            f.write(f"{indent}- {os.path.basename(root)}/\n")
            
            sub_indent = '  ' * (level + 1)
            for file in files:
                if file != 'debug_report.md':  # 排除报告文件本身
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    f.write(f"{sub_indent}- {file} ({size} bytes)\n")
        
        f.write("\n## 建议的后续步骤\n\n")
        f.write("1. 检查生成的Python代码文件了解图结构\n")
        f.write("2. 分析日志文件中的警告和错误信息\n")
        f.write("3. 查看内核代码了解底层优化\n")
        f.write("4. 根据性能分析结果优化代码结构\n")
    
    print(f"📋 调试报告已生成: {report_path}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print("torch.compile 全阶段深度调试指南")
    print("=" * 80)
    
    print("\n🎯 本指南涵盖:")
    print("1. TorchDynamo阶段：Graph Break分析、Guard调试、动态行为处理")
    print("2. AOTAutograd阶段：反向图生成、内存优化、函数式变换")
    print("3. Inductor阶段：代码生成、内核融合、性能优化")
    print("4. 错误模式：编译失败、性能回退、内存异常、数值精度")
    
    # 运行综合演示
    debug_dir = comprehensive_debug_demonstration()
    
    print(f"\n💡 关键调试工具总结:")
    print("- torch._dynamo.explain(): 分析graph break")
    print("- TORCH_LOGS环境变量: 控制日志输出")
    print("- torch.compile()的backend参数: 测试不同编译阶段")
    print("- 调试文件: 检查生成的代码和日志")
    
    print(f"\n🛠️ 推荐调试流程:")
    print("1. 使用explain()快速分析函数结构")
    print("2. 设置适当的日志级别")
    print("3. 逐阶段测试：eager -> aot_eager -> inductor")
    print("4. 分析生成的调试文件")
    print("5. 根据问题类型应用相应解决方案")
    
    return debug_dir

if __name__ == "__main__":
    main()