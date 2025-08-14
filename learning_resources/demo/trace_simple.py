"""
演示追踪PyTorch操作的完整流程
从Python API到C++实现
"""
import torch

print("=" * 60)
print("PyTorch 操作追踪演示")
print("=" * 60)

# 1. 创建张量
print("\n[1] 创建张量")
print("-" * 60)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

print(f"x = {x}")
print(f"y = {y}")
print(f"x.requires_grad = {x.requires_grad}")
print(f"x.device = {x.device}")
print(f"x.dtype = {x.dtype}")

# 2. 执行加法操作
print("\n[2] 执行加法: z = x + y")
print("-" * 60)
z = torch.add(x, y)  # 等价于 x + y

print(f"z = {z}")
print(f"z.grad_fn = {z.grad_fn}")
print(f"z.grad_fn 类型 = {type(z.grad_fn).__name__}")

# 3. 查看计算图连接
print("\n[3] 查看计算图连接")
print("-" * 60)
if z.grad_fn:
    print(f"z.grad_fn.next_functions = {z.grad_fn.next_functions}")
    for i, (fn, idx) in enumerate(z.grad_fn.next_functions):
        print(f"  输入{i}: {fn} (输出索引: {idx})")

# 4. 继续构建计算图
print("\n[4] 继续计算: w = z * 2")
print("-" * 60)
w = z * 2
print(f"w = {w}")
print(f"w.grad_fn = {w.grad_fn}")

# 5. 计算loss
print("\n[5] 计算loss: loss = w.sum()")
print("-" * 60)
loss = w.sum()
print(f"loss = {loss}")
print(f"loss.grad_fn = {loss.grad_fn}")

# 6. 反向传播
print("\n[6] 执行反向传播: loss.backward()")
print("-" * 60)
print("开始反向传播...")
loss.backward()
print("反向传播完成!")

# 7. 查看梯度
print("\n[7] 查看计算的梯度")
print("-" * 60)
print(f"x.grad = {x.grad}")
print(f"y.grad = {y.grad}")

# 手动验证梯度
print("\n[8] 验证梯度计算")
print("-" * 60)
print("计算过程:")
print("  loss = sum((x + y) * 2)")
print("  loss = sum(2x + 2y)")
print("  ∂loss/∂x = 2 (对每个元素)")
print("  ∂loss/∂y = 2 (对每个元素)")
print(f"\n预期 x.grad = [2.0, 2.0, 2.0]")
print(f"实际 x.grad = {x.grad}")
print(f"\n预期 y.grad = [2.0, 2.0, 2.0]")
print(f"实际 y.grad = {y.grad}")

# 9. 查看张量的内部属性
print("\n[9] 张量的内部属性")
print("-" * 60)
print(f"x.is_leaf = {x.is_leaf}")
print(f"z.is_leaf = {z.is_leaf}")
print(f"x.grad_fn = {x.grad_fn} (叶子节点没有grad_fn)")
print(f"z.grad_fn = {z.grad_fn} (非叶子节点有grad_fn)")

print("\n" + "=" * 60)
print("追踪完成!")
print("=" * 60)

print("\n提示:")
print("- grad_fn 记录了创建张量的操作")
print("- next_functions 连接到计算图中的前一个节点")
print("- 叶子节点(requires_grad=True)会累积梯度")
print("- backward()从loss反向遍历整个计算图")
