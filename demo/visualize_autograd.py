"""
可视化Autograd计算图

展示如何查看和理解PyTorch的自动微分图结构
"""
import torch


def print_grad_fn_tree(tensor, indent=0, visited=None):
    """
    递归打印计算图的树形结构

    参数:
        tensor: 要打印的张量
        indent: 缩进级别
        visited: 已访问的节点集合(避免循环)
    """
    if visited is None:
        visited = set()

    prefix = "  " * indent

    if tensor.grad_fn is None:
        print(f"{prefix}└─ 叶子节点 (requires_grad={tensor.requires_grad})")
        return

    grad_fn = tensor.grad_fn
    grad_fn_name = grad_fn.__class__.__name__

    # 避免重复访问
    if id(grad_fn) in visited:
        print(f"{prefix}└─ {grad_fn_name} (已访问)")
        return

    visited.add(id(grad_fn))

    print(f"{prefix}└─ {grad_fn_name}")

    # 递归打印输入节点
    if hasattr(grad_fn, 'next_functions'):
        for i, (next_fn, output_nr) in enumerate(grad_fn.next_functions):
            if next_fn is not None:
                print(f"{prefix}  ├─ 输入{i} (输出索引:{output_nr})")
                # 创建一个虚拟张量来继续遍历
                if hasattr(next_fn, 'variable'):
                    print_grad_fn_tree_from_fn(next_fn, indent+2, visited)
                else:
                    print_grad_fn_tree_from_fn(next_fn, indent+2, visited)


def print_grad_fn_tree_from_fn(grad_fn, indent=0, visited=None):
    """从grad_fn开始打印"""
    if visited is None:
        visited = set()

    prefix = "  " * indent

    if grad_fn is None:
        print(f"{prefix}└─ None")
        return

    if id(grad_fn) in visited:
        print(f"{prefix}└─ {grad_fn.__class__.__name__} (已访问)")
        return

    visited.add(id(grad_fn))

    print(f"{prefix}└─ {grad_fn.__class__.__name__}")

    if hasattr(grad_fn, 'next_functions'):
        for i, (next_fn, output_nr) in enumerate(grad_fn.next_functions):
            if next_fn is not None:
                print_grad_fn_tree_from_fn(next_fn, indent+1, visited)


def example_1_simple():
    """示例1: 简单的加法和乘法"""
    print("=" * 70)
    print("示例1: 简单计算图 - (x + y) * 2")
    print("=" * 70)

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([3.0, 4.0], requires_grad=True)

    z = x + y  # AddBackward
    w = z * 2  # MulBackward

    print("\n计算图结构:")
    print_grad_fn_tree(w)

    print("\n执行反向传播...")
    w.sum().backward()

    print(f"\nx的梯度: {x.grad}")
    print(f"y的梯度: {y.grad}")
    print("\n说明: 梯度都是2,因为 d/dx(2(x+y)) = 2")


def example_2_complex():
    """示例2: 更复杂的计算图"""
    print("\n" + "=" * 70)
    print("示例2: 复杂计算图 - (x^2 + y^2) * (x + y)")
    print("=" * 70)

    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)

    # 构建复杂计算图
    x_sq = x ** 2      # PowBackward
    y_sq = y ** 2      # PowBackward
    sum_sq = x_sq + y_sq  # AddBackward
    sum_xy = x + y     # AddBackward
    result = sum_sq * sum_xy  # MulBackward

    print(f"\nx = {x.item()}")
    print(f"y = {y.item()}")
    print(f"x^2 = {x_sq.item()}")
    print(f"y^2 = {y_sq.item()}")
    print(f"x^2 + y^2 = {sum_sq.item()}")
    print(f"x + y = {sum_xy.item()}")
    print(f"result = {result.item()}")

    print("\n计算图结构:")
    print_grad_fn_tree(result)

    print("\n执行反向传播...")
    result.backward()

    print(f"\nx的梯度: {x.grad}")
    print(f"y的梯度: {y.grad}")

    # 手动验证
    x_val = 2.0
    y_val = 3.0
    # result = (x^2 + y^2) * (x + y)
    # d/dx = 2x(x+y) + (x^2+y^2)
    expected_grad_x = 2*x_val*(x_val+y_val) + (x_val**2+y_val**2)
    expected_grad_y = 2*y_val*(x_val+y_val) + (x_val**2+y_val**2)

    print(f"\n验证 - 预期x的梯度: {expected_grad_x}")
    print(f"验证 - 预期y的梯度: {expected_grad_y}")


def example_3_shared_variable():
    """示例3: 共享变量的计算图"""
    print("\n" + "=" * 70)
    print("示例3: 共享变量 - x + x * x")
    print("=" * 70)

    x = torch.tensor([2.0], requires_grad=True)

    # x被使用了多次
    y = x + x * x  # x出现3次!

    print(f"\nx = {x.item()}")
    print(f"y = x + x*x = {y.item()}")

    print("\n计算图结构:")
    print_grad_fn_tree(y)

    print("\n执行反向传播...")
    y.backward()

    print(f"\nx的梯度: {x.grad}")

    # 手动验证: y = x + x^2, dy/dx = 1 + 2x = 1 + 4 = 5
    expected_grad = 1 + 2*x.item()
    print(f"验证 - 预期梯度: {expected_grad}")
    print("说明: x被使用多次时,梯度会累加")


def example_4_no_grad():
    """示例4: 不需要梯度的部分"""
    print("\n" + "=" * 70)
    print("示例4: 部分不需要梯度")
    print("=" * 70)

    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=False)  # 不需要梯度

    z = x * y

    print(f"\nx.requires_grad = {x.requires_grad}")
    print(f"y.requires_grad = {y.requires_grad}")
    print(f"z.requires_grad = {z.requires_grad}")

    print("\n计算图结构:")
    print_grad_fn_tree(z)

    z.backward()

    print(f"\nx的梯度: {x.grad}")
    print(f"y的梯度: {y.grad} (因为requires_grad=False)")


def example_5_detach():
    """示例5: 使用detach()截断计算图"""
    print("\n" + "=" * 70)
    print("示例5: 使用detach()截断计算图")
    print("=" * 70)

    x = torch.tensor([2.0], requires_grad=True)

    # 第一部分计算
    y = x ** 2
    print(f"y = x^2 = {y.item()}")
    print(f"y.requires_grad = {y.requires_grad}")

    # 截断计算图
    y_detached = y.detach()
    print(f"\ny_detached.requires_grad = {y_detached.requires_grad}")

    # 使用截断后的值继续计算
    z = y_detached * 3
    print(f"z = y_detached * 3 = {z.item()}")
    print(f"z.requires_grad = {z.requires_grad}")

    print("\n说明: detach()后,z不再依赖于x,无法反向传播到x")


if __name__ == "__main__":
    # 运行所有示例
    example_1_simple()
    example_2_complex()
    example_3_shared_variable()
    example_4_no_grad()
    example_5_detach()

    print("\n" + "=" * 70)
    print("所有示例完成!")
    print("=" * 70)

    print("\n关键要点:")
    print("1. grad_fn 记录了创建张量的操作")
    print("2. next_functions 链接到计算图的前驱节点")
    print("3. 叶子节点(手动创建的张量)累积梯度")
    print("4. 非叶子节点的grad_fn指向创建它的操作")
    print("5. requires_grad=False的张量不参与梯度计算")
    print("6. detach()可以截断计算图")
