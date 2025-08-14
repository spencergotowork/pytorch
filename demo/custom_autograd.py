"""
自定义Autograd函数示例

展示如何实现自定义的前向和反向传播
"""
import torch
from torch.autograd import Function
import math


class MyReLU(Function):
    """
    自定义ReLU激活函数

    前向: y = max(0, x)
    反向: dy/dx = 1 if x > 0 else 0
    """

    @staticmethod
    def forward(ctx, input):
        """
        前向传播

        Args:
            ctx: context对象,用于保存反向传播需要的信息
            input: 输入张量

        Returns:
            输出张量
        """
        print(f"\n[MyReLU Forward]")
        print(f"  输入: {input}")

        # 保存input以便反向传播使用
        ctx.save_for_backward(input)

        # 计算 max(0, x)
        output = input.clamp(min=0)

        print(f"  输出: {output}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播

        Args:
            ctx: 包含保存的张量
            grad_output: 上游传来的梯度 (dL/dy)

        Returns:
            对每个输入的梯度 (dL/dx)
        """
        print(f"\n[MyReLU Backward]")
        print(f"  上游梯度 (dL/dy): {grad_output}")

        # 取出保存的input
        input, = ctx.saved_tensors

        # dy/dx = 1 if x > 0 else 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0

        print(f"  计算的梯度 (dL/dx): {grad_input}")
        return grad_input


class MySigmoid(Function):
    """
    自定义Sigmoid激活函数

    前向: y = 1 / (1 + exp(-x))
    反向: dy/dx = y * (1 - y)
    """

    @staticmethod
    def forward(ctx, input):
        print(f"\n[MySigmoid Forward]")
        print(f"  输入: {input}")

        # 计算sigmoid
        output = 1 / (1 + torch.exp(-input))

        # 保存output用于反向传播
        # 注意:这里保存output而不是input,因为反向公式需要output
        ctx.save_for_backward(output)

        print(f"  输出: {output}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print(f"\n[MySigmoid Backward]")
        print(f"  上游梯度: {grad_output}")

        # 取出保存的output
        output, = ctx.saved_tensors

        # dy/dx = y * (1 - y)
        grad_input = grad_output * output * (1 - output)

        print(f"  计算的梯度: {grad_input}")
        return grad_input


class MyLinear(Function):
    """
    自定义线性层: y = xW + b

    前向: y = xW + b
    反向:
        dL/dx = dL/dy * W^T
        dL/dW = x^T * dL/dy
        dL/db = sum(dL/dy)
    """

    @staticmethod
    def forward(ctx, input, weight, bias):
        print(f"\n[MyLinear Forward]")
        print(f"  input shape: {input.shape}")
        print(f"  weight shape: {weight.shape}")
        print(f"  bias shape: {bias.shape}")

        # 保存用于反向传播
        ctx.save_for_backward(input, weight, bias)

        # 计算 y = xW + b
        output = input.mm(weight) + bias

        print(f"  output shape: {output.shape}")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        print(f"\n[MyLinear Backward]")
        print(f"  上游梯度 shape: {grad_output.shape}")

        # 取出保存的张量
        input, weight, bias = ctx.saved_tensors

        # 计算各个参数的梯度
        grad_input = grad_output.mm(weight.t())
        grad_weight = input.t().mm(grad_output)
        grad_bias = grad_output.sum(0)

        print(f"  grad_input shape: {grad_input.shape}")
        print(f"  grad_weight shape: {grad_weight.shape}")
        print(f"  grad_bias shape: {grad_bias.shape}")

        return grad_input, grad_weight, grad_bias


def example_1_relu():
    """示例1: 测试自定义ReLU"""
    print("=" * 70)
    print("示例1: 自定义ReLU函数")
    print("=" * 70)

    # 创建输入
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    print(f"输入x: {x}")

    # 应用自定义ReLU
    y = MyReLU.apply(x)
    print(f"输出y: {y}")

    # 反向传播
    loss = y.sum()
    print(f"\nloss: {loss}")
    print("\n开始反向传播:")
    loss.backward()

    print(f"\n最终x的梯度: {x.grad}")
    print("说明: 只有x>0的位置梯度为1,其他为0")


def example_2_sigmoid():
    """示例2: 测试自定义Sigmoid"""
    print("\n" + "=" * 70)
    print("示例2: 自定义Sigmoid函数")
    print("=" * 70)

    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    print(f"输入x: {x}")

    y = MySigmoid.apply(x)
    print(f"输出y: {y}")

    loss = y.sum()
    print(f"\nloss: {loss}")
    print("\n开始反向传播:")
    loss.backward()

    print(f"\n最终x的梯度: {x.grad}")

    # 验证
    print("\n验证:")
    with torch.no_grad():
        y_val = MySigmoid.apply(x)
        expected_grad = y_val * (1 - y_val)
        print(f"预期梯度 (y*(1-y)): {expected_grad}")


def example_3_linear():
    """示例3: 测试自定义线性层"""
    print("\n" + "=" * 70)
    print("示例3: 自定义线性层")
    print("=" * 70)

    # 创建输入和参数
    batch_size, in_features, out_features = 2, 3, 4

    x = torch.randn(batch_size, in_features, requires_grad=True)
    W = torch.randn(in_features, out_features, requires_grad=True)
    b = torch.randn(out_features, requires_grad=True)

    print(f"x: {x}")
    print(f"W: {W}")
    print(f"b: {b}")

    # 应用自定义线性层
    y = MyLinear.apply(x, W, b)
    print(f"\ny: {y}")

    # 反向传播
    loss = y.sum()
    print(f"\nloss: {loss}")
    print("\n开始反向传播:")
    loss.backward()

    print(f"\nx的梯度:\n{x.grad}")
    print(f"\nW的梯度:\n{W.grad}")
    print(f"\nb的梯度:\n{b.grad}")


def example_4_compare_with_builtin():
    """示例4: 与PyTorch内置函数比较"""
    print("\n" + "=" * 70)
    print("示例4: 与PyTorch内置函数比较")
    print("=" * 70)

    x1 = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)

    # 使用自定义ReLU
    y1 = MyReLU.apply(x1)
    y1.sum().backward()

    # 使用PyTorch内置ReLU
    y2 = torch.relu(x2)
    y2.sum().backward()

    print(f"自定义ReLU输出: {y1}")
    print(f"内置ReLU输出:   {y2}")
    print(f"\n自定义ReLU梯度: {x1.grad}")
    print(f"内置ReLU梯度:   {x2.grad}")
    print(f"\n结果相同: {torch.allclose(x1.grad, x2.grad)}")


class MyExp(Function):
    """
    自定义指数函数: y = exp(x)
    反向: dy/dx = exp(x) = y
    """

    @staticmethod
    def forward(ctx, input):
        output = torch.exp(input)
        ctx.save_for_backward(output)  # 保存output而不是input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output * output  # dy/dx = exp(x)


class MyPower(Function):
    """
    自定义幂函数: y = x^n
    反向: dy/dx = n * x^(n-1)
    """

    @staticmethod
    def forward(ctx, input, exponent):
        ctx.save_for_backward(input)
        ctx.exponent = exponent
        return input ** exponent

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        exponent = ctx.exponent
        grad_input = grad_output * exponent * (input ** (exponent - 1))
        return grad_input, None  # 对exponent不求梯度


def example_5_more_functions():
    """示例5: 更多自定义函数"""
    print("\n" + "=" * 70)
    print("示例5: 更多自定义函数")
    print("=" * 70)

    print("\n--- 指数函数 ---")
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = MyExp.apply(x)
    y.sum().backward()
    print(f"x: {x.data}")
    print(f"exp(x): {y.data}")
    print(f"梯度: {x.grad}")
    print(f"验证: exp(x)的导数是exp(x)本身")

    print("\n--- 幂函数 ---")
    x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y2 = MyPower.apply(x2, 3)  # x^3
    y2.sum().backward()
    print(f"x: {x2.data}")
    print(f"x^3: {y2.data}")
    print(f"梯度: {x2.grad}")
    print(f"验证 (3x^2): {3 * x2.data ** 2}")


if __name__ == "__main__":
    # 运行所有示例
    example_1_relu()
    example_2_sigmoid()
    example_3_linear()
    example_4_compare_with_builtin()
    example_5_more_functions()

    print("\n" + "=" * 70)
    print("所有示例完成!")
    print("=" * 70)

    print("\n关键要点:")
    print("1. 继承torch.autograd.Function类")
    print("2. 实现静态方法forward()和backward()")
    print("3. 使用ctx保存反向传播需要的张量")
    print("4. backward()返回与forward()输入对应的梯度")
    print("5. 使用.apply()方法调用自定义函数")
    print("6. 可以保存非张量属性(如ctx.exponent)")
