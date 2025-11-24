import torch

# 验证DispatchKey
x = torch.randn(3, 3, requires_grad=True)
print("BackendKey:", torch._C._dispatch_keyset_full_after(torch._C.DispatchKey.BackendSelect))

# 检查特定算子的内核是否存在
op_name = "aten::add.Tensor"
for key in ["CPU", "CUDA", "AutogradCPU"]:
    # has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key)
    # print(f"{op_name} has {key} kernel: {has_kernel}")
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, key)
    print(f"{op_name} has {key} kernel: {has_kernel}")
