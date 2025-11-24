import torch

a = torch.tensor([1.0, 2.0], requires_grad=True, device='cuda')
b = torch.tensor([2.0, 3.0], requires_grad=True, device='cuda')
c = a * a+b

# c.backward()
c.retain_grad()
loss = torch.sum(c)
loss.backward()

print(a.grad)
print(b.grad)
print(c.grad)
