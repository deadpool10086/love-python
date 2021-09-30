import torch

# 定义a b c x的值，abc指定为需要求导requires_grad=True
x = torch.tensor(2., requires_grad=True)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)
# 定义y函数
y = a * x ** 2 + b * x + c;
# 使用autograd.grad自定求导
grads = torch.autograd.grad(y, [a, b, c, x])
# 打印abc分别的导数值（带入x的值）
print('after', grads[0],grads[1],grads[2],grads[3])