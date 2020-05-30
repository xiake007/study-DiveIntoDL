import torch
import numpy as np

x=torch.randn(5,3)
x=torch.empty(5,3,dtype=torch.int8)
x=torch.zeros(5,3)
x=torch.randn(1)
# print(x)
# print(x.item())
x=torch.arange(1,3).view(1,2)
# print(x)
y=torch.arange(1,4).view(3,1)
# print(y)
# print(x+y)
id_before=id(y)
y=x+y
id_after=id(y)
# print(id_before==id_after)

# numpy互转
x=torch.randn(3,2)
x=x.numpy()
x=torch.from_numpy(x)
# print(x)
if torch.cuda.is_available():
    dev=torch.device('cuda')
    y=torch.ones_like(x,device=dev)
    x=x.to(device=dev)
    z=x+y
    # print(z)
    # print(z.to(device='cpu',dtype=torch.double))

x=torch.ones(2,2,requires_grad=True)
# print(x)
# print(x.grad_fn)
y=x+2
# print(y.grad_fn)
# print(x.grad_fn)
# print(x.is_leaf,y.is_leaf)
z=y*y*3
out=z.mean()
# print(z)
# print(out)

#梯度
out.backward()
# print(x.grad)

out2=x.sum()
out2.backward()
print(x.grad)

out3=x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
