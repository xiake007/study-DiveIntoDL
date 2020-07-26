import torch
import torch.nn as nn
import utils

def test():
    X=torch.arange(9).reshape(3,3)
    K=torch.arange(4).reshape(2,2)
    Y=utils.corr2d(X,K)
    print(X.shape,K.shape,Y.shape)
    print(X,K,Y)

class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D,self).__init__()
        self.weight=nn.Parameter(torch.randn(kernel_size))
        self.bias=nn.Parameter(torch.randn(1))

    def forward(self, x):
        return utils.corr2d(x,self.weight)+self.bias

def test_img_border_detect():
    X=torch.ones(6,8)
    X[:,2:6]=0
    print(X)
    K=torch.tensor([[1.,-1.]])
    print(K)
    Y=utils.corr2d(X,K)
    print(Y)

def train():
    X=torch.ones(6,8)
    X[:,2:6]=0
    K=torch.tensor([[1.,-1.]])
    Y=utils.corr2d(X,K)
    conv2d=Conv2D((1,2))
    lr=0.01
    for i in range(20):
        y_hat=conv2d(X)
        l=((Y-y_hat)**2).sum()
        l.backward()
        #
        conv2d.weight.data -= lr*conv2d.weight.grad
        conv2d.bias.data -= lr*conv2d.bias.grad
        #
        conv2d.weight.grad.fill_(0)
        conv2d.bias.grad.fill_(0)
        if (i+1)%5==0:
            print(f'step={i+1}: loss is {l.item()}')
    print(conv2d.weight.data,conv2d.bias.data)

if __name__=='__main__':
    # test()
    # test_img_border_detect()
    train()

