import numpy as np
import torch
from IPython import display
import matplotlib.pyplot as plt
import random

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize']=figsize

def plot(x,y):
    set_figsize()
    plt.scatter(x,y,1)
    plt.show()

def gen_datasets(num_samples,num_input,true_w,true_b):
    """
    y=w1*x1+w2*x2+b, 即：Y=XW+b
    :return:
    """
    features=torch.randn(num_samples,num_input,dtype=torch.float32)
    labels=features[:,0]*true_w[0]+features[:,1]*true_w[1]+true_b
    labels += torch.tensor(np.random.normal(0,0.01,size=len(labels)),dtype=torch.float32)
    return features,labels

def data_iter(batch_size,features,labels):
    size=len(labels)
    indices=list(range(size))
    random.shuffle(indices)
    for i in range(0,size,batch_size):
        bb=torch.LongTensor(indices[i:min(i+batch_size,size)])
        yield features.index_select(0,bb),labels.index_select(0,bb)

def linreg(X,w,b):
    return torch.mm(X,w)+b

def squared_loss(y_hat,y):
    return (y_hat-y.view(y_hat.size()))**2/2

def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size

def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,
             legend=None,figsize=(3.5,2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=':')
        plt.legend(legend)

    plt.show()

import torchvision
import torchvision.transforms as transforms
import sys
def download_FashionMNIST(is_download=True):
    mnist_train=torchvision.datasets.FashionMNIST(root='../dataset/FashionMNIST',train=True,transform=transforms.ToTensor(),download=is_download);
    mnist_test=torchvision.datasets.FashionMNIST(root='../dataset/FashionMNIST',train=False,transform=transforms.ToTensor(),download=is_download);
    return mnist_train,mnist_test

mnist_train,mnist_test=download_FashionMNIST(is_download=False)

def load_data_fashion_mnist(batch_size=256):
    if sys.platform.startswith('win'):
        num_workers=0
    else:
        num_workers=4
    train_iter=torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_iter=torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    return train_iter,test_iter

def evaluate_accuracy(data_iter,net):
    acc_sum=0.
    m=0
    for X,y in data_iter:
        if isinstance(net,torch.nn.Module):
            net.eval()
            acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
            net.train()
        else:
            if('is_trainning' in net.__code__.co_varnames):
                acc_sum += (net(X,is_trainning=False).argmax(dim=1)==y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        m += y.shape[0]
    return acc_sum/m

def train_ch3(net,train_iter,test_iter,loss,num_epoch,batch_size,params=None,lr=None,optimizer=None):
    for epoch in range(num_epoch):
        loss_sum,acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X)
            l=loss(y_hat,y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()

            acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            loss_sum += l.item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter,net)
        print(f'epoch={epoch}, loss={loss_sum/n}, train_acc={acc_sum/n}, test_acc={test_acc}')

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self,x):
        return x.view(x.shape[0],-1)

def corr2d(X,K):
    h,w=K.shape
    Y=torch.zeros(X.shape[0]-h+1,X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y
