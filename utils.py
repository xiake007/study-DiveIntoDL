import numpy as np
import torch
from IPython import display
import matplotlib.pyplot as plt
import random

def plot(x,y):
    def use_svg_display():
        display.set_matplotlib_formats('svg')
    def set_figsize(figsize=(3.5,2.5)):
        use_svg_display()
        plt.rcParams['figure.figsize']=figsize

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

