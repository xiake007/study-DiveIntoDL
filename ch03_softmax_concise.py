import torch
import torch.nn as nn
from torch.nn import init
import utils
import numpy as np
import sys

class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(num_inputs,num_outputs)

    def forward(self, x):
        y=self.linear(x.view(x.shape[0],-1))
        return y

from collections import OrderedDict
class SoftmaxConcise(object):
    def __init__(self):
        self.num_inputs = 784 #28x28
        self.num_outputs = 10
        self.batch_size=256
        self.train_iter,self.test_iter=utils.load_data_fashion_mnist()
        self.num_epoch=3
        self.net = LinearNet(self.num_inputs,self.num_outputs)

    def init_model(self):
        init.normal_(self.net.linear.weight,mean=0,std=0.01)
        init.constant_(self.net.linear.bias,val=0)

    def loss(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return torch.optim.SGD(self.net.parameters(),lr=0.1)

    def train(self):
        self.init_model()
        utils.train_ch3(self.net,self.train_iter,self.test_iter,self.loss(), self.num_epoch, \
                        self.batch_size,None,None,optimizer=self.optimizer())

if __name__ =='__main__':
    obj = SoftmaxConcise()
    obj.train()
