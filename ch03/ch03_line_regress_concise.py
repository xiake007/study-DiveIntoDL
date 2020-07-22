from utils import *
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from collections import OrderedDict

class LineRegressConcise(object):
    """
    realize line regress neural networks ,using concise type
    """
    def __init__(self):
        self.true_w=[2,-3.4]
        self.true_b=4.2
        self.num_samples=1000
        self.num_input=2
        self.lr=0.3
        self.epochs=3
        self.batch_size=10

        self.features,self.labels=gen_datasets(self.num_samples,self.num_input,self.true_w,self.true_b)
        self.dataset = Data.TensorDataset(self.features,self.labels)
        self.data_iter=Data.DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True)

        # self.net = LinearNet(self.num_input)
        self.net = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.num_input, 1))
            # ......
        ]))
        # print(self.net)
        init.normal_(self.net.linear.weight,mean=0,std=0.01)
        init.constant_(self.net.linear.bias,val=0)
        # print(self.net)
        self.loss=nn.MSELoss()
        self.optim = self.sgd()
        # print(self.optim)

    def sgd(self):
        return optim.SGD(self.net.parameters(),lr=0.01)

    def train(self):
        num_epochs = 3
        for epoch in range(1,num_epochs+1):
            for X,y in self.data_iter:
                output=self.net(X)
                loss =self.loss(output,y.view(-1,1))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # print('epoch %d, loss: %f'%(epoch,loss.item()),self.net.linear.weight)

    def pipline(self):
        # for x,y in self.data_iter:
        #     print(x,y)
        #     break

        # for param in self.net.parameters():
        #     print(param)

        self.train()
        dense=self.net[0]
        # dense=self.net.linear
        print(self.true_w,dense.weight)
        print(self.true_b,dense.bias)

class LinearNet(nn.Module):
    def __init__(self,n_features):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(n_features,1)

    def forward(self,x):
        y=self.linear(x)
        return y

def main():
    obj=LineRegressConcise()
    obj.pipline()

if __name__ == '__main__':
    main()