import torch
import torch.nn as nn
import torch.utils.data as Dataset
import numpy as np
import matplotlib.pyplot as plt
import random

class LineRegOrg(object):
    """
    y=X*w+b+noise_e
    1.确定真实的true_w，true_b，随机生成数据X，计算得到真实的y
    2.使用LineRegress算法训练数据X，得到预测的y_hat
    3.使用y_hat、y计算loss，从而预测 predict_w,predict_b
    """
    def __init__(self):
        self.num_epochs = 30
        self.batch_size=10
        self.num_samples=1000
        self.num_inputs=2
        self.lr = 0.03
        self.true_w=[2,-3.4]
        self.true_b=4.2
        self.features,self.labels=self.gen_data()

    def gen_data(self):
        features = torch.rand(self.num_samples,self.num_inputs,dtype=torch.float32)
        labels = self.true_w[0]*features[:,0]+self.true_w[1]*features[:,1]+self.true_b
        noise_e = torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)
        labels += noise_e
        return features,labels

    def show(self):
        plt.scatter(self.features[:,1].numpy(),self.labels.numpy(),1)
        plt.show()

    def data_iter(self):
        """
        :return: 每次返回一个batch-size的数据
        """
        index=list(range(self.num_samples))
        random.shuffle(index)
        for i in range(0,self.num_samples,self.batch_size):
            j=torch.LongTensor(index[i:min(i+self.batch_size,self.num_samples)])
            yield self.features.index_select(0,j),self.labels.index_select(0,j)

    def print_data_iter(self):
        for i in self.data_iter():
            print(i)
            break

    def net(self,X,w,b):
        return torch.mm(X,w)+b

    def loss(self,y_hat,y):
        return (y_hat-y.view(y_hat.size()))**2/2

    def sgd(self,params):
        for param in params:
            param.data -= self.lr*param.grad/self.batch_size

    def train(self):
        # init w,b
        w = torch.tensor(np.random.normal(0,0.01,size=(self.num_inputs,1)),dtype=torch.float32)
        b = torch.zeros(1,dtype=torch.float32)
        w.requires_grad_(requires_grad=True)
        b.requires_grad_(requires_grad=True)
        for epoch in range(self.num_epochs):
            for X,y in self.data_iter():
                pre_y = self.net(X,w,b)
                ls = self.loss(pre_y,y).sum()
                ls.backward()
                self.sgd([w,b])

                w.grad.data.zero_()
                b.grad.data.zero_()
            epoch_loss = self.loss(pre_y,y).sum()
            print('epoch={}, loss={}, w={}, b={}'.format(epoch,epoch_loss,w,b))

    def test(self):
        # print(self.__class__)
        # self.print_data_iter()
        # self.show()
        self.train()

if __name__ == '__main__':
    LR = LineRegOrg()
    LR.test()