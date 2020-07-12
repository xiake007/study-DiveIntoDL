import numpy as np
import torch

from utils import *

class LineRegressOrg(object):
    """
    realize line regress neural networks ,only using Tensor,autograd
    """
    def __init__(self):
        self.true_w=[2,-3.4]
        self.true_b=4.2
        self.num_samples=1000
        self.num_input=2
        self.lr=0.3
        self.epochs=30
        self.features,self.labels=gen_datasets(self.num_samples,self.num_input,self.true_w,self.true_b)

    def test_data_iter(self):
        for X,Y in self.data_iter(10,self.features,self.labels):
            print(X,Y)
            break

    def pipline(self):
        """
        #1.generate feature data,read data
        #2.define network
        #3.define loss function
        #4.define optimize Algorithm
        #5.train

        :return:
        """
        print(self.features[0],self.labels[0])
        # plot(self.features[:,1].numpy(),self.labels.numpy())
        # self.test_data_iter()
        self.train()

    def init_wb(self):
        w=torch.tensor(np.random.normal(0,0.01,(self.num_input,1)),dtype=torch.float32)
        b=torch.zeros(1,dtype=torch.float32)
        w.requires_grad_(requires_grad=True)
        b.requires_grad_(requires_grad=True)
        return w,b

    def train(self):
        lr = 0.03
        epochs=3
        batch_size=10
        w,b =self.init_wb()
        Net=linreg
        Loss=squared_loss
        print('Init w1=%f,w2=%f,b=%f'%(w[0].data.numpy(),w[1].data.numpy(),b.data.numpy()))
        for epoch in range(epochs):
            for X,y in data_iter(batch_size=batch_size,
                                      features=self.features,labels=self.labels):
                y_hat=Net(X,w,b)
                epoch_loss = Loss(y_hat,y).sum()
                epoch_loss.backward()
                sgd([w,b],lr,batch_size)
                w.grad.data.zero_()
                b.grad.data.zero_()

            train_loss = Loss(Net(self.features,w,b),self.labels)
            print('epoch %d: loss is %f, w1=%f,w2=%f,b=%f'%(epoch,train_loss.mean().item(),
                                                            w[0].data.numpy(),w[1].data.numpy(),b.data.numpy()))

def main():
    obj=LineRegressOrg()
    obj.pipline()

if __name__ == '__main__':
    main()