import numpy as np
import utils
import torch
import torch.nn

class PerceptronOrg(object):
    def __init__(self):
        self.batch_size=256
        self.train_iter,self.test_iter=utils.load_data_fashion_mnist()
        self.input_num=784  #28x28
        self.hide_num=256
        self.output_num=256
        self.params=self.init_weight()

    def init_weight(self):
        w1=torch.tensor(np.random.normal(0.0,0.01,(self.input_num,self.hide_num)),dtype=torch.float)
        b1=torch.zeros(self.hide_num,dtype=torch.float)
        w2=torch.tensor(np.random.normal(0.0,0.01,(self.hide_num,self.output_num)),dtype=torch.float)
        b2=torch.zeros(self.output_num,dtype=torch.float)
        params=[w1,b1,w2,b2]
        for param in params:
            param.requires_grad_(True)
        return params

    def relu(self,x):
        return torch.max(x,torch.tensor(0.0))

    def net(self,x):
        [w1,b1,w2,b2] = self.params
        x=x.view((-1,self.input_num))
        h=self.relu(torch.mm(x,w1)+b1)
        o=torch.matmul(h,w2)+b2
        return o

    def loss(self):
        return torch.nn.CrossEntropyLoss()

    def train(self):
        num_epochs,lr=5,100.
        utils.train_ch3(self.net,self.train_iter,self.test_iter,self.loss(), \
                        num_epochs,self.batch_size,self.params,lr)

class PerceptronConcise(PerceptronOrg):
    def __init__(self):
        super(PerceptronConcise,self).__init__()
        self.net = self.create_model()

    def create_model(self):
        net=torch.nn.Sequential(
            utils.FlattenLayer(),
            torch.nn.Linear(self.input_num,self.hide_num),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hide_num,self.output_num),
        )
        for param in net.parameters():
            torch.nn.init.normal_(param,mean=0.,std=0.01)

        return net

    def train(self):
        optimizer=torch.optim.SGD(params=self.net.parameters(),lr=0.5)
        utils.train_ch3(self.net,self.train_iter,self.test_iter,self.loss(),5,self.batch_size,None,None,optimizer)

if __name__=='__main__':
    obj=PerceptronConcise()
    # obj=PerceptronOrg()
    obj.train()