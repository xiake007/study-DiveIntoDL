import torch
import torch.nn as nn
import numpy as np
import utils

class DropOutOrg(object):
    def __init__(self):
        self.drop_prob=0
        self.n_input,self.n_hidden1,self.n_hidden2,self.n_output=784,256,256,10
        self.drop_prob1,self.drop_prob2=0.2,0.5
        self.params = self.init_weight()

    def dropout(self,X,drop_prob):
        X=X.float()
        keep_prob=1-drop_prob
        if keep_prob==0:
            return torch.zeros_like(X)
        mask = (torch.rand(X.shape)<keep_prob).float()
        return mask*X/keep_prob

    def init_weight(self):
        n_input,n_hidden1,n_hidden2,n_output=self.n_input,self.n_hidden1,self.n_hidden2,self.n_output
        w1=torch.tensor(np.random.normal(0.,0.01,(n_input,n_hidden1)),dtype=torch.float,requires_grad=True)
        b1=torch.zeros(n_hidden1,dtype=torch.float,requires_grad=True)
        w2=torch.tensor(np.random.normal(0.,0.01,(n_hidden1,n_hidden2)),dtype=torch.float,requires_grad=True)
        b2=torch.zeros(n_hidden2,dtype=torch.float,requires_grad=True)
        w3=torch.tensor(np.random.normal(0.,0.01,(n_hidden2,n_output)),dtype=torch.float,requires_grad=True)
        b3=torch.zeros(n_output,dtype=torch.float,requires_grad=True)
        return [w1,b1,w2,b2,w3,b3]

    def net(self,X,is_trainning=True):
        X=X.view(-1,self.n_input)
        H1 = (torch.matmul(X,self.params[0])+self.params[1]).relu()
        if is_trainning:
            H1 = self.dropout(H1,self.drop_prob1)
        H2= (torch.matmul(H1,self.params[2])+self.params[3]).relu()
        if is_trainning:
            H2 = self.dropout(H2,self.drop_prob2)
        return torch.matmul(H2,self.params[4])+self.params[5]

    def train(self):
        n_epochs,lr,batch_size=5,100,256
        loss = nn.CrossEntropyLoss()
        train_iter,test_iter=utils.load_data_fashion_mnist(batch_size)
        utils.train_ch3(self.net,train_iter,test_iter,loss,n_epochs,batch_size,self.params,lr)

class DropoutConcise(DropOutOrg):
    def __init__(self):
        super(DropoutConcise,self).__init__()

    def net(self):
        net=nn.Sequential(
            utils.FlattenLayer(),
            nn.Linear(self.n_input,self.n_hidden1),
            nn.ReLU(),
            nn.Dropout(self.drop_prob1),
            nn.Linear(self.n_hidden1,self.n_hidden2),
            nn.ReLU(),
            nn.Dropout(self.drop_prob2),
        )
        for param in net.parameters():
            nn.init.normal_(param,mean=0,std=0.01)
        return net

    def train(self):
        net = self.net()
        optimizer=torch.optim.SGD(net.parameters(),lr=0.5)
        n_epochs,lr,batch_size=5,100,256
        loss = nn.CrossEntropyLoss()
        train_iter,test_iter=utils.load_data_fashion_mnist(batch_size)
        utils.train_ch3(net,train_iter,test_iter,loss,n_epochs,batch_size,None,None,optimizer=optimizer)

def test_concise():
    obj=DropoutConcise()
    obj.train()

def test_org():
    obj=DropOutOrg()
    # X=torch.arange(16).view(2,8)
    # print(obj.dropout(X,0))
    # print(obj.dropout(X,0.5))
    # print(obj.dropout(X,1))
    obj.train()

if __name__=='__main__':
    # test_org()
    test_concise()
