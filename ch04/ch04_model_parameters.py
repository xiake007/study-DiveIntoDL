import torch
import torch.nn as nn
import torch.nn.init as init

def test1():
    net=nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))
    for name,param in net.named_parameters():
        if 'weight' in name:
            print(param)
    print(net)
    x=torch.rand(2,4)
    y=net(x).sum()
    print(net(x),y)

    print(type(net.named_parameters()))
    for name,param in net.named_parameters():
        print(name,param.size())
    for name,param in net[0].named_parameters():
        print(name,param.size())
    weight0=list(net[0].parameters())[0]
    print(weight0.data)
    print(weight0.grad)
    y.backward()
    print(weight0.grad)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight1=nn.Parameter(torch.rand(2,3))
        self.weight2=torch.rand(3,4)

    def forward(self, *input):
        pass

class ModelInit(nn.Module):
    def __init__(self):
        super(ModelInit,self).__init__()
        self.net=nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))
    def init_1(self):
        for name,param in self.net.named_parameters():
            if 'weight' in name:
                print(name,param.data)
                nn.init.normal_(param,mean=0,std=0.01)
                print(name,param.data)
            if 'bias' in name:
                print(name,param.data)
                nn.init.constant_(param,val=0.)
                print(name,param.data)

    def init_my(self):
        #[-10,-5],[5,10]
        def init_weight(tensor):
            with torch.no_grad():
                tensor.uniform_(-10,10)
                tensor *= (tensor.abs()>=5.).float()
        for name,param in self.net.named_parameters():
            if 'weight' in name:
                init_weight(param)
                print(name,param.data)
            if 'bias' in name:
                param.data += 1
                print(name,param.data)

def test_modelinit():
    net=ModelInit()
    # net.init_1()
    # net.init_my()

def test_mymodel():
    net=MyModel()
    for name,param in net.named_parameters():
        print(name,param)

def test_share_params():
    linear=nn.Linear(1,1,bias=False)
    net=nn.Sequential(linear,linear)
    print(net)
    for name,param in net.named_parameters():
        nn.init.constant_(param,val=3)
        print(name,param)
    print(id(net[0])==id(net[1]))
    print(id(net[0].weight)==id(net[1].weight))

    x=torch.ones(1,1)
    y=net(x).sum()
    print(y)
    print(net[0].weight.grad)
    y.backward()
    print(net[0].weight.grad)

if __name__=='__main__':
    # test1()
    # test_mymodel()
    # test_modelinit()
    test_share_params()