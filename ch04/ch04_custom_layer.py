import torch
import torch.nn as nn

class CenterLay(nn.Module):
    def __init__(self):
        super(CenterLay,self).__init__()

    def forward(self, x):
        return x-x.mean()

def test_centerlay():
    net=CenterLay()
    x=torch.tensor(([1,2,3,4]),dtype=torch.float)
    # print(net(x))

    net=nn.Sequential(nn.Linear(3,128),CenterLay())
    x=torch.rand(2,3)
    y=net(x).mean().item()
    print(y)

class MylistDense(nn.Module):
    def __init__(self):
        super(MylistDense,self).__init__()
        self.params=nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4,1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x=torch.mm(x,self.params[i])
        return x

def test_mylistdense():
    x=torch.rand(3,4)
    net=MylistDense()
    print(net)
    print(net(x))

class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense,self).__init__()
        self.params=nn.ParameterDict({
            'linear1':nn.Parameter(torch.randn(4,4)),
            'linear2':nn.Parameter(torch.randn(4,2)),
        })
        self.params.update({'linear3':nn.Parameter(torch.randn(4,1))})

    def forward(self, x,choice='linear1'):
        x=torch.mm(x,self.params[choice])
        return x

def test_mydictdense():
    net=MyDictDense()
    print(net)
    # x=torch.rand(2,4)
    x=torch.ones(2,4)
    print(net(x,'linear1'))
    print(net(x,'linear2'))
    print(net(x,'linear3'))

def test_list_dict_lay():
    x=torch.ones(2,4)
    net=nn.Sequential(
        MyDictDense(),
        MylistDense(),
    )
    print(net)
    print(net(x))
if __name__=='__main__':
    # test_centerlay()
    # test_mylistdense()
    # test_mydictdense()
    test_list_dict_lay()

