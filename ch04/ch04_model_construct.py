import torch
import torch.nn as nn
import sys
sys.path.append('../')
import utils

class MPL(nn.Module):
    def __init__(self,**kwargs):
        super(MPL,self).__init__(**kwargs)
        n_input=784 #28x28
        self.hide_lay=nn.Linear(in_features=784,out_features=256)
        self.hide_relu=nn.ReLU()
        self.output_lay=nn.Linear(256,10)

    def forward(self, x):
        x=self.hide_lay(x)
        x=self.hide_relu(x)
        x=self.output_lay(x)
        return x

from collections import OrderedDict
class MySequential(nn.Module):
    def __init__(self,*args):
        super(MySequential,self).__init__()
        if len(args)==0 and isinstance(args[0],OrderedDict):
            for key,module in args[0]:
                self.add_module(key,module)
        else:
            for i,module in enumerate(args):
                self.add_module(str(i),module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

def test_mpl():
    x = torch.rand(2,784)
    net=MPL()
    print(net)
    print(net(x))

def test_mysequential():
    net=MySequential(
        nn.Linear(784,256),
        nn.ReLU(),
        nn.Linear(256,10)
    )
    print(net)
    x = torch.rand(2,784)
    print(net(x))

def test_modulelist():
    net=nn.ModuleList([nn.Linear(784,256),nn.ReLU()])
    net.append(nn.Linear(256,10))
    print(net)
    print(net[-1])
if __name__=='__main__':
    # test_mpl()
    # test_mysequential()
    test_modulelist()


