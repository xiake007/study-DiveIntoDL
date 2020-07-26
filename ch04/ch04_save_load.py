import torch
import torch.nn as nn

def test_rw_tensor():
    x=torch.ones(4)
    torch.save(x,'x.pt')
    print(torch.load('x.pt'))
    y=torch.zeros(3)
    torch.save([x,y],'xy.pt')
    print(torch.load('xy.pt'))
    torch.save({'x':x,'y':y},'xy_dict.pt')
    print(torch.load('xy_dict.pt'))

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.linear1=nn.Linear(4,3)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(3,2)

    def forward(self, x):
        x=self.linear1(x)
        return self.linear2(self.relu(x))

    def opitim(self):
        return torch.optim.SGD(self.parameters(),lr=0.01,momentum=0.9)

def test_rw_model():
    net=MLP()
    print(net.state_dict())
    print(net.opitim().state_dict())

    x=torch.rand(2,4)
    y=net(x)
    torch.save(net,'net.pth')
    net2=torch.load('net.pth')
    y2=net2(x)
    print(y==y2)

# test_rw_tensor()

test_rw_model()
