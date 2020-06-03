import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(1024, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool1(nn.ReLU(self.conv1(x)))
        x = self.pool2(nn.ReLU(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = nn.Softmax(self.fc3(x))
        return x

#######
if __name__ == '__main__':
    print('hello world')
    net = AlexNet()
    torch.save(net,'test.pth')