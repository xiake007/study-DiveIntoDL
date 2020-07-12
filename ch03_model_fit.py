import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.data.dataloader
import numpy as np
import utils

class PolyFeatures(object):
    def __init__(self):
        self.n_train=100
        self.n_epochs=100
        self.true_w=[1.2,-3.4,5.6]
        self.true_b=5
        self.gen_data()


    def gen_data(self):
        # y=1.2x-3.4x^2+5.6x^3+5+sigma
        true_w=self.true_w
        true_b=self.true_b
        features=torch.randn((self.n_train*2,1))
        poly_features=torch.cat((features,torch.pow(features,2),torch.pow(features,3)),dim=1)
        labels=poly_features[:,0]*true_w[0]+poly_features[:,1]*true_w[1]+poly_features[:,2]*true_w[2]+true_b
        labels += torch.tensor(np.random.normal(0.,0.01,size=labels.size()),dtype=torch.float)
        # print(features[:2],poly_features[:2],labels[:2])
        self.features=features
        self.poly_features=poly_features
        self.labels=labels

    def loss(self):
        return nn.MSELoss()

    def fit_and_plot(self,train_features,test_features,train_labels,test_labels):
        net = nn.Linear(train_features.shape[-1],1)
        batch_size=min(10,train_features.shape[0])
        dataset=torch.utils.data.TensorDataset(train_features,train_labels)
        train_iter=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)
        optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
        train_loss,test_loss=[],[]
        for epoch in range(self.n_epochs):
            for X,y in train_iter:
                loss = self.loss()(net(X),y.view(-1,1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_labels = train_labels.view(-1,1)
            test_labels = test_labels.view(-1,1)
            train_loss.append(self.loss()(net(train_features),train_labels).mean().item())
            test_loss.append(self.loss()(net(test_features),test_labels).mean().item())
        print(f'final epoch train loss:{train_loss[-1]},test loss:{test_loss[-1]}')

        utils.semilogy(range(1,self.n_epochs+1),train_loss,'epoch','loss', \
                      range(1,self.n_epochs+1),test_loss,['train','test'])

        print('gt_weight: {}, gt_bias:{}'.format(self.true_w,self.true_b))
        print('pre_weight: {},pre_bias:{}'.format(net.weight.data.numpy(),net.bias.data.numpy()))

    def train_underfit(self):
        self.fit_and_plot(self.features[:self.n_train,:],self.features[self.n_train:,:], \
                          self.labels[:self.n_train],self.labels[self.n_train:])
    def train_normal(self):
        self.fit_and_plot(self.poly_features[:self.n_train,:],self.poly_features[self.n_train:,:],\
                          self.labels[:self.n_train],self.labels[self.n_train:])
    def train_overfit(self):
        self.fit_and_plot(self.poly_features[:2,:],self.poly_features[self.n_train:,:], \
                          self.labels[:2],self.labels[self.n_train:])

def test():
    obj=PolyFeatures()
    # obj.train_normal()
    # obj.train_underfit()
    obj.train_overfit()


if __name__ == '__main__':
    test()










def test():
    obj=PolyFeatures()
    obj.gen_data()

if __name__=='__main__':
    test()
