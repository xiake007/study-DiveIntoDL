import utils
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

class WeightDecayOrg(object):
    def __init__(self):
        #y=\sum_{i=1}^p0.01*x_i + 0.05 +sigma
        self.n_train,self.n_test,self.n_intput=20,100,200
        self.true_w,self.true_b=torch.ones(self.n_intput,1)*0.01, 0.05
        self.batch_size,self.n_epochs,self.lr=1,100,0.003


    def generate_features(self):
        features=torch.randn((self.n_train+self.n_test,self.n_intput))
        labels=torch.matmul(features,self.true_w)+self.true_b
        labels += torch.tensor(np.random.normal(0.,0.01,size=labels.size()),dtype=torch.float)
        return features,labels

    def init_weights(self):
        w=torch.randn((self.n_intput,1), requires_grad=True)
        b=torch.zeros(1,requires_grad=True)
        return [w,b]

    def l2_penalty(self,w):
        return (w**2).sum()/2

    def net(self):
        return utils.linreg

    def loss(self):
        return utils.squared_loss

    def load_dataset(self):
        n_train=self.n_train
        features,labels=self.generate_features()
        train_features,test_features=features[:n_train,:],features[n_train:,:]
        train_labels,test_labels=labels[:n_train],labels[n_train:]
        dataset = Data.TensorDataset(train_features,train_labels)
        train_iter = Data.DataLoader(dataset,self.batch_size,shuffle=True)
        return train_iter,train_features,test_features,train_labels,test_labels

    def fit_and_plot(self,lambd):
        [w,b] = self.init_weights()
        train_iter,train_features,test_features,train_labels,test_labels = self.load_dataset()
        train_ls,test_ls=[],[]
        for _ in range(self.n_epochs):
            for X,y in train_iter:
                l = self.loss()(self.net()(X,w,b),y)+lambd*self.l2_penalty(w)
                l = l.sum()

                if w.grad is not None:
                    w.grad.data.zero_()
                    b.grad.data.zero_()
                l.backward()
                utils.sgd([w,b],self.lr,self.batch_size)
            train_ls.append(self.loss()(self.net()(train_features,w,b),train_labels).mean().item())
            test_ls.append(self.loss()(self.net()(test_features,w,b),test_labels).mean().item())
        utils.semilogy(range(1,self.n_epochs+1),train_ls,'epoch','loss',
                       range(1,self.n_epochs+1),test_ls,['train','test'])
        print('L2 norm of w:',w.norm().item())

    def fit_and_plot_pytorch(self,wd):
        net=nn.Linear(self.n_intput,1)
        nn.init.normal_(net.weight,mean=0.,std=1)
        nn.init.normal_(net.bias,mean=0.,std=1)

        optim_w=torch.optim.SGD(params=[net.weight],lr=self.lr,weight_decay=wd)
        optim_b=torch.optim.SGD(params=[net.bias],lr=self.lr)

        train_iter,train_features,test_features,train_labels,test_labels = self.load_dataset()

        train_ls,test_ls=[],[]
        for _ in range(self.n_epochs):
            for X,y in train_iter:
                l=self.loss()(net(X),y).mean()
                optim_w.zero_grad()
                optim_b.zero_grad()

                l.backward()
                optim_w.step()
                optim_b.step()
            train_ls.append(self.loss()(net(train_features),train_labels).mean().item())
            test_ls.append(self.loss()(net(test_features),test_labels).mean().item())
        utils.semilogy(range(1,self.n_epochs+1),train_ls,'epoch','loss',
                       range(1,self.n_epochs+1),test_ls,['train','test'])
        print('L2 norm w:',net.weight.data.norm().item())


def test():
    obj=WeightDecayOrg()
    # obj.fit_and_plot(lambd=0)
    # obj.fit_and_plot(lambd=3)
    # obj.fit_and_plot_pytorch(wd=0)
    obj.fit_and_plot_pytorch(wd=3)

if __name__=='__main__':
    test()


