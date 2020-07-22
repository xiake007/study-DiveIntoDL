import torch
import torch.utils.data
import matplotlib.pyplot as plt
import time,sys
import utils
from  utils import mnist_train,mnist_test
import numpy as np

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    utils.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def test_download_MNIST():
    print(type(mnist_train))
    print(len(mnist_train),len(mnist_test))
    feature,label=mnist_train[0]
    print(feature.shape,label)

def test_plot_10imgs():
    X,y=[],[]
    for i in range(10):
        X.append(mnist_train[i][0])
        y.append(mnist_test[i][1])
    y=get_fashion_mnist_labels(y)
    show_fashion_mnist(X,y)

def test_read_dataset_time():
    train_iter,_ = utils.create_dataloader()
    start=time.time()
    for X,y in train_iter:
        continue
    print('%.2f sec.'%(time.time()-start))

class SoftmaxOrg(object):
    def __init__(self):
        self.train_iter,self.test_iter=utils.load_data_fashion_mnist(256)
        self.input_num=28*28
        self.output_num=10
        self.W,self.b=self.init_model_param()

    def init_model_param(self):
        W = torch.tensor(np.random.normal(0,0.01,(self.input_num,self.output_num)),dtype=torch.float)
        b = torch.zeros(self.output_num,dtype=torch.float)
        W.requires_grad_(True)
        b.requires_grad_(True)
        return W,b

    def softmax(self,X):
        def test():
            X=torch.Tensor([[1,2,3],[4,5,6]])
            print(X.sum(dim=0,keepdim=True))
            print(X.sum(dim=1,keepdim=True))
        X_exp=X.exp()
        partition = X_exp.sum(dim=1,keepdim=True)
        return X_exp/partition

    def net(self,X):
        return self.softmax(torch.mm(X.view(-1,self.input_num),self.W)+self.b)

    def cross_entropy(self,y_hat,y):
        def test():
            y_hat=torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
            y=torch.LongTensor([0,2])
            return y_hat.gather(1, y.view(-1,1))
        return - torch.log(y_hat.gather(1,y.view(-1,1)))

    def accuracy(self,y_hat,y):
        return (y_hat.argmax(dim=1)==y).float().mean().item()
    def train(self):
        num_epoch,lr=5,0.1
        batch_size=256
        train_iter,test_iter=utils.load_data_fashion_mnist()
        utils.train_ch3(self.net,train_iter,test_iter,self.cross_entropy,num_epoch,batch_size,[self.W,self.b],lr)

    def test(self):
        _,test_iter=utils.load_data_fashion_mnist()
        X,y=iter(test_iter).next()
        true_labels=get_fashion_mnist_labels(y)
        pre_labels=get_fashion_mnist_labels(self.net(X).argmax(dim=1).numpy())
        titles=[true+'\n'+pre for true,pre in zip(true_labels,pre_labels)]
        show_fashion_mnist(X[0:9],titles[0:9])


def test_softmax_org():
    obj=SoftmaxOrg()
    x = torch.rand((2,5))
    print(x)
    # print(obj.softmax(x))
    y_hat=torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    y=torch.LongTensor([0,2])
    # print(obj.cross_entropy(y_hat,y))
    # print(obj.accuracy(y_hat,y))
    _,test_iter=utils.load_data_fashion_mnist()
    print(utils.evaluate_accuracy(test_iter,obj.net))
    # obj.train()
    # obj.test()

if __name__=='__main__':
    # test_download_MNIST()
    # test_plot_10imgs()
    # test_read_dataset_time()
    test_softmax_org()
