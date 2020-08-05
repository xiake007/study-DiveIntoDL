import torch
import numpy as np
import utils

def conv2d_multi_in(X,K):
    res = utils.corr2d(X[0,:,:],K[0,:,:])
    for i in range(1,X.shape[0]):
        res += utils.corr2d(X[i,:,:],K[i,:,:])
    return res

def test_multi_in():
    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    Y=conv2d_multi_in(X,K)
    print(Y)

def conv2d_multi_out(X,K):
    return torch.stack([conv2d_multi_in(X,k) for k in K])

def test_multi_out():
    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
    K=torch.stack([K,K+1,K+2])
    Y=conv2d_multi_out(X,K)
    print(Y.shape)
    print(Y)

def conv2d_multi_1x1(X,K):
    ci,h,w=X.shape
    co=K.shape[0]
    X=X.view(ci,h*w)
    K=K.view(co,ci)
    Y=torch.mm(K,X)
    return Y.view(co,h,w)

def test_conv2d_1x1():
    X=torch.rand(3,3,3)
    K=torch.rand(2,3,1,1)
    y1=conv2d_multi_1x1(X,K)
    y2=conv2d_multi_out(X,K)
    print((y1-y2).norm().item()<1e-5)
    print(y1.shape,y2.shape)

if __name__=='__main__':
    # test_multi_in()
    # test_multi_out()
    test_conv2d_1x1()

