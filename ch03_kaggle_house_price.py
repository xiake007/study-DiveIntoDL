import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import utils
import os.path as osp

class HousePrice(object):
    def __init__(self):
        self.dataset_path='./dataset/kaggle_house'
        self.train_data,self.test_data,self.all_features=self.load_dataset()
        self.train_features,self.test_features,self.train_labels= self.preprocess_data()

    def load_dataset(self):
        train_data=pd.read_csv(osp.join(self.dataset_path,'train.csv'))
        test_data=pd.read_csv(osp.join(self.dataset_path,'test.csv'))
        print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])
        all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
        print(train_data.shape,test_data.shape,all_features.shape)
        return train_data,test_data,all_features

    def preprocess_data(self):
        all_features=self.all_features
        num_feature_index=all_features.dtypes[all_features.dtypes!='object'].index
        all_features[num_feature_index] = all_features[num_feature_index].apply(lambda x:(x-x.mean())/x.std())
        all_features[num_feature_index]=all_features[num_feature_index].fillna(0)
        self.all_features = pd.get_dummies(all_features,dummy_na=True)
        print(all_features.shape)
        n_train=self.train_data.shape[0]
        train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float)
        test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float)
        train_labels=torch.tensor(self.train_data.SalePrice.values,dtype=torch.float)
        return train_features,test_features,train_labels

    def loss(self,net,X,y):
        mse_ls=nn.MSELoss()
        pre_labels = torch.max(net(X),torch.tensor(1.))
        log_rmse=torch.sqrt(mse_ls(pre_labels.log(),y.log))
        return log_rmse.item()

    def net(self,n_input):
        net=nn.Linear(n_input,1)
        for param in net.parameters():
            nn.init.normal_(param,mean=0,std=0.01)

    def train(self):
        pass



def test():
    obj=HousePrice()
    obj.preprocess_data()
if __name__=='__main__':
    test()