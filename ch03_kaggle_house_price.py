import torch
import torch.nn as nn
import torch.utils.data as Data
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
        all_features[num_feature_index] = all_features[num_feature_index].apply(lambda x:(x-x.mean())/(x.std()))
        all_features[num_feature_index]=all_features[num_feature_index].fillna(0)
        all_features = pd.get_dummies(all_features,dummy_na=True)
        print(all_features.shape)
        n_train=self.train_data.shape[0]
        train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float)
        test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float)
        # train_labels=torch.tensor(self.train_data.SalePrice.values,dtype=torch.float)
        train_labels=torch.tensor(self.train_data.SalePrice.values,dtype=torch.float).view(-1,1)
        return train_features,test_features,train_labels

    def loss(self):
        return nn.MSELoss()

    def log_rmse(self,net,X,y):
        with torch.no_grad():
            pre_labels = torch.max(net(X),torch.tensor(1.))
            log_rmse=torch.sqrt(self.loss()(pre_labels.log(),y.log()))
        return log_rmse.item()

    def test_log(self):
        a=torch.tensor([[2.]]).log()
        b=torch.tensor([[3.]]).log()
        b2=torch.tensor([3.]).log()
        c=self.loss()(a,b)
        c2=self.loss()(a,b2)
        print(a,b,b2,c,c2)

    def net(self,n_input):
        net=nn.Linear(n_input,1)
        for param in net.parameters():
            nn.init.normal_(param,mean=0,std=0.01)
        return net

    def train(self,net,train_features,train_labels,test_features,test_labels,
              n_epochs,lr,weight_decay,batch_size):
        train_ls,test_ls=[],[]
        dataset=Data.TensorDataset(train_features,train_labels)
        train_iter=Data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
        optimization=torch.optim.Adam(params=net.parameters(),lr=lr,weight_decay=weight_decay)
        net=net.float()
        for epoch in range(n_epochs):
            for X,y in train_iter:
                loss = self.loss()(net(X.float()),y.float())
                optimization.zero_grad()
                loss.backward()
                optimization.step()
            train_ls.append(self.log_rmse(net,train_features,train_labels))
            if test_features is not None:
                test_ls.append(self.log_rmse(net,test_features,test_labels))
        return train_ls,test_ls

    def get_k_fold_data(self,k,i,X,y):
        size_per_fold = X.shape[0]//k
        train_data,train_label=None,None
        for j in range(k):
            idx=slice(j*size_per_fold,(j+1)*size_per_fold)
            X_part,y_part=X[idx,:],y[idx]
            if j==i:
                val_data,val_label=X_part,y_part
            elif train_data is None:
                train_data,train_label=X_part,y_part
            else:
                train_data=torch.cat((train_data,X_part),dim=0)
                train_label=torch.cat((train_label,y_part),dim=0)
        return train_data,train_label,val_data,val_label

    def k_fold(self,k,X_train,y_train,n_epochs,lr,weight_decay,batch_size):
        train_sum,val_sum=0,0
        for i in range(k):
            data = self.get_k_fold_data(k,i,X_train,y_train)
            net = self.net(X_train.shape[1])
            train_ls,val_ls=self.train(net,*data,n_epochs,lr,weight_decay,batch_size)
            train_sum += train_ls[-1]
            val_sum += val_ls[-1]
            if i==0:
                utils.semilogy(range(1,n_epochs+1),train_ls,'epoch','rmse-loss',
                               range(1,n_epochs+1),val_ls,['train','val'])
            print(f'{i} fold train rmse: {train_ls[-1]}, val rmse: {val_ls[-1]}')
        mean_train_ls,mean_val_ls=train_sum/k,val_sum/k
        print(f'{k} fold mean train rmse: {mean_train_ls}, val rmse: {mean_val_ls}')

    def train_k_fold(self):
        k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 16
        self.k_fold(k,self.train_features,self.train_labels,num_epochs,lr,
                    weight_decay,batch_size)

    def train_and_pred(self):
        k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 16
        net = self.net(self.train_features.shape[1])
        train_ls,_=self.train(net,self.train_features,self.train_labels,None,None,n_epochs=num_epochs,
                   lr=lr,weight_decay=weight_decay,batch_size=batch_size)
        utils.semilogy(range(1,1+num_epochs),train_ls,'epoch','loss')
        print('train rmse is: {}'.format(train_ls[-1]))
        preds=net(self.test_features).detach().numpy()
        self.test_data['SalePrice']=pd.Series(preds.reshape(1,-1)[0])
        submission=pd.concat([self.test_data['Id'],self.test_data['SalePrice']],axis=1)
        submission.to_csv('./submission.csv',index=False)

def test():
    obj=HousePrice()
    # obj.train_k_fold()
    obj.train_and_pred()
    # obj.test_log()
if __name__=='__main__':
    test()
