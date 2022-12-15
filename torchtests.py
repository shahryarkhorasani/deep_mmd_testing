from abc import abstractmethod
import scipy
from scipy import stats
import numpy as np
from copy import deepcopy

from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from base import *


class TorchmodelTest(TwoSampleTest):
    '''Test using a torch.nn.Module network as a test'''
    def __init__(self, model, d='ada', reshape=None, device=None, pca=True):
        '''
        # Parameters
        model (torch.nn.Module): neural network to be used as feature extractor
        d (int or str): maximum number of neurons of last layer of model to be used. If d=='ada', choose d = sqrt(m).round()

        '''
        super(TorchmodelTest, self).__init__()
        self.d = d
        if device is None:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device('cuda' if use_cuda else 'cpu')
        else:
            self.device = device
        self.reshape = reshape
        self.pca = pca
        self.eps_ridge = 1e-8    # add ridge to Covariance for numerical stability

        self.n_perm = 1000
        
        self.model = model


    def _pca(self, data):
        if self.d == 'ada':
            d = np.sqrt(len(data)/2.).round().astype(int)
        else:
            d = self.d
        pca = PCA(n_components=d)
        return torch.from_numpy(pca.fit_transform(data))
        '''
        data = data - data.mean(0)
        U, S, V = torch.svd(data)
        U = U[:, :d]
        return U[:, :d] * S[:d]
        '''

    def compute_p_value(self, features_X, features_Y):
        n, d = features_X.shape
        m = len(features_Y)
        mean_fX = features_X.mean(0)
        mean_fY = features_Y.mean(0)
        D = mean_fX - mean_fY

        all_features = np.concatenate([features_X, features_Y])
        Cov_D = (1./n + 1./m) * np.cov(all_features.T) + self.eps_ridge * np.eye(d)

        statistic = D.dot(np.linalg.solve(Cov_D, D))
        p_val = 1. - stats.chi2.cdf(statistic, d)
        return p_val
    
    def compute_mmd(self, features_X, features_Y):

        #if self.mmd:
        mean_fX = features_X.mean(0)
        mean_fY = features_Y.mean(0)
        D = mean_fX - mean_fY
        statistic = np.linalg.norm(D)
        '''
        else:
            n, d = features_X.shape
            m = len(features_Y)
            all_features = np.concatenate([features_X, features_Y])
            mean_fX = all_features[:n].mean(0)
            mean_fY = all_features[n:].mean(0)
            D = mean_fX - mean_fY

            Cov_D = (1./n + 1./m) * np.cov(all_features.T) + self.eps_ridge * np.eye(d)

            statistic = D.dot(np.linalg.solve(Cov_D, D))
        '''
        return statistic
 

    def compute_p_value_rs(self, features_X, features_Y):
        stat = self.compute_mmd(features_X, features_Y)

        n, m = len(features_X), len(features_Y)
        l = n + m
        features_Z = np.vstack((features_X, features_Y))

        # compute null samples
        resampled_vals = np.empty(self.n_perm)
        for i in range(self.n_perm):
            index = np.random.permutation(l)
            feats_X, feats_Y = features_Z[index[:n]], features_Z[index[n:]]
            resampled_vals[i] = self.compute_mmd(feats_X, feats_Y)
        resampled_vals.sort()
        
        p_val = np.mean(stat < resampled_vals)
        return p_val


    def test_rs(self, X, Y):
        with torch.no_grad():
            X, Y = torch.Tensor(X), torch.Tensor(Y)
            if not self.reshape is None:
                X = X.view(*self.reshape)
                Y = Y.view(*self.reshape)
            self.model.eval()
            self.model = self.model.to(self.device)
            X = X.to(self.device)
            Y = Y.to(self.device)
            
            features_X = self.model(X).cpu().numpy()
            features_Y = self.model(Y).cpu().numpy()

        '''
        if self.pca:
            features_Z = np.vstack((features_X, features_Y))
            features_Z = self._pca(features_Z).numpy()
            features_X, features_Y = features_Z[:len(X)], features_Z[len(X):]
        '''

        return self.compute_p_value_rs(features_X, features_Y)

    def test(self, X, Y):
        if self.d == 'mmd':
            return self.test_rs(X, Y)
        else:
            return self.test_param(X, Y)

    def test_param(self, X, Y):
        with torch.no_grad():
            X, Y = torch.Tensor(X), torch.Tensor(Y)
            if not self.reshape is None:
                X = X.view(*self.reshape)
                Y = Y.view(*self.reshape)
            self.model.eval()
            self.model = self.model.to(self.device)
            X = X.to(self.device)
            Y = Y.to(self.device)
            
            features_X = self.model(X).cpu()
            features_Y = self.model(Y).cpu()

            if self.pca:
                feats_Z = torch.cat((features_X, features_Y))
                feats_Z = self._pca(feats_Z).numpy().astype(np.float64)
                features_X = feats_Z[:len(features_X)]
                features_Y = feats_Z[len(features_X):]
            else:
                features_X = features_X.numpy().astype(np.float64)
                features_Y = features_Y.numpy().astype(np.float64)

                n_feats = features_X.shape[1]
                index = np.arange(n_feats)
                np.random.shuffle(index)
                if self.d == 'ada':
                    d = np.sqrt(len(X)).round().astype(int)
                else:
                    d = self.d
                index = index[:d]

                features_X = features_X[:, index]
                features_Y = features_Y[:, index]

            return self.compute_p_value(features_X, features_Y)

class TMTSplit(TorchmodelTest):
    def __init__(self, model_type, model_params, eval_type='output', d='ada', reshape=None, device=None, pca=True):
        super(TMTSplit, self).__init__(None, d, reshape, device, pca)
        self.model_type = model_type
        self.model_params = model_params
        self.epochs = 100
        self.eval_type = eval_type

    def split_data(self, X, Y):
        perm_X = torch.randperm(self.n)
        perm_Y = torch.randperm(self.m)
        X_tr, X_te = X[perm_X[:(self.n//2)]], X[perm_X[(self.n//2):]]
        Y_tr, Y_te = Y[perm_Y[:(self.n//2)]], Y[perm_Y[(self.n//2):]]
        return (X_tr, Y_tr), (X_te, Y_te)

    def preprocess(self, X, Y):
        X, Y = torch.Tensor(X), torch.Tensor(Y)
        if not self.reshape is None:
            X = X.view(*self.reshape)
            Y = Y.view(*self.reshape)
        return X, Y

    def load_model(self):
        self.model = self.model_type(**self.model_params).to(self.device)
    def train_model(self, X, Y):
        Z = torch.cat((X, Y))
        labels = torch.cat((torch.zeros(len(X)), torch.ones(len(Y))))

        tset = torch.utils.data.TensorDataset(Z, labels)
        loader = torch.utils.data.DataLoader(tset, batch_size=64, shuffle=True)
        self.model.train()
        loss_func = nn.BCELoss()

        opt = optim.Adam(self.model.parameters())
        for ep in range(self.epochs):
            for i, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                opt.zero_grad()
                out = self.model(data)
                loss = loss_func(out.flatten(), target.flatten())
                loss.backward()
                opt.step()

    def eval_model(self, X, Y):
        self.model.eval()
        with torch.no_grad():
            if self.eval_type == 'embedding':
                model = next(self.model.children())
                model = model[:-2]
            elif self.eval_type == 'output':
                model = self.model
            pred_X = model(X.to(self.device)).cpu().numpy()
            pred_Y = model(Y.to(self.device)).cpu().numpy()
        return pred_X, pred_Y

    def test(self, X, Y):
        self.n, self.m = len(X), len(Y)
        X, Y = self.preprocess(X, Y)
        (X_tr, Y_tr), (X_te, Y_te) = self.split_data(X, Y)

        self.load_model()
        self.train_model(X_tr, Y_tr)

        features_X, features_Y = self.eval_model(X_te, Y_te)
        if self.d == 'mmd':
            p_val = self.compute_p_value_rs(features_X, features_Y)
        else:
            if self.eval_type == 'embedding':
                feats_Z = torch.cat((torch.from_numpy(features_X), torch.from_numpy(features_Y)))
                feats_Z = self._pca(feats_Z).numpy().astype(np.float64)
                features_X = feats_Z[:len(features_X)]
                features_Y = feats_Z[len(features_X):]
         
            p_val = self.compute_p_value(features_X, features_Y)
        return p_val

    
