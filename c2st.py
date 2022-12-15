'''reimplementation of classifier two-sample test'''

from scipy import stats
#from testing import *
from base import *
#from audio import *

from sklearn.linear_model import LogisticRegression
# fix sklearn deprecated warning
import warnings
warnings.warn = lambda *args, **kwargs: None

import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchmodels import Lambda, FlattenLayer, set_parameters_grad


class Classifier2SampleTest(TwoSampleTest):
    def __init__(self, model_type, model_params={}, device='cpu', reshape=None):
        super(Classifier2SampleTest, self).__init__()
        self.model_type = model_type
        self.model_params = model_params
        self.device = device
        self.reshape = reshape
        self.epochs = 100

    def preprocess(self, X, Y):
        X, Y = torch.Tensor(X), torch.Tensor(Y)
        if not self.reshape is None:
            X = X.view(*self.reshape)
            Y = Y.view(*self.reshape)
        return X, Y

    def split_data(self, X, Y):
        perm_X = torch.randperm(self.n)
        perm_Y = torch.randperm(self.m)
        X_tr, X_te = X[perm_X[:(self.n//2)]], X[perm_X[(self.n//2):]]
        Y_tr, Y_te = Y[perm_Y[:(self.m//2)]], Y[perm_Y[(self.m//2):]]
        return (X_tr, Y_tr), (X_te, Y_te)

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
            pred_X = self.model(X.to(self.device)).cpu().numpy()
            pred_Y = self.model(Y.to(self.device)).cpu().numpy()
            correct_X = (pred_X < 0.5)
            correct_Y = (pred_Y >= 0.5)
            acc = np.concatenate((correct_X, correct_Y)).mean()
            return acc

    def compute_p_value(self, accuracy):
        p_val = 1. - stats.norm.cdf(accuracy, loc=0.5, scale=np.sqrt(0.25/self.n))
        return p_val

    def load_model(self):
        self.model = self.model_type(**self.model_params).to(self.device)
 
    def test(self, X, Y):
        self.n, self.m = len(X), len(Y)
        assert self.n == self.m
        self.d = X.shape[1]
        X, Y = self.preprocess(X, Y)
        (X_tr, Y_tr), (X_te, Y_te) = self.split_data(X, Y)

        self.load_model()

        self.train_model(X_tr, Y_tr)

        accuracy = self.eval_model(X_te, Y_te)
        p_val = self.compute_p_value(accuracy)
        return p_val

class TransferC2ST(Classifier2SampleTest):
    def __init__(self, pretrained_model, model_d, device='cpu', reshape=None):
        super(TransferC2ST, self).__init__(None, None, device=device, reshape=reshape)
        self.pre_model = pretrained_model.to(device)
        self.pre_model.eval()
        self.model_d = model_d

    def load_model(self):
        self.model = nn.Sequential(nn.Linear(self.model_d, 1), nn.Sigmoid())

    def train_model(self, X, Y):
        Z = torch.cat((X, Y))
        labels = torch.cat((torch.zeros(len(X)), torch.ones(len(Y))))
        with torch.no_grad():
            Z_feats = self.pre_model(Z.to(self.device)).cpu()
        
        tset = torch.utils.data.TensorDataset(Z_feats, labels)
        loader = torch.utils.data.DataLoader(tset, batch_size=64, shuffle=True)

        self.model.train()
        loss_func = nn.BCELoss()

        opt = optim.Adam(self.model.parameters())
        for ep in range(self.epochs):
            for i, (data, target) in enumerate(loader):
                opt.zero_grad()
                out = self.model(data)
                loss = loss_func(out.flatten(), target.flatten())
                loss.backward()
                opt.step()

    def eval_model(self, X, Y):
        self.model.eval()
        with torch.no_grad():
            feats_X = self.pre_model(X.to(self.device)).cpu()
            feats_Y = self.pre_model(Y.to(self.device)).cpu()
            pred_X = self.model(feats_X).numpy()
            pred_Y = self.model(feats_Y).numpy()
            correct_X = (pred_X < 0.5)
            correct_Y = (pred_Y >= 0.5)
            acc = np.concatenate((correct_X, correct_Y)).mean()
            return acc
    

class TransferLRC2ST(TransferC2ST):
    def __init__(self, pretrained_model, model_d, device='cpu', reshape=None):
        super(TransferLRC2ST, self).__init__(
                pretrained_model=pretrained_model,
                model_d=model_d,
                device=device,
                reshape=reshape)
        self.pre_model = pretrained_model.to(device)
        self.pre_model.eval()
        self.model_d = model_d

    def load_model(self):
        self.model = LogisticRegression(C=1.)

    def train_model(self, X, Y):
        Z = torch.cat((X, Y))
        with torch.no_grad():
            Z_feats = self.pre_model(Z.to(self.device)).cpu().numpy()

        labels = np.concatenate((np.zeros(len(X)), np.ones(len(Y))))
        self.model.fit(Z_feats, labels)

    def eval_model(self, X, Y):
        with torch.no_grad():
            feats_X = self.pre_model(X.to(self.device)).cpu().numpy()
            feats_Y = self.pre_model(Y.to(self.device)).cpu().numpy()
            pred_X = self.model.predict(feats_X)
            pred_Y = self.model.predict(feats_Y)
            correct_X = (pred_X < 0.5)
            correct_Y = (pred_Y >= 0.5)
            acc = np.concatenate((correct_X, correct_Y)).mean()
            return acc
 

class DefaultModel(nn.Module):
    def __init__(self, d, h=20):
        super(DefaultModel, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(d, h),
                nn.ReLU(True),
                nn.Linear(h, 1),
                nn.Sigmoid()
                )
    def forward(self, x):
        return self.net(x)

def get_tc2st_cae_test(path, model_d, device='cpu', lr=False):
    model = torch.load(path, map_location='cpu')
    model = nn.Sequential(model.encoder[:-1], FlattenLayer(), nn.Tanh())
    set_parameters_grad(model, False)
    model = model.to(device)
    model.eval()
    if lr:
        test = TransferLRC2ST(model, model_d, device=device, reshape=None)
    else:
        test = TransferC2ST(model, model_d, device=device, reshape=None)
    return test

def get_tc2st_res_test(path='res18', device='cpu', lr=False):
    if path == 'res18':
        model = models.resnet18(pretrained=True)
        model_d = 512
    elif path == 'res34':
        model = models.resnet34(pretrained=True)
        model_d = 512
    elif path == 'res50':
        model = models.resnet50(pretrained=True)
        model_d = 2048 
    elif path == 'res152':
        model = models.resnet152(pretrained=True)
        model_d = 2048 
    else:
        dic = torch.load(path, map_location='cpu')
        model = dic['model']
        model_d = 512

    model.fc = Lambda(lambda x: x)
    set_parameters_grad(model, False)
    model = model.to(device)
    model.eval()
    if lr:
        test = TransferLRC2ST(model, model_d, device=device, reshape=None)
    else:
        test = TransferC2ST(model, model_d, device=device, reshape=None)
    return test


