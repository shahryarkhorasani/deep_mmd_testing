''' This module is a mess - needs refactoring '''

import abc
from abc import  abstractmethod
import scipy
from scipy import stats
import numpy as np
from copy import deepcopy

from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class TestData:
    '''Base class for all data modules
    
    Every subclass needs to implement a .get_data(H0) method, that takes a
    boolean (whether to sample from H0 or H1) and outputs two sets of observations
    from the two distributions to be tested. Output should be two np.ndarray's of
    size (m, *d) where m = #observations/population, d = dimensionality (might be
    non-scalar, e.g. d = (3, 128, 128) for images, d = 10 for 10-dim features)
    '''
    def __init__(self):
        pass

    def reset_seed(self):
        np.random.seed(self.seed)

    def set_seed(self, seed):
        self.seed = seed

    @abstractmethod
    def get_data(self, H0=True):
        pass

class MVNormalData(TestData):
    '''Testing on two multivariate normal distributions with mean 0
    
    x ~ N(0, I)
    y ~ N(0, I) [under H0]
    y ~ N(0, Sigma), where Sigma = Sigma(h1eps) != I [under H1]
    
    '''
    def __init__(self, m=200, d=10, h1eps=1e-1, seed=123, h1_type=2):
        self.m = m
        self.d = d
        self.seed = seed
        self.reset_seed()
        self.params0 = {'mean':np.zeros(d), 'cov':np.eye(d), 'size':m}
        self.c0_data = np.empty(10*m)

        if h1_type == 0:
            cov_alt = np.eye(d)
            x = np.arange(d-1)
            cov_alt[x, x+1] = cov_alt[x+1, x] = h1eps
        elif h1_type == 1:
            cov_alt = (1+h1eps)*np.eye(d)
        elif h1_type == 2:
            cov_alt = np.eye(d)
            cov_alt[0,0] = 1 + h1eps
        self.params1 = {'mean':np.zeros(d), 'cov':cov_alt, 'size':m}

    def get_data(self, H0=True):
        X = np.random.multivariate_normal(**self.params0)
        if H0:
            Y = np.random.multivariate_normal(**self.params0)
        else:
            Y = np.random.multivariate_normal(**self.params1)
        return X, Y
         

class IndData(TestData):
    '''Independence testing on 2-variate data

    x = N(0,1)
    eps = N(0, gamma^2)
    y = cos(delta*x) + eps [under H1]
    y = x [under H0]
    see: "Revisiting Classifier 2-sample tests"

    '''
    def __init__(self, gamma=0.25, delta=1., m=200, seed=123):
        self.m = m
        self.gamma = gamma
        self.delta = delta
        self.seed = seed
        self.reset_seed()
        self.c0_data = np.empty(2*m)

    def get_data(self,H0=True):
        X1 = np.random.normal(loc=0, scale=1, size=(self.m, 1))
        eps = np.random.normal(loc=0, scale=self.gamma, size=(self.m, 1))
        X2 = np.cos(self.delta*X1) + eps
        X = np.hstack([X1, X2])
        if H0:
            Y1 = np.random.normal(loc=0, scale=1, size=(self.m, 1))
            eps = np.random.normal(loc=0, scale=self.gamma, size=(self.m, 1))
            Y2 = np.cos(self.delta*X1) + eps
            Y = np.hstack([X1, X2])
        else:
            randperm = np.random.permutation(self.m)
            Y = np.hstack([X1, X2[randperm]])
        return X, Y

class TorchData(TestData):
    def __init__(self, m=200, seed=123):
        super(TorchData, self).__init__()

        self.m = m
        self.seed = seed
        self.reset_seed()

    def get_data(self, H0=True):
        perm0 = torch.randperm(len(self.c0_data))
        X = self.c0_data[perm0[:self.m]]
        if H0:
            Y = self.c0_data[perm0[self.m:(2*self.m)]]
        else:
            perm1 = torch.randperm(len(self.c1_data))
            Y = self.c1_data[perm1[:self.m]]
        return np.array(X), np.array(Y)

class PlanesData(TorchData):
    def __init__(self, m=20, c0='A300', c1='C-47', seed=123, gray=False, target_shape=(48,34)):
        super(PlanesData, self).__init__(m=m, seed=seed)
        self.target_shape = target_shape
        if c0 == 'A300' and c1 == 'A330':
            self.path = 'data/aircrafts/fgvc-aircraft-2013b/testing_val/'
        elif c0 == 'Boeing 737' and c1 == 'Boeing 747':
            self.path = 'data/aircrafts/fgvc-aircraft-2013b/testing_test/'
        else:
            self.path = 'data/aircrafts/fgvc-aircraft-2013b/testing/'
        self.gray = False
        self.c0_data = self.load_class(c0)
        self.c1_data = self.load_class(c1)

    def load_class(self, c):
        bs = 200
        if self.gray:
            transform = transforms.Compose([
                    transforms.Resize(self.target_shape),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else: 
            transform = transforms.Compose([
                    transforms.Resize(self.target_shape),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        tset = torchvision.datasets.ImageFolder(self.path, transform=transform)

        loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=8)
        all_class_c = []
        c = tset.classes.index(c)
        for data, target in tqdm(loader):
            all_class_c.append(data[target==c])
        return torch.cat(all_class_c)

class BirdsData(TorchData):
    def __init__(self, m=20, c0='161.Blue_winged_Warbler', c1='167.Hooded_Warbler', seed=123, gray=False, target_shape=(224,224)):
        super(BirdsData, self).__init__(m=m, seed=seed)
        self.target_shape = target_shape
        self.gray = gray
        self.c0_data = self.load_class(c0)
        self.c1_data = self.load_class(c1)

    def load_class(self, c):
        bs = 200
        if self.gray:
            transform = transforms.Compose([
                    transforms.Resize(self.target_shape),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            transform = transforms.Compose([
                    transforms.Resize(self.target_shape),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        tset = torchvision.datasets.ImageFolder('./data/birds/CUB_200_2011/images/', transform=transform)

        loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=8)
        all_class_c = []
        c = tset.classes.index(c)
        for data, target in tqdm(loader):
            all_class_c.append(data[target==c])
        return torch.cat(all_class_c)


class FMNISTData(TorchData):
    def __init__(self, m=200, c0=1, c1=7, seed=123):
        super(MNISTData, self).__init__(m=m, seed=seed)
        self.c0_data = get_fmnist_class(c0)
        self.c1_data = get_fmnist_class(c1)
        self.d = 28*28

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)


class STL10Data(TorchData):
    '''3 & 5 = cats & dogs'''
    def __init__(self, m=200, c0=0, c1=1, seed=123):
        super(STL10Data, self).__init__(m=m, seed=seed)
        self.c0_data = get_stl10_class(c0)
        self.c1_data = get_stl10_class(c1)
        self.d = 3*96*96

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)

        self.seed = seed
        self.reset_seed()

class CIFAR100Data(TorchData):
    def __init__(self, m=200, c0=0, c1=1, seed=123):
        super(CIFAR100Data, self).__init__(m=m, seed=seed)
        self.c0_data = get_cifar_class(c0, CIFAR100=True)
        self.c1_data = get_cifar_class(c1, CIFAR100=True)
        self.d = 3*32*32

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)

        self.seed = seed
        self.reset_seed()

class CIFAR10Data(TorchData):
    def __init__(self, m=200, c0=0, c1=1, seed=123):
        super(CIFAR10Data, self).__init__(m=m, seed=seed)
        self.c0_data = get_cifar_class(c0, CIFAR100=False)
        self.c1_data = get_cifar_class(c1, CIFAR100=False)
        self.d = 3*32*32

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)

        self.seed = seed
        self.reset_seed()

class MNISTData(TorchData):
    def __init__(self, m=200, c0=1, c1=7, seed=123):
        super(MNISTData, self).__init__(m=m, seed=seed)
        self.c0_data = get_mnist_class(c0)
        self.c1_data = get_mnist_class(c1)
        self.d = 28*28

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)

class TinyImagenet(TorchData):
    def __init__(self, m=200, target_shape=(224, 224), normalization=None, seed=123):
        super(TinyImagenet, self).__init__(m=m, seed=seed)
        self.target_shape = target_shape
        if normalization is None:
            self.normalization = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        else:
            self.normalization = normalization
        self.c0_data, self.c1_data = self.get_classes()

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)
    def get_dim(self):
        return 224*224*3

    def get_classes(self):
        bs = 512
        transform = transforms.Compose([
            transforms.Resize(self.target_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization['mean'],
                                    std=self.normalization['std'])
            ])
        tset = torchvision.datasets.ImageFolder('data/imagenet-tiny2/', transform=transform)
        loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=12)
        all_class_c0 = []
        all_class_c1 = []
        k = 0
        for data, target in loader:
            all_class_c0.append(data[target==0])
            all_class_c1.append(data[target==1])
            k += 1
            if k > 4:
                break
        return torch.cat(all_class_c0), torch.cat(all_class_c1)
 
class XrayData(TorchData):
    def __init__(self, m=200, target_shape=(128, 128), normalization=None, seed=123):
        super(XrayData, self).__init__(m=m, seed=seed)
        self.target_shape = target_shape
        if normalization is None:
            self.normalization = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        else:
            self.normalization = normalization
        self.c0_data, self.c1_data = self.get_classes()

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)
    def get_dim(self):
        return 128*128*3

    def get_classes(self):
        bs = 512
        transform = transforms.Compose([
            transforms.Resize(self.target_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization['mean'],
                                    std=self.normalization['std'])
            ])
        tset = torchvision.datasets.ImageFolder('data/xray/', transform=transform)
        loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=12)
        all_class_c0 = []
        all_class_c1 = []
        k = 0
        for data, target in loader:
            all_class_c0.append(data[target==0])
            all_class_c1.append(data[target==1])
            k += 1
            if k > 4:
                break
        return torch.cat(all_class_c0), torch.cat(all_class_c1)
     
class DogData(TorchData):
    def __init__(self, m=200, target_shape=(224, 224), seed=123, gray=False, normalize=True, var=False):
        super(DogData, self).__init__(m=m, seed=seed)
        self.target_shape = target_shape
        self.c0_data, self.c1_data = get_2dog_data(target_shape=target_shape, gray=gray, normalize=normalize, var=var)


class MalariaData(TorchData):
    def __init__(self, m=200, c0=1, c1=7, target_shape=(128, 128), normalization=None, seed=123):
        super(MalariaData, self).__init__(m=m, seed=seed)
        self.target_shape = target_shape
        if normalization is None:
            self.normalization = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
        else:
            self.normalization = normalization
        self.c0_data, self.c1_data = self.get_classes()

        assert 2*m <= len(self.c0_data)
        assert 2*m <= len(self.c1_data)
    def get_dim(self):
        return 136*136*3

    def get_classes(self):
        bs = 512
        transform = transforms.Compose([
            transforms.Resize(self.target_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization['mean'],
                                    std=self.normalization['std'])
            ])
        tset = torchvision.datasets.ImageFolder('data/cell_images/', transform=transform)
        loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=12)
        all_class_c0 = []
        all_class_c1 = []
        k = 0
        for data, target in loader:
            all_class_c0.append(data[target==0])
            all_class_c1.append(data[target==1])
            k += 1
            if k > 10:
                break
        return torch.cat(all_class_c0), torch.cat(all_class_c1)

def get_2dog_data(target_shape=(224, 224), gray=False, normalize=True, var=False):
    '''alternatively: siberian huskey vs malamute or belgian malinois vs german shepherd'''
    if var:
        path = 'data/two-dogs-var2'
    else:
        path = 'data/two-dogs'
    bs = 1024
    if gray:

        transform = transforms.Compose([
            transforms.Resize(target_shape),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    else:

        if normalize:
            transform = transforms.Compose([
                transforms.Resize(target_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(target_shape),
                transforms.ToTensor(),
                ])
 
    tset = torchvision.datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=12)
    all_class_0 = []
    all_class_1 = []
    for data, target in loader:
        all_class_0.append(data[target==0])
        all_class_1.append(data[target==1])
    return torch.cat(all_class_0), torch.cat(all_class_1)

class FacesData(TorchData, object):
    def __init__(self, m=200, target_shape=(224, 224), seed=123, crop=True, gray=False):
        #super(FacesData, self).__init__(m=m, seed=seed)
        self.m = m
        self.target_shape = target_shape

        bs = 1024
        if crop:
            if gray:
                transform = transforms.Compose([
                    transforms.CenterCrop((762-300, 562-200)),
                    transforms.Resize(target_shape),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            else:
                transform = transforms.Compose([
                    transforms.CenterCrop((762-300, 562-200)),
                    transforms.Resize(target_shape),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
 
        else:
            transform = transforms.Compose([
                transforms.Resize(target_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
 
        tset = torchvision.datasets.ImageFolder('data/faces', transform=transform)
        loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=12)
        all_class_0 = []
        all_class_1 = []
        for data, target in loader:
            all_class_0.append(data[target==0])
            all_class_1.append(data[target==1])
        self.c0_data = torch.cat(all_class_0)
        self.c1_data = torch.cat(all_class_1)


def get_cifar_class(c=0, CIFAR100=False):
    bs = 1024
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if CIFAR100:
        tset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    else:
        tset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=8)
    all_class_c = []
    for data, target in loader:
        all_class_c.append(data[target==c])
    return torch.cat(all_class_c)

def get_stl10_class(c=0):
    bs = 1024
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    tset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=8)
    all_class_c = []
    for data, target in loader:
        all_class_c.append(data[target==c])
    return torch.cat(all_class_c)

def get_mnist_class(c=0):
    bs = 1024
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    tset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=8)
    all_class_c = []
    for data, target in loader:
        all_class_c.append(data[target==c])
    return torch.cat(all_class_c)

def get_fmnist_class(c=0):
    bs = 1024
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    tset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True, num_workers=8)
    all_class_c = []
    for data, target in loader:
        all_class_c.append(data[target==c])
    return torch.cat(all_class_c)





