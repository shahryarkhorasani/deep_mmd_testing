from tqdm import tqdm
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch import optim


from torchtests import TorchmodelTest

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        
        return x.view(len(x), -1)

def set_parameters_grad(model, requires_grad):
    '''update requires_grad for all paramters in model'''
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_cae_test(path, device='cpu', d='ada'):
    model = torch.load(path, map_location='cpu')
    model = nn.Sequential(model.encoder[:-1], FlattenLayer(), nn.Tanh())
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    T = TorchmodelTest(model=model, d=d, device=device, pca=True)
    return T

def get_res_test(path='res18', device='cpu', d='ada'):
    if path == 'res18':
        model = models.resnet18(pretrained=True)
    elif path == 'res34':
        model = models.resnet34(pretrained=True)
    elif path == 'res50':
        model = models.resnet50(pretrained=True)
    elif path == 'res152':
        model = models.resnet152(pretrained=True)
    else:
        dic = torch.load(path, map_location='cpu')
        model = dic['model']

    model.fc = nn.Tanh()
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    T = TorchmodelTest(model=model, d=d, device=device, pca=True)
    return T

def get_vgg_test(path='vgg11', device='cpu', d='ada'):
    if path == 'vgg11':
        model = models.vgg11_bn(pretrained=True)
    elif path == 'vgg13':
        model = models.vgg13_bn(pretrained=True)
    elif path == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    else:
        dic = torch.load(path, map_location='cpu')
        model = dic['model']
    model = nn.Sequential(model.features, FlattenLayer(), model.classifier[:4], nn.Tanh())
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    T = TorchmodelTest(model=model, d=d, device=device, pca=True)
    return T


