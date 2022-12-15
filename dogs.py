import pickle
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler

from torchmodels import FlattenLayer, set_parameters_grad, Lambda
from testing import *
from analytic import *
from c2st import *
from classification import *





def compare_convs_dogs(device='cpu', n_runs=100, alpha=0.05, out_path='results/dogs_nets.pkl'):
    paths_to_models = [
            'models/supervised_dogs_res50_ep100_aug.pt',
            'models/supervised_dogs_vgg11_ep100_aug.pt'
            ]
    d = 10

    tests = {
            'vgg11-imagenet':get_vgg_test('vgg11', device=device, d=d, aggreg='avg'),
            'vgg11-scratch':get_vgg_test(paths_to_models[1], device=device, d=d, aggreg='avg'),
            'res50-imagenet':get_res_test('res50', device=device, d=d, aggreg='avg'),
            'res50-scratch':get_res_test(paths_to_models[0], device=device, d=d, aggreg='avg'),
            'c2st-def':get_c2st_test(device=device),
            'c2st-vgg11':get_c2st_vgg_test(path=None, device=device),
            'c2st-vgg11-transfer':get_c2st_vgg_test(path='vgg11', device=device),
            'c2st-vgg11-transfer-imagenet':get_c2st_vgg_test(path=paths_to_models[1], device=device),
            'c2st-res50':get_c2st_res_test(path=None, device=device),
            'c2st-res50-transfer':get_c2st_res_test(path='res50', device=device),
            'c2st-res50-transfer-imagenet':get_c2st_res_test(path=paths_to_models[0], device=device),
            }
    data = DogData()
    data_params = ('m', [10, 15, 20, 25, 50, 75, 100])

    results = benchmark_pipe(
            data,
            tests,
            data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True
            )

    pickle.dump(results, open(out_path, 'wb'))

    return results



def run_dogs_experiment(alpha=0.05, n_runs=100, device='cpu', out_path='results/power_dogs.pkl'):
    paths_to_models = [
            'models/supervised_dogs_res50_ep100_aug.pt',
            'models/supervised_dogs_vgg11_ep100_aug.pt'
            ]
    d = 10

    tests = {
            'vgg11-imagenet':get_vgg_test('vgg11', device=device, d=d, aggreg='avg'),
            'vgg11-scratch':get_vgg_test(paths_to_models[1], device=device, d=d, aggreg='avg'),
            'res50-imagenet':get_res_test('res50', device=device, d=d, aggreg='avg'),
            'res50-scratch':get_res_test(paths_to_models[0], device=device, d=d, aggreg='avg'),
            }

    analytic_tests = get_all_tests(alpha=alpha, Js=[1], split_ratios=[0.5])
    tests.update(analytic_tests)

    data = DogData()

    data_params = ('m', [10, 15, 20, 25, 50, 75, 100])

    results = benchmark_pipe(
            data,
            tests,
            data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True
            )

    pickle.dump(results, open(out_path, 'wb'))

    return results


def train_model(model, epochs=10, bs=64, lr=0.01, device='cpu', save_str='', var=False):

    target_shape = 224
    train, val = get_dogs_data(bs=bs, target_shape=target_shape, var=var)
    n_classes = 118

    #model = get_vgg(vgg=vgg, n_classes=n_classes).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    for epoch in range(1, epochs+1):
        print('starting epoch %d/%d' % (epoch, epochs))
        scheduler.step()
        train_one_epoch(model, opt, train, device=device)
        val_acc = evaluate(model, val, device=device)
        if epoch % 5 == 0:
            torch.save({'model':model, 'val_acc':val_acc}, save_str % epoch)

def train_vgg(vgg=11, epochs=10, bs=64, device='cpu'):
    model = get_vgg(vgg=vgg, n_classes=118).to(device)
    save_str = 'supervised_dogs_vgg' + str(vgg) + '_ep%d_aug.pt'
    train_model(model, epochs=epochs, bs=bs, device=device, save_str=save_str)
    return model

def train_res(res=18, epochs=10, bs=64, device='cpu', lr=0.01):
    model = get_resnet(res=res, n_classes=118).to(device)
    save_str = 'supervised_dogs_res' + str(res) + '_ep%d_var.pt'
    train_model(model, epochs=epochs, bs=bs, lr=lr, device=device, save_str=save_str, var=True)
    return model

def train_dense(epochs=10, bs=64, device='cpu', lr=0.01):
    model = get_dense(n_classes=118).to(device)
    save_str = 'supervised_dogs_dense121_ep%d_aug.pt'
    train_model(model, epochs=epochs, bs=bs, lr=lr, device=device, save_str=save_str)
    return model


def train_one_epoch(model, optimizer, loader, device='cpu'):
    model.train()
    running_loss = 0.
    criterion = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 5 == 0:
            print('running loss@%d: %.4f' % (i, running_loss/(i+1)))


def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        val_accuracy = correct / len(loader.sampler.indices)
    print('validation accuracy: %.4f' % (val_accuracy))
    return val_accuracy

def get_dense(dense=121, n_classes=118):
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(1024, n_classes, bias=True)
    return model

def get_resnet(res=50, n_classes=118):
    if res == 18:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
    if res == 34:
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
    if res == 50:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
    return model

def get_vgg(vgg=11, n_classes=118):
    model = models.vgg11_bn(pretrained=False)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=n_classes, bias=True)
    return model

def get_vgg_test(path=None, device='cpu', d=10, aggreg=None):
    if path == 'vgg11':
        model = models.vgg11_bn(pretrained=True)
    elif path == 'vgg13':
        model = models.vgg13_bn(pretrained=True)
    elif path == 'vgg16':
        model = models.vgg16_bn(pretrained=True)
    elif path == 'vgg19':
        model = models.vgg19_bn(pretrained=True)
    else:
        dic = torch.load(path, map_location='cpu')
        model = dic['model']

    if aggreg is None or aggreg=='pca':
        model = nn.Sequential(
            model.features,
            FlattenLayer(),
            model.classifier[:4],
            nn.Tanh()
            )
    elif aggreg == 'max':
        model = nn.Sequential(
                model.features,
                FlattenLayer(),
                model.classifier[:4],
                Lambda(lambda x: x.view(len(x), -1, 1)),
                nn.AdaptiveMaxPool2d((d, 1)),
                FlattenLayer(),
                nn.Tanh()
                )
    elif aggreg == 'avg':
        model = nn.Sequential(
                model.features,
                FlattenLayer(),
                model.classifier[:4],
                Lambda(lambda x: x.view(len(x), -1, 1)),
                nn.AdaptiveAvgPool2d((d, 1)),
                FlattenLayer(),
                nn.Tanh()
                )
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    test = TorchmodelTest(model=model, d=d, device=device, pca=(aggreg=='pca'))
    return test

def get_res_clf(path='res18', device='cpu'):
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

    model.fc = FlattenLayer()
    model = nn.Sequential(model, Lambda(lambda x: x[:, 0]), nn.Tanh())
    #model.fc = Lambda(lambda x: x.view(len(x), -1, 1))
    #model = nn.Sequential(model, nn.AdaptiveAvgPool2d((1, 1)), FlattenLayer(), nn.Tanh())

    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    classifier = TorchClassificationTest(model=model, device=device)
    return classifier


def get_tc2st_res_test(path='res18', device='cpu'):
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
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    test = TransferC2ST(model, model_d, device=device, reshape=None)
    return test


def get_res_test(path='res18', device='cpu', d=10, aggreg=None):
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

    if aggreg is None or aggreg=='pca':
        model.fc = nn.Tanh()
    elif aggreg == 'max':
        model.fc = Lambda(lambda x: x.view(len(x), -1, 1))
        model = nn.Sequential(model, nn.AdaptiveMaxPool2d((d, 1)), FlattenLayer(), nn.Tanh())
    elif aggreg == 'avg':
        model.fc = Lambda(lambda x: x.view(len(x), -1, 1))
        model = nn.Sequential(model, nn.AdaptiveAvgPool2d((d, 1)), FlattenLayer(), nn.Tanh())
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    test = TorchmodelTest(model=model, d=d, device=device, pca=(aggreg=='pca'))
    return test

def get_dense_test(path, device='cpu', d=10):
    dic = torch.load(path, map_location='cpu')
    model = dic['model']
    model.classifier = nn.Tanh()
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    test = TorchmodelTest(model=model, d=d)
    return test

def get_dogs_data(bs=16, target_shape=224, train_ratio=0.8, var=False):
    if var:
        path = 'data/dogs-without2-var2'
    else:
        path = 'data/dogs-without2'
    transform_tr = transforms.Compose([
        transforms.Resize((target_shape, target_shape)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        # same normalization as normal vgg for comparison
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    transform_val = transforms.Compose([
        transforms.Resize((target_shape, target_shape)),
        transforms.ToTensor(),
        # same normalization as normal vgg for comparison
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    tset = torchvision.datasets.ImageFolder(path, transform=transform_tr)
    vset = torchvision.datasets.ImageFolder(path, transform=transform_val)
    n = len(tset)
    n_train = int(n*train_ratio)
    indices = np.arange(n)
    np.random.shuffle(indices)
    ind_tr, ind_val = indices[:n_train], indices[n_train:]
    sampler_tr = SubsetRandomSampler(ind_tr)
    sampler_val = SubsetRandomSampler(ind_val)

    loader_train = torch.utils.data.DataLoader(tset, batch_size=bs, sampler=sampler_tr, num_workers=12)
    loader_val = torch.utils.data.DataLoader(vset, batch_size=bs, sampler=sampler_val, num_workers=12)
    
    return loader_train, loader_val

def get_c2st_test(device='cpu'):
    model_type = DefaultModel
    model_params = {'d':(224*224*3), 'h':1000}
    reshape = (-1, 224*224*3)
    T = Classifier2SampleTest(model_type, model_params, device=device, reshape=reshape)
    return T

def get_c2st_vgg_test(path='vgg11', device='cpu'):
    model_type = PretrainedVGG11
    model_params = {'path':path}
    T = Classifier2SampleTest(model_type, model_params, device=device)
    return T

def get_c2st_res_test(path='res50', device='cpu'):
    model_type = PretrainedRes50
    model_params = {'path':path}
    T = Classifier2SampleTest(model_type, model_params, device=device)
    return T

class PretrainedRes50(nn.Module):
    def __init__(self, path):
        super(PretrainedRes50, self).__init__()
        if path is None:
            model = models.resnet50(pretrained=False)
        elif path == 'res50':
            model = models.resnet50(pretrained=True)
        else:
            model = torch.load(path, map_location='cpu')['model']
        model.fc = nn.Linear(2048, 1, bias=True)
        self.model = nn.Sequential(
                model,
                nn.Sigmoid()
                )
    def forward(self, x):
        return self.model(x)

class PretrainedVGG11(nn.Module):
    def __init__(self, path):
        super(PretrainedVGG11, self).__init__()
        if path is None:
            model = models.vgg11_bn(pretrained=False)
        elif path == 'vgg11':
            model = models.vgg11_bn(pretrained=True)
        else:
            model = torch.load(path, map_location='cpu')['model']

        self.model = nn.Sequential(
                model.features,
                FlattenLayer(),
                model.classifier[:-1],
                nn.Linear(4096, 1),
                nn.Sigmoid()
                )
    def forward(self, x):
        return self.model(x)



