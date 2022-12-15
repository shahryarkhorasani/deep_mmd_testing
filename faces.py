import pickle
from testing import *
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from dogs import get_res_test, get_vgg_test, get_c2st_test, get_c2st_vgg_test, get_c2st_res_test

from torchmodels import FlattenLayer, set_parameters_grad, Lambda
from analytic import *


### TODO prune from here
def compare_convs_faces(device='cpu', n_runs=100, alpha=0.05, out_path='results/faces_nets.pkl'):

    tests = {
            'res50-imagenet':get_res_test('res50', device=device, aggreg='avg'),
            'res152-imagenet':get_res_test('res152', device=device, aggreg='avg'),
            'vgg19-imagenet':get_vgg_test('vgg19', device=device, aggreg='avg'),
            'c2st-def':get_c2st_test(device=device),
            'c2st-vgg11':get_c2st_vgg_test(path=None, device=device),
            'c2st-vgg11-imagenet':get_c2st_vgg_test(path='vgg11', device=device),
            'c2st-res50':get_c2st_res_test(path=None, device=device),
            'c2st-res50-imagenet':get_c2st_res_test(path='res50', device=device),
            }
    data = FacesData(m=10, crop=False)
    data_params = ('m', [10, 15, 20, 25, 50, 75, 100, 150, 200])

    results = benchmark_pipe(
            data,
            tests,
            data_params=data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True)

    pickle.dump(results, open(out_path, 'wb'))
    return results


def run_face_experiments(alpha=0.05, n_runs=100, device='cpu', out_path='results/power_faces.pkl'):
    d = 10

    nn_tests = {
            'res50-imagenet':get_res_test('res50', device=device, aggreg='avg'),
            'res152-imagenet':get_res_test('res152', device=device, aggreg='avg'),
            'vgg19-imagenet':get_vgg_test('vgg19', device=device, aggreg='avg')
            }
    analytic_tests = get_all_tests(alpha=alpha, Js=[1], split_ratios=[0.5])

    # different preprocessing for deep and kernel tests
    data_crop = FacesData(m=10, crop=True)
    data_nocrop = FacesData(m=10, crop=False)

    data_params = ('m', [10, 15, 20, 25, 50, 75, 100, 150, 200])
    results_nn = benchmark_pipe(
            data_nocrop,
            nn_tests,
            data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True
            )

    results_ana = benchmark_pipe(
            data_crop,
            analytic_tests,
            data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True
            )
    results_nn.update(results_ana)

    pickle.dump(results_nn, open(out_path, 'wb'))
    return results_nn




def load_imdbwiki(path=None, d=10, device='cpu'):
    if path is None:
        path = '/home/matthias/Downloads/imdb-wiki/dex_imdb_wiki.caffemodel.pth'
    model = IMDB_Wiki()
    model.load_state_dict(torch.load(path), strict=False)
    model = nn.Sequential(
            model.features,
            FlattenLayer(),
            model.classifier,
            Lambda(lambda x: x.view(len(x), -1, 1)), 
            nn.AdaptiveAvgPool2d((d, 1)),
            FlattenLayer(),
            nn.Tanh())

    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()
    test = TorchmodelTest(model=model, d=d)
    return test

class IMDB_Wiki(nn.Module):
    def __init__(self,):
        super(IMDB_Wiki, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                )

        self.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096, bias=True),
                )

        self.clf = nn.Linear(4096, 101, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)
        x = self.classifier(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.clf(x)
        return F.softmax(x)
