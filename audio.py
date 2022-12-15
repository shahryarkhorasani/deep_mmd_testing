import os
import pickle
import numpy as np
from scipy.io import wavfile

from torchmodels import FlattenLayer, Lambda
from torchtests import *
from base import *
from data import *
from analytic import *
from c2st import *

#from classification import *

from tqdm import tqdm

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch import optim



DATA_DIR_VAR = 'data/sound/taor'
#PATH_TO_1_VAR = os.path.join(DATA_DIR_VAR, '03 Gramatik - Bluestep (Album Version).wav')
PATH_TO_1_VAR = os.path.join(DATA_DIR_VAR, '05 Gramatik - We Used To Dream.wav')
PATH_TO_2_VAR = os.path.join(DATA_DIR_VAR, '04 Gramatik - Pardon My French.wav')
othervar = [os.path.join(DATA_DIR_VAR, s) for s in [ '04 Gramatik - Pardon My French.wav', '05 Gramatik - We Used To Dream.wav']]

OTHER_PATHS_VAR = [os.path.join(DATA_DIR_VAR, song) for song in
    ['01 Gramatik - Brave Men.wav',
     '02 Gramatik - Torture.wav',
     '03 Gramatik - Bluestep (Album Version).wav',
     "06 Gramatik - You Don't Understand.wav",
     '07 Gramatik - Obviously.wav',
     '08 Gramatik - Control Room Before You.wav',
     '09 Gramatik - Prime Time.wav',
     '10 Gramatik - Get A Grip Feat. Gibbz.wav',
     "11 Gramatik - Just Jammin' NYC.wav",
     '12 Gramatik - Expect Us.wav',
     '13 Gramatik - Faraway.wav',
     '14 Gramatik - No Turning Back.wav',
     "15 Gramatik - It's Just A Ride.wav"]
    ]


'''
DATA_DIR_VAR = 'data/sound/epigram'
PATH_TO_1_VAR = os.path.join(DATA_DIR_VAR, '02 Gramatik - Satoshi Nakamoto feat. Adrian Lau & ProbCause.wav')
PATH_TO_2_VAR = os.path.join(DATA_DIR_VAR, '03 Gramatik - War Of The Currents.wav')

OTHER_PATHS_VAR = [os.path.join(DATA_DIR_VAR, song) for song in
    ['01 Gramatik - Tempus Illusio.wav',
     '04 Gramatik - Native Son feat. Raekwon & Leo Napier.wav',
     '05 Gramatik - Native Son Prequel feat. Leo Napier.wav',
     '06 Gramatik - Room 3327.wav',
     '07 Gramatik - Eat Liver! feat. Laibach.wav',
     '08 Gramatik - Back To The Future feat. ProbCause.wav',
     '09 Gramatik - Corporate Demons feat. Luxas.wav',
     '10 Gramatik - Anima Mundi feat. Russ Liquid.wav']
    ]
'''

DATA_DIR = 'data/sound/digitalfreedom/'
PATH_TO_1 = os.path.join(DATA_DIR, '01 Gramatik - Fist Up.wav')
PATH_TO_2 = os.path.join(DATA_DIR, '02 Gramatik - 23 Flavors.wav')

OTHER_PATHS = [os.path.join(DATA_DIR, song) for song in 
    ['03 Gramatik - Illusion Of Choice.wav',
     '04 Gramatik - Born Ready.wav',
     '05 Gramatik - Solidified Feat. Jay Fresh.wav',
     '06 Gramatik - Talkbox Intended Feat. Temu.wav',]
    ]

### TODO prune from here
def compare_convs_am(device='cpu', n_runs=100, alpha=0.05, out_path='results/audio_nets.pkl'):
    Tm5 = get_M5_test('models/M5_m50000_val5635.pt', d=10, device=device, aggreg='avg')
    T_def = get_c2st_am_test(type=0, device=device)
    T_c2st = get_c2st_am_test(type=1, device=device)
    T_c2st_transfer = get_c2st_am_test(type=2, device=device)
    tests = {
            'Deep MMD':Tm5,
            'C2ST, def':T_def,
            'C2ST, conv':T_c2st,
            'C2ST transfer, conv':T_c2st_transfer
            }

    data = AMData(m=10, noise_level=0.5, d=1000)
    data_params_m = ('m', [10, 15, 20, 25, 50, 75, 100, 150, 200, 300, 500, 1000])
    data_params_noise = ('noise_level', [0.1, 0.5, 0.75, 1, 1.5, 2, 3, 4])

    '''
    data.noise_level = 0.5
    results_m = benchmark_pipe(
            data,
            tests,
            data_params_m,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True)
    '''

    data.m = 500
    results_noise = benchmark_pipe(
            data,
            tests,
            data_params_noise,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True)

    #pickle.dump({'results m':results_m, 'results noise':results_noise}, open(out_path, 'wb'))
    pickle.dump(results_noise, open(out_path, 'wb'))
    return results_m, results_noise



def run_am_experiment(alpha=0.05, n_runs=100, device='cpu', noise_level=0.1, out_path='results/power_AM.pkl'):
    path_to_model = 'models/M5_m50000_val5635.pt'
    d = 10

    tests = {'M5 (1d-conv)':get_M5_test(path_to_model, d=d, device=device, aggreg='avg')}
    analytic_tests = get_all_tests(alpha=alpha, Js=[1], split_ratios=[0.5])
    tests.update(analytic_tests)
    
    data = AMData(m=10, noise_level=noise_level, d=1000)

    data_params = ('m', [10, 15, 20, 25, 50, 75, 100, 150, 200, 300, 500, 1000])

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
### TODO prune till here
    
def get_tc2st_am_test(path, device='cpu', lr=False):
    reshape = (-1, 1, 1000)
    m5 = torch.load(path, map_location='cpu')
    #m5 = torch.load('models/M5_m50000_val5635.pt', map_location='cpu')
    pretrained = nn.Sequential(m5.features, nn.ReLU(inplace=True), FlattenLayer())
    model_d = 512
    if lr:
        T = TransferLRC2ST(pretrained, model_d, device=device, reshape=reshape)
    else:
        T = TransferC2ST(pretrained, model_d, device=device, reshape=reshape)
    return T

def get_c2st_am_test(type=0, device='cpu'):
    reshape = (-1, 1, 1000)
    if type == 0:
        model_type = DefaultModel
        model_params = {'d':1000, 'h':100}
    elif type == 1:
        model_type = M5Net
        model_params = {'classes':1}
    elif type == 2:
        model_type = PretrainedM5
        model_params = {'path':'models/M5_m50000_val5635.pt'}

    T = Classifier2SampleTest(model_type, model_params, device=device, reshape=reshape)
    return T
    
'''
def get_M5_test(path, d=10, device='cpu', aggreg=None):
    if path is not None:
        model = torch.load(path, map_location='cpu').to(device)
    else:
        model = M5Net()

    if aggreg is None or aggreg=='pca':
        model = nn.Sequential(model.features, nn.MaxPool1d(4), FlattenLayer(), nn.Tanh())
    elif aggreg == 'max':
        model = nn.Sequential(model.features, nn.MaxPool1d(4), nn.AdaptiveMaxPool2d((d, 1)), FlattenLayer(), nn.Tanh())
    elif aggreg == 'avg':
        model = nn.Sequential(model.features, nn.MaxPool1d(4), nn.AdaptiveAvgPool2d((d, 1)), FlattenLayer(), nn.Tanh())

    test = TorchmodelTest(model=model, d=d, reshape=(-1, 1, 1000), device=device, pca=(aggreg=='pca'))
    return test
'''


def get_M5_test(path, d='ada', device='cpu'):
    if path is not None:
        model = torch.load(path, map_location='cpu').to(device)
    else:
        model = M5Net()

    model = nn.Sequential(model.features, nn.MaxPool1d(4), FlattenLayer(), nn.Tanh())
    test = TorchmodelTest(model=model, d=d, reshape=(-1, 1, 1000), device=device, pca=True)
    return test


### TODO to data.py
class AMData(TestData):
    def __init__(self, path_to_audio_1=PATH_TO_1, path_to_audio_2=PATH_TO_2, m=10, noise_level=0.1, d=1000):
        self.c0_data = self.load_process_audio(path_to_audio_1)        
        self.c1_data = self.load_process_audio(path_to_audio_2)        
        self.m = m
        self.d = d
        self.noise_level = noise_level

    def get_data(self, H0=True):
        X = self._select_slices(self.c0_data)
        noise_X = np.random.normal(0, self.noise_level, size=(self.m, self.d))
        X += noise_X
        if H0:
            Y = self._select_slices(self.c0_data)
        else:
            Y = self._select_slices(self.c1_data)
        noise_Y = np.random.normal(0, self.noise_level, size=(self.m, self.d))
        Y += noise_Y
        return X, Y

    def _select_slices(self, signal):
        ind_Z = np.random.choice(len(signal)-self.d, self.m, replace=True)
        Z = np.empty((self.m, self.d))
        for m in range(self.m):
            start = ind_Z[m]
            end = start + self.d
            Z[m, :] = signal[start:end]

        return Z
   
    def load_process_audio(self, path_to_audio):
        fs, original_audio = self.load_audio(path_to_audio)
        am_signal = amplitude_modulation(original_audio, fs)
        return am_signal

    def load_audio(self, path_to_1):
        sampling_rate, audio_1 = wavfile.read(path_to_1)
        audio_1 = audio_1.mean(1)
        audio_1 /= audio_1.std()
        return sampling_rate, audio_1

def do(path_to_audio, noise_level=0.1):
    fs, original_audio = load_audio(path_to_audio)
    am_signal = amplitude_modulation(original_audio, fs)
    am_signal = am_signal + np.random.normal(0, noise_level**2, len(am_signal))
    return original_audio, am_signal

def amplitude_modulation(signal, fs, multiple=3, transmit_multiple=5, offset=2, envelope=1):
    upsampled_signal = interp(signal, multiple*transmit_multiple)

    carrier_frequency = fs*multiple

    t = np.arange(len(upsampled_signal)) / (fs*multiple*transmit_multiple)

    carrier_signal = np.sin(2*np.pi*carrier_frequency*t)
    
    am_signal = carrier_signal * (offset + upsampled_signal*envelope)
    return am_signal

def interp(y, factor):
    index = np.arange(len(y))
    xx = np.linspace(index[0], index[-1], len(y)*factor)
    interpolated_y = np.interp(xx, index, y)
    return interpolated_y
### TODO end to data.py

def train_one_epoch(model, opt, loader, device='cpu', add_noise=1):
    model.train()
    running_loss = 0.
    for i, (data, target) in enumerate(loader):
        opt.zero_grad()
        data, target = data.to(device), target.to(device)
        
        if add_noise:
            n_data, _, d = data.shape
            noise_levels = add_noise*torch.rand(n_data, device=device)
            noise = torch.Tensor(n_data, d).to(device)
            noise.normal_()
            noise.t().mul_(noise_levels)
            data = data + noise.t().contiguous().view(n_data, 1, d)
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        if i % 250 == 0:
            print('batch %d, running loss: %.3f' % (i, running_loss / (i+1)))

def evaluate(model, loader, device='cpu'):
    model.eval()
    full_loss = 0.
    correct = 0.
    n = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=-1)
            full_loss += loss.item()
            correct += (pred.flatten().to(torch.float32) == target.flatten().to(torch.float32)).sum().item()
            n += len(target)
        
        accuracy = correct / float(n)
    print('val loss: %4f, val accuracy: %.4f' % (full_loss, accuracy))

def load_audio_data(m_train, m_val, bs=128, d=1000, var=True):
    validation_noise = 0.1
    if var:
        paths = OTHER_PATHS_VAR
    else:
        paths = OTHER_PATHS

    train = []
    val = []
    for i, path in enumerate(paths):
        snippets_train = extract_snippets(path, m=m_train, d=d, noise_level=0.)
        labels_train = i * torch.ones(m_train, dtype=torch.long)
        train.append((snippets_train, labels_train))
        snippets_val = extract_snippets(path, m=m_val, d=d, noise_level=validation_noise)
        labels_val = i * torch.ones(m_val, dtype=torch.long)
        val.append((snippets_val, labels_val))
    Z_train = torch.cat([torch.from_numpy(a.astype(np.float32)) for a, b in train]).view(-1, 1, d)
    labels_train = torch.cat([b for a, b in train])
    Z_val = torch.cat([torch.from_numpy(a.astype(np.float32)) for a, b in val]).view(-1, 1, d)
    labels_val = torch.cat([b for a, b in val])

    train = torch.utils.data.TensorDataset(Z_train, labels_train)
    loader_train = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    val = torch.utils.data.TensorDataset(Z_val, labels_val)
    loader_val = torch.utils.data.DataLoader(val, batch_size=bs, shuffle=False)

    return loader_train, loader_val

def extract_snippets(path_to_file, m=100, d=1000, noise_level=0.):
    sampling_rate, audio = wavfile.read(path_to_file)
    audio = audio.mean(1)
    audio /= audio.std()
    am_signal = amplitude_modulation(audio, sampling_rate)
    ind = np.random.choice(len(am_signal)-d, m, replace=True)
    Z = np.empty((m, d))
    for i in range(m):
        start = ind[i]
        end = start + d
        Z[i, :] = am_signal[start:end]

    if noise_level:
        noise = np.random.normal(0, noise_level, size=(m, d))
        Z += noise
    return Z
#TODO remove !
'''
def load_audio_data_other(m_train, m_val, bs=128, d=1000, var=True):
    if var:
        paths = OTHER_PATHS_VAR
    else:
        paths = OTHER_PATHS

    data1 = AMData(path_to_audio_1=paths[0], path_to_audio_2=paths[1],
            m=m_train, d=d, noise_level=0)
    data2 = AMData(path_to_audio_1=paths[2], path_to_audio_2=paths[3],
            m=m_train, d=d, noise_level=0)
    if var:
        data3 = AMData(path_to_audio_1=paths[4], path_to_audio_2=paths[5],
                m=m_train, d=d, noise_level=0)
        data4 = AMData(path_to_audio_1=paths[6], path_to_audio_2=paths[7],
                m=m_train, d=d, noise_level=0)

    A, B = data1.get_data(H0=False)
    C, D = data2.get_data(H0=False)
    if var:
        E, F = data3.get_data(H0=False)
        G, H = data4.get_data(H0=False)
    A = torch.tensor(A.reshape(m_train, 1, d), dtype=torch.float32)
    B = torch.tensor(B.reshape(m_train, 1, d), dtype=torch.float32)
    C = torch.tensor(C.reshape(m_train, 1, d), dtype=torch.float32)
    D = torch.tensor(D.reshape(m_train, 1, d), dtype=torch.float32)
    if var:
        E = torch.tensor(E.reshape(m_train, 1, d), dtype=torch.float32)
        F = torch.tensor(F.reshape(m_train, 1, d), dtype=torch.float32)
        G = torch.tensor(G.reshape(m_train, 1, d), dtype=torch.float32)
        H = torch.tensor(H.reshape(m_train, 1, d), dtype=torch.float32)

    if var:
        Z = torch.cat((A, B, C, D, E, F, G, H))
        labels = torch.cat((
            torch.zeros(m_train, dtype=torch.long),
            torch.ones(m_train, dtype=torch.long),
            2*torch.ones(m_train, dtype=torch.long),
            3*torch.ones(m_train, dtype=torch.long),
            4*torch.ones(m_train, dtype=torch.long),
            5*torch.ones(m_train, dtype=torch.long),
            6*torch.ones(m_train, dtype=torch.long),
            7*torch.ones(m_train, dtype=torch.long)),
            )

    else:
        Z = torch.cat((A, B, C, D))
        labels = torch.cat((
            torch.zeros(m_train, dtype=torch.long),
            torch.ones(m_train, dtype=torch.long),
            2*torch.ones(m_train, dtype=torch.long),
            3*torch.ones(m_train, dtype=torch.long)))

    tset = torch.utils.data.TensorDataset(Z, labels)
    loader_tr = torch.utils.data.DataLoader(tset, batch_size=bs, shuffle=True)

    validation_noise = 0.1
    data1 = AMData(path_to_audio_1=paths[0], path_to_audio_2=paths[1],
            m=m_val, d=d, noise_level=validation_noise)
    data2 = AMData(path_to_audio_1=paths[2], path_to_audio_2=paths[3],
            m=m_val, d=d, noise_level=validation_noise)
    if var:
        data3 = AMData(path_to_audio_1=paths[4], path_to_audio_2=paths[5],
                m=m_val, d=d, noise_level=0)
        data4 = AMData(path_to_audio_1=paths[6], path_to_audio_2=paths[7],
                m=m_val, d=d, noise_level=0)

 
    A, B = data1.get_data(H0=False)
    C, D = data2.get_data(H0=False)
    if var:
        E, F = data3.get_data(H0=False)
        G, H = data4.get_data(H0=False)
    A = torch.tensor(A.reshape(m_val, 1, d), dtype=torch.float32)
    B = torch.tensor(B.reshape(m_val, 1, d), dtype=torch.float32)
    C = torch.tensor(C.reshape(m_val, 1, d), dtype=torch.float32)
    D = torch.tensor(D.reshape(m_val, 1, d), dtype=torch.float32)
    if var:
        E = torch.tensor(E.reshape(m_val, 1, d), dtype=torch.float32)
        F = torch.tensor(F.reshape(m_val, 1, d), dtype=torch.float32)
        G = torch.tensor(G.reshape(m_val, 1, d), dtype=torch.float32)
        H = torch.tensor(H.reshape(m_val, 1, d), dtype=torch.float32)

    if var:
        Z = torch.cat((A, B, C, D, E, F, G, H))
        labels = torch.cat((
            torch.zeros(m_val, dtype=torch.long),
            torch.ones(m_val, dtype=torch.long),
            2*torch.ones(m_val, dtype=torch.long),
            3*torch.ones(m_val, dtype=torch.long),
            4*torch.ones(m_val, dtype=torch.long),
            5*torch.ones(m_val, dtype=torch.long),
            6*torch.ones(m_val, dtype=torch.long),
            7*torch.ones(m_val, dtype=torch.long)),
            )

    else:
        Z = torch.cat((A, B, C, D))
        labels = torch.cat((
            torch.zeros(m_val, dtype=torch.long),
            torch.ones(m_val, dtype=torch.long),
            2*torch.ones(m_val, dtype=torch.long),
            3*torch.ones(m_val, dtype=torch.long)))

    vset = torch.utils.data.TensorDataset(Z, labels)
    loader_val = torch.utils.data.DataLoader(vset, batch_size=bs, shuffle=False)

    return loader_tr, loader_val
'''

def train_model(device='cpu', epochs=10, add_noise=1, var=True, MXN=5, m_train=10000, bs=128, lr=0.001):
    #train, val = load_audio_data_other(m_train=m_train, m_val=1000, bs=bs, var=var)
    train, val = load_audio_data(m_train=m_train, m_val=1000, bs=bs, var=var)
    if var:
        n_classes = 13
    else:
        n_classes = 4

    if MXN == 5:
        model = M5Net(n_classes).to(device)
    elif MXN == 11:
        model = M11Net(n_classes).to(device)
    
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    #opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001)
    #sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    for epoch in range(1, epochs+1):
        print('starting epoch %d/%d' % (epoch, epochs))
        #sched.step()
        train_one_epoch(model, opt, train, device=device, add_noise=add_noise)
        evaluate(model, val, device=device)
    return model



class MXNet(nn.Module):
    def __init__(self, classes=1):
        super(MXNet, self).__init__()
        if classes == 2:
            classes = 1
        self.classes = classes

        self.classifier = nn.Sequential(
                nn.ReLU(True),
                nn.MaxPool1d(4),
                FlattenLayer(),
                nn.Linear(512, classes)
                )

    def forward(self, x):
        x = self.features(x)
        #x = x.view(len(x), -1)
        x = self.classifier(x)
        if self.classes == 1:
            x = torch.sigmoid(x)
        return x

class M11Net(MXNet):
    def __init__(self, classes=1):
        super(M11Net, self).__init__(classes=classes)
        self.features = nn.Sequential(
                # kernel_size = 80 originally
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=20, stride=4),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 64, 3),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Conv1d(64, 64, 3),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Conv1d(128, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.MaxPool1d(2),

                nn.Conv1d(128, 256, 3),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Conv1d(256, 256, 3),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Conv1d(256, 256, 3),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.MaxPool1d(2),

                nn.Conv1d(256, 512, 3),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Conv1d(512, 512, 3),
                nn.BatchNorm1d(512),
                )

class M5Net(MXNet):
    def __init__(self, classes=1):
        super(M5Net, self).__init__(classes=classes)
        self.features = nn.Sequential(
                # kernel_size = 80 originally
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=20, stride=4),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.MaxPool1d(4),

                nn.Conv1d(128, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.MaxPool1d(4),

                nn.Conv1d(128, 256, 3),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.MaxPool1d(4),

                nn.Conv1d(256, 512, 3),
                nn.BatchNorm1d(512),
                )

class PretrainedM5(nn.Module):
    def __init__(self, path):
        super(PretrainedM5, self).__init__()
        m5 = torch.load(path, map_location='cpu')
        self.model = nn.Sequential(
                m5.features,
                m5.classifier[:-1],
                nn.Linear(512, 1, bias=True),
                nn.Sigmoid())
    def forward(self, x):
        return self.model(x)


