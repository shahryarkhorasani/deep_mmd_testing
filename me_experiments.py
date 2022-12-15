'''own file, since me are terribly slow, don't want it to run with others'''



from testing import *
from analytic import *

from audio import *


def get_me_tests(Js=[1], alpha=0.05, split_ratios=[0.1, 0.5]):
    ME_v1 = {
            ('ME var1, J=%d' % J):METest_var1(J=J, alpha=alpha) for J in Js
            }
    ME_v2 = {
            ('ME var2, J=%d, ratio=%.2f' % (J, ratio)):METest_var2(J=J, alpha=alpha, split_ratio=ratio) for J in Js for ratio in split_ratios
            }
    ME_v3 = {
            ('ME var3, J=%d, ratio=%.2f' % (J, ratio)):METest_var3(J=J, alpha=alpha, split_ratio=ratio) for J in Js for ratio in split_ratios
            }
    ME_v1.update(ME_v2)
    ME_v1.update(ME_v3)
    return ME_v1



def benchmark_dogs_me(alpha=0.05, n_runs=100, out_path='results/me_power_dogs.pkl'):
    Js = [1]
    split_ratios = [0.5]
    tests = get_me_tests(alpha=alpha, Js=Js, split_ratios=split_ratios)
    data = DogData(gray=True, target_shape=(64, 64))

    data_params = ('m', [50, 75, 100])

    results = benchmark_pipe(
            data,
            tests, 
            data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True)
    pickle.dump(results, open(out_path, 'wb'))
    return results

def benchmark_faces_me(alpha=0.05, n_runs=100, out_path='results/me_power_faces.pkl'):
    Js = [1]
    split_ratios = [0.5]
    tests = get_me_tests(alpha=alpha, Js=Js, split_ratios=split_ratios)
    data = FacesData(crop=True, gray=True, target_shape=(48,34))

    data_params = ('m', [50, 75, 100, 150, 200])

    results = benchmark_pipe(
            data,
            tests, 
            data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True)
    pickle.dump(results, open(out_path, 'wb'))
    return results

def benchmark_am_me(alpha=0.05, n_runs=100, out_path='results/me_power_AM.pkl'):
    Js = [1]
    split_ratios = [0.5]
    tests = get_me_tests(alpha=alpha, Js=Js, split_ratios=split_ratios)
    data = AMData(noise_level=0.1)


    data_params = ('m', [50, 75, 100, 150, 200, 500, 1000])

    results = benchmark_pipe(
            data,
            tests, 
            data_params,
            n_runs=n_runs,
            alpha=alpha,
            verbose=True)
    pickle.dump(results, open(out_path, 'wb'))
    return results

