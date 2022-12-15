import abc
from abc import abstractmethod
import numpy as np
from tqdm import tqdm

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class ClassificationPipe:
    def __init__(self, data, clf, alpha=0.05, n_runs=100):
        self.data = data
        self.clf = clf 
        self.alpha = alpha
        self.n_runs = n_runs

    def evaluate_classification(self):
        p_vals_H0 = self.clf.classify(*self.data.get_data(H0=True))
        p_vals_H1 = self.clf.classify(*self.data.get_data(H0=False))
        T1ER = (p_vals_H0 < self.alpha).mean()
        T2ER = 1 - (p_vals_H1 < self.alpha).mean()
        return {'T1ER': T1ER, 'T2ER': T2ER}, p_vals_H0, p_vals_H1, self.alpha




class TestPipe:
    '''Pipeline class to fully evaluate a test on a given data set over n_runs'''
    def __init__(self, data, test, alpha=0.05, n_runs=100):
        self.data = data
        self.test = test
        self.alpha = alpha
        self.n_runs = n_runs

    def evaluate_test(self):
        if len(self.data.c0_data) < 2*self.data.m:
            test_h0 = False
        else:
            test_h0 = True
        p_values_H0 = np.empty(self.n_runs)
        p_values_H1 = np.empty(self.n_runs)
        for run in tqdm(range(self.n_runs)):
            if test_h0:
                p_values_H0[run] = self.test.test(*self.data.get_data(H0=True))
            self.test.reset()
            p_values_H1[run] = self.test.test(*self.data.get_data(H0=False))
            self.test.reset()
        if test_h0:
            T1ER = (p_values_H0 < self.alpha).mean()
        else:
            T1ER = np.nan
        T2ER = 1 - (p_values_H1 < self.alpha).mean()
        return {'T1ER': T1ER, 'T2ER': T2ER}, p_values_H0, p_values_H1, self.alpha

def benchmark_pipe(data, test, data_params=None, n_runs=10, alpha=0.05, verbose=True):
    '''

    # Parameters
    data (param_name, param_values):
    '''
    results = []

    if data_params:
        param = data_params[0]
        values = data_params[1]
    else:
        values = [0]

    for value in values:
        if data_params:
            if verbose:
                print('starting %s = %s' % (param, value))
            setattr(data, param, value)

        try:
            errs = TestPipe(data, test, alpha=alpha, n_runs=n_runs).evaluate_test()[0]
        except Exception as e:
            print('throwing exception', e)
            errs = {'T1ER':np.nan, 'T2ER':np.nan}

        results.append((value, errs))
        if verbose:
            print(value, errs)
                
    return results


class TwoSampleTest(ABC):
    '''Base class for all testing procedures
    
    Every subclass needs to implement the .test(X, Y) method, that takes in two sets
    of observations X and Y and outputs a (scalar) p-value
    '''
    def __init__(self):
        pass

    @abstractmethod
    def test(self, X, Y):
        pass

    def reset(self):
        pass

