
import contextlib
import numpy as np

import freqopttest.util as fot_util
import freqopttest.data as fot_data
import freqopttest.kernel as fot_kernel
import freqopttest.tst as fot_tst

from base import *

def get_all_tests(alpha=0.05, Js=[4, 7, 10], split_ratios=[0.1, 0.5]):
    n_permute = 1000
    tests = {
            'Linear MMD var1':LinearMMDTest_var1(alpha=alpha),
            'Quad MMD var1':MMDTest_var1(alpha=alpha, n_permute=n_permute),
            }

    MMD_v2 = {
            ('Quad MMD var2, ratio=%.2f' % ratio):MMDTest_var2(alpha=alpha, split_ratio=ratio, n_permute=n_permute) for ratio in split_ratios
            }
    LMMD_v2 = {
            ('Linear MMD var2, ratio=%.2f' % ratio):LinearMMDTest_var2(alpha=alpha, split_ratio=ratio) for ratio in split_ratios
            }

    ME_v1 = {
            ('ME var1, J=%d' % J):METest_var1(J=J, alpha=alpha) for J in Js
            }
    ME_v2 = {
            ('ME var2, J=%d, ratio=%.2f' % (J, ratio)):METest_var2(J=J, alpha=alpha, split_ratio=ratio) for J in Js for ratio in split_ratios
            }
    ME_v3 = {
            ('ME var3, J=%d, ratio=%.2f' % (J, ratio)):METest_var3(J=J, alpha=alpha, split_ratio=ratio) for J in Js for ratio in split_ratios
            }

    SCF_v1 = {
            ('SCF var1, J=%d' % J):SCFTest_var1(J=J, alpha=alpha) for J in Js
            }
    SCF_v2 = {
            ('SCF var2, J=%d, ratio=%.2f' % (J, ratio)):SCFTest_var2(J=J, alpha=alpha, split_ratio=ratio) for J in Js for ratio in split_ratios
            }
    SCF_v3 = {
            ('SCF var3, J=%d, ratio=%.2f' % (J, ratio)):SCFTest_var3(J=J, alpha=alpha, split_ratio=ratio) for J in Js for ratio in split_ratios
            }


    tests.update(MMD_v2)
    tests.update(LMMD_v2)

    #tests.update(ME_v1)
    #tests.update(ME_v2)
    #tests.update(ME_v3)

    tests.update(SCF_v1)
    tests.update(SCF_v2)
    #tests.update(SCF_v3)
    return tests

class FOTTest(TwoSampleTest):
    def __init__(self):
        pass

    def preprocess(self, X, Y):
        if len(X.shape) > 2:
            X = X.reshape(len(X), -1)
            Y = Y.reshape(len(Y), -1)
        XY = fot_data.TSTData(X, Y)
        return XY
 

class METest_var1(FOTTest):
    def __init__(self, J=10, alpha=0.05):
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        
        #locations = fot_tst.MeanEmbeddingTest.init_locs_randn(XY, self.J)
        locations = fot_tst.MeanEmbeddingTest.init_locs_subset(XY, self.J)
        med = fot_util.meddistance(XY.stack_xy(), 1000)
        kernel = fot_kernel.KGauss(med)
        ME = fot_tst.MeanEmbeddingTest(locations, med, alpha=self.alpha)

        result = ME.perform_test(XY)
        p_val = result['pvalue']
        return p_val


class METest_var2(FOTTest):
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        #locations = fot_tst.MeanEmbeddingTest.init_locs_randn(XY, self.J)
        locations = fot_tst.MeanEmbeddingTest.init_locs_subset(train, self.J)
        med = fot_util.meddistance(train.stack_xy(), 1000)
        gwidth, info = fot_tst.MeanEmbeddingTest.optimize_gwidth(
                train, locations, med**2)
        
        ME = fot_tst.MeanEmbeddingTest(locations, gwidth, alpha=self.alpha)

        result = ME.perform_test(test)
        p_val = result['pvalue']
        return p_val


class METest_var3(FOTTest):
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        
        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        with contextlib.redirect_stdout(None):
            test_locs, gwidth, info = fot_tst.MeanEmbeddingTest.optimize_locs_width(
                    train,
                    self.alpha,
                    n_test_locs=self.J,
                    )

        ME = fot_tst.MeanEmbeddingTest(test_locs, gwidth, alpha=self.alpha)

        result = ME.perform_test(test)
        p_val = result['pvalue']
        return p_val


class LinearMMDTest_var1(FOTTest):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def test(self, X, Y):
        XY = self.preprocess(X, Y)
 
        med = fot_util.meddistance(XY.stack_xy(), 1000)
        kernel = fot_kernel.KGauss(med)

        MMD = fot_tst.LinearMMDTest(kernel, alpha=self.alpha)

        result = MMD.perform_test(XY)
        p_val = result['pvalue']
        return p_val


class LinearMMDTest_var2(FOTTest):
    def __init__(self, alpha=0.05, split_ratio=0.5):
        self.alpha = alpha
        self.split_ratio = split_ratio

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)
        med = fot_util.meddistance(train.stack_xy(), 1000)

        bandwidths = (med**2) * (2.**np.linspace(-4, 4, 20))
        kernels = [fot_kernel.KGauss(width) for width in bandwidths]
        with contextlib.redirect_stdout(None):
            best_i, powers = fot_tst.LinearMMDTest.grid_search_kernel(
                    train, kernels, alpha=self.alpha)
        best_kernel = kernels[best_i]

        MMD = fot_tst.LinearMMDTest(best_kernel, alpha=self.alpha)

        result = MMD.perform_test(test)
        p_val = result['pvalue']
        return p_val


class MMDTest_var1(FOTTest):
    def __init__(self, n_permute=200, alpha=0.05):
        self.n_permute = n_permute
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        med = fot_util.meddistance(XY.stack_xy(), 1000)
        kernel = fot_kernel.KGauss(med)

        MMD = fot_tst.QuadMMDTest(kernel, n_permute=self.n_permute, alpha=self.alpha)

        result = MMD.perform_test(XY)
        p_val = result['pvalue']
        return p_val


class MMDTest_var2(FOTTest):
    def __init__(self, n_permute=200, alpha=0.05, split_ratio=0.5):
        self.n_permute = n_permute
        self.alpha = alpha
        self.split_ratio = split_ratio

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)
        med = fot_util.meddistance(train.stack_xy(), 1000)

        bandwidths = (med**2) * (2.**np.linspace(-4, 4, 20))
        kernels = [fot_kernel.KGauss(width) for width in bandwidths]
        with contextlib.redirect_stdout(None):
            best_i, powers = fot_tst.QuadMMDTest.grid_search_kernel(
                    train, kernels, alpha=self.alpha)
        best_kernel = kernels[best_i]

        MMD = fot_tst.QuadMMDTest(best_kernel, n_permute=self.n_permute, alpha=self.alpha)

        result = MMD.perform_test(test)
        p_val = result['pvalue']
        return p_val


class SCFTest_var1(FOTTest):
    def __init__(self, J=10, alpha=0.05):
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        SCF = fot_tst.SmoothCFTest.create_randn(XY, self.J, alpha=self.alpha, seed=1)
        result = SCF.perform_test(XY)
        p_val = result['pvalue']
        return p_val


class SCFTest_var2(FOTTest):
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        freqs = np.random.randn(self.J, XY.dim())

        mean_sd = train.mean_std()
        scales = 2.**np.linspace(-4, 4, 30)
        list_gwidth = np.hstack([mean_sd*scales*(XY.dim()**0.5), 2**np.linspace(-8, 8, 20)])
        list_gwidth.sort()
        with contextlib.redirect_stdout(None):
            best_i, powers = fot_tst.SmoothCFTest.grid_search_gwidth(
                    train, freqs, list_gwidth, self.alpha)
        best_width = list_gwidth[best_i]

        SCF = fot_tst.SmoothCFTest(freqs, best_width, self.alpha)

        result = SCF.perform_test(test)
        p_val = result['pvalue']
        return p_val


class SCFTest_var3(FOTTest):
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        with contextlib.redirect_stdout(None):
            with contextlib.redirect_stderr(None):
                test_freqs, gwidth, info = fot_tst.SmoothCFTest.optimize_freqs_width(
                        train,
                        self.alpha,
                        n_test_freqs=self.J
                        )
        SCF = fot_tst.SmoothCFTest(test_freqs, gwidth, alpha=self.alpha)
        result = SCF.perform_test(test)
        p_val = result['pvalue']
        return p_val


