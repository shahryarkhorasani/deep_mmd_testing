# Deep MMD Testing

### (everything Work in Progress and still buggy and ugly code)

## Dependencies:

Everything uses Python 3.6; Dependencies are in `requirements.txt` and can be installed (e.g. in a virtualenv) via `pip install -r requirements.txt`


## Using the interface

To import everything, just `from testing import *`, i.e. `testing.py` just imports all necessary stuff.

The Base components are in `base.py`. The basic pipeline should be to create an instance of `TestPipe` and then to `evaluate_test()`. This requires a `TestData` and a `TwoSampleTest` to be passed. Every test should inherit from `TwoSampleTest`, every data set from `TestData` (see `data.py`).

For examples of `TestData` objects, see `data.py`: Take e.g. `MVNormalData`; if you call `get_data(H0=True)` you get a pair X, Y of samples from the null distribution (i.e. X, Y are both samples from the same multivariate normal distribution N(0,I)). If you call `get_data(H0=False)`, you get a pair `X, Y` of samples from the alternative distribution, i.e. X ~ N(0, I), Y ~ N(0, \Sigma), where Sigma is some non-identity matrix, depending on the parameter `h1eps` passed to the object in the `__init__`.

Subclasses of `TorchData` can be ignored, they are just for easy loading of standard image data sets (MNIST, CIFAR, ...) with pytorch.

For examples of `TwoSampleTest` objects, look at `kmmd.py`.
A two-sample-test only needs to implement the `test(X, Y)` method, that takes in two sets of observations and returns a p-value.
In this case, `KMMDBootstrapTest` and `KMMDSpectralTest` only differ in how they estimate the null-hypothesis, so they share the common super-class `KMMDTest` and only differ in the implementation of `compute_null_samples`.

Examples for neural networks are implemented in `nnmmd.py`, but this is still very rough and experimental code. For starters, we need an implementation so that in the `test(X, Y)` method, we compute the output of the last layer, and then input this into a `test_features(X, Y)` method to compute the p-values, see implementation in the `CustomNNTest` class.



### Example:

```python
from testing import *
m = 250     # number of samples per population
d = 10      # input data dimension

data = MVNormalData(m=m, d=d, h1eps=0.2)    # create multivariate normal data object,
                                            # h1eps specifies how far H1 deviates from H0
mmd = KMMDBootstrapTest()                   # object that performs the test

alpha = 0.05    # significance value
n_runs = 10     # how often to run the test
error_rates, p_values_H0, p_values_H1, _ = TestPipe(data, mmd, alpha=alpha, n_runs=n_runs).evaluate_test()
print('Type I error rate = %.2f' % error_rates['T1ER'])
print('Type II error rate = %.2f' % error_rates['T2ER'])
```


