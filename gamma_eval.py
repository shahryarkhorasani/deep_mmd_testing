import numpy as np
from scipy.special import gammainc


def cdf_gaussian_norm(t0, cov):
    eig_vals, _ = np.linalg.eigh(cov)
    d = len(eig_vals)
    alphas = np.ones(d) / 2.
    betas = 2*eig_vals
    return cdf_sum_gamma(alphas, betas, t0, m=1000)
    
def cdf_sum_gamma(alphas, betas, t0, m=1000):
    '''
    compute cdf of
        sum_i=1^n X_i
    where X_i ~ Gamma(alpha_i, beta_i) (with beta parameter from paper
    '''
    n = len(alphas)
    beta1 = betas.min()

    C = np.prod( (beta1/betas)**alphas )
    rho = np.sum(alphas)

    ks = np.arange(1, m+1)
    gammas = np.array([alphas.dot((1 - beta1/betas)**k) for k in ks]) / ks

    #print('rho:', rho)
    #print('C:', C)
    #print('gammas:', gammas)
    deltas = np.ones(m+1)
    for k in range(1, m+1):
        deltas[k] = (np.arange(1, k+1) * gammas[:k] * deltas[:k][::-1]).mean()
    #print('deltas:', deltas)

    cdf_partial = 0
    for i in range(m+1):
        cdf_partial += deltas[i] * gammainc(rho+i, t0/beta1)
    return C*cdf_partial
