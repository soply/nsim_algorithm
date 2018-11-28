# coding: utf8
"""
A collection of functions used to test the estimator. Except random polynomial increments,
the functions are basic.
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.2
params = {'legend.fontsize': 'small',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'lines.markersize' : 7}
mpl.rcParams.update(params)

class RandomPolynomialIncrements(object):
    """ Use with area of definition > 0 to avoid non-monotonicity. """
    def __init__(self, tlower, tupper, deg, n_increments = 10,
                 coefficient_bound = [1.0, 2.0]):
        self.tlower_ = tlower
        self.tupper_ = tupper
        self.deg_ = deg
        self.n_increments_  = n_increments
        self.coefficient_bound_ = coefficient_bound
        self.bases_, self.coeffs_ = self._initialize_()

    def _initialize_(self):
        """ Set random bases and coefficients of polynom. """
        bases = np.zeros(self.n_increments_ + 1)
        bases[0] = self.tlower_
        bases[1:-1] = np.sort(np.random.uniform(self.tlower_, self.tupper_,
                                                size = self.n_increments_ - 1))
        bases[-1] = self.tupper_
        coeffs = np.zeros((self.deg_ + 1, self.n_increments_))
        for i in range(self.n_increments_):
            coeffs[0,i] = np.random.uniform(low = self.coefficient_bound_[0],
                                            high = self.coefficient_bound_[1],
                                            size = 1)
            if i == 0:
                # Make slope in first interval positive
                coeffs[1,i] = 3.0 * coeffs[0,i] * (0.5 * (bases[i] + 0.05 * (self.tupper_ - self.tlower_)) - self.tlower_)
                # Make value at 0 equal to 0
                coeffs[-1,i] = -np.polyval(coeffs[:,i], self.tlower_ - 0.5 * (bases[i] + 0.05 * (self.tupper_ - self.tlower_)))
            elif i == 1:
                coeffs[-1,i] = np.polyval(coeffs[:,0], bases[1] - 0.5 * (bases[0] + 0.05 * (self.tupper_ - self.tlower_))) - \
                                np.polyval(coeffs[:,i], bases[1] - 0.5 * (bases[1] + bases[0]))
            elif i > 1:
                coeffs[-1,i] = np.polyval(coeffs[:,i-1], bases[i] - 0.5 * (bases[i-1] + bases[i-2])) - \
                                np.polyval(coeffs[:,i], bases[i] - 0.5 * (bases[i] + bases[i-1]))
        return bases, coeffs

    def get_bases(self):
        return self.bases_

    def get_coeffs(self):
        return self.coeffs_

    def eval(self, x):
        if np.abs(x - self.tupper_) < 1e-15:
            x = self.tupper_ - 1e-15
        idx = np.digitize(x, self.bases_) - 1
        if idx == 0:
            return np.polyval(self.coeffs_[:,idx], x - 0.5 * (self.bases_[idx] + 0.05 * (self.tupper_ - self.tlower_)))
        else:
            return np.polyval(self.coeffs_[:,idx], x - 0.5 * (self.bases_[idx] + self.bases_[idx-1]))


    def plot(self, white_noise_var = 0.0, n = 100):
        x = np.linspace(self.tlower_, self.tupper_, n)
        fval = np.zeros(x.shape)
        for i in range(len(x)):
            fval[i] = self.eval(x[i])
        fig = plt.figure()
        plt.xlabel(r'Intrinsic curve parameter $t$')
        plt.ylabel(r'$g \circ \gamma^{-1}(t)$')
        plt.plot(x, fval)
        if white_noise_var > 0.0:
            noisy_fval = np.zeros(x.shape)
            noisy_fval = fval + np.random.normal(scale=np.sqrt(white_noise_var), size = n)
            plt.plot(x, noisy_fval, 'g', alpha = 0.5)


def randomPolynomialIncrements_for_parallel(x, tlower, tupper, bases, coeffs):
        if np.abs(x - tupper) < 1e-15:
            x = tupper - 1e-15
        idx = np.digitize(x, bases) - 1
        if idx == 0:
            return np.polyval(coeffs[:,idx], x - 0.5 * \
                    (bases[idx] + 0.05 * (tupper - tlower)))
        else:
            return np.polyval(coeffs[:,idx], x - 0.5 * \
                    (bases[idx] + bases[idx-1]))


def identity_function(x, derivative = 0):
    if derivative == 0:
        return x[0]
    elif derivative == 1:
        retr = np.zeros(x.shape[0])
        retr[0] = 1.0
        return retr


def injective_on_first_coordinate(x, derivative = 0):
    if derivative == 0:
        return 2.0/3.0 * x[0]
    elif derivative == 1:
        retr = np.zeros(x.shape[0])
        retr[0] = 2.0/3.0
        return retr
