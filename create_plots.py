# coding: utf8

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
logfmt = ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)


import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.2
params = {'legend.fontsize': 'small',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
mpl.rcParams.update(params)


__color_rotation__ = ['red', 'gold','coral','darkturquoise','royalblue','darkblue','m','hotpink','lightpink','dimgrey']
__marker_rotation__ = ['o', 'H', 's', '^', '+', '.', 'D', 'x', 'o', 'H', 's']
__linestyle_rotation__ = ['-', '--', ':', '-.']


def create_noisefree_plots():
    # Load files for NSIM
    savestr_base_nsim = 'for_plotting'
    filename_errors_nsim = '../img/' + savestr_base_nsim + '/errors'
    tan_err_nsim = np.load(filename_errors_nsim + '/tangent_error.npy')
    f_test_nsim = np.load(filename_errors_nsim + '/f_error_test.npy')
    # Load files for kNN
    savestr_base_kNN = 'for_plotting_kNN'
    filename_errors_kNN = '../img/' + savestr_base_kNN + '/errors'
    f_test_kNN = np.load(filename_errors_kNN + '/f_error_test.npy')
    N = [1000, 2000, 4000, 8000, 16000, 32000]
    D = [6, 12, 24]
    tan_err_nsim_mean = np.mean(tan_err_nsim, axis = 7)
    tan_err_nsim_std = np.std(tan_err_nsim, axis = 7)
    f_test_nsim_mean = np.mean(f_test_nsim, axis = 7)
    f_test_nsim_std = np.std(f_test_nsim, axis = 7)
    f_test_kNN_mean = np.mean(f_test_kNN, axis = 5)
    f_test_kNN_std = np.std(f_test_kNN, axis = 5)
    # Make plot for tangent error
    fig1 = plt.figure()
    ax1 = plt.gca()
    for i, d in enumerate(D):
        plt.errorbar(N, tan_err_nsim_mean[:,i,:,:,:,:], tan_err_nsim_std[:,i,:,:,:,:], fmt = __marker_rotation__[i] + '-', label = "D = {0}".format(d))
    ax1.plot(N, np.divide(1.0, N) * 1000, 'k',label = r'N^{-1}')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(logfmt)
    ax1.yaxis.set_major_formatter(logfmt)
    plt.legend()
    ax1.set_xlabel(r'$N$')
    ax1.set_ylabel(r'Mean$\Vert a_j - \hat a_j\Vert$')
    # Make plot for function error
    fig2 = plt.figure()
    ax2 = plt.gca()
    for i, d in enumerate(D):
        ax2.errorbar(N, f_test_nsim_mean[:,i,:,:,:,:], f_test_nsim_std[:,i,:,:,:,:], fmt = __marker_rotation__[i] + '-', label = "(NSIM) $D = {0}$".format(d))
    for i, d in enumerate(D):
        ax2.errorbar(N, f_test_kNN_mean[0,:,i,:,:], f_test_kNN_std[0,:,i,:,:], fmt='--', label = "(KNN) $D = {0}$".format(d))
    ax2.plot(N, np.divide(1.0, N) * 100, 'k', label = r'N^{-1}')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.xaxis.set_major_formatter(logfmt)
    ax2.yaxis.set_major_formatter(logfmt)
    plt.legend(ncol = 2)
    ax2.set_xlabel(r'$N$')
    ax2.set_ylabel(r'RMSE$(\hat f_1 - f_1)$')
    plt.show()

if __name__ == "__main__":
    create_noisefree_plots()
    import pdb; pdb.set_trace()
