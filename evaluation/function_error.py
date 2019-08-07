# coding: utf8
import json

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
logfmt = ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)


import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 4.0
params = {
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'lines.markersize' : 7,
         'font.size' : 20}
mpl.rcParams.update(params)


__color_rotation__ = ['red', 'gold','coral','darkturquoise','royalblue','darkblue','m','hotpink','lightpink','dimgrey']
__marker_rotation__ = ['o', '^', 's', 'H', '+', '.', 'D', 'x', 'o', 'H', 's']
__linestyle_rotation__ = ['--', ':', '-', '-.']


def plot_error(folder):
    # Load parameters
    with open(folder + '/log.txt') as f:
        run_for = json.load(f)
    error_cv = np.load(folder + 'f_error_CV.npy')
    error_test = np.load(folder + 'f_error_test.npy')
    # Perform cross validation
    error_final = np.zeros(error_cv[:,:,:,:,:,0,:].shape)
    ind_cv = np.argmin(error_cv, axis = 5)
    shape = error_cv.shape
    aux_iter = [(i1,i2,i3,i4,i5,i6) for i1 in range(shape[0]) for i2 in range(shape[1]) for i3 in range(shape[2]) for i4 in range(shape[3]) for i5 in range(shape[4]) for i6 in range(shape[6])]
    for index in aux_iter:
        error_final[index] = error_test[index[0], index[1], index[2], index[3], index[4], ind_cv[index], index[5]]
    mean_error_final = np.mean(error_final, axis = 5)
    std_error_final = np.std(error_final, axis = 5)
    plt.figure(figsize = (12,8))
    for i, D in enumerate(run_for['D']):
        for j, sigma_f in enumerate(run_for['sigma_f']):
            if i < len(run_for['D']) - 1 and j > 0:
                continue
            plt.errorbar(run_for['N'][1:], mean_error_final[1:,i,0,j,0], std_error_final[1:,i,0,j,0],
                         color = __color_rotation__[j],
                         linestyle = __linestyle_rotation__[i],
                         label = r'$D = {:d},\ \sigma_\varepsilon = {:.0E}$'.format(D, sigma_f))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(ncol = 2, prop={'size': 15})
    plt.xlabel(r'$N$')
    plt.ylabel(r'$RMSE(\hat f - f)$')
    plt.tight_layout()
    plt.show()
