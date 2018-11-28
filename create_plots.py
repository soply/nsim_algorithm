# coding: utf8
"""
Auxiliary file to create the plots for NSIM article.
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
logfmt = ticker.LogFormatterExponent(base=10.0, labelOnlyBase=True)


import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.2
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'lines.markersize' : 7}
mpl.rcParams.update(params)


__color_rotation__ = ['red', 'gold','coral','darkturquoise','royalblue','darkblue','m','hotpink','lightpink','dimgrey']
__marker_rotation__ = ['o', '^', 's', 'H', '+', '.', 'D', 'x', 'o', 'H', 's']
__linestyle_rotation__ = ['-', '--', ':', '-.']


def create_noisefree_plots():
    # Load files for NSIM
    savestr_base_nsim = 'noise_free_helix4'
    filename_errors_nsim = '../img/' + savestr_base_nsim + '/errors'
    tan_err_nsim = np.load(filename_errors_nsim + '/tangent_error.npy')
    f_test_nsim = np.load(filename_errors_nsim + '/f_error_test.npy')
    # Load files for kNN
    savestr_base_kNN = 'helix_without_noise_kNN'
    filename_errors_kNN = '../img/' + savestr_base_kNN + '/errors'
    f_test_kNN = np.load(filename_errors_kNN + '/f_error_test.npy')
    N = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
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
        plt.errorbar(N, tan_err_nsim_mean[:-1,i,:,:,:,:], tan_err_nsim_std[:-1,i,:,:,:,:], fmt = __marker_rotation__[i] + '-', label = "D = {0}".format(d))
    ax1.plot(N, np.divide(1.0, N) * 10000, 'k',label = r'$N^{-1}$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(logfmt)
    ax1.yaxis.set_major_formatter(logfmt)
    plt.legend()
    ax1.set_xlabel(r'$N$')
    ax1.set_ylabel(r'RMSE$(\Vert a_j - \hat a_j\Vert)$')
    # Make plot for function error
    fig2 = plt.figure()
    ax2 = plt.gca()
    for i, d in enumerate(D):
        ax2.errorbar(N, f_test_nsim_mean[:-1,i,:,:,:,:], f_test_nsim_std[:-1,i,:,:,:,:], fmt = __marker_rotation__[i] + '-', label = "(NSIM) $D = {0}$".format(d))
    for i, d in enumerate(D):
        ax2.errorbar(N, f_test_kNN_mean[0,:-1,i,:,:], f_test_kNN_std[0,:-1,i,:,:], fmt=__marker_rotation__[i+2] + '--', label = "(KNN) $D = {0}$".format(d))
    ax2.plot(N, np.divide(1.0, N) * 100, 'k', label = r'$N^{-1}$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.xaxis.set_major_formatter(logfmt)
    ax2.yaxis.set_major_formatter(logfmt)
    plt.legend(ncol = 1)
    ax2.set_xlabel(r'$N$')
    ax2.set_ylabel(r'RMSE$(|\hat f_1 - f_1|)$')
    fig1.savefig(filename_errors_nsim + '/noisefree_tangent_error.pdf', format = 'pdf', )
    fig2.savefig(filename_errors_nsim + '/noisefree_fun_error.pdf', format = 'pdf', )
    plt.show()


def create_noisy_plots():
    # Load files for NSIM
    savestr_base_nsim = 'noisy_helix10'
    filename_errors_nsim = '../img/' + savestr_base_nsim + '/errors'
    tan_err_nsim = np.load(filename_errors_nsim + '/tangent_error.npy')
    f_CV_nsim = np.load(filename_errors_nsim + '/f_error_CV.npy')
    f_test_nsim = np.load(filename_errors_nsim + '/f_error_test.npy')
    # Load files for SIM
    savestr_base_sim = 'noisy_identity2'
    filename_errors_sim = '../img/' + savestr_base_sim + '/errors'
    tan_err_sim = np.load(filename_errors_sim + '/tangent_error.npy')
    f_CV_sim = np.load(filename_errors_sim + '/f_error_CV.npy')
    f_test_sim = np.load(filename_errors_sim + '/f_error_test.npy')
    # Perform crossvalidation, i.e. search for best k + level set combination in f_CV_nsim for each run
    tan_errCVed_nsim = np.zeros(f_test_nsim.shape[0:4] + (1, 1, f_test_nsim.shape[6], f_test_nsim.shape[7]))
    f_testCVed_nsim = np.zeros(f_test_nsim.shape[0:4] + (1, 1, f_test_nsim.shape[6], f_test_nsim.shape[7]))
    tan_errCVed_sim = np.zeros(f_test_sim.shape[0:4] + (1, 1, f_test_sim.shape[6], f_test_sim.shape[7]))
    f_testCVed_sim = np.zeros(f_test_sim.shape[0:4] + (1, 1, f_test_sim.shape[6], f_test_sim.shape[7]))
    for i1 in range(f_test_nsim.shape[0]):
        for i2 in range(f_test_nsim.shape[1]):
            for i3 in range(f_test_nsim.shape[2]):
                for i4 in range(f_test_nsim.shape[3]):
                    for i6 in range(f_test_nsim.shape[6]):
                        for i7 in range(f_test_nsim.shape[7]):
                            # Get indices of minimum for nsim
                            row, col = np.unravel_index(f_test_nsim[i1,i2,i3,i4,:,:,i6,i7].argmin(), f_test_nsim[i1,i2,i3,i4,:,:,i6,i7].shape)
                            print row, col
                            tan_errCVed_nsim[i1,i2,i3,i4,0,0,i6,i7] = tan_err_nsim[i1,i2,i3,i4,row,col,i6,i7]
                            f_testCVed_nsim[i1,i2,i3,i4,0,0,i6,i7] = f_test_nsim[i1,i2,i3,i4,row,col,i6,i7]
                            # Get indices of minimum for sim
                            row, col = np.unravel_index(f_test_sim[i1,i2,i3,i4,:,:,i6,i7].argmin(), f_test_sim[i1,i2,i3,i4,:,:,i6,i7].shape)
                            tan_errCVed_sim[i1,i2,i3,i4,0,0,i6,i7] = tan_err_sim[i1,i2,i3,i4,row,col,i6,i7]
                            f_testCVed_sim[i1,i2,i3,i4,0,0,i6,i7] = f_test_sim[i1,i2,i3,i4,row,col,i6,i7]

    N = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]#, 512000]
    var_f = [0.05, 0.1, 0.2, 0.4]
    # Mean and std for nsim
    tan_errCVed_nsim_mean = np.mean(tan_errCVed_nsim, axis = 7)
    tan_errCVed_nsim_std = np.std(tan_errCVed_nsim, axis = 7)
    f_testCVed_nsim_mean = np.mean(f_testCVed_nsim, axis = 7)
    f_testCVed_nsim_std = np.std(f_testCVed_nsim, axis = 7)
    # Mean and std for sim
    tan_errCVed_sim_mean = np.mean(tan_errCVed_sim, axis = 7)
    tan_errCVed_sim_std = np.std(tan_errCVed_sim, axis = 7)
    f_testCVed_sim_mean = np.mean(f_testCVed_sim, axis = 7)
    f_testCVed_sim_std = np.std(f_testCVed_sim, axis = 7)
    # Make plot for tangent error
    fig1 = plt.figure()
    ax1 = plt.gca()
    for i, var_f_local in enumerate(var_f):
        plt.errorbar(N, tan_errCVed_nsim_mean[:,:,:,:,:,:,i], tan_errCVed_nsim_std[:,:,:,:,:,:,i], c = __color_rotation__[i], fmt = __marker_rotation__[i] + '-', label = r'$c = {0}$'.format(var_f_local))
        plt.errorbar(N, tan_errCVed_sim_mean[:,:,:,:,:,:,i], tan_errCVed_sim_std[:,:,:,:,:,:,i], c = __color_rotation__[i], fmt = __marker_rotation__[i] + '-', label = r'$c = {0}$'.format(var_f_local))
    # ax1.plot(N, np.divide(1.0, np.sqrt(N)) * 1000, 'k',label = r'N^{-1}')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(logfmt)
    ax1.yaxis.set_major_formatter(logfmt)
    plt.legend()
    ax1.set_xlabel(r'$N$')
    ax1.set_ylabel(r'RMSE$(\Vert a_j - \hat a_j\Vert)$')
    # Make plot for function error
    fig2 = plt.figure()
    ax2 = plt.gca()
    for i, var_f_local in enumerate(var_f):
        ax2.errorbar(N, f_testCVed_nsim_mean[:,:,:,:,:,:,i], f_testCVed_nsim_std[:,:,:,:,:,:,i], c = __color_rotation__[i], fmt = __marker_rotation__[i] + '-', label = r'(Helix) $c= {0}$'.format(var_f_local))
        ax2.errorbar(N, f_testCVed_sim_mean[:,:,:,:,:,:,i], f_testCVed_sim_std[:,:,:,:,:,:,i], c = __color_rotation__[i], fmt = __marker_rotation__[i] + '--', label = r'(SIM) $c= {0}$'.format(var_f_local))
    ax2.plot(N, np.divide(1.0, np.power(N, 1.0/3.0)) , 'k', label = r'$N^{-1/3}$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.xaxis.set_major_formatter(logfmt)
    ax2.yaxis.set_major_formatter(logfmt)
    plt.legend(ncol = 2)
    ax2.set_xlabel(r'$N$')
    ax2.set_ylabel(r'RMSE$(|\hat f_{CV} - f_{CV}|)$')
    fig1.savefig(filename_errors_nsim + '/noisy_tangent_error.pdf', format = 'pdf', )
    fig2.savefig(filename_errors_nsim + '/noisy_fun_error.pdf', format = 'pdf', )
    plt.show()




if __name__ == "__main__":
    create_noisy_plots()
