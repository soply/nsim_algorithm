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


def plot_all_errors(x_dict, errors, cols = ['var_f', 'n_samples',
                    'ambient_dim', 'n_noise', 'eucl_ball_frac'],
                    polyfit = True,
                    save = [],
                    savestr = None, additional_savestr = '',
                    yaxis_scaling = 'log',
                    ylabel = 'Rel L2 Error',
                    legend_pos =  'best',
                    **kwargs):
    errors_mean = np.mean(errors, axis = 5)
    errors_std = np.std(errors, axis = 5)
    # Noise variance
    fig = plt.figure(figsize = (9,6))
    ax1 = plt.gca()
    for i2, n_samples in enumerate(x_dict['n_samples']):
        for i3, ambient_dim in enumerate(x_dict['ambient_dim']):
            for i4, n_noise in enumerate(x_dict['n_noise']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax1.errorbar(x_dict['var_f'], errors_mean[:,i2,i3,i4,i5,i6],
                                     errors_std[:,i2,i3,i4,i5,i6], fmt='--',
                                 label = 'N={0} D={1} sig={2} lam={3}'.format(n_samples,
                                 ambient_dim, n_noise, lam))
    ax1.semilogy(x_dict['var_f'], np.log(x_dict['var_f'])/np.array(x_dict['var_f']),
             label = 'LOG/LIN-Rate')
    ax1.semilogy(x_dict['var_f'], 1.0/np.array(x_dict['var_f']),
             label = 'LIN-Rate')
    ax1.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax1.set_yscale('linear')
    ax1.legend(loc = legend_pos, labelspacing=0.2)
    ax1.set_xlabel('var_f')
    ax1.set_ylabel(ylabel)
    if 'var_f' in save:
        fig.savefig('../img/' + savestr + '/' + additional_savestr + 'vs_var_f.png')
    # n_samples on x axis
    fig = plt.figure(figsize = (12,8))
    ax2 = plt.gca()
    ctr = 0
    for i2, var_f in enumerate(x_dict['var_f']):
        for i3, ambient_dim in enumerate(x_dict['ambient_dim']):
            for i4, n_noise in enumerate(x_dict['n_noise']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        if polyfit:
                            pp = np.polyfit(np.log(x_dict['n_samples'][-4:]),
                                              np.log(errors_mean[i2,-4:,i3,i4,i5,i6]), deg = 1)
                            polyfit_x = np.array((x_dict['n_samples'][0],x_dict['n_samples'][-1]))
                            ax2.semilogy(polyfit_x, np.exp(pp[1]) * polyfit_x ** pp[0],
                                         color =__color_rotation__[ctr])

                            ax2.errorbar(x_dict['n_samples'], errors_mean[i2,:,i3,i4,i5,i6],
                                     errors_std[i2,:,i3,i4,i5,i6], fmt= '--'+__marker_rotation__[ctr],
                                     color =__color_rotation__[ctr],
                                     label = r'$D={:d}$ $\sigma={:.1f}$, $\sigma_f={:.5f}$, $\lambda={:.2f}$ Rate={:.2f}'.format(
                                     ambient_dim, n_noise, np.sqrt(var_f), lam, pp[0]))
                        else:
                            ax2.semilogy(x_dict['n_samples'], var_f * np.ones(len(x_dict['n_samples'])),
                                         '--', color =__color_rotation__[ctr])
                            ax2.errorbar(x_dict['n_samples'], errors_mean[i2,:,i3,i4,i5,i6],
                                     errors_std[i2,:,i3,i4,i5,i6], fmt= '-'+__marker_rotation__[ctr],
                                     color =__color_rotation__[ctr],
                                     label = (r'$\sigma_f^2 = {:.0e}$'.format(
                                        var_f)))
                            ax2.set_yscale('log')
                        ctr += 1

    # Fit line through plot
    # ax2.semilogy(x_dict['n_samples'], np.log(x_dict['n_samples'])/np.array(x_dict['n_samples']), color = 'k',
    #          label = r'$\log(N)/N^{-2/3}$-Rate')
    # ax2.semilogy(x_dict['n_samples'], np.power(np.array(x_dict['n_samples']), -2.0/3.0), color = 'k',
    #          label = r'$N^{-2/3}$-Rate')
    # ####################### Add KNN plot #####################
    # __knn_color_rotation__ = ['k', 'gray', 'sandybrown', 'tan']
    # ctr_knn = 0
    # # for i4, n_noise in enumerate(x_dict['n_noise']):
    # #     knn_comp = kwargs.get('knn_comp')
    # #     knn_comp_mean = np.mean(knn_comp, axis = 4)
    # #     knn_comp_std = np.std(knn_comp, axis = 4)
    # #     ax2.errorbar(x_dict['n_samples'], knn_comp_mean[0,:,0,i4],
    # #              knn_comp_std[0,:,0,i4], fmt= '--'+__marker_rotation__[ctr_knn],
    # #              color =__knn_color_rotation__[ctr_knn],
    # #              label = (r'kNN-Reg: $\sqrt{\sigma}$' + r'$={:.1f}$'.format(
    # #              n_noise)))
    # #     ctr_knn += 1
    # for i3, ambient_dim in enumerate(x_dict['ambient_dim']):
    #     knn_comp = kwargs.get('knn_comp')
    #     knn_comp_mean = np.mean(knn_comp, axis = 4)
    #     knn_comp_std = np.std(knn_comp, axis = 4)
    #     ax2.errorbar(x_dict['n_samples'], knn_comp_mean[0,:,i3,0],
    #              knn_comp_std[0,:,i3,0], fmt= '--'+__marker_rotation__[ctr_knn],
    #              color =__knn_color_rotation__[ctr_knn],
    #              label = r'1-NN, $D={:d}$'.format(
    #              ambient_dim))
    #     ctr_knn += 1
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.xaxis.set_major_formatter(logfmt)
    ax2.yaxis.set_major_formatter(logfmt)

    if yaxis_scaling == 'lin':
        ax2.set_yscale('linear')
    ax2.legend(loc = legend_pos, ncol = 1, labelspacing = 0.1)
    ax2.set_xlabel(r'$N$ $[log_{10}]$')
    ax2.set_ylabel(ylabel)# + r' $[log_{10}]$')
    if 'n_samples' in save:
        fig.savefig('../img/' + savestr + '/' + additional_savestr + 'vs_n_samples.png')
    # ambient_dim on x axis
    fig = plt.figure(figsize = (9,6))
    ax3 = plt.gca()
    for i2, var_f in enumerate(x_dict['var_f']):
        for i3, n_samples in enumerate(x_dict['n_samples']):
            for i4, n_noise in enumerate(x_dict['n_noise']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax3.errorbar(x_dict['ambient_dim'], errors_mean[i2,i3,:,i4,i5,i6],
                                 errors_std[i2,i3,:,i4,i5,i6], fmt='--',
                                 label = 'N={0} sig={1} lam={2} var_f={3}'.format(
                                 n_samples, n_noise, lam, var_f))
    ax3.semilogy(x_dict['ambient_dim'], np.sqrt(x_dict['ambient_dim']),
             label = 'SQRT-Rate')
    ax3.semilogy(x_dict['ambient_dim'], np.array(x_dict['ambient_dim']),
             label = 'LIN-Rate')
    ax3.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax3.set_yscale('linear')
    ax3.legend(loc = legend_pos, labelspacing=0.2)
    ax3.set_xlabel('D')
    ax3.set_ylabel(ylabel)
    if 'ambient_dim' in save:
        fig.savefig('../img/' + savestr + '/' + additional_savestr + 'vs_ambient_dim.png')
    # n_noise on x axis
    fig = plt.figure(figsize = (9,6))
    ax4 = plt.gca()
    for i2, var_f in enumerate(x_dict['var_f']):
        for i3, n_samples in enumerate(x_dict['n_samples']):
            for i4, ambient_dim in enumerate(x_dict['ambient_dim']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax4.errorbar(x_dict['n_noise'], errors_mean[i2,i3,i4,:,i5,i6],
                                 errors_std[i2,i3,i4,:,i5,i6], fmt='--',
                                 label = 'N={0} D={1} lam={2} var_f={3}'.format(
                                 n_samples, ambient_dim, lam, var_f))
    ax4.semilogy(x_dict['n_noise'], np.sqrt(x_dict['n_noise']),
             label = 'SQRT-Rate')
    ax4.semilogy(x_dict['n_noise'], np.array(x_dict['n_noise']),
             label = 'LIN-Rate')
    ax4.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax4.set_yscale('linear')
    ax4.legend(loc =legend_pos, labelspacing=0.2)
    ax4.set_xlabel('sigma')
    ax4.set_ylabel(ylabel)
    if 'n_noise' in save:
        fig.savefig('../img/' + '/' + additional_savestr + 'vs_n_noise.png')
    # lam on x axi
    fig = plt.figure(figsize = (9,6))
    ax5 = plt.gca()
    for i2, var_f in enumerate(x_dict['var_f']):
        for i3, n_samples in enumerate(x_dict['n_samples']):
            for i4, ambient_dim in enumerate(x_dict['ambient_dim']):
                for i5, n_noise in enumerate(x_dict['n_noise']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax5.errorbar(x_dict['eucl_ball_frac'], errors_mean[i2,i3,i4,i5,:,i6],
                             errors_std[i2,i3,i4,i5,:,i6], fmt='--',
                             label = 'N={0} D={1} sig={2} var_f={3}'.format(
                             n_samples, ambient_dim, n_noise, var_f))
    ax5.semilogy(x_dict['eucl_ball_frac'], np.log(x_dict['eucl_ball_frac'])/np.array(x_dict['eucl_ball_frac']),
             label = 'LOG/LIN-Rate')
    ax5.semilogy(x_dict['eucl_ball_frac'], 1.0/np.array(x_dict['eucl_ball_frac']),
             label = 'LIN-Rate')
    ax5.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax5.set_yscale('linear')
    ax5.legend(loc =legend_pos, labelspacing=0.2)
    ax5.set_xlabel('eucl_ball_frac')
    ax5.set_ylabel(ylabel)
    if 'eucl_ball_frac' in save:
        fig.savefig('../img/' + '/' + additional_savestr + 'vs_lam.png')


def plot_all_errors_old(x_dict, errors, cols = ['n_level_sets', 'n_samples',
                    'ambient_dim', 'n_noise', 'eucl_ball_frac'], save = [],
                    savestr = None, additional_savestr = '',
                    yaxis_scaling = 'log',
                    ylabel = 'Rel L2 Error',
                    legend_pos =  'best'):
    errors_mean = np.mean(errors, axis = 5)
    errors_std = np.std(errors, axis = 5)
    # Level sets on x axis
    fig = plt.figure(figsize = (9,6))
    ax1 = plt.gca()
    for i2, n_samples in enumerate(x_dict['n_samples']):
        for i3, ambient_dim in enumerate(x_dict['ambient_dim']):
            for i4, n_noise in enumerate(x_dict['n_noise']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax1.errorbar(x_dict['n_level_sets'], errors_mean[:,i2,i3,i4,i5,i6],
                                     errors_std[:,i2,i3,i4,i5,i6], fmt='--',
                                 label = 'N={0} D={1} sig={2} lam={3}'.format(n_samples,
                                 ambient_dim, n_noise, lam))
    ax1.semilogy(x_dict['n_level_sets'], np.log(x_dict['n_level_sets'])/np.array(x_dict['n_level_sets']),
             label = 'LOG/LIN-Rate')
    ax1.semilogy(x_dict['n_level_sets'], 1.0/np.array(x_dict['n_level_sets']),
             label = 'LIN-Rate')
    ax1.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax1.set_yscale('linear')
    ax1.legend(loc = legend_pos, labelspacing=0.2)
    ax1.set_xlabel('n_level_sets')
    ax1.set_ylabel(ylabel)
    if 'level_sets' in save:
        fig.savefig('../img/' + savestr + '/' + additional_savestr + 'vs_level_sets.png')
    # n_samples on x axis
    fig = plt.figure(figsize = (9,6))
    ax2 = plt.gca()
    for i2, n_level_sets in enumerate(x_dict['n_level_sets']):
        for i3, ambient_dim in enumerate(x_dict['ambient_dim']):
            for i4, n_noise in enumerate(x_dict['n_noise']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax2.errorbar(x_dict['n_samples'], errors_mean[i2,:,i3,i4,i5,i6],
                                 errors_std[i2,:,i3,i4,i5,i6], fmt='--',
                                 label = 'D={0} sig={1} lam={2}'.format(ambient_dim, n_noise, lam))
    ax2.semilogy(x_dict['n_samples'], np.log(x_dict['n_samples'])/np.array(x_dict['n_samples']),
             label = 'LOG/LIN-Rate')
    ax2.semilogy(x_dict['n_samples'], 1.0/np.array(x_dict['n_samples']),
             label = 'LIN-Rate')
    ax2.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax2.set_yscale('linear')
    ax2.legend(loc = legend_pos, labelspacing=0.2)
    ax2.set_xlabel('N')
    ax2.set_ylabel(ylabel)
    if 'n_samples' in save:
        fig.savefig('../img/' + savestr + '/' + additional_savestr + 'vs_n_samples.png')
    # ambient_dim on x axis
    fig = plt.figure(figsize = (9,6))
    ax3 = plt.gca()
    for i2, n_level_sets in enumerate(x_dict['n_level_sets']):
        for i3, n_samples in enumerate(x_dict['n_samples']):
            for i4, n_noise in enumerate(x_dict['n_noise']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax3.errorbar(x_dict['ambient_dim'], errors_mean[i2,i3,:,i4,i5,i6],
                                 errors_std[i2,i3,:,i4,i5,i6], fmt='--',
                                 label = 'N={0} sig={1} lam={2}'.format(n_samples, n_noise, lam))
    ax3.semilogy(x_dict['ambient_dim'], np.sqrt(x_dict['ambient_dim']),
             label = 'SQRT-Rate')
    ax3.semilogy(x_dict['ambient_dim'], np.array(x_dict['ambient_dim']),
             label = 'LIN-Rate')
    ax3.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax3.set_yscale('linear')
    ax3.legend(loc = legend_pos, labelspacing=0.2)
    ax3.set_xlabel('D')
    ax3.set_ylabel(ylabel)
    if 'ambient_dim' in save:
        fig.savefig('../img/' + savestr + '/' + additional_savestr + 'vs_ambient_dim.png')
    # n_noise on x axis
    fig = plt.figure(figsize = (9,6))
    ax4 = plt.gca()
    for i2, n_level_sets in enumerate(x_dict['n_level_sets']):
        for i3, n_samples in enumerate(x_dict['n_samples']):
            for i4, ambient_dim in enumerate(x_dict['ambient_dim']):
                for i5, lam in enumerate(x_dict['eucl_ball_frac']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax4.errorbar(x_dict['n_noise'], errors_mean[i2,i3,i4,:,i5,i6],
                                 errors_std[i2,i3,i4,:,i5,i6], fmt='--',
                                 label = 'N={0} D={1} lam={2}'.format(n_samples, ambient_dim, lam))
    ax4.semilogy(x_dict['n_noise'], np.sqrt(x_dict['n_noise']),
             label = 'SQRT-Rate')
    ax4.semilogy(x_dict['n_noise'], np.array(x_dict['n_noise']),
             label = 'LIN-Rate')
    ax4.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax4.set_yscale('linear')
    ax4.legend(loc =legend_pos, labelspacing=0.2)
    ax4.set_xlabel('sigma')
    ax4.set_ylabel(ylabel)
    if 'n_noise' in save:
        fig.savefig('../img/' + '/' + additional_savestr + 'vs_n_noise.png')
    # lam on x axi
    fig = plt.figure(figsize = (9,6))
    ax5 = plt.gca()
    for i2, n_level_sets in enumerate(x_dict['n_level_sets']):
        for i3, n_samples in enumerate(x_dict['n_samples']):
            for i4, ambient_dim in enumerate(x_dict['ambient_dim']):
                for i5, n_noise in enumerate(x_dict['n_noise']):
                    for i6, oversample in enumerate(x_dict['fac_oversample_lvset']):
                        ax5.errorbar(x_dict['eucl_ball_frac'], errors_mean[i2,i3,i4,i5,:,i6],
                             errors_std[i2,i3,i4,i5,:,i6], fmt='--',
                             label = 'N={0} D={1} sig={2}'.format(n_samples, ambient_dim, n_noise))
    ax5.semilogy(x_dict['eucl_ball_frac'], np.log(x_dict['eucl_ball_frac'])/np.array(x_dict['eucl_ball_frac']),
             label = 'LOG/LIN-Rate')
    ax5.semilogy(x_dict['eucl_ball_frac'], 1.0/np.array(x_dict['eucl_ball_frac']),
             label = 'LIN-Rate')
    ax5.set_xscale('log')
    if yaxis_scaling == 'lin':
        ax5.set_yscale('linear')
    ax5.legend(loc =legend_pos, labelspacing=0.2)
    ax5.set_xlabel('eucl_ball_frac')
    ax5.set_ylabel(ylabel)
    if 'eucl_ball_frac' in save:
        fig.savefig('../img/' + '/' + additional_savestr + 'vs_lam.png')
