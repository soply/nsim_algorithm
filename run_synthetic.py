# coding: utf8

import json
# I/O import
import os
import shutil
import sys
import tempfile
import time

import numpy as np
# For parallelization
from joblib import Parallel, delayed

# My imports
from estimator import NSIM_Estimator
from synthethic_problem_factory.functions_on_manifolds import (RandomPolynomialIncrements,
                                                               ahalf_polynomial_on_first,
                                                               identity_function,
                                                               injective_on_first_coordinate,
                                                               norm_function,
                                                               quadratic_on_first_coordinate,
                                                               randomPolynomialIncrements_for_parallel,
                                                               sinus)
from synthethic_problem_factory.manifolds_as_curves_classes import (Circle_Piece_2D,
                                                                    Circle_Segment_Builder,
                                                                    Helix_Curve_3D,
                                                                    Identity_kD,
                                                                    S_Curve_2D)
from synthethic_problem_factory.sample_synthetic_data import \
    sample_1D_fromClass
# from visualisation.error_plots import plot_all_errors
# from visualisation.vis_nD import *


# Loss function
def MSE(prediction, reference):
    """ Mean squared error """
    return np.sum(np.linalg.norm(prediction - reference, axis = 0) ** 2)/ \
                np.sum(np.linalg.norm(reference, axis = 0) ** 2)

def RMSE(prediction, reference):
    """ Root Mean squared error """
    return np.sqrt(MSE(prediction, reference))

def UAE(prediction, reference):
    """ Uniform absolute error max_i |v_i - hat v_i| """
    return np.max(np.divide(np.linalg.norm(prediction - reference, axis = 0),
                     np.linalg.norm(reference, axis = 0)))


def run_example(n_samples,
                ambient_dim,
                noise,
                ball_radius,
                n_neighbors,
                n_levelsets,
                var_f,
                xlow,
                xhigh,
                f_on_manifold,
                f_tangent_error,
                f_f_error_CV,
                f_f_error_test,
                comp_time,
                rep, i1, i2, i3, i4, i6, i7, # Indices to write into
                xlow_error = None,
                xhigh_error = None,
                levelset_modus = 'number',
                neighbor_modus = 'number',
                CV_split = 0.1,
                savestr_base = None,
                apply_rotation = None,
                random_state = None,
                args_f = None):
    # Set segment in which to compute error
    if xlow_error is None:
        xlow_error = xlow
    elif xhigh_error is None:
        xhigh_error = xhigh
    # Construct manifold
    sequence = [(1,0),(1,1),(1,0)]
    f_manifold = Helix_Curve_3D(ambient_dim)
    if random_state is not None:
        np.random.set_state(random_state)
    # Split n_samples into training and CV
    n_samples_CV = np.floor(CV_split * n_samples).astype('int')
    n_samples_train = n_samples - n_samples_CV
    # Sample training data
    pdisc, points, normalspaces, fval, fval_clean, tangentspaces, basepoints = \
                                            sample_1D_fromClass(xlow,
                                                                xhigh,
                                                                f_manifold,
                                                                f_on_manifold,
                                                                n_samples_train,
                                                                noise,
                                                                var_f = var_f,
                                                                tube = 'l2',
                                                                args_f = args_f)
    idx_train_error = np.where(np.logical_and(pdisc > xlow_error - 1e-10,
                                           pdisc < xhigh_error + 1e-10))[0]
    # Sample CV data
    pdisc_CV, points_CV, normalspaces_CV, fval_CV, fval_clean_CV, tangentspaces_CV, basepoints_CV = \
                                            sample_1D_fromClass(xlow,
                                                                xhigh,
                                                                f_manifold,
                                                                f_on_manifold,
                                                                n_samples_CV,
                                                                noise,
                                                                var_f = var_f,
                                                                tube = 'l2',
                                                                args_f = args_f)
    idx_cv_error = np.where(np.logical_and(pdisc_CV > xlow_error - 1e-10,
                                           pdisc_CV < xhigh_error + 1e-10))[0]
    # Get test samples
    n_test_samples = 1000
    pdisc_test, points_test, normalspaces_test, fval_test, fval_clean_test, tangentspaces_test, basepoints_test = \
                            sample_1D_fromClass(xlow,
                                                xhigh, f_manifold, f_on_manifold,
                                                n_test_samples, noise,
                                                var_f = 0.00,
                                                tube = 'l2',
                                                args_f = args_f)
    idx_test_error = np.where(np.logical_and(pdisc_test > xlow_error - 1e-10,
                                             pdisc_test < xhigh_error + 1e-10))[0]
    if apply_rotation is not None:
        # Apply rotation to basepoints
        basepoints = apply_rotation.dot(basepoints)
        basepoints_CV = apply_rotation.dot(basepoints_CV)
        basepoints_test = apply_rotation.dot(basepoints_test)
        # Apply rotation to points
        points = apply_rotation.dot(points)
        points_CV = apply_rotation.dot(points_CV)
        points_test = apply_rotation.dot(points_test)
    # Calculate number of level sets
    if levelset_modus == 'factor':
        n_levelsets_mod = np.floor(float(n_samples_train)/float(ambient_dim * n_levelsets)).astype('int')
    else:
        n_levelsets_mod = n_levelsets
    print "n_levelsets = {0}".format(n_levelsets)
    # Turn nNeighbors into number of samples
    if neighbor_modus == 'factor':
        nNei = np.ceil(np.array(n_neighbors) * np.power(n_samples, 2.0/3.0)).astype('int')
    else:
        nNei = n_neighbors
    start = time.time()
    nsim_kNN = NSIM_Estimator(n_neighbors = nNei,
                            n_levelsets = n_levelsets_mod,
                            ball_radius = ball_radius,
                            split_by = 'dyadic')
    try:
        nsim_kNN = nsim_kNN.fit(points.T, fval)
        fval_predict_CV = nsim_kNN.predict(points_CV.T)
        fval_predict_test = nsim_kNN.predict(points_test.T)
        end = time.time()
        # Error Calculations
        # Function Error
        if isinstance(n_neighbors, (int,long)):
            f_f_error_CV[i1,i2,i3,i4,0,i6,i7,rep] = RMSE(np.reshape(fval_predict_CV, (1,-1))[0,idx_cv_error], np.reshape(fval_CV, (1,-1))[0,idx_cv_error])
            f_f_error_test[i1,i2,i3,i4,0,i6,i7,rep] = RMSE(np.reshape(fval_predict_test, (1,-1))[0,idx_test_error], np.reshape(fval_test, (1,-1))[0,idx_test_error])
        else:
            for l, _ in enumerate(n_neighbors):
                f_f_error_CV[i1,i2,i3,i4,l,i6,i7,rep] = RMSE(np.reshape(fval_predict_CV[:,l], (1,-1))[0,idx_cv_error], np.reshape(fval_CV, (1,-1))[0,idx_cv_error])
                f_f_error_test[i1,i2,i3,i4,l,i6,i7,rep] = RMSE(np.reshape(fval_predict_test[:,l], (1,-1))[0,idx_test_error], np.reshape(fval_test, (1,-1))[0,idx_test_error])
        # Tangent Error
        J =  len(set(nsim_kNN.labels_))
        tan_errs = np.zeros(J)
        for i in range(J):
            if len(np.intersect1d(np.where(nsim_kNN.labels_ == i), idx_train_error)) > 0:
                tmean = np.mean(pdisc[nsim_kNN.labels_ == i])
                real_tangent = apply_rotation.dot(f_manifold.get_tangent(tmean))
                tan_errs[i] = np.minimum(np.linalg.norm(real_tangent - nsim_kNN.tangents_[i,:]),
                                np.linalg.norm(real_tangent + nsim_kNN.tangents_[i,:]))
        # Compute RMSE
        f_tangent_error[i1,i2,i3,i4,:,i6,i7,rep] = np.sqrt(np.mean(np.square(tan_errs)))
    except RuntimeError as e:
        print e
        f_f_error_CV[i1,i2,i3,i4,:,i6,i7,rep] = 1e16
        f_f_error_test[i1,i2,i3,i4,:,i6,i7,rep] = 1e16
        f_tangent_error[i1,i2,i3,i4,:,i6,i7,rep] = 1e16
    # Computational time
    end = time.time()
    comp_time[i1,i2,i3,i4,:,i6,i7,rep] = end - start
    print "N : {0}  D : {1}    R : {2}   sigma_eps : {3}    rep : {4}   Time : {5} s".format(n_samples, ambient_dim, noise, var_f, rep, end - start)
    print "Tangential error: ", f_tangent_error[i1,i2,i3,i4,:,i6,i7,rep]
    print "Fval error (CV): ", f_f_error_CV[i1,i2,i3,i4,:,i6,i7,rep]
    print "Fval error (Test): ", f_f_error_test[i1,i2,i3,i4,:,i6,i7,rep]

if __name__ == "__main__":
    # Get number of jobs from sys.argv
    if len(sys.argv) > 1:
        n_jobs = int(sys.argv[1])
    else:
        n_jobs = 4 # Default 4 jobs
    print 'Using n_jobs = {0}'.format(n_jobs)
    # Cuve start and endpoint
    xlow = 0.0 * np.pi
    xhigh = 3 * np.pi
    np.random.seed(123123)
    fun_obj = RandomPolynomialIncrements(xlow, xhigh, 2, 100,
                                         coefficient_bound = [1.0, 1.5])
    # Calculate variance
    flower, fupper = fun_obj.eval(xlow), fun_obj.eval(xhigh)
    avg_grad = (fupper - flower)/(xhigh - xlow)
    # fun_obj.plot(white_noise_var=1e-4, n = 1000)
    # Parameters
    run_for = {
        'n_samples' : [1000, 2000, 4000],
        'n_noise' : [0.25],
        'ambient_dim' : [12],
        'var_f' : [0.0],
        'ball_radius' : [0.5],
        'n_levelsets' : [2 ** j for j in range(11)],
        'n_neighbors' : [1],
        'levelset_modus' : 'number',
        'neighbor_modus' : 'number',
        'CV_split' : 0.01,
    }

    # Sample rotations for each dimension
    rotations = {}
    for j, D in enumerate(run_for['ambient_dim']):
        from scipy.stats import special_ortho_group
        rotations[D] = special_ortho_group.rvs(D)

    repititions = 3
    savestr_base = 'test'
    filename_errors = '../img/' + savestr_base + '/errors'

    try:
        f_tangent_error = np.load(filename_errors + '/tangent_error.npy')
        f_f_error_CV = np.load(filename_errors + '/f_error_CV.npy')
        f_f_error_test = np.load(filename_errors + '/f_error_test.npy')
    except IOError:
        if not os.path.exists(filename_errors):
            os.makedirs(filename_errors)
            # Save a log file
        with open(filename_errors + '/log.txt', 'w') as file:
            file.write(json.dumps(run_for, indent=4)) # use `json.loads` to do the reverse
        tmp_folder = tempfile.mkdtemp()
        dummy_for_shape = np.zeros((len(run_for['n_samples']),
                                      len(run_for['ambient_dim']),
                                      len(run_for['n_noise']),
                                      len(run_for['ball_radius']),
                                      len(run_for['n_neighbors']),
                                      len(run_for['n_levelsets']),
                                      len(run_for['var_f']),
                                      repititions))
        try:
            # Create error containers
            f_tangent_error = np.memmap(os.path.join(tmp_folder, 'tangent_error'), dtype='float64',
                                       shape=dummy_for_shape.shape, mode='w+')
            f_f_error_CV = np.memmap(os.path.join(tmp_folder, 'f_error_CV'), dtype='float64',
                                       shape=dummy_for_shape.shape, mode='w+')
            f_f_error_test = np.memmap(os.path.join(tmp_folder, 'f_error_test'), dtype='float64',
                                       shape=dummy_for_shape.shape, mode='w+')
            comp_time = np.memmap(os.path.join(tmp_folder, 'comp_time'), dtype='float64',
                                       shape=dummy_for_shape.shape, mode='w+')
            random_state = np.random.get_state()
            # Run experiments in parallel
            Parallel(n_jobs=n_jobs)(delayed(run_example)(
                                run_for['n_samples'][i1],
                                run_for['ambient_dim'][i2],
                                run_for['n_noise'][i3],
                                run_for['ball_radius'][i4],
                                run_for['n_neighbors'],
                                run_for['n_levelsets'][i6],
                                run_for['var_f'][i7],
                                xlow,
                                xhigh,
                                randomPolynomialIncrements_for_parallel,
                                f_tangent_error,
                                f_f_error_CV,
                                f_f_error_test,
                                comp_time,
                                rep, i1, i2, i3, i4, i6, i7,
                                xlow_error = 0.0,
                                xhigh_error = 3.0 * np.pi,
                                levelset_modus = run_for['levelset_modus'],
                                neighbor_modus = run_for['neighbor_modus'],
                                CV_split = run_for['CV_split'],
                                savestr_base = savestr_base + '/' + str(rep) + '/',
                                apply_rotation = rotations[run_for['ambient_dim'][i2]],
                                args_f = (xlow, xhigh, fun_obj.get_bases(),
                                          fun_obj.get_coeffs()))
                                for rep in range(repititions)
                                for i7 in range(len(run_for['var_f']))
                                for i1 in range(len(run_for['n_samples']))
                                for i2 in range(len(run_for['ambient_dim']))
                                for i3 in range(len(run_for['n_noise']))
                                for i4 in range(len(run_for['ball_radius']))
                                for i6 in range(len(run_for['n_levelsets'])))
            # Dump memmaps to files
            f_tangent_error.dump(filename_errors + '/tangent_error.npy')
            f_f_error_CV.dump(filename_errors + '/f_error_CV.npy')
            f_f_error_test.dump(filename_errors + '/f_error_test.npy')
            comp_time.dump(filename_errors + '/comp_time.npy')
        finally:
            try:
                shutil.rmtree(tmp_folder)
            except:
                print('Failed to delete: ' + tmp_folder)
    #
    # """ Post-Processing: Read errors and create error plots after model selection """
    # tangent_error = np.load(filename_errors + '/tangent_error.npy')
    # f_error_CV = np.load(filename_errors + '/f_error_CV.npy')
    # f_error_test = np.load(filename_errors + '/f_error_test.npy')
    # # Select the best according to the f_error_CV and plot those in plot
    # best_tangent_error = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
    #                   len(run_for['ambient_dim']), len(run_for['n_noise']),
    #                   len(run_for['ball_radius']),
    #                   len(run_for['n_levelsets']), repititions))
    # best_f_error_CV = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
    #                   len(run_for['ambient_dim']), len(run_for['n_noise']),
    #                   len(run_for['ball_radius']),
    #                   len(run_for['n_levelsets']), repititions))
    # best_f_error_test = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
    #                   len(run_for['ambient_dim']), len(run_for['n_noise']),
    #                   len(run_for['ball_radius']),
    #                   len(run_for['n_levelsets']), repititions))
    # # Store selected layers
    # chosen_layers = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
    #                   len(run_for['ambient_dim']), len(run_for['n_noise']),
    #                   len(run_for['ball_radius']),
    #                   len(run_for['n_levelsets']), repititions))
    # for rep in range(repititions):
    #     for j, var_f in enumerate(run_for['var_f']):
    #         for k, N in enumerate(run_for['n_samples']):
    #             for kk, D in enumerate(run_for['ambient_dim']):
    #                 for n, noise in enumerate(run_for['n_noise']):
    #                     for m, ball_radius in enumerate(run_for['ball_radius']):
    #                         for nn, n_levelsets in enumerate(run_for['n_levelsets']):
    #                             # Exclude zeros if layer was not useful anymore due to
    #                             # insufficient points
    #                             idx = np.argmin(f_error_CV[j,k,kk,n,m,:,nn,rep])
    #                             best_tangent_error[j,k,kk,n,m,nn,rep] = tangent_error[j,k,kk,n,m,idx,nn,rep]
    #                             best_f_error_CV[j,k,kk,n,m,nn,rep] = f_error_CV[j,k,kk,n,m,idx,nn,rep]
    #                             best_f_error_test[j,k,kk,n,m,nn,rep] = f_error_test[j,k,kk,n,m,idx,nn,rep]
    #                             chosen_layers[j,k,kk,n,m,nn,rep] = idx
    #                             print "N = {0}, D = {1}, (idx, value) = ({2}, {3})".format(N, D, idx, run_for['n_neighbors'][idx])
    # ############################## KNN STUFF ###########################
    # filename_errors_kNN =  '../img/normalKnn_without_noise2/errors/'
    # f_error_CV_kNN = np.load(filename_errors_kNN + '/f_error_CV.npy')
    # f_error_test_kNN = np.load(filename_errors_kNN + '/f_error_test.npy')
    # #Select the best according to the f_error_CV and plot those in plot
    # run_for_kNN = {
    #     'n_samples' : [2000, 4000, 8000, 16000, 32000, 64000, 128000],
    #     'n_noise' : [0.5],
    #     'ambient_dim' : [5,10,20],
    #     'var_f' : [0.0],
    #     'n_neighbors' : [1e-10],
    # }
    # best_f_error_CV_kNN = np.zeros((len(run_for_kNN['var_f']), len(run_for_kNN['n_samples']),
    #                               len(run_for_kNN['ambient_dim']), len(run_for_kNN['n_noise']),
    #                               repititions))
    # best_f_error_test_kNN = np.zeros((len(run_for_kNN['var_f']), len(run_for_kNN['n_samples']),
    #                               len(run_for_kNN['ambient_dim']), len(run_for_kNN['n_noise']),
    #                               repititions))
    #
    # # Store selected layers
    # chosen_K_kNN = np.zeros((len(run_for_kNN['var_f']), len(run_for_kNN['n_samples']),
    #                       len(run_for_kNN['ambient_dim']), len(run_for_kNN['n_noise']),
    #                       repititions))
    # for rep in range(repititions):
    #     for j, var_f in enumerate(run_for_kNN['var_f']):
    #         for k, N in enumerate(run_for_kNN['n_samples']):
    #             for kk, D in enumerate(run_for_kNN['ambient_dim']):
    #                 for n, noise in enumerate(run_for_kNN['n_noise']):
    #                     # Exclude zeros if layer was not useful anymore due to
    #                     # insufficient points
    #                     print rep, j, k, kk, n
    #                     idx = np.argmin(f_error_CV_kNN[j,k,kk,n,:,rep])
    #                     best_f_error_CV_kNN[j,k,kk,n,rep] = f_error_CV_kNN[j,k,kk,n,idx,rep]
    #                     best_f_error_test_kNN[j,k,kk,n,rep] = f_error_test_kNN[j,k,kk,n,idx,rep]
    #                     chosen_K_kNN[j,k,kk,n,rep] = idx
    # import pdb; pdb.set_trace()
    # plot_all_errors(run_for, best_f_error_CV[:,:,:,:,:,:,:], save = ['n_samples'],
    #                                    savestr = savestr_base,
    #                                    additional_savestr = 'f_error_CV_',
    #                                    legend_pos = 'lower left',
    #                                    polyfit = False)
    # plot_all_errors(run_for, best_f_error_test[:,:,:,:,:,:,:], save = ['n_samples'],
    #                                    savestr = savestr_base,
    #                                    additional_savestr = 'f_error_',
    #                                    legend_pos = 'lower left',
    #                                    ylabel = r'$MSE(f,\hat{f})$',
    #                                    polyfit = False)
