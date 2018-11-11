# coding: utf8

# I/O import
import os
import sys
import json
import time

# For parallelization
from joblib import Parallel, delayed
import tempfile
import shutil

# Specific imports
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from synthethic_problem_factory.functions_on_manifolds import (RandomPolynomialIncrements,
                                                    randomPolynomialIncrements_for_parallel,
                                                    identity_function,
                                                    injective_on_first_coordinate,
                                                    norm_function,
                                                    quadratic_on_first_coordinate,
                                                    sinus,
                                                    ahalf_polynomial_on_first)
from synthethic_problem_factory.manifolds_as_curves_classes import (Circle_Piece_2D,
                                                         Circle_Segment_Builder,
                                                         Helix_Curve_3D,
                                                         Identity_kD,
                                                         S_Curve_2D)
from synthethic_problem_factory.sample_synthetic_data import sample_1D_fromClass


# Loss function
def MSE(prediction, reference):
    return np.sum(np.linalg.norm(prediction - reference, axis = 0) ** 2)/ \
                np.sum(np.linalg.norm(reference, axis = 0) ** 2)

def RMSE(prediction, reference):
    """ Root Mean squared error """
    return np.sqrt(MSE(prediction, reference))

def UAE(prediction, reference):
    """ Uniform absolute error max_i |v_i - hat v_i| """
    return np.max(np.linalg.norm(prediction - reference, axis = 0))


def run_example(n_samples,
                noise,
                ambient_dim,
                var_f,
                n_neighbors,
                xlow,
                xhigh,
                f_on_manifold,
                f_f_error_CV,
                f_f_error_test,
                comp_time,
                rep, ctr_j, ctr_k, ctr_kk, ctr_n, # Indices to write into
                neighbor_modus = 'number',
                CV_split = 0.1,
                savestr_base = None,
                apply_rotation = None,
                random_state = None,
                args_f = None):
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
    # Sample CV data
    pdisc_CV, points_CV, normalspaces_CV, fval_CV, fval_CV_clean, tangentspaces_CV, basepoints_CV = \
                                            sample_1D_fromClass(xlow,
                                                                xhigh,
                                                                f_manifold,
                                                                f_on_manifold,
                                                                n_samples_CV,
                                                                noise,
                                                                var_f = var_f,
                                                                tube = 'l2',
                                                                args_f = args_f)
    # Get test samples
    n_test_samples = 1000
    pdisc_test, points_test, normalspaces_test, fval_test, fval_test_clean, tangentspaces_test, basepoints_test = \
                                            sample_1D_fromClass(xlow,
                                                                xhigh, f_manifold, f_on_manifold,
                                                                n_test_samples, noise,
                                                                var_f = 0.00,
                                                                tube = 'l2',
                                                                args_f = args_f)
    if apply_rotation is not None:
        # Apply rotation to basepoints
        basepoints = apply_rotation.dot(basepoints)
        basepoints_CV = apply_rotation.dot(basepoints_CV)
        basepoints_test = apply_rotation.dot(basepoints_test)
        # Apply rotation to points
        points = apply_rotation.dot(points)
        points_CV = apply_rotation.dot(points_CV)
        points_test = apply_rotation.dot(points_test)
    for idx, k in enumerate(n_neighbors):
        if neighbor_modus == 'factor':
            nNei = np.floor(k * np.power(n_samples, 2.0/(2.0 + ambient_dim))).astype('int')
        else:
            nNei = k
        start = time.time()
        knn_reg = KNeighborsRegressor(n_neighbors = nNei)
        knn_reg = knn_reg.fit(points.T, fval)
        fval_predict_CV = knn_reg.predict(points_CV.T)
        fval_predict_test = knn_reg.predict(points_test.T)
        end = time.time()
        f_f_error_CV[ctr_j,ctr_k,ctr_kk,ctr_n,idx,rep] = RMSE(np.reshape(fval_predict_CV, \
                                                                        (1,-1)), np.reshape(fval_CV, (1,-1)))
        f_f_error_test[ctr_j,ctr_k,ctr_kk,ctr_n,idx,rep] = RMSE(np.reshape(fval_predict_test, \
                                                                        (1,-1)), np.reshape(fval_test, (1,-1)))
        comp_time[ctr_j,ctr_k,ctr_kk,ctr_n,idx,rep] = end - start
        start = end
        print "Fval error (CV): ", f_f_error_CV[ctr_j,ctr_k,ctr_kk,ctr_n,idx,rep]
        print "Fval error (Test): ", f_f_error_test[ctr_j,ctr_k,ctr_kk,ctr_n,idx,rep]

if __name__ == "__main__":
    # Get number of jobs from sys.argv
    if len(sys.argv) > 1:
        n_jobs = int(sys.argv[1])
    else:
        n_jobs = 4 # Default 4 jobs
    print 'Using n_jobs = {0}'.format(n_jobs)
    # Set parameters
    xlow = 0.0 * np.pi
    xhigh = 3.0 * np.pi
    np.random.seed(123123)
    fun_obj = RandomPolynomialIncrements(xlow, xhigh, 2, 100,
                                         coefficient_bound = [1.0, 1.5])
    # Calculate variance
    flower, fupper = fun_obj.eval(xlow), fun_obj.eval(xhigh)
    avg_grad = (fupper - flower)/(xhigh - xlow)
    # fun_obj.plot(white_noise_var=(avg_grad * 0.04) ** 2, n = 1000)

    # Parameters
    run_for = {
        'n_samples' : [1000, 2000, 4000, 8000, 16000, 32000],
        'n_noise' : [0.25],
        'ambient_dim' : [6, 12, 24],
        'var_f' : [0.0],
        'n_neighbors' : [1],
        'neighbor_modus' : 'number'
    }
    # Sample rotations for each dimension
    rotations = {}
    for j, D in enumerate(run_for['ambient_dim']):
        from scipy.stats import special_ortho_group
        rotations[D] = special_ortho_group.rvs(D)

    repititions = 2
    savestr_base = 'for_plotting_kNN'
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
        dummy_for_shape = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
                          len(run_for['ambient_dim']), len(run_for['n_noise']),
                          len(run_for['n_neighbors']), repititions))
        try:
            # Create error containers
            f_f_error_CV = np.memmap(os.path.join(tmp_folder, 'f_error_CV'), dtype='float64',
                                       shape=dummy_for_shape.shape, mode='w+')
            f_f_error_test = np.memmap(os.path.join(tmp_folder, 'f_error_test'), dtype='float64',
                                       shape=dummy_for_shape.shape, mode='w+')
            comp_time = np.memmap(os.path.join(tmp_folder, 'comp_time'), dtype='float64',
                                       shape=dummy_for_shape.shape, mode='w+')
            random_state = np.random.get_state()
            # Run experiments in parallel
            Parallel(n_jobs=n_jobs)(delayed(run_example)(
                                run_for['n_samples'][k],
                                run_for['n_noise'][n],
                                run_for['ambient_dim'][kk],
                                run_for['var_f'][j],
                                run_for['n_neighbors'],
                                xlow,
                                xhigh,
                                randomPolynomialIncrements_for_parallel,
                                f_f_error_CV,
                                f_f_error_test,
                                comp_time,
                                rep, j, k, kk, n,
                                neighbor_modus = run_for['neighbor_modus'],
                                CV_split = 0.1,
                                savestr_base = savestr_base + '/' + str(rep) + '/',
                                apply_rotation = rotations[run_for['ambient_dim'][kk]],
                                args_f = (xlow, xhigh, fun_obj.get_bases(),
                                          fun_obj.get_coeffs()))
                                for rep in range(repititions)
                                for j in range(len(run_for['var_f']))
                                for k in range(len(run_for['n_samples']))
                                for kk in range(len(run_for['ambient_dim']))
                                for n in range(len(run_for['n_noise'])))
            # Dump memmaps to files
            f_f_error_CV.dump(filename_errors + '/f_error_CV.npy')
            f_f_error_test.dump(filename_errors + '/f_error_test.npy')
            comp_time.dump(filename_errors + '/comp_time.npy')
        finally:
            try:
                shutil.rmtree(tmp_folder)
            except:
                print('Failed to delete: ' + tmp_folder)

    """ Post-Processing: Read errors and create error plots after model selection """
    f_error_CV = np.load(filename_errors + '/f_error_CV.npy')
    f_error_test = np.load(filename_errors + '/f_error_test.npy')
    # Select the best according to the f_error_CV and plot those in plot
    best_f_error_CV = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
                      len(run_for['ambient_dim']), len(run_for['n_noise']),
                      repititions))
    best_f_error_test = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
                      len(run_for['ambient_dim']), len(run_for['n_noise']),
                      repititions))

    # Store selected layers
    chosen_layers = np.zeros((len(run_for['var_f']), len(run_for['n_samples']),
                      len(run_for['ambient_dim']), len(run_for['n_noise']),
                      repititions))
    for rep in range(repititions):
        for j, var_f in enumerate(run_for['var_f']):
            for k, N in enumerate(run_for['n_samples']):
                for kk, D in enumerate(run_for['ambient_dim']):
                    for n, noise in enumerate(run_for['n_noise']):
                        # Exclude zeros if layer was not useful anymore due to
                        # insufficient points
                        idx = np.argmin(f_error_CV[j,k,kk,n,:,rep])
                        best_f_error_CV[j,k,kk,n,rep] = f_error_CV[j,k,kk,n,idx,rep]
                        best_f_error_test[j,k,kk,n,rep] = f_error_test[j,k,kk,n,idx,rep]
                        chosen_layers[j,k,kk,n,rep] = idx
    #
    # plot_all_errors(run_for, best_f_error_CV[:,:,:,:,:,:], save = ['n_samples'],
    #                                    savestr = savestr_base,
    #                                    additional_savestr = 'f_error_CV_',
    #                                    legend_pos = 'lower left')
    # plot_all_errors(run_for, best_f_error_test[:,:,:,:,:,:], save = ['n_samples'],
    #                                    savestr = savestr_base,
    #                                    additional_savestr = 'f_error_',
    #                                    legend_pos = 'lower left',
    #                                    ylabel = r'$MSE(f,\hat{f})$')
    # plot_all_errors(run_for, chosen_layers[:,:,:,:,:,:], save = ['n_samples'],
    #                                    polyfit = False,
    #                                    savestr = savestr_base,
    #                                    additional_savestr = 'chosen_layers',
    #                                    ylabel = 'Chosen scale',
    #                                    legend_pos = 'lower left',
    #                                    yaxis_scaling = 'lin')
