# coding: utf8
"""
Run file to run kNN experiments on synthethic problems. Problems are created
using the synthethic problem factory.
"""
# coding: utf8
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
# I/O import
import os
import shutil
import sys
import tempfile
import time

# Specific imports
import numpy as np
# For parallelization
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid

from synthethic_problem_factory.curves import *
from synthethic_problem_factory.functions_on_manifolds import (RandomPolynomialIncrements,
                                                               randomPolynomialIncrements_for_parallel)
from synthethic_problem_factory.sample_synthetic_data import \
    sample_1D_fromClass
from benchmarks.utils import estimate

# Score functions
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
                ambient_dim,
                noise,
                var_f,
                random_seeds,
                manifold,
                f_on_manifold,
                estimator,
                parametergrid,
                f_f_error_CV, # File function error crossvalidation
                f_f_error_test, # File function error test
                comp_time, # File computational time
                rep, i1, i2, i3, i4, # Indices to write into
                savestr_base = None,
                args_f = None):
    """
    Main function to run a single experiment. Saves the results into the
    given files. The test manifold is set below.
    """
    assert 'options' in estimator, "MAIN: No 'options' key in estimator dict"
    # Check if crossvalidation is done
    CV_split = estimator['options'].get('CV_split', 0.0)
    np.random.seed(random_seeds[i1, i2, i3, i4, rep])
    # Setting the test manifold, check synthethic_problem_factory.curves
    manifold = get_manifold(ambient_dim, **manifold)
    # Split n_samples into training and CV
    n_samples_CV = np.floor(CV_split * n_samples).astype('int')
    n_samples_train = n_samples - n_samples_CV
    # Get training samples
    all_pdisc, all_points, all_normalspaces, all_fval, all_fval_clean, all_tangentspaces, all_basepoints = \
                                            sample_1D_fromClass(manifold,
                                                                f_on_manifold,
                                                                n_samples,
                                                                noise,
                                                                var_f = var_f,
                                                                tube = 'l2',
                                                                args_f = args_f)
    # Extract training and CV set
    points, fval = all_points[:,:n_samples_train], all_fval[:n_samples_train]
    points_CV, fval_CV = all_points[:,n_samples_train:], all_fval[n_samples_train:]
    # Get test samples
    n_test_samples = 1000
    pdisc_test, points_test, normalspaces_test, fval_test, fval_test_clean, tangentspaces_test, basepoints_test = \
                                            sample_1D_fromClass(manifold,
                                                                f_on_manifold,
                                                                n_test_samples, noise,
                                                                var_f = 0.00,
                                                                tube = 'l2',
                                                                args_f = args_f)
    noisy = (var_f > 0.0) # Required for some estimator to choose parameters optimally (knn, sirknn)
    for idx, param in enumerate(parametergrid):

        start = time.time()
        fval_predict_CV, fval_predict_test = estimate(points.T, fval, points_CV.T, points_test.T,
                                                      estimator, param, N = n_samples, D = ambient_dim,
                                                      noisy = noisy)
        end = time.time()
        if CV_split == 0.0:
            f_f_error_CV[i1,i2,i3,i4,idx,rep] = 1e16
        else:
            f_f_error_CV[i1,i2,i3,i4,idx,rep] = RMSE(np.reshape(fval_predict_CV, \
                                                                        (1,-1)), np.reshape(fval_CV, (1,-1)))
            print "Function error (CV): ", f_f_error_CV[i1,i2,i3,i4,idx,rep]
        f_f_error_test[i1,i2,i3,i4,idx,rep] = RMSE(np.reshape(fval_predict_test, \
                                                                        (1,-1)), np.reshape(fval_test, (1,-1)))
        comp_time[i1,i2,i3,i4,idx,rep] = end - start
        start = end
        print "Function error (Test): ", f_f_error_test[i1,i2,i3,i4,idx,rep]
    print "Finished N = {0}     D = {1}     sigma = {2}     sigma_f = {3}   rep = {4}".format(
        n_samples, ambient_dim, noise, var_f, rep)


if __name__ == "__main__":
    # Get number of jobs from sys.argv
    if len(sys.argv) > 1:
        n_jobs = int(sys.argv[1])
    else:
        n_jobs = 1 # Default 1 jobs
    print 'Using n_jobs = {0}'.format(n_jobs)
    # Define manifolds to test
    manifolds = [{'start' : 0, 'end' : 1.0, 'manifold_id' : 'identity'},
                {'start' : 0, 'end' : np.pi/2.0, 'manifold_id' : 'scurve'},
                {'start' : 0, 'end' : 2.0 * np.pi, 'manifold_id' : 'helix'}]
    for manifold in manifolds:
        # Sample random function, or load if already exist
        if not os.path.exists('random_polynomials/random_polynomial_for_' + manifold['manifold_id'] + '.npz'):
            fun_obj = RandomPolynomialIncrements(manifold['start'], manifold['end'], 2, 100,
                                                 coefficient_bound = [1.0, 1.5])
            bases = fun_obj.get_bases()
            coeffs = fun_obj.get_coeffs()
            np.savez('random_polynomials/random_polynomial_for_' + manifold['manifold_id'] + '.npz', bases = bases, coeffs = coeffs)
        else:
            data = np.load('random_polynomials/random_polynomial_for_' + manifold['manifold_id'] + '.npz')
            bases = data['bases']
            coeffs = data['coeffs']
        print "Considering manifold {0}".format(manifold['manifold_id'])
        # Parameters
        run_for = {
            'N' : [200 * (2 ** i) for i in range(10)],
            'D' : [4,8,16],
            'sigma_X' : [0.25],
            'sigma_f' : [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
            'repititions' : 20,
            # Estimator information
            'estimator' : {
                'estimator_id' : 'knn',
                'options' : {
                    'CV_split' : 0.15,
                    # 'learning_rate' : 0.1,
                },
                'params' : {
                    'n_neighbors' : [0.5],
                }
            }
        }
        parametergrid = ParameterGrid(run_for['estimator']['params'])
        random_seeds = np.random.randint(0, high = 2**32 - 1, size = (len(run_for['N']),
                                                              len(run_for['D']),
                                                              len(run_for['sigma_X']),
                                                              len(run_for['sigma_f']),
                                                              run_for['repititions']))
        savestr_base = 'abc1'
        filename_errors = 'results/' + manifold['manifold_id'] + '/' + run_for['estimator']['estimator_id'] + savestr_base
        try:
            f_f_error_CV = np.load(filename_errors + '/f_error_CV.npy')
            f_f_error_test = np.load(filename_errors + '/f_error_test.npy')
            comp_time = np.load(filename_errors + '/comp_time.npy')
        except IOError:
            if not os.path.exists(filename_errors):
                os.makedirs(filename_errors)
                # Save a log file
            with open(filename_errors + '/log.txt', 'w') as file:
                file.write(json.dumps(run_for, indent=4)) # use `json.loads` to do the reverse
            tmp_folder = tempfile.mkdtemp()
            dummy_for_shape = np.zeros((len(run_for['N']),
                                        len(run_for['D']),
                                        len(run_for['sigma_X']),
                                        len(run_for['sigma_f']),
                                        len(list(parametergrid)),
                                        run_for['repititions']))
            try:
                # Create error containers
                f_f_error_CV = np.memmap(os.path.join(tmp_folder, 'f_error_CV'), dtype='float64',
                                           shape=dummy_for_shape.shape, mode='w+')
                f_f_error_test = np.memmap(os.path.join(tmp_folder, 'f_error_test'), dtype='float64',
                                           shape=dummy_for_shape.shape, mode='w+')
                comp_time = np.memmap(os.path.join(tmp_folder, 'comp_time'), dtype='float64',
                                           shape=dummy_for_shape.shape, mode='w+')
                # Run experiments in parallel
                Parallel(n_jobs=n_jobs, backend = "multiprocessing")(delayed(run_example)(
                                    run_for['N'][i1],
                                    run_for['D'][i2],
                                    run_for['sigma_X'][i3],
                                    run_for['sigma_f'][i4],
                                    random_seeds,
                                    manifold,
                                    randomPolynomialIncrements_for_parallel,
                                    run_for['estimator'],
                                    parametergrid,
                                    f_f_error_CV,
                                    f_f_error_test,
                                    comp_time,
                                    rep, i1, i2, i3, i4,
                                    savestr_base = savestr_base + '/' + str(rep) + '/',
                                    args_f = (bases, coeffs))
                                    for rep in range(run_for['repititions'])
                                    for i1 in range(len(run_for['N']))
                                    for i2 in range(len(run_for['D']))
                                    for i3 in range(len(run_for['sigma_X']))
                                    for i4 in range(len(run_for['sigma_f'])))
                # Dump memmaps to files
                f_f_error_CV.dump(filename_errors + '/f_error_CV.npy')
                f_f_error_test.dump(filename_errors + '/f_error_test.npy')
                comp_time.dump(filename_errors + '/comp_time.npy')
            finally:
                try:
                    shutil.rmtree(tmp_folder)
                except:
                    print('Failed to delete: ' + tmp_folder)
        import pdb; pdb.set_trace()
