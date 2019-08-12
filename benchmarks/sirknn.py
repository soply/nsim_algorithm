# coding: utf8
import sys

import numpy as np


def sirknn(X, Y, X_CV, X_test, estimator, param, path, noisy = True, **kwargs):
    """ Implementation of SIR kNN estimator specific for the NSIM model """
    sys.path.insert(0, path)
    from simple_estimation.estimators.SIRKnn import SIRKnn

    options = estimator.get('options')
    assert 'n_neighbors' in param, "SIRKNN: 'n_neighbors' not in param"
    assert 'n_components' in param, "SIRKNN: 'n_components' not in param "
    assert 'n_levelsets' in param, "SIRKNN: 'n_levelsets' not in param"
    if noisy:
        # Optimal choice is 1 in the noise free case
<<<<<<< Updated upstream
        nNei = np.maximum(np.floor(param['n_neighbors'] * np.power(kwargs['N'], 2.0/3.0)).astype('int'), 1)
=======
        nNei = np.maximum(np.floor(param['n_neighbors'] * np.power(kwargs['N'], 2.0/(2.0 + kwargs['D']))).astype('int'), 1)
>>>>>>> Stashed changes
    else:
        nNei = 1
    sirknn = SIRKnn(n_neighbors = nNei, n_components = param['n_components'],
                    n_levelsets = param['n_levelsets'], rescale = options.get('rescale', True))
    sirknn = sirknn.fit(X, Y)
    if X_CV.shape[0] > 0:
        Y_CV = sirknn.predict(X_CV)
    else:
        Y_CV = np.zeros([0])
    Y_test = sirknn.predict(X_test)
    return Y_CV, Y_test
