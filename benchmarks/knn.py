# coding: utf8

import numpy as np

from sklearn.neighbors import KNeighborsRegressor


def knn(X, Y, X_CV, X_test, estimator, param, noisy = True, **kwargs):
    """ Implementation of ordinary kNN estimator specific for the NSIM model """
    options = estimator.get('options')
    assert 'n_neighbors' in param, "KNN: 'n_neighbors' not in param"
    if noisy:
        # Optimal choice is 1 in the noise free case
        nNei = np.maximum(np.floor(param['n_neighbors'] * np.power(kwargs['N'], 2.0/3.0).astype('int'), 1))
    else:
        nNei = 1
    knn_reg = KNeighborsRegressor(n_neighbors = nNei)
    knn_reg = knn_reg.fit(X, Y)
    print nNei
    if X_CV.shape[0] > 0:
        Y_CV = knn_reg.predict(X_CV)
    else:
        Y_CV = np.zeros([0])
    Y_test = knn_reg.predict(X_test)
    return Y_CV, Y_test
