# coding: utf8

import numpy as np

from sklearn.linear_model import LinearRegression


def linreg(X, Y, X_CV, X_test, estimator, param, **kwargs):
    """ Implementation of ordinary kNN estimator specific for the NSIM model """
    options = estimator.get('options')
    linreg = LinearRegression()
    linreg = linreg.fit(X, Y)
    if X_CV.shape[0] > 0:
        Y_CV = linreg.predict(X_CV)
    else:
        Y_CV = np.zeros([0])
    Y_test = linreg.predict(X_test)
    return Y_CV, Y_test
