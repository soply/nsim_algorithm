# coding: utf8
import sys

import numpy as np


def ffnn(X, Y, X_CV, X_test, estimator, param, path, **kwargs):
    """ Implementation of SIR kNN estimator specific for the NSIM model """
    sys.path.insert(0, path)
    from simple_estimation.estimators.FeedForwardNetwork import FeedForwardNetwork

    options = estimator.get('options')
    assert 'n_hidden' in param, "SFFNNIRKNN: 'n_hidden' not in param"
    FFNN = FeedForwardNetwork(n_hidden = param['n_hidden'], general_options = options)
    FFNN = FFNN.fit(X, Y)
    if X_CV.shape[0] > 0:
        Y_CV = FFNN.predict(X_CV)
    else:
        Y_CV = np.zeros([0])
    Y_test = FFNN.predict(X_test)
    return Y_CV, Y_test
