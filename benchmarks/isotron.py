# coding: utf8
import sys

import numpy as np


def isotron(X, Y, X_CV, X_test, estimator, param, path, **kwargs):
    """ Implementation of SIR kNN estimator specific for the NSIM model """
    sys.path.insert(0, path)
    from simple_estimation.estimators.Isotron import IsotronCV

    options = estimator.get('options')
    assert 'max_iter' in options, "ISOTRON: 'max_iter' not in estimator options"
    assert 'CV_split' in options, "ISOTRON: 'CV_split' not in estimator options"
    isotron = IsotronCV(max_iter = options['max_iter'], cv_split = options['CV_split'])
    isotron = isotron.fit(X, Y)
    if X_CV.shape[0] > 0:
        Y_CV = isotron.predict(X_CV)
    else:
        Y_CV = np.zeros([0])
    Y_test = isotron.predict(X_test)
    return Y_CV, Y_test
