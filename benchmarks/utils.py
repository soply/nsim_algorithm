# coding: utf8
import sys

import numpy as np

from knn import knn
from linreg import linreg
from sirknn import sirknn
from ffnn import ffnn
from isotron import isotron
from elm_handler import elm_regressor

__path_to_source__ = '../../simple_estimation/'
__path_to_elm__ = '../../PythonELM/'
def estimate(X, Y, X_CV, X_test, estimator, param, **kwargs):
    """ Handler method to distribute estimate to correct estimation method. """
    sys.path.insert(0, __path_to_source__)
    sys.path.insert(0, __path_to_elm__)
    assert 'estimator_id' in estimator, "estimate_handler: 'estimator_id' not in estimator"
    if estimator['estimator_id'] == 'knn':
        return knn(X, Y, X_CV, X_test, estimator, param, **kwargs)
    elif estimator['estimator_id'] == 'linreg':
        return linreg(X, Y, X_CV, X_test, estimator, param, **kwargs)
    elif estimator['estimator_id'] == 'sirknn':
        return sirknn(X, Y, X_CV, X_test, estimator, param, __path_to_source__, **kwargs)
    elif estimator['estimator_id'] == 'ffnn':
        return ffnn(X, Y, X_CV, X_test, estimator, param, __path_to_source__, **kwargs)
    elif estimator['estimator_id'] == 'isotron':
        return isotron(X, Y, X_CV, X_test, estimator, param, __path_to_source__, **kwargs)
    elif estimator['estimator_id'] == 'elm':
        return elm_regressor(X, Y, X_CV, X_test, estimator, param, __path_to_elm__, **kwargs)
    else:
        raise NotImplementedError("Estimator {0} is not implemented".format(estimator.get('estimator_id', 'NONE')))
