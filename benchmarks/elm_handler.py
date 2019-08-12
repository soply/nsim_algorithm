# coding: utf8
import sys

import numpy as np


def elm_regressor(X, Y, X_CV, X_test, estimator, param, path, **kwargs):
    """ Implementation of SIR kNN estimator specific for the NSIM model """
    assert 'n_hidden' in param, "SFFNNIRKNN: 'n_hidden' not in param"
    sys.path.insert(0, path)
    options = estimator.get('options')
    from elm import ELMRegressor
    """
    ELM-Parameters (copied from file)
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate in the SimpleRandomLayer

    `alpha` : float, optional (default=0.5)
        Mixing coefficient for distance and dot product input activations:
        activation = alpha*mlp_activation + (1-alpha)*rbf_width*rbf_activation

    `rbf_width` : float, optional (default=1.0)
        multiplier on rbf_activation

    `activation_func` : {callable, string} optional (default='tanh')
        Function used to transform input activation

        It must be one of 'tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid',
        'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric' or
        a callable.  If none is given, 'tanh' will be used. If a callable
        is given, it will be used to compute the hidden unit activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `user_components`: dictionary, optional (default=None)
        dictionary containing values for components that woud otherwise be
        randomly generated.  Valid key/value pairs are as follows:
           'radii'  : array-like of shape [n_hidden]
           'centers': array-like of shape [n_hidden, n_features]
           'biases' : array-like of shape [n_hidden]
           'weights': array-like of shape [n_hidden, n_features]

    `regressor`    : regressor instance, optional (default=None)
        If provided, this object is used to perform the regression from hidden
        unit activations to the outputs and subsequent predictions.  If not
        present, an ordinary linear least squares fit is performed

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.
    """
    elm_reg = ELMRegressor(n_hidden = param['n_hidden'], activation_func = options.get('activation_func', 'tanh'))
    elm_reg = elm_reg.fit(X, Y)
    Y_CV = elm_reg.predict(X_CV)
    Y_test = elm_reg.predict(X_test)
    return Y_CV, Y_test
