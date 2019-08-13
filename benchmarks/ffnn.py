# coding: utf8
import sys

import numpy as np


def ffnn(X, Y, X_CV, X_test, estimator, param, path, **kwargs):
    """ Implementation of SIR kNN estimator specific for the NSIM model """
    sys.path.insert(0, path)
    from simple_estimation.estimators.FeedForwardNetwork import FeedForwardNetwork
    """
    Activation functions:

    activation : String
        Identifier for the activation function of the hidden layern. Can be
        'Rectifier', 'Sigmoid', 'Tanh', 'ExpLin', 'Linear', 'Softmax'.
    """
    options = {}
    options['learning_rate'] = estimator.get('options').get('learning_rate', 0.01)
    options['valid_size'] = estimator.get('options').get('valid_size', 0.1)
    options['batch_size'] = estimator.get('options').get('batch_size', 1)
    options['learning_rule'] = estimator.get('options').get('learning_rule', 'sgd')

    assert 'n_hidden' in param, "FFNN: 'n_hidden' not in param"
    FFNN = FeedForwardNetwork(n_hidden = param['n_hidden'],
                              activation = estimator.get('options').get('activation_func', 'Sigmoid'),
                              general_options = options)
    FFNN = FFNN.fit(X, Y)
    if X_CV.shape[0] > 0:
        Y_CV = FFNN.predict(X_CV)
    else:
        Y_CV = np.zeros([0])
    Y_test = FFNN.predict(X_test)
    return Y_CV, Y_test
