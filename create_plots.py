# coding: utf8
"""
Auxiliary file to create the plots for NSIM article.
"""

from evaluation.function_error import plot_error, plot_error_nsim
from evaluation.tangent_error import plot_error as plot_error_tan



if __name__ == "__main__":
    # plot_error_tan('results/identity/nsim/run_1/')
    manifolds = ['identity', 'scurve', 'helix']
    estimator_ids = ['linreg','knn','sirknn','nsim','isotron','ffnn']
    for manifold in manifolds:
        for estimator_id in estimator_ids:
            if estimator_id == 'nsim':
                plot_error_tan('results/' + manifold + '/' + estimator_id + '/run_1/')
                plot_error_nsim('results/' + manifold + '/' + estimator_id + '/run_1/')
                continue
            elif estimator_id == 'ffnn' and manifold != 'identity':
                continue
            plot_error('results/' + manifold + '/' + estimator_id + '/run_1/')
