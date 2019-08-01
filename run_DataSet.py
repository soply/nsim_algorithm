# coding: utf8
"""
Run file to run NSIM experiments on real data problems. Real data sets
are loaded from github/soply/db_hand
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from estimator import NSIM_Estimator


# Path to Data Sets
path_to_source = '../..'


if __name__ == '__main__':
    sys.path.insert(0, path_to_source + '/DataSets/')
    # from handler_UCI_AirQuality import read_all
    #from handler_UCI_Superconduct import read_all
    from handler_UCI_RealEstate import read_all
    data = read_all(scaling = 'MeanVar')
    X, Y = data[:,:-1], data[:,-1]
    Y = np.log(Y)
    # Parameters
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    for k in range(1,30):
        for L in range(1,6):
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15)
            nsim_kNN = NSIM_Estimator(n_neighbors = k,
                                n_levelsets = L,
                                split_by = 'stateq')
            nsim_kNN.fit(X_train, Y_train)
            Y_predict = nsim_kNN.predict(X_test)
            linreg = linreg.fit(X_train, Y_train)
            Y_predict_linreg = linreg.predict(X_test)
            print k, L, np.sqrt(np.mean(np.square(np.exp(Y_predict) - np.exp(Y_test)))), np.sqrt(np.mean(np.square(np.exp(Y_predict_linreg) - np.exp(Y_test))))
        print "========================\n\n"
        # plt.plot(Y_predict, label = 'Prediction')
        # plt.plot(Y_test)
        # plt.legend()
        # plt.show()
    import pdb; pdb.set_trace()
    # Compute Grammian
    gram = nsim_kNN.tangents_.dot(nsim_kNN.tangents_.T)
    plt.matshow(gram)
    plt.colorbar()
    plt.clim(-1,1)
    plt.show()
    import pdb; pdb.set_trace()
