# coding: utf8
import inspect

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import empirical_covariance
from sklearn.metrics.pairwise import euclidean_distances


"""
NSIM estimator where we partition the data into several level sets, and then
learn the geometry in each of the level sets. The number of level sets is
provided as a parameter.
"""

class NSIM_Estimator(BaseEstimator, RegressorMixin):
    """
    """

    def __init__(self,
                 n_levelsets = 1,
                 n_neighbors = 1,
                 ball_radius = None,
                 split_by = 'dyadic',
                 verbose_ = False):
        """
        n_levelsets: int
            Number of level sets to create

        n_neighbors: int
            Number of neighbors used in the kNN regression step.

        split_by : string
            Type of splitting that is used. Possibilities:
                'statistically_equivalent' : Splits such that each subset has
                    the same number of samples (statistically equivalent blocks).
                'Y_equivalent' : Separates the range of the response into
                    equivalently large intervals.

        verbose : bool
            Amount of print outs
        """
        # Set attributes of object to the same name as given in the argument
        # list.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)


    def fit(self, X, y=None):
        """
        """
        self.N, self.D = X.shape
        # Sort samples according to Y
        order = np.argsort(y)
        self.X_ = X[order,:]
        self.Y_ = y[order]
        # Create level set partitioning
        if self.split_by == 'stateq':
            self._construct_dyadic_partition() # Sets self.labels_
        elif self.split_by == 'dyadic':
            self._construct_statistically_equivalent_blocks() # Sets self.labels_
        # Check samples per level set
        n_samples_per_levelset = np.bincount(self.labels_).astype('int')
        if any(n_samples_per_levelset <= self.D):
            critical_LVsets = np.where(n_samples_per_levelset < 2 * self.D)[0]
            raise RuntimeError("Level sets {0} have only {1} <= {2} samples. Fitting not possible.".format(
                critical_LVsets, n_samples_per_levelset[critical_LVsets], self.D
            ))
        # Find smallest singular vectors
        self._calculate_tangents() # sets self.tangents_
        self.PX_ = np.zeros(self.N) # Storing projections of training points
        for i in range(self.n_levelsets):
            self.PX_[self.labels_ == i] = self.tangents_[i,:].dot(self.X_[self.labels_ == i, :].T)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "X_")
            getattr(self, "Y_")
            getattr(self, "PX_")
            getattr(self, "labels_")
            getattr(self, "tangents_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        # Get prediction
        predictions = self._predict(X)
        if isinstance(self.n_neighbors, (int,long)):
            return predictions[:,0]
        else:
            return predictions

    # Auxiliary methods below
    def _construct_dyadic_partition(self):
        """
        Partitions given data (X,Y) based on level set partitioning on the Y-values.
        To split the Y values, we create n_splits disjoint intervals spanning the
        range of Y and having equivalent width. Afterwards, we assign each sample
        based on its Y value to one of the intervals.

        Parameters
        -------------


        Returns
        ------------
        labels: np.array (size: n_samples) with ints
            Contains labels of each sample

        edges: np.array (size n_splits + 1) with floats
            Contains the Y-values at which we separate the data.

        """
        hist, edges = np.histogram(self.Y_, bins = self.n_levelsets)
        # Correct for upper and lower edge to include all samples
        edges[0] -= 1e-10
        edges[-1] += 1e-10
        self.labels_ = np.digitize(Y, edges) - 1


    def _construct_statistically_equivalent_blocks(self):
        """ Splits the given data (X,Y) into statistically equivalent blocks
        (i.e. #points of two blocks differs at most by 1) based on the order
        of Y. Returns 1 vector with labels from 0 to n_splits - 1.

        Parameters
        -------------


        Returns
        ------------
        labels: np.array (size: n_samples) with ints
            Contains labels of each sample

        edges: np.array (size n_splits + 1) with floats
            Contains the Y-values at which we separate the data.
        """
        # Assuming ordered Y here
        pieces = np.array_split(np.arange(self.N), self.n_levelsets)
        self.labels_ = np.zeros(self.N).astype('int')
        for i, piece in enumerate(pieces):
            self.labels_[piece] = i


    def _calculate_tangents(self):
    	"""
    	"""
    	self.tangents_ = np.zeros((self.n_levelsets, self.D))
    	for i in range(self.n_levelsets):
    		cov = empirical_covariance(self.X_[self.labels_ == i,:])
    		pinv = np.linalg.pinv(cov)
    		rhs = np.mean((self.X_[self.labels_ == i,:] - np.mean(self.X_[self.labels_ == i,:])).T * (self.Y_[self.labels_ == i] -  np.mean(self.Y_[self.labels_ == i])), axis = 1)
    		self.tangents_[i,:] = pinv.dot(rhs)
    		self.tangents_[i,:] /= np.linalg.norm(self.tangents_[i,:])


    def _predict(self, X_predict):
        """
        Can handle both cases where X_predict is a single sample (vector of shape D),
        or an input matrix.
        """
        # Handle only case where n_neighbors is a list here
        if isinstance(self.n_neighbors, (int,long)):
            tmp_n_neighbors = [self.n_neighbors]
        else:
            tmp_n_neighbors = self.n_neighbors
        if len(X_predict.shape) == 1: # Single sample case
            tmp_X_predict = np.reshape(X_predict, (1, -1))
        else:
            tmp_X_predict = X_predict
        n_test_samples = tmp_X_predict.shape[0]
        if self.ball_radius is None:
            ball_radius = 1e16
        else:
            ball_radius = self.ball_radius
        # Container
        prediction = np.zeros((n_test_samples, len(tmp_n_neighbors)))
        # Boolean matrix with 1 in (i,j) if training sample X_j is inside the Euclidean ball around test sample i
        inside_euclidean_ball = np.less(euclidean_distances(tmp_X_predict, self.X_), ball_radius).astype('bool')
        # Get training samples belonging to a certain level set
        assignment = [[] for _ in range(self.n_levelsets)]
        for j in range(self.n_levelsets):
            assignment[j] = np.where(self.labels_ == j)[0]
        # Get maximum number of points in the radius for any test point
        min_idx = tmp_n_neighbors
        for k in range(n_test_samples):
            distances = 100.0 * np.ones(self.N)
            for i in range(self.n_levelsets):
                idx = np.where(inside_euclidean_ball[k,assignment[i]])[0] # Find indices that are inside euclidean ball
                PX_predict = self.tangents_[i, :].dot(tmp_X_predict[k,:].T)
                ind_levelset = np.where(self.labels_[idx] == i)[0]
                # Setting distances inside level set and euclidean ball
                distances[assignment[i][idx]] = np.abs(self.PX_[assignment[i][idx]] - PX_predict)
                # distances[assignemnt[ind_levelset]] = np.abs(self.PX_[idx][ind_levelset] - PX_predict)
            for l, nNei in enumerate(tmp_n_neighbors):
                idx_for_pred = np.argpartition(distances, tmp_n_neighbors[l])[:tmp_n_neighbors[l]]
                idx_below_bound = np.where(distances[idx_for_pred] < 90.0)[0]
                if len(idx_below_bound) < min_idx[l]:
                    min_idx[l] = len(idx_below_bound)
                if len(idx_for_pred[idx_below_bound]) == 0:
                    raise RuntimeError("kNN Prediction: No neighbours satisfy the requirements.")
                prediction[k,l] = np.mean(self.Y_[idx_for_pred[idx_below_bound]])
        if any(np.array(min_idx) < np.array(tmp_n_neighbors)):
            print "Could use only {0} samples for some predictions".format(min_idx)
        return prediction
