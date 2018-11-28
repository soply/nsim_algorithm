# coding: utf8
"""
Implementation of some examplary curves. Identity_kD corresponds to a SIM model,
the rest are actual curves with nonzero curvature. Circle_Segment_Builder can
be used to build a curve as a sequence of quarter circles.
"""
import numpy as np

""" General utilities """
def normal_nd(vectors):
    """
    Parameters
    -------------
    vectors : np.array, shape (space_dim, n_vectors)
        Matrix that contains columnwise vectors to which we want to search the
        orthogonal basis.

    Returns
    ------------
    Returns a matrix with space_dim - n_vectors columns, where each column is
    orthonogal to the given vectors and the matrix is orthonormal itself.
    """
    U, S, V = np.linalg.svd(vectors.T, full_matrices = 1, compute_uv = 1)
    compl_basis_idx = np.where(S > 1e-14)[0]
    basis_idx = np.setdiff1d(np.arange(vectors.shape[0]), compl_basis_idx)
    return V.T[:, basis_idx] # columns of V to small S are orthogonal basis

class Identity_kD(object):
    """
        gamma : I -> R^D
    with
        gamma(t) = (t,...,t,0,...0)/sqrt(k).

    Normalization to ensure arc-length parametrization.
    """
    def __init__(self, n_active_features, n_features):
        self._k = n_active_features
        self._n = n_features
        self._plot_dim = np.minimum(self._k, 3).astype('int')

    def get_plot_dim(self):
        return self._plot_dim

    def get_n_features(self):
        return self._n

    def get_basepoint(self, t):
        vec = np.zeros(self._n)
        vec[0:self._k] = t/np.sqrt(self._k)
        return vec

    def get_tangent(self, t):
        vec = np.zeros(self._n)
        vec[0:self._k] = 1.0/np.sqrt(self._k)
        return vec

    def get_normal(self, t):
        return normal_nd(np.reshape(self.get_tangent(t), (self._n, -1)))

    def get_length(self, t0, t1):
        return np.abs(t1-t0)


class Circle_Piece_2D(object):
    """
        gamma : I -> R^D
    with
        gamma(t) = (cos(t),sin(t),...0)

    # Easily arc-length parameterized
    """
    def __init__(self, n_features):
        self._plot_dim = 2
        self._n = n_features

    def get_plot_dim(self):
        return self._plot_dim

    def get_n_features(self):
        return self._n

    def get_basepoint(self, t):
        vec = np.zeros(self._n)
        vec[0], vec[1] = np.cos(t), np.sin(t)
        return vec

    def get_tangent(self, t):
        vec = np.zeros(self._n)
        vec[0], vec[1] = -np.sin(t), np.cos(t)
        return vec

    def get_curvature_vector(self, t):
        vec = np.zeros(self._n)
        vec[0], vec[1] = -np.cos(t), -np.sin(t)
        return vec

    def get_normal(self, t):
        return normal_nd(np.reshape(self.get_tangent(t), (self._n, -1)))

    def get_length(self, t0, t1):
        return np.abs(t1-t0)


class S_Curve_2D(object):
    """
        gamma : [-pi/2,pi/2] -> R^D
    with
        gamma(t) = (cos(-t),sin(-t)) if t <= 0 and
        gamma(t) = (2,0) - (cos(t), sin(t)) if t >= 0

    # Note that [-pi/2,pi/2] or a subset of this should be used but not an
    extended version.
    """
    def __init__(self, n_features):
        self._plot_dim = 2
        self._n = n_features

    def get_plot_dim(self):
        return self._plot_dim

    def get_n_features(self):
        return self._n

    def get_basepoint(self, t):
        vec = np.zeros(self._n)
        if t <= 0.0:
            vec[0], vec[1] = np.cos(t), np.sin(t)
        else:
            vec[0], vec[1] = 2.0 - np.cos(t), np.sin(t)
        return vec

    def get_tangent(self, t):
        vec = np.zeros(self._n)
        if t <= 0.0:
            vec[0], vec[1] = -np.sin(t), np.cos(t)
        else:
            vec[0], vec[1] = np.sin(t), np.cos(t)
        return vec

    def get_curvature_vector(self, t):
        vec = np.zeros(self._n)
        if t <= 0.0:
            vec[0], vec[1] = -np.cos(t), -np.sin(t)
        else:
            vec[0], vec[1] = np.cos(t), -np.sin(t)
        return vec

    def get_normal(self, t):
        return normal_nd(np.reshape(self.get_tangent(t), (self._n, -1)))

    def get_length(self, t0, t1):
        return np.abs(t1-t0)


class Helix_Curve_3D(object):
    """
        gamma : I -> R^D
    with
        gamma(t) = (a * cos(alpha * t), a * sin(alpha * t), alpha * b * t)

    and alpha = 1/(a^2 + b^2)^{1/2}.

    # Arc-length parameterized
    """
    def __init__(self, n_features, radius = 1, pitch = 1):
        self._plot_dim = 3
        self._n = n_features
        self._radius = radius
        self._pitch = pitch
        self._alpha = 1.0/np.sqrt(self._radius ** 2 + self._pitch ** 2)

    def get_plot_dim(self):
        return self._plot_dim

    def get_n_features(self):
        return self._n

    def get_basepoint(self, t):
        vec = np.zeros(self._n)
        vec[0] = self._radius * np.cos(self._alpha * t)
        vec[1] = self._radius * np.sin(self._alpha * t)
        vec[2] = self._alpha * self._pitch * t
        return vec

    def get_tangent(self, t):
        vec = np.zeros(self._n)
        vec[0] = (-1.0) * self._radius * self._alpha * np.sin(self._alpha * t)
        vec[1] = self._radius * self._alpha * np.cos(self._alpha * t)
        vec[2] = self._alpha * self._pitch
        return vec

    def get_curvature_vector(self, t):
        vec = np.zeros(self._n)
        vec[0] = (-1.0) * self._radius * (self._alpha ** 2) * np.cos(self._alpha * t)
        vec[1] = (-1.0) * self._radius * (self._alpha ** 2) * np.sin(self._alpha * t)
        vec[2] = self._alpha * self._pitch
        return vec

    def get_normal(self, t):
        return normal_nd(np.reshape(self.get_tangent(t), (self._n, -1)))

    def get_length(self, t0, t1):
        return np.abs(t1-t0)


class Circle_Segment_Builder(object):
    """
    Builds a curve

        gamma : [0, t_K] -> R^D

    where t_K = K * pi/2. The curve always starts at (0,0), and for each segment
    we are given two tuples (s_k1, k_1), (s_k2, k_2) where k_1 and k_2 in
    {1,...,D} are the directions in which the gradient points at the beginning
    respectively the end of the segment, and s_k1, s_k2 are signs. Thus, the
    curve is uniquely defined by a sequence like

        [(-1,4), (-1,5), (1,6)].

    Example:

        [(1,0)(+1,1),(-1,0),(-1,1)]

    represents the circle in the 0/1 plane.
    """
    def __init__(self, sequence, n_features):
        self._plot_dim = 3
        self._sequence = sequence
        self._n_segments = len(sequence)
        self._k = len(set([x[1] for x in sequence]))
        self._n = n_features
        """
        In each segment, we represent the curve as
        gamma(t)|_[tk,tk+1] = a_k + s_ki sin(t-k*pi/2)*e_i + s_kj cos(t-k*pi/2)*e_j.
        We store the a_k's for each segment in self._translations.
        """
        self._translations = np.zeros((n_features, self._n_segments)) # Stores the translations at each segment beginning
        # Compute translations
        self._init_curve()

    def get_plot_dim(self):
        return self._plot_dim

    def get_n_features(self):
        return self._n

    def _init_curve(self):
        # Build the curve for easy access, i.e. get the translations for each
        # segment and the coefficients for each segment.
        for j in range(1, self._n_segments):
            self._translations[:,j] = self._translations[:,j-1]
            self._translations[self._sequence[j-1][1],j] += self._sequence[j-1][0]
            self._translations[self._sequence[j][1],j] += self._sequence[j][0]

    def get_basepoint(self, t):
        vec = np.zeros(self._n)
        # Get segment
        seg = np.floor(t/(np.pi/2.0)).astype('int')
        # Add translation
        vec += self._translations[:, seg]
        si = self._sequence[seg][0]
        ei = self._sequence[seg][1]
        sj = self._sequence[seg+1][0]
        ej = self._sequence[seg+1][1]
        vec[ei] = vec[ei] + float(si) * np.sin(t - float(seg) * np.pi/2.0)
        vec[ej] = vec[ej] - float(sj) * np.cos(t - float(seg) * np.pi/2.0) + float(sj)
        return vec


    def get_tangent(self, t):
        vec = np.zeros(self._n)
        # Get segment
        seg = np.floor(t/(np.pi/2.0)).astype('int')
        # Add the curve segment
        si = self._sequence[seg][0]
        ei = self._sequence[seg][1]
        sj = self._sequence[seg+1][0]
        ej = self._sequence[seg+1][1]
        vec[ei] = float(si) * np.cos(t - float(seg) * np.pi/2.0)
        vec[ej] = float(sj) * np.sin(t - float(seg) * np.pi/2.0)
        return vec

    def get_normal(self, t):
        return normal_nd(np.reshape(self.get_tangent(t), (self._n, -1)))

    def get_length(self, t0, t1):
        return np.abs(t1-t0)
