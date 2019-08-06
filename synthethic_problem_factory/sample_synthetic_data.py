# coding: utf8
import numpy as np
"""
Auxiliary function to sample synthethic data.
"""

def sample_1D_fromClass(manifold, f_on_manifold,
                        n_samples, noise_level, tube = 'l2', var_f = 0.0,
                        return_original = False, args_f = None):
    """
    Points are generated according to

        X = gamma(t) + F(gamma(t))U

    where t is sampled uniformly in the domain of the curve, U are coefficients
    for sampling a random normal vector F(gamma(t))U (normal to the curve at
    gamma(t)).

    Parameters
    -----------
    manifold: Curve object from curves.py
        Instantiation of a curve using classes provided in curves.py

    f_on_manifold: python function
        1D link function that is evaluated for all t points.

    n_samples : integer
        Number of samples

    noise_level : float
        Bound for the norm of U. If tube = 'linfinity', this is the maximum size
        of 1 coordinate. If tube = 'l2', this is the maximum distance from gamma(t).

    tube : string
        Type of noise F(gamma(t))U. Can be 'linf' or 'l2'.

    var_f : float
        Variance of Gaussian noise added to the function values

    return_original: Boolean
        If true, returns also the points sampled exactly on the curve.

    args_f : Additional arguments for the function f_on_manifold if necessary.


    Returns problem data:
    ---------------------
    s_disc: np.array of size N
        Samples t

    points : np.array of size D x N
        X samples

    points_original : np.array of size D x N
        Returns [t, U] for each sample (only if reurn_original is true)

    normalspaces : np.array of size D x D-1 x N
        Orthonormal basis for the normal space of the curve at each sample gamma(t)

    fval : np.array of size N
        Function values

    fval_clean : np.array of size N
        Function values without additive noise

    tangentspaces : np.array of size D x 1 x N
        True tangent of the curve at each sample gamma(t)

    basepoints : np.array of size D x N
        gamma(t) samples (only if reurn_original is true)
    """
    # Find length corresponding to end parameter b
    s_disc = np.random.uniform(low = manifold.get_start(),
                               high = manifold.get_end(), size = (n_samples))
    # s_disc = np.sort(s_disc)
    n_features = manifold.get_n_features()
    # Containers
    basepoints = np.zeros((n_features, n_samples))
    points = np.zeros((n_features, n_samples))
    points_original = np.zeros((n_features, n_samples)) # Contains t + normal coefficients
    points_original[0,:] = s_disc
    tangentspaces = np.zeros((n_features, 1, n_samples))
    normalspaces = np.zeros((n_features, n_features - 1, n_samples))
    fval = np.zeros(n_samples)
    # Sample Detachment Coefficients
    if tube == 'linfinity':
        # Sample detachment coefficients from ||k||_inf < noise_level.
        random_coefficients = np.random.uniform(-noise_level, noise_level,
                                            size = (n_features - 1, n_samples))
    elif tube == 'l2':
        # Sample detachment coefficients from ||k||_2 < noise_level
        rand_sphere = np.random.normal(size = (n_features - 1, n_samples))
        rand_sphere = rand_sphere/np.linalg.norm(rand_sphere, axis = 0)
        radii = np.random.uniform(0, 1, size = n_samples)
        radii = noise_level * np.power(radii, 1.0/(n_features - 1))
        random_coefficients = rand_sphere * radii
    points_original[1:,:] = random_coefficients
    for i in range(n_samples):
            basepoints[:,i] = manifold.get_basepoint(s_disc[i])
            tangentspaces[:,0,i] = manifold.get_tangent(s_disc[i])
            normalspaces[:,:,i] = manifold.get_normal(s_disc[i])
            normal_vector = np.sum(normalspaces[:,:,i] * random_coefficients[:,i],
                                   axis=1)
            points[:,i] = basepoints[:,i] + normal_vector
            if args_f is not None:
                fval[i] = f_on_manifold(np.array([s_disc[i]]), manifold.get_start(), manifold.get_end(), *args_f)
            else:
                fval[i] = f_on_manifold(np.array([s_disc[i]]))
    # Apply noise to the function values
    fval_clean = np.copy(fval)
    if var_f > 0.0:
        Ymin, Ymax = np.min(fval_clean), np.max(fval_clean)
        avg_grad = (Ymax - Ymin)/(manifold.get_end() - manifold.get_start())
        fval_noise = np.random.uniform(low = -avg_grad * np.sqrt(var_f), high = avg_grad * np.sqrt(var_f),
                                      size = n_samples)
        fval += fval_noise
    if return_original:
        return s_disc, points, points_original, normalspaces, fval, fval_clean,\
                tangentspaces, basepoints
    else:
        return s_disc, points, normalspaces, fval, fval_clean,\
                tangentspaces, basepoints
