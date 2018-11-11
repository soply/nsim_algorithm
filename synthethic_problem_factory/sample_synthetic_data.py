# coding: utf8
import numpy as np


"""
Several methods to sample synthethic data for a given experiment.
"""

def sample_1D_fromClass(a, b, manifold, f_on_manifold,
                        n_samples, noise_level, tube = 'l2', var_f = 0.0,
                        return_original = False, args_f = None):
    # Find length corresponding to end parameter b
    s_disc = np.random.uniform(low = a, high = b, size = (n_samples))
    s_disc = np.sort(s_disc)
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
                fval[i] = f_on_manifold(np.array([s_disc[i]]), *args_f)
            else:
                fval[i] = f_on_manifold(np.array([s_disc[i]]))
    # Apply noise to the function values
    fval_clean = np.copy(fval)
    if var_f > 0.0:
        fval_noise = np.random.normal(loc = 0.0, scale = np.sqrt(var_f),
                                      size = n_samples)
        fval += fval_noise
    if return_original:
        return s_disc, points, points_original, normalspaces, fval, fval_clean,\
                tangentspaces, basepoints
    else:
        return s_disc, points, normalspaces, fval, fval_clean,\
                tangentspaces, basepoints
