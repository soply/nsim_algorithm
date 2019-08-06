import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker


"""
Some auxiliary methods for plotting, if needed.
"""


def handle_2D_plot():
    fig = plt.figure()
    ax = fig.gca()
    return fig, ax


def handle_3D_plot():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    return fig, ax


def add_function(point_cloud, fval, ax, cb = True, dim = 2):
    if dim == 2:
        sc = ax.scatter(point_cloud[0, :], point_cloud[1, :],
                        c=fval, s = 50.0)
    elif dim == 3:
        sc = ax.scatter(point_cloud[0, :], point_cloud[1, :], point_cloud[2, :],
                        c=fval, s = 50.0)
    if cb == True:
        cb = plt.colorbar(sc)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

def add_scattered_pointcloud(point_cloud, labels, ax, label = None, dim = 2):
    if dim == 2:
        if len(set(labels)) == 1:
            sc = ax.scatter(point_cloud[0, :], point_cloud[1, :], c=labels[:],
                            s = 50.0)
        else:
            # define the colormap
            cmap = plt.cm.gist_ncar
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
            # define the bins and normalize
            bounds = np.linspace(0,len(np.unique(labels)),
                len(np.unique(labels)) + 1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            if label is not None:
                idx = (labels == label)
            else:
                idx = np.arange(point_cloud.shape[1])
            sc = ax.scatter(point_cloud[0, idx], point_cloud[1, idx],
                            c=labels[idx], s = 50.0, cmap = cmap, norm = norm)
    elif dim == 3:
        if len(set(labels)) == 1:
            sc = ax.scatter(point_cloud[0, :], point_cloud[1, :], point_cloud[2, :],
                            c=labels[:], s = 50.0)
        else:
            # define the colormap
            cmap = plt.cm.gist_ncar
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
            # define the bins and normalize
            bounds = np.linspace(0,len(np.unique(labels)),
                len(np.unique(labels)) + 1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            if label is not None:
                idx = (labels == label)
            else:
                idx = np.arange(point_cloud.shape[1])
            sc = ax.scatter(point_cloud[0, idx], point_cloud[1, idx],
                            point_cloud[2, idx],
                            c=labels[idx], s = 50.0, cmap = cmap, norm = norm)

def add_scattered_pointcloud_simple(point_cloud, ax, color = 'b', dim = 2):
    if dim == 2:
        sc = ax.scatter(point_cloud[0, :], point_cloud[1, :],
                        c=color, s = 50.0)
    elif dim == 3:
        sc = ax.scatter(point_cloud[0, :], point_cloud[1, :], point_cloud[2, :],
                        c=color, s = 50.0)

def add_affine_space(center, vector, length, ax, color = 'g', dim = 2):
    if dim == 2:
        ax.plot((center[0] -length * vector[0], center[0] + length * vector[0]),
                 (center[1] -length * vector[1], center[1] + length * vector[1]),
                 c = color, linewidth = 4)
    elif dim == 3:
        ax.plot((center[0] -length * vector[0], center[0] + length * vector[0]),
                (center[1] -length * vector[1], center[1] + length * vector[1]),
                zs=(center[2] -length * vector[2], center[2] + length * vector[2]))
