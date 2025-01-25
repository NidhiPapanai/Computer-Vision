import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import normalize_function
import numpy as np

def visualize_map_lines(S1, S2, map, samples,title):
    X1 = S1['X'][:, 0]
    Y1 = S1['X'][:, 1]
    Z1 = S1['X'][:, 2]

    X2 = S2['X'][:, 0]
    Y2 = S2['X'][:, 1]
    Z2 = S2['X'][:, 2]

    # Normalize the coordinates for color mapping
    g1 = normalize_function(0.1, 0.99, Y2)
    g2 = normalize_function(0.1, 0.99, Z2)
    g3 = normalize_function(0.1, 0.99, X2)

    f1 = g1[map]
    f2 = g2[map]
    f3 = g3[map]

    # Plot semi-transparent meshes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Source mesh
    ax.plot_trisurf(X1, Y1, Z1, triangles=S1['T'], cmap='viridis', facecolors=np.column_stack([f1, f2, f3]), alpha=0.6)

    # Target mesh (shifted to the right for clarity)
    xdiam = 3 / 2 * (np.max(X2) - np.min(X2))
    ax.plot_trisurf(X2 + xdiam, Y2, Z2, triangles=S2['T'], cmap='viridis', facecolors=np.column_stack([g1, g2, g3]), alpha=0.6)

    if samples is not None and len(samples) > 0:
        target_samples = map[samples]

        Xstart = X1[samples]
        Ystart = Y1[samples]
        Zstart = Z1[samples]
        Xend = X2[target_samples]
        Yend = Y2[target_samples]
        Zend = Z2[target_samples]

        Xend += xdiam
        Colors = np.column_stack([f1, f2, f3])
        ColorSet = Colors[samples, :]

        # Plot the lines between corresponding points
        for i in range(len(samples)):
            ax.plot([Xstart[i], Xend[i]], [Ystart[i], Yend[i]], [Zstart[i], Zend[i]], color=ColorSet[i], lw=2)
    plt.title(title)
    plt.show()