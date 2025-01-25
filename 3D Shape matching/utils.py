import numpy as np
from sklearn.neighbors import NearestNeighbors

def normalize_function(min_new, max_new, f):
    fnew = f - np.min(f)
    fnew = (max_new - min_new) * fnew / np.max(fnew) + min_new
    return fnew

def icp_refine(L1, L2, C, nk):
    Vs = L1
    Vt = L2
    n1 = L1.shape[1]  # Number of points in L1
    n2 = L2.shape[1]  # Number of points in L2
    C=np.array(C)

    for k in range(nk):
        # Find the nearest neighbors
        transformed_Vs = (C @ Vs.T).T  # Transform source points
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(transformed_Vs)
        matches = nbrs.kneighbors(Vt, return_distance=False).flatten()
        # Compute the transformation matrix W
        W = np.linalg.lstsq(L2, L1[matches, :], rcond=None)[0]

        # Compute SVD and update C
        U, _, Vt_ = np.linalg.svd(W)
        C = U @ np.eye(n2, n1) @ Vt_

    return C, matches

def euclidean_fps(surface, k, seed=None):
    C = surface['X']
    nv = C.shape[0]

    if seed is None:
        idx = np.random.randint(nv)
    else:
        idx = seed

    dists = np.sum((C - C[idx, :]) ** 2, axis=1)

    for i in range(k):
        maxi = np.argmax(dists)
        idx = np.append(idx, maxi)
        new_dists = np.sum((C - C[maxi, :]) ** 2, axis=1)
        dists = np.minimum(dists, new_dists)
    
    return idx[1:]

