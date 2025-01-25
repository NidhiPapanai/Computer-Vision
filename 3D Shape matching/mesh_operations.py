import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

def connectivity(T):
    nf = T.shape[0]  # Number of faces
    nv = np.max(T)   # Number of vertices

    # Step 1: Create edges
    E2V = np.vstack((T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]))
    id1=np.argsort(E2V,axis=1)
    E2V = np.sort(E2V, axis=1)  # Sort vertices in each edge
    E2V, ia, ic = np.unique(E2V, axis=0, return_index=True, return_inverse=True)
    # Step 2: Edge signs
    edgeSg = id1[:,0]-id1[:,1]
    t2es = np.column_stack((edgeSg[:nf], edgeSg[nf:2*nf], edgeSg[2*nf:]))
    edgeSg = edgeSg[ia]

    # Step 3: Map edges to triangles
    e1 = ic[:nf]
    e2 = ic[nf:2*nf]
    e3 = ic[2*nf:]
    T2E = np.column_stack((e1, e2, e3))
    # Step 4: Find boundary edges
    edge_counts = np.bincount(T2E.flatten())
    bound = np.where(edge_counts == 1)[0]
    nB = len(bound)

    if nB > 0:
        print(f"Warning: Mesh with {nB} boundary edges.")
    
    temp=np.hstack((e1,e2,e3,bound))
    idx = np.argsort(np.hstack((e1, e2, e3, bound)))
    idx[idx < 3 * nf] = ((idx[idx < 3 * nf]) % nf)
    idx[idx >= 3 * nf] = 0
    E2T = idx.reshape(2, len(E2V)).T
    E2T[E2T > nf] = 0
    E2T = np.column_stack((E2T, edgeSg, -edgeSg))

    # Step 5: Triangle-to-triangle adjacency
    T2T = np.column_stack((
        E2T[T2E[:, 0], 0], E2T[T2E[:, 1], 0], E2T[T2E[:, 2], 0],
        E2T[T2E[:, 0], 1], E2T[T2E[:, 1], 1], E2T[T2E[:, 2], 1]
    ))
    T2T = np.sort((T2T != np.arange(0, nf )[:, None]) * T2T, axis=1)
    T2T = T2T[:, 3:6]
    print(T2T)

    # Adjust T2E to include edge signs
    T2E = T2E * t2es
    return E2V, T2E, E2T, T2T

def per_edge_laplacian(T2E, E2V, area, perEdge):
    # nv = X.shape[0]
    nv = np.max(E2V)+1 # Number of vertices
    ne = E2V.shape[0]  # Number of edges
    print("ne")
    print(ne)

    # Step 1: Compute sparse matrix for edge Laplacian (Le)
    ILe = np.hstack((T2E[:, 0], T2E[:, 1], T2E[:, 2]))
    JLe = np.hstack((T2E[:, 1], T2E[:, 2], T2E[:, 0]))
    SLe = (1 / 8) * (1 / np.hstack((area, area, area)))

    # Create sparse matrix for Le
    Le = coo_matrix((-SLe, (ILe, JLe)), shape=(ne, ne))
    Le = Le + Le.T + coo_matrix((SLe, (ILe, ILe)), shape=(ne, ne))

    # Step 2: Compute sparse matrix for per-edge Laplacian (W)
    Ie = E2V[:, 0]
    Je = E2V[:, 1]
    S = Le @ perEdge

    W = coo_matrix(
        (np.hstack((S, S, -S, -S)), 
         (np.hstack((Ie, Je, Ie, Je)), np.hstack((Je, Ie, Ie, Je)))),
        shape=(nv, nv)
    )

    return W, Le

def per_edge_area(E2T, E2V, area):
    nv = np.max(E2V)+1  # Number of vertices
    ne = E2V.shape[0]  # Number of edges
    nf = area.shape[0]  # Number of triangles

    # Step 1: Construct sparse edge-area matrix (Le)
    I = np.hstack((np.where(E2T[:, 0] != 0)[0], np.where(E2T[:, 1] != 0)[0]))
    J = np.hstack((E2T[E2T[:, 0] != 0, 0], E2T[E2T[:, 1] != 0, 1]))
    S = np.ones(len(I)) / 12

    Le = coo_matrix((S, (I, J)), shape=(ne, nf))

    # Step 2: Compute sparse weight matrix (W)
    Ie = E2V[:, 0]
    Je = E2V[:, 1]
    S = Le @ area

    W = coo_matrix(
        (np.hstack((S, S, S, S)),
         (np.hstack((Ie, Je, Ie, Je)), np.hstack((Je, Ie, Ie, Je)))),
        shape=(nv, nv)
    )

    return W, Le


def mesh_info(X, T, num_eigs):
    mesh = {}
    mesh['X'] = X
    mesh['T'] = T

    # Connectivity
    mesh['E2V'], mesh['T2E'], mesh['E2T'], mesh['T2T'] = connectivity(T)

    mesh['nf'] = T.shape[0]
    mesh['nv'] = X.shape[0]
    mesh['ne'] = mesh['E2V'].shape[0]
    num_eigs = min(num_eigs, mesh['nv'])

    # Normals and areas
    normals = np.cross(X[T[:, 0], :] - X[T[:, 1], :], X[T[:, 0], :] - X[T[:, 2], :])
    areas = np.sqrt(np.sum(normals**2, axis=1)) / 2
    normals = normals / np.sqrt(np.sum(normals**2, axis=1))[:, np.newaxis]

    mesh['normal'] = normals
    mesh['area'] = areas
    mesh['sqrt_area'] = np.sqrt(np.sum(areas))

    # Vertex normals
    row_indices = T.flatten()
    col_indices = np.repeat(np.arange(mesh['nf']), 3)
    data = np.tile(areas, 3)
    A = csr_matrix((data, (row_indices, col_indices)), shape=(mesh['nv'], mesh['nf']))
    Nv = A @ normals
    Nv = Nv / np.sqrt(np.sum(Nv**2, axis=1))[:, np.newaxis]
    mesh['Nv'] = Nv

    # Edge lengths
    edge_lengths_squared = np.sum((X[mesh['E2V'][:, 0], :] - X[mesh['E2V'][:, 1], :])**2, axis=1)
    mesh['SqEdgeLength'] = edge_lengths_squared

    # Eigen decomposition
    cot_laplacian, _ = per_edge_laplacian(np.abs(mesh['T2E']), mesh['E2V'], areas, edge_lengths_squared)
    cot_laplacian = -cot_laplacian
    mesh['cotLaplacian'] = cot_laplacian

    Ae, _ = per_edge_area(mesh['E2T'], mesh['E2V'], areas)
    mesh['Ae'] = Ae

    try:
        eigenvalues, laplace_basis = eigsh(cot_laplacian, k=num_eigs, M=Ae, sigma=1e-5)
    except:
        # In case of trouble, make the Laplacian definite
        eigenvalues, laplace_basis = eigsh(cot_laplacian - 1e-8 * np.eye(mesh['nv']), k=num_eigs, M=Ae, which='SM')

    mesh['laplaceBasis'] = laplace_basis
    mesh['eigenvalues'] = eigenvalues

    return mesh
