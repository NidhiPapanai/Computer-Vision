import numpy as np

def wave_kernel_signature(laplace_basis, eigenvalues, Ae, num_times):
    num_eigenfunctions = eigenvalues.shape[0]
    
    # Compute D matrix
    D = (laplace_basis.T @ (Ae @ (laplace_basis ** 2)))
    
    # Logarithmic scaling for the eigenvalues
    absolute_eigenvalues = np.abs(eigenvalues)
    log_eigenvalues = np.log(absolute_eigenvalues)
    
    emin = log_eigenvalues[1]
    emax = log_eigenvalues[-1]
    s = 7 * (emax - emin) / num_times  # Constant scaling factor
    
    # Adjust bounds with padding
    emin += 2 * s
    emax -= 2 * s
    
    # Create the time steps for the wave kernel signature
    es = np.linspace(emin, emax, num_times)
    
    # Compute the wave kernel signature matrix
    log_diff = log_eigenvalues[:, None] - es[None, :]  # Broadcasting instead of tiling
    T = np.exp(-log_diff ** 2 / (2 * s ** 2))
    
    # Compute WKS
    wks = laplace_basis @ (D @ T)
    
    return wks


def wave_kernel_map(laplace_basis, eigenvalues, Ae, num_times, landmarks):
    num_eigenfunctions = eigenvalues.shape[0]
    num_vertices = laplace_basis.shape[0]
    
    # Absolute eigenvalues and logarithmic scaling
    absolute_eigenvalues = np.abs(eigenvalues)
    log_eigenvalues = np.log(absolute_eigenvalues)
    
    emin = log_eigenvalues[1]
    emax = log_eigenvalues[-1]
    s = 7 * (emax - emin) / num_times  # Constant scaling factor
    
    # Adjust bounds with padding
    emin += 2 * s
    emax -= 2 * s
    
    # Create the time steps for the wave kernel map
    es = np.linspace(emin, emax, num_times)
    
    # Precompute the time kernel matrix (T)
    log_diff = log_eigenvalues[:, None] - es[None, :]  # Broadcasting
    T = np.exp(-log_diff ** 2 / (2 * s ** 2))
    
    # Initialize list for wave kernel maps
    wkms = []
    
    for landmark in landmarks:
        # Create the segment vector for the landmark
        segment = np.zeros(num_vertices)
        segment[landmark] = 1
        
        # Compute the wave kernel map for the landmark
        segment_projection = laplace_basis.T @ segment  # Shape: (num_eigenfunctions,)
        wkm = T * segment_projection[:, None]  # Broadcasting over time steps
        wkm = laplace_basis @ wkm  # Shape: (num_vertices, num_times)
        
        wkms.append(wkm)
    
    # Concatenate the wave kernel maps for all landmarks
    wkms = np.hstack(wkms)  # Combine along the time dimension
    
    return wkms