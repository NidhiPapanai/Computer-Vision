from file_io import read_off
from mesh_operations import mesh_info
from descriptors import wave_kernel_signature, wave_kernel_map
from visualization import visualize_map_lines
from utils import icp_refine,euclidean_fps
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

print("start")
numEigsSrc = 60
numEigsTar = 60
#print("reading the source shape...")
#start_time = time.time()
#file=open('Mesh/tr_reg_089.off','r')
file='C:/Users/NIDHI/Downloads/Map3D/SGP_dataset/holes/null/cat0.mat'
X, T = read_off(file)
Src = mesh_info(X, T, 200)
#dict_keys(['X', 'T', 'E2V', 'T2E', 'E2T', 'T2T', 'nf', 'nv', 'ne', 'normal', 'area', 'sqrt_area', 'Nv', 'SqEdgeLength', 'cotLaplacian', 'Ae', 'laplaceBasis', 'eigenvalues'])
print('done (found {} vertices)'.format(Src['Nv']))
#print(f'Time taken: {time.time() - start_time:.2f} seconds')

# Reading the target shape
print("reading the target shape...", end='')
#start_time = time.time()
file='C:/Users/NIDHI/Downloads/Map3D/SGP_dataset/holes/null/cat0.mat'
X, T = read_off(file)
Tar = mesh_info(X, T, 200)
print('done (found {} vertices)'.format(Tar['Nv']))
#print(f'Time taken: {time.time() - start_time:.2f} seconds')

#check landmark values.
landmarks1 = np.arange(500, 6001, 1000).reshape(-1, 1)
landmarks2 = landmarks1
landmarks = np.hstack((landmarks1, landmarks2[:, :1]))

# Compute descriptors
#print(Src['laplaceBasis'][0][0])
fct_src = wave_kernel_signature(Src['laplaceBasis'], Src['eigenvalues'], Src['Ae'], 200)
fct_src = np.column_stack([fct_src, wave_kernel_map(Src['laplaceBasis'], Src['eigenvalues'], Src['Ae'], 200, landmarks[:, 0])])

fct_tar = wave_kernel_signature(Tar['laplaceBasis'], Tar['eigenvalues'], Tar['Ae'], 200)
fct_tar = np.column_stack([fct_tar, wave_kernel_map(Tar['laplaceBasis'], Tar['eigenvalues'], Tar['Ae'], 200, landmarks[:, 1])])

#Subsample descriptors/dimensionality reduction.
fct_src = fct_src[:, ::10]
fct_tar = fct_tar[:, ::10]
print(f'done computing descriptors ({fct_src.shape[1]} on source and {fct_tar.shape[1]} on target)')
assert fct_src.shape[1] == fct_tar.shape[1]

# Normalization
no = np.sqrt(np.diag(fct_src.T @ Src['Ae'] @ fct_src))
fct_src = fct_src / no[np.newaxis, :]
fct_tar = fct_tar / no[np.newaxis, :]
print('Pre-computing the multiplication operators...', end='')

# Multiplication Operators
numFct = fct_src.shape[1]
OpSrc = [None] * numFct
OpTar = [None] * numFct

SrcLaplaceBasis = Src['laplaceBasis'] 
SrcEigenvalues = Src['eigenvalues']  

TarLaplaceBasis = Tar['laplaceBasis'] 
TarEigenvalues = Tar['eigenvalues']  

# Perform slicing
Src['laplaceBasis'] = SrcLaplaceBasis[:, :numEigsSrc]
Src['eigenvalues'] = SrcEigenvalues[:numEigsSrc]

Tar['laplaceBasis'] = TarLaplaceBasis[:, :numEigsTar]
Tar['eigenvalues'] = TarEigenvalues[:numEigsTar]

for i in range(numFct):
    OpSrc[i] = Src['laplaceBasis'].T @ Src['Ae'] @ (fct_src[:, i][:, np.newaxis] * Src['laplaceBasis'])
    OpTar[i] = Tar['laplaceBasis'].T @ Tar['Ae'] @ (fct_tar[:, i][:, np.newaxis] * Tar['laplaceBasis'])

Fct_src = Src['laplaceBasis'].T @ Src['Ae'] @ fct_src
Fct_tar = Tar['laplaceBasis'].T @ Tar['Ae'] @ fct_tar
print('done')

# Fmap Computation
print('Optimizing the functional map...')

# Compute Dlb (distance between eigenvalues)
Dlb = (np.tile(Src['eigenvalues'], (numEigsTar, 1)) - np.tile(Tar['eigenvalues'].T, (numEigsSrc, 1)))**2
Dlb = Dlb / np.linalg.norm(Dlb, 'fro')**2

# Constant functional map
constFct = np.sign(Src['laplaceBasis'][0, 0] * Tar['laplaceBasis'][0, 0]) * np.array([np.sqrt(np.sum(Tar['area']) / np.sum(Src['area'])), *np.zeros(numEigsTar - 1)])

# Parameters
a = 1e-1  # Descriptors preservation
b = 1     # Commutativity with descriptors
c = 1e-3  # Commutativity with Laplacian

# Define the functional map objective function
def funObj(F):
    # Reshape F and compute terms
    F_reshaped = F.reshape((numEigsTar, numEigsSrc))
    
    # Term 1: Objective function involving the difference between Fct_src and Fct_tar
    term1 = a * np.sum((F_reshaped @ Fct_src - Fct_tar)**2) / 2
    
    # Term 2: Regularization term involving OpTar and OpSrc
    OpTerm = sum([np.sum((X @ F_reshaped - F_reshaped @ Y)**2) for X, Y in zip(OpTar, OpSrc)])
    term2 = b * OpTerm / 2
    
    # Term 3: Penalty term involving Dlb
    term3 = c * np.sum((F**2 * Dlb.flatten()) / 2)
    
    # Gradients
    grad1 = (a * np.ravel((F_reshaped @ Fct_src - Fct_tar) @ Fct_src.T)).flatten()
    
    # Efficient gradient computation for term2 (using numpy instead of list comprehension)
    grad2 = (b * np.sum([np.ravel(X.T @ (X @ F_reshaped - F_reshaped @ Y) - (X @ F_reshaped - F_reshaped @ Y) @ Y) for X, Y in zip(OpTar, OpSrc)], axis=0)).flatten()
    
    grad3 = c * (F * Dlb.flatten())
    
    # Return the objective and gradients
    return term1 + term2 + term3, grad1 + grad2 + grad3

# Projection function to ensure the constraints on F
def funProj(F):
    return np.concatenate([constFct, F[numEigsTar:]])



# Initial guess for F_lb
F_lb = np.zeros(numEigsTar * numEigsSrc)
F_lb[0] = constFct[0]

print('done')

# Define the options for the optimization
options = {'disp': True}  # equivalent to verbose = 1 in MATLAB
# Perform the optimization using a quasi-Newton method (BFGS)
# Optimization using BFGS
result = minimize(lambda F: funObj(F)[0], F_lb, jac=lambda F: funObj(F)[1], method='CG', options=options)

# Reshape the result into the desired shape
F_lb = result.x.reshape((numEigsTar, numEigsSrc))

print('done fmap optimization.')

print('ICP refinement...', end='')

# Call the icp_refine function
F_lb2, _ = icp_refine(Src['laplaceBasis'], Tar['laplaceBasis'], F_lb, 5)

print('done')

print('Converting to p2p map...', end='')

# fmap before ICP (for comparison)
transformed_src_before_icp = (F_lb @ Src['laplaceBasis'].T).T
transformed_src_before_icp*=1e5
nbrs_before_icp = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Tar['laplaceBasis'])
pF_lb = nbrs_before_icp.kneighbors(transformed_src_before_icp, return_distance=False).flatten()

# fmap after ICP
transformed_src_after_icp = (F_lb2 @ Src['laplaceBasis'].T).T
nbrs_after_icp = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Tar['laplaceBasis'])
pF_lb2 = nbrs_after_icp.kneighbors(transformed_src_after_icp, return_distance=False).flatten()

print('done')


print('Visualizing results')

figure = plt.figure(figsize=(16, 4))

# Number of basis functions
nbasis = Src['laplaceBasis'].shape[1]

# Generate random smooth function
kf = np.random.rand(nbasis) - 0.5
kf = kf * ((np.arange(nbasis, 0, -1)) ** 2)

f = Src['laplaceBasis'] @ kf
f_normalized = (f - np.min(f)) / (np.max(f) - np.min(f))
facecolors = plt.cm.viridis(f_normalized)
# Subplot 1: f = random smooth function
ax1 = figure.add_subplot(1, 4, 1, projection='3d')
ax1.plot_trisurf(Src['X'][:, 0], Src['X'][:, 1], Src['X'][:, 2], triangles=Src['T'], cmap='viridis', shade=True, antialiased=True, linewidth=0,facecolors=facecolors)
ax1.view_init(elev=90, azim=-90)
ax1.set_title('f = random smooth function')
ax1.axis('off')

# Subplot 2: f(p2pmap) before ICP
ax2 = figure.add_subplot(1, 4, 2, projection='3d')
ax2.plot_trisurf(Tar['X'][:, 0], Tar['X'][:, 1], Tar['X'][:, 2], triangles=Tar['T'], cmap='viridis', shade=True, antialiased=True, linewidth=0)
ax2.view_init(elev=90, azim=-90)
ax2.set_title('f(p2pmap) before ICP')
ax2.axis('off')

# Subplot 3: f(p2pmap) after ICP
ax3 = figure.add_subplot(1, 4, 3, projection='3d')
ax3.plot_trisurf(Tar['X'][:, 0], Tar['X'][:, 1], Tar['X'][:, 2], triangles=Tar['T'], cmap='viridis', shade=True, antialiased=True, linewidth=0)
ax3.view_init(elev=90, azim=-90)
ax3.set_title('f(p2pmap) after ICP')
ax3.axis('off')

# Subplot 4: Fmap*f (without p2p)
ax4 = figure.add_subplot(1, 4, 4, projection='3d')
ax4.plot_trisurf(Tar['X'][:, 0], Tar['X'][:, 1], Tar['X'][:, 2], triangles=Tar['T'], cmap='viridis', shade=True, antialiased=True, linewidth=0)
ax4.view_init(elev=90, azim=-90)
ax4.set_title('Fmap*f (without p2p)')
ax4.axis('off')

plt.tight_layout()
plt.show()

# Create figure
#plt.figure(figsize=(12, 6))

# Subplot 1: Map before ICP
#plt.subplot(1, 2, 1)
samples = euclidean_fps(Src, 200)
visualize_map_lines(Tar, Src, pF_lb, samples,"Map before ICP")
#plt.view_init(elev=90, azim=-90)  # Adjust view
# plt.title('Map before ICP')
# plt.axis('equal')
# plt.axis('off')

# Compute and print the mean Euclidean map error (without ICP)
error_before_icp = np.mean(np.sqrt(np.sum((Tar['X'] - Tar['X'][pF_lb, :]) ** 2, axis=1))) / Tar['sqrt_area']
print(f'Mean Euclidean map error (without ICP): {error_before_icp:.6f}')

# Subplot 2: Map after ICP
#plt.subplot(1, 2, 2)
samples = euclidean_fps(Src, 200)
visualize_map_lines(Tar, Src, pF_lb2, samples,"Map after ICP")
#plt.view_init(elev=90, azim=-90)  # Adjust view
# plt.title('Map after ICP')
# plt.axis('equal')
# plt.axis('off')

# Compute and print the mean Euclidean map error (with ICP)
error_after_icp = np.mean(np.sqrt(np.sum((Tar['X'] - Tar['X'][pF_lb2, :]) ** 2, axis=1))) / Tar['sqrt_area']
print(f'Mean Euclidean map error (with ICP): {error_after_icp:.6f}')
