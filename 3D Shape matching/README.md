# 3D Mesh Analysis and Visualization

This project implements a comprehensive framework for reading, processing, and visualizing 3D meshes. It includes methods for mesh connectivity, Laplacian computations, wave kernel signature (WKS), and iterative closest point (ICP) refinement. The code also supports visualizing correspondences between two meshes using wave kernel maps.

---

## Features

- **Read OFF Files**: Load 3D meshes from `.off` files.
- **Connectivity Analysis**: Compute connectivity structures such as edges, triangles, and adjacency relationships.
- **Wave Kernel Signature**: Compute the Wave Kernel Signature (WKS) for meshes.
- **Iterative Closest Point (ICP)**: Refine correspondences between two meshes using ICP.
- **Mesh Visualization**: Render meshes in 3D with correspondence lines.
- **Geometric and Spectral Analysis**: Compute Laplacians, eigenvalues, and eigenfunctions for meshes.

---

## Requirements

The project uses Python and requires the following libraries:

- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`

To install all dependencies, run:
```bash
pip install numpy matplotlib scipy scikit-learn
