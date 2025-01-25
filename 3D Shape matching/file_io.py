import numpy as np
from scipy.io import loadmat

def read_off(filename):
    surface = loadmat(filename)
    #print(surface)
    x,y,z,tri = surface['N'][0][0][0][:,0],surface['N'][0][0][0][:,1],surface['N'][0][0][0][:,2],surface['N'][0][0][1]
    # x=np.array(x)
    # y=np.array(y)
    # z=np.array(z)
    #tri=np.array(tri)
    tri -= 1
    ver = np.c_[x,y,z]
    return ver,tri
# import numpy as np
# def read_off(file):
#     if 'OFF' != file.readline().strip():
#         raise('Not a valid OFF header')
#     n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
#     verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
#     faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
#     verts=np.array(verts)
#     faces=np.array(faces)
#     return verts, faces