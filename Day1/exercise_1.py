# Lab exercise 1:

# Ajad Chhatkuli
# Uses python 2.7 -> just change the print for 3
# Note: You might need additional library to plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate 3D data -> 3x50 is preferable over 50x3
N=50
p = np.random.uniform(0,1,(3,N))

# add [1,1,4]
# move 3D object away from the camera with some random vector
offset = np.repeat([[1],[1],[4]], N, axis=1)
p = p + offset

# bonus visualize 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p[0,:], p[1,:], p[2,:], zdir='z', s=20, c=None, depthshade=True)
plt.interactive(False)
plt.show()


# Add one to the point vectors
v_ones = np.ones((1,N))
p = np.vstack((p,v_ones))
print p.shape

# Generate cameras of 3X4
v_zeros = np.zeros((3,1))
# First camera at origin
M1 = np.hstack((np.eye(3),v_zeros))

# Second camera at R, t
# R is pi/6 rotation x, y
R = np.array([[0.8660, 0.2500, 0.4330],
     [0, 0.8660, -0.5000],
      [-0.5000, 0.4330, 0.7500]])

t = np.array([[-0.4, 0.3, 0.6]])
t = t.T
print R.shape
print t.shape
M2 = np.hstack((R,t))

# Now project all points at once:


# Image points in camera 1
q1 = np.dot(M1,p)
q2 = np.dot(M2,p)

# output is w*[u, v, 1] => ok for epipolar equation, not ok for actual camera projections

# to visualize image projections you need [u,v,1], therefore divide by last homogeneous element
u1 = q1[0,:]/q1[2,:]
v1 = q1[1,:]/q1[2,:]

u2 = q2[0,:]/q2[2,:]
v2 = q2[1,:]/q2[2,:]


# form the epipolar equations

# equation Matrix:
A = np.vstack((u2 * u1,  u2 * v1 , u2 , v2 * u1 , v2 * v1 , v2 , u1 , v1 , np.ones((1,N))))
A = A.T
print A.shape

# To solve A* f = 0, ||f||=1

u_svd,s_svd,vh_svd = np.linalg.svd(A,full_matrices=False)

# Note python gives vh_svd.T instead of vh_svd
# A = u*s*vh in python instead of u*s*vh.T

# optional :# Check svd decomposition
s_svd = np.diag(s_svd)
Ah = np.dot(np.dot(u_svd,s_svd),vh_svd)
print np.linalg.norm((A-Ah))

# get solution: last row of vh

f = vh_svd[-1,:]

# get the essential matrix:
E = f.reshape((3,3))

print E
# The actual Essential Matrix from the is:
tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
E_actual = np.dot(tx,R)

# scale is not relevant since if E satisfies epipolar eqn w*E also does.
E_actual = E_actual/np.linalg.norm(E_actual)
print E_actual