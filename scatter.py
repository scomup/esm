import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib

print(matplotlib.__version__)
# 3.0.3
#for 3D plotting
from mpl_toolkits.mplot3d import Axes3D


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.eye(3)
        self.K[0,0] = fx
        self.K[1,1] = fy
        self.K[0,2] = cx
        self.K[1,2] = cy

cam = PinholeCamera(0.,0.,1.,1.,0.,0.)

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])       
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])          
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

if __name__ == '__main__':
    Pw = np.array([
        [-1,-4,0],
        [-1,-3,0],
        [-1,-2,0],
        [-1,-1,0],
        [-1,0,0],
        [-1,1,0],
        [-1,2,0],
        [-1,3,0],
        [-1,4,0],
        [1,-4,0],
        [1,-3,0],
        [1,-2,0],
        [1,-1,0],
        [1,0,0],
        [1,1,0],
        [1,2,0],
        [1,3,0],
        [1,4,0]]) 
    Pw_ =  np.append(Pw, np.ones((Pw.shape[0],1)), axis=1)

    Twc1 = np.eye(4)
    Twc1[0:3,0:3] = eulerAnglesToRotationMatrix([0.3,0,0])
    Twc1[2,3] = 1.
    Tc1w = np.linalg.inv(Twc1)

    Pc1 = np.dot(Tc1w,Pw_.T).T
    Ic1 = np.dot(cam.K,Pc1[:,0:3].T).T
    print(Ic1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(Ic1[:, 0], Ic1[:, 1], c='g') 
    plt.show()
    """   
    fig = plt.figure()
    ax1 = fig.add_subplot(111 , projection='3d')
    ax1.scatter(X[:, 0], X[:, 1],zs=X[:, 2],zdir='z', s=50, vmin=0, vmax=1, c='r', cmap=plt.cm.jet) 

    R = eulerAnglesToRotationMatrix([0.1,0.1,0.2])
    X2 = np.dot(R,X.T).T
    ax1.scatter(X2[:, 0], X2[:, 1],zs=X2[:, 2],zdir='z', s=50, vmin=0, vmax=1, c='g', cmap=plt.cm.jet) 
    T = np.zeros([3,4])
    T[0:3,0:3]= R
    T[:,3] = -R[:,2]
    M = np.zeros([3,4])
    M[0:3,0:3] = np.eye(3)
    M[2,3] = 0.

    N = np.zeros([4,3])
    N[0,0] = 1.
    N[1,1] = 1.
    N[2,2] = 1.
    N[3,2] = -0.
    Tvc = np.dot(N,np.linalg.inv(R))
    T = np.dot(R,M)
    print(np.linalg.inv(R))
    print(Tvc)
    print(np.dot(T,Tvc))
    print(np.dot(M,N))


    
    X =  np.append(X, np.ones((X.shape[0],1)), axis=1)
    X3 = np.dot(T,X.T).T
    X4 = np.dot(Tvc,X3.T).T
    ax1.scatter(X3[:, 0], X3[:, 1],zs=X3[:, 2],zdir='z', s=50, vmin=0, vmax=1, c='b', cmap=plt.cm.jet) 
    ax1.scatter(X4[:, 0], X4[:, 1],zs=X4[:, 2],zdir='z', s=50, vmin=0, vmax=1, c='b', cmap=plt.cm.jet) 

    plt.show()
    """