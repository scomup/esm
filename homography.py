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

cam = PinholeCamera(640,480,200,200,320,240)

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
        [1,4,0],
        [0,-4,0]])
    Pw_ =  np.append(Pw, np.ones((Pw.shape[0],1)), axis=1)

    Twvr = np.eye(4)
    Twvr[1,3] = -7
    Twvr[2,3] = 1.
    Twvl = np.eye(4)
    Twvl[0:3,0:3] = eulerAnglesToRotationMatrix([0,0,0.3])
    Twvl[1,3] = -6.5
    Twvl[2,3] = 1.
    Tvc = np.eye(4)
    Tvc[0:3,0:3] = eulerAnglesToRotationMatrix([-np.pi/2,0,0])
    Twcr = np.dot(Twvr, Tvc)
    Twcl = np.dot(Twvl, Tvc)

    Tcrw = np.linalg.inv(Twcr)
    Tclw = np.linalg.inv(Twcl)
    Tcv = np.linalg.inv(Tvc)

    Tvlvr = np.dot(Twvr,np.linalg.inv(Twvl))


    M = np.zeros([3,4])
    M[0:3,0:3] = np.eye(3)
    M[2,3] = -1.

    N = np.zeros([4,3])
    N[0,0] = 1.
    N[1,1] = 1.
    N[3,2] = -1.

    Rcv = Tcv[0:3,0:3]
    Rvc = np.linalg.inv(Rcv)

    tmp = np.dot(cam.K,Rcv)
    tmp = np.dot(tmp,M)
    tmp = np.dot(tmp,Tvlvr)
    tmp = np.dot(tmp,N)
    tmp = np.dot(tmp,Rvc)
    Hlr = np.dot(tmp,np.linalg.inv(cam.K))
    Hlr /= Hlr[2,2]
    print(Hlr)

    Pr = np.dot(Tcrw,Pw_.T).T
    Pr = Pr[Pr[:,2]>0]
    Ir = np.dot(cam.K,Pr[:,0:3].T)
    Ir = Ir / Ir[2,:]
    Ir = Ir.T

    Pl = np.dot(Tclw,Pw_.T).T
    Pl = Pl[Pl[:,2]>0]
    Il = np.dot(cam.K,Pl[:,0:3].T)
    Il = Il / Il[2,:]
    Il = Il.T

    Il_ = np.dot( Hlr, Ir[:,0:3].T)
    Il_ = Il_[0:3,:]/Il_[2,:]
    Il_ = Il_.T
    print(Ir)
    print(Il)









    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(Ir[:, 0], Ir[:, 1], c='r') 
    ax1.scatter(Il[:, 0], Il[:, 1], c='g') 
    ax1.scatter(Il_[:, 0], Il_[:, 1], c='b') 
    #ax1.scatter(Ir__[:, 0], Ir__[:, 1], c='b') 
    plt.xlim(0,640)
    plt.ylim(0,480)
    plt.gca().invert_yaxis()
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
    M[2,3] = -1.

    N = np.zeros([4,3])
    N[0,0] = 1.
    N[1,1] = 1.
    N[3,2] = -1.
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