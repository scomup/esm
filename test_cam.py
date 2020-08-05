import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

d = 3
K = np.matrix([329.749817,0,341.199105,0,401.966949,178.383924,0,0,1]).reshape([3,3])
K_inv = np.linalg.inv(K)
M = np.matrix([ 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., -1.*d]).reshape(3,4)
N = np.matrix([ 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., -1./d]).reshape(4,3)

land_mark = np.matrix([
     0,1,0,1,
    -1,1,0,1,
    -1,2,0,1,
    -1,3,0,1,
     1,1,0,1,
     1,2,0,1,
     1,3,0,1]).reshape([-1,4]).T


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

Rvc = eulerAnglesToRotationMatrix([np.pi/2,0,0])


def Pose(pose,Rvc,d):
    x,y,yaw=pose
    R_z = np.array([[np.cos(yaw),    -np.sin(yaw), 0],
                [np.sin(yaw),    np.cos(yaw),     0],
                [0,               0,              1]])  

    Rwc = np.dot(Rvc, R_z)
    T = np.eye(4)
    T[0:3,0:3] = Rwc
    T[0,3] = x
    T[1,3] = y
    T[2,3] = d
    return np.matrix(T)

def get_img_point(Twc):
    Tcw = np.linalg.inv(Twc)
    Pc  = Tcw*land_mark
    Ic  = K*Pc[0:3,:]
    Ic /= Ic[2,:]
    return Ic

def Hlr(vr,vl):
    Tvr = Pose(vr,np.eye(3),0)
    Tvl = Pose(vl,np.eye(3),0)
    Tvlvr = np.linalg.inv(Tvl) * Tvr
    Rcv = np.linalg.inv(Rvc)
    Hlr = K * Rcv * M * Tvlvr * N * Rvc * K_inv
    #Hlr = K * Rcv * Tvlvr * Rvc *(I|-ndc)* K_inv
    return Hlr

def H0r(v):
    Twv = Pose(v,np.eye(3),0)
    Rwv = Twv[0:3,0:3]
    t = Twv[0:3,3]
    Rcv = np.linalg.inv(Rvc)
    Rwc = Rwv * Rvc
    n = Rcv * np.matrix([0,0,1]).reshape([3,1])
    A = Rwc - t * n.T/d
    H = K * A * K_inv
    return H


land_mark = np.matrix([
     0,1,0,1,
    -1,1,0,1,
    -1,2,0,1,
    -1,3,0,1,
     1,1,0,1,
     1,2,0,1,
     1,3,0,1]).reshape([-1,4]).T

Twc = np.eye(4)

vr=[0,0,0]
vl=[0,0.5,0]

Twcr = Pose(vr,Rvc,d)
Icr = get_img_point(Twcr)


Twcl = Pose(vl,Rvc,d)
Icl = get_img_point(Twcl)

H = Hlr(vr,vl)
H_inv = np.linalg.inv(H)
Icr_topview = H *Icr
Icr_topview/= Icr_topview[2,:]


#Hlr = Hlr(vr,vl)
#npl = Hlr*Icr
#npl /= npl[2,:]
plt.scatter(Icr_topview[0,:].A1,Icr_topview[1,:].A1,c='r')
plt.scatter(Icl[0,:].A1,Icl[1,:].A1,c='g',label='live image')
plt.scatter(Icr[0,:].A1,Icr[1,:].A1,c='b',label='reference image')
plt.legend()
#plt.xlim([0,680])
#plt.ylim([0,390])
plt.gca().invert_yaxis()
plt.show()
print(Ic1.T)
print(Tcw1)
