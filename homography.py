import numpy as np
import math
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

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])       
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])          
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


R = eulerAnglesToRotationMatrix([0.3,0,0])
t = np.array([[0,0,0]]).T
n = np.array([1,0,1.]).T

cam = PinholeCamera(0.,0.,1.,1.,0.,0.)

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
