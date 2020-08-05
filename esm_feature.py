#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from math import *

def euler2mat(theta):
    R_x = np.matrix([1, 0, 0, 0, cos(theta[0]), -sin(theta[0]), 0, sin(theta[0]), cos(theta[0])]).reshape([3,3])
    R_y = np.matrix([cos(theta[1]), 0, sin(theta[1]), 0, 1, 0, -sin(theta[1]), 0, cos(theta[1])]).reshape([3,3])
    R_z = np.matrix([cos(theta[2]), -sin(theta[2]), 0, sin(theta[2]), cos(theta[2]), 0, 0, 0, 1]).reshape([3,3])
    R = R_z * R_y * R_x
    return R

Rvc = euler2mat([0.8674,-0.0275,-0.0127])
K = np.matrix([329.749817,0,341.199105,0,401.966949,178.383924,0,0,1]).reshape([3,3])
Kinv = np.linalg.inv(K)
Rcv = np.linalg.inv(Rvc)

M = np.matrix([ 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., -1.]).reshape([3,4])
N = np.matrix([ 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., -1.]).reshape([4,3])

M1 = K*Rcv*M
M2 = N*Rvc*Kinv


class esm:
    def __init__(self, desc, img, rect):
        self.ref_img = img
        self.ref_desc_full = desc
        self.ref_desc = self.ref_desc_full[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        self.rect = rect
        self.precompute()
        self.ref_dxdy = self.desc_gradient(self.ref_desc)
        self.H0 = np.eye(3)
        self.H0[0,2] = rect[0]
        self.H0[1,2] = rect[1]
        self.T = np.eye(4)
        self.H = np.eye(3)
        self.last_err = np.inf
    
        
    def show_process(self):
        plt.text(-200,-600, 'image:\n', size = 10, color = "blue")
        plt.text(-300, 660, 'feature space(256 channel)\n', size = 10, color = "blue")

        self.ax1.imshow(self.ref_img,'gray',vmin=0,vmax=255)
        self.ax3.imshow(np.sum(self.ref_desc_full, 2),'hot',vmin=0,vmax=1)
        p0 = [0.,0.,1.]
        p1 = [0.,self.rect[2],1.]
        p2 = [self.rect[3],0.,1.]
        p3 = [self.rect[3],self.rect[2],1.]
        p = np.array([p0,p1,p2,p3])
        p_ref = np.dot(self.H0,p.T).T
        poly = plt.Polygon(((p_ref[0,0],p_ref[0,1]),
            (p_ref[2,0],p_ref[2,1]),
            (p_ref[3,0],p_ref[3,1]),
            (p_ref[1,0],p_ref[1,1])),
            fill=False,color='lime',linewidth=2)
        self.ax1.add_patch(poly)
        poly = plt.Polygon(((p_ref[0,0],p_ref[0,1]),
            (p_ref[2,0],p_ref[2,1]),
            (p_ref[3,0],p_ref[3,1]),
            (p_ref[1,0],p_ref[1,1])),
            fill=False,color='lime',linewidth=2)
        self.ax3.add_patch(poly)
        
        poly = plt.Polygon(((p_ref[0,0],p_ref[0,1]),
            (p_ref[2,0],p_ref[2,1]),
            (p_ref[3,0],p_ref[3,1]),
            (p_ref[1,0],p_ref[1,1])),
            fill=False,color='lime',linewidth=1)
        self.ax2.add_patch(poly)
        
        self.ax2.imshow(self.cur_img,'gray',vmin=0,vmax=255)
        self.ax4.imshow(np.sum(self.cur_desc, 2),'hot',vmin=0,vmax=1)
        #H = np.dot(self.H0,self.H)
        

        p = np.array([p0,p1,p2,p3])
        #H_inv = np.linalg.inv(self.H)
        tempH = np.dot(self.H0, self.H)
        p_cur = np.dot(tempH,p.T)
        p_cur = (p_cur/p_cur[2,:]).T
        poly = plt.Polygon(((p_cur[0,0],p_cur[0,1]),
            (p_cur[2,0],p_cur[2,1]),
            (p_cur[3,0],p_cur[3,1]),
            (p_cur[1,0],p_cur[1,1])),
            fill=False,color='blue',linewidth=2)
        self.ax2.add_patch(poly)

        poly = plt.Polygon(((p_cur[0,0],p_cur[0,1]),
            (p_cur[2,0],p_cur[2,1]),
            (p_cur[3,0],p_cur[3,1]),
            (p_cur[1,0],p_cur[1,1])),
            fill=False,color='blue',linewidth=2)
        self.ax4.add_patch(poly)


        plt.pause(0.001)
        

    def track(self, desc, img, show=False):
        self.cur_img = img
        self.cur_desc = desc
        itr = 0
        if(show):
            fig = plt.figure()
            self.ax1 = fig.add_subplot(221)
            self.ax2 = fig.add_subplot(222)
            self.ax3 = fig.add_subplot(223)
            self.ax4 = fig.add_subplot(224)
            

        while(True):
            if(show):
                self.show_process()

            cur_desc = self.get_cur_desc()
            err,residuals = self.residuals(cur_desc, self.ref_desc)
            if self.last_err - err < 0.0000001:
                print("OK!")
                break
            else:
                if(show):
                    self.ax2.cla()
                    self.ax4.cla()
            self.last_err = err
            print('itr %d, err:%f'%(itr,err))
            Ji = (self.desc_gradient(cur_desc) + self.ref_dxdy)/2.
            J = np.zeros([self.rect[2],self.rect[3],256,3])
            for u in range(self.rect[2]):
                for v in range(self.rect[3]):
                    J[v,u] = np.dot(Ji[v,u,:], self.JwJg[v,u,:,:])
            J = J.reshape(-1,3)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T,residuals)
            x0 = np.dot(hessian_inv,temp)

            A = np.zeros([4,4])
            for i in range(len(self.A)):
                A += x0[i] * self.A[i]

            dT = self.exp(A)
            #X = np.matrix([1,0,0,0, 0,1,0,-0.4, 0,0,1,0, 0,0,0,1.]).reshape([4,4])
            self.T = np.dot(self.T,dT)
            self.H = M1 * self.T * M2
            itr+=1

    def exp(self,A):
        G = np.zeros([4,4])
        A_factor = np.eye(4)
        i_factor = 1.
        for i in range(9):
            G += A_factor/i_factor
            A_factor = np.dot(A_factor, A)
            i_factor*= float(i+1) 
        return G

    def residuals(self, data1, data2):
        residuals = data1 - data2
        m = np.sum(residuals*residuals)
        return np.sqrt(m/(self.rect[2]*self.rect[3])), residuals.reshape(-1)


    def get_cur_desc(self):
        H = np.dot(self.H0, self.H)
        return cv2.warpPerspective(self.cur_desc, H,(self.rect[2],self.rect[3]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    def desc_gradient(self, desc):
        dx = np.roll(desc,-1,axis=1) - desc
        dy = np.roll(desc,-1,axis=0) - desc
        dx[:,-1] = 0.
        dx[-1,:] = 0.
        dy[:,-1] = 0.
        dy[-1,:] = 0.
        return np.stack([dx,dy],3)




    def precompute(self):
        A1 = np.matrix([0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0.]).reshape([4,4])
        A2 = np.matrix([0,0,0,0, 0,0,0,1, 0,0,0,0, 0,0,0,0.]).reshape([4,4])
        A3 = np.matrix([0,-1,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0.]).reshape([4,4])

        H1 = M1 * A1 * M2
        H2 = M1 * A2 * M2
        H3 = M1 * A3 * M2

        self.Jg = np.vstack([H1.flatten(),
            H2.flatten(),
            H3.flatten()]).T
        self.Jg = np.array(self.Jg)
        self.A = [A1,A2,A3]

        u, v = np.meshgrid( range(self.rect[2]), range(self.rect[3]))
        self.Jw = np.zeros([self.rect[2],self.rect[3],2,9])
        self.Jw[:,:,0,0] = u
        self.Jw[:,:,0,1] = v
        self.Jw[:,:,0,2] = 1.
        self.Jw[:,:,0,6] = -u*u
        self.Jw[:,:,0,7] = -u*v
        self.Jw[:,:,0,8] = -u

        self.Jw[:,:,1,3] = u
        self.Jw[:,:,1,4] = v
        self.Jw[:,:,1,5] = 1.
        self.Jw[:,:,1,6] = -u*v
        self.Jw[:,:,1,7] = -v*v
        self.Jw[:,:,1,8] = -v

        self.JwJg = np.dot(self.Jw,self.Jg)

from superpoint import SuperPointNet
import torch
from torchvision.transforms import ToTensor, ToPILImage

net = SuperPointNet()
net.load_state_dict(torch.load('/home/liu/workspace/superpoint/weight/superpoint_v1.pth',
                               map_location=lambda storage, loc: storage))


def getdesc(img):
    input_img = ToTensor()(img)
    input_img = input_img[None, :, :]

    _, desc = net.forward(input_img)
    return desc[0].permute(1, 2, 0).cpu().detach().numpy()

def shift_x(image, shift):
   h, w = image.shape[:2]
   src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
   dest = src.copy()
   dest[:,0] += shift # シフトするピクセル値
   affine = cv2.getAffineTransform(src, dest)
   return cv2.warpAffine(image, affine, (w, h))

def get_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (640,360))
    return img

    
if __name__ == "__main__":

    ref_img = get_img('/home/liu/bag/wlo60/frame0100.png')
    tar_img = get_img('/home/liu/bag/wlo60/frame0103.png')

    ref_desc = getdesc(ref_img)
    ref_desc = cv2.resize(ref_desc, (640,360),interpolation=cv2.INTER_CUBIC)
    tar_desc = getdesc(tar_img)
    tar_desc = cv2.resize(tar_desc, (640,360),interpolation=cv2.INTER_CUBIC)
    #ref_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0260.png')
    #tar_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0261.png')
    esm = esm(ref_desc,ref_img, [240, 100, 160, 160]) #x,y,weight,height
    esm.track(tar_desc,tar_img,True)
    #print(esm.H)
    plt.show()