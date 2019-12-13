#!/usr/bin/env python3

import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

class esm:
    def __init__(self, ref_img, tar_img, rect):
        self.ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(float)/255.
        self.tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY).astype(float)/255.
        t = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)
        t1 = t[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        t2 = t[100:260,400:560]
        self.tar_img = self.tar_img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        #cv2.imshow("win",cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]])
        #cv2.waitKey(0)
        self.rect = rect
        self.precompute()
        self.tar_dxdy = self.image_gradient(self.tar_img)
        self.H0 = np.eye(3)
        self.H0[0,2] = rect[0]
        self.H0[1,2] = rect[1]
        self.H = np.eye(3)
    
        

    def track(self):
        while(True):
            self.cur_img = self.get_cur_image()
            r,residuals = self.residuals()
            print(r)
            Ji = (self.image_gradient(self.cur_img) + self.tar_dxdy)/2.
            J = np.zeros([self.rect[2],self.rect[3],8])
            for u in range(self.rect[2]):
                for v in range(self.rect[3]):
                    J[v,u] = np.dot(Ji[v,u,:], self.JwJg[v,u,:,:])
            J = J.reshape(-1,8)
            hessian = np.dot(J.T,J)
            hessian_inv = np.linalg.inv(hessian)
            temp = -np.dot(J.T,residuals)
            x0 = np.dot(hessian_inv,temp)

            A = np.zeros([3,3])
            for i in range(len(self.A)):
                A += x0[i] * self.A[i]

            dH = self.exp(A)

            self.H = np.dot(self.H,dH)

    def exp(self,A):
        G = np.zeros([3,3])
        A_factor = np.eye(3)
        i_factor = 1.
        for i in range(9):
            G += A_factor/i_factor
            A_factor = np.dot(A_factor, A)
            i_factor*= float(i+1) 
        return G

    def residuals(self):
        residuals = self.cur_img - self.tar_img
        m = np.sum(residuals*residuals)
        return np.sqrt(m/self.rect[2]*self.rect[3]), residuals.reshape(-1)


    def get_cur_image(self):
        H = np.dot(self.H0, self.H)
        return cv2.warpPerspective(self.ref_img, H,(self.rect[2],self.rect[3]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    def image_gradient(self, img):
        dx = np.roll(self.tar_img,1,axis=1) - self.tar_img
        dy = np.roll(self.tar_img,1,axis=0) - self.tar_img
        dx[:,0] = 0.
        dx[:,-1] = 0.
        dy[0,:] = 0.
        dy[-1,:] = 0.
        return np.dstack([dx,dy])




    def precompute(self):
        A1 = np.array([0,0,1,0,0,0,0,0,0.]).reshape([3,3])
        A2 = np.array([0,0,0,0,0,1,0,0,0.]).reshape([3,3])
        A3 = np.array([0,1,0,0,0,0,0,0,0.]).reshape([3,3])
        A4 = np.array([0,0,0,1,0,0,0,0,0.]).reshape([3,3])
        A5 = np.array([1,0,0,0,-1,0,0,0,0.]).reshape([3,3])
        A6 = np.array([0,0,0,0,-1,0,0,0,0.]).reshape([3,3])
        A7 = np.array([0,0,0,0,0,0,1,0,0.]).reshape([3,3])
        A8 = np.array([0,0,0,0,0,0,0,1,0.]).reshape([3,3])
        self.A = [A1,A2,A3,A4,A5,A6,A7,A8]

        self.Jg = np.vstack([A1.flatten(),
            A2.flatten(),
            A3.flatten(),
            A4.flatten(),
            A5.flatten(),
            A6.flatten(),
            A7.flatten(),
            A8.flatten()]).T
        self.A = [A1,A2,A3,A4,A5,A6,A7,A8]

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






    
if __name__ == "__main__":
    ref_img = cv2.imread('/home/liu/workspace/LK20_ImageAlignment/lenna.png')
    tar_img = cv2.imread('/home/liu/workspace/LK20_ImageAlignment/lenna.png')

    #ref_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0261.png')
    #tar_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0261.png')
    esm = esm(ref_img, tar_img, [100, 100, 160, 160]) #x,y,weight,height
    esm.track()