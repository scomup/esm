#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

class esm:
    def __init__(self, img, rect):
        self.ref_img_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)/255.
        self.ref_img = self.ref_img_full[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        self.rect = rect
        self.precompute()
        self.ref_dxdy = self.image_gradient(self.ref_img)
        self.H0 = np.eye(3)
        self.H0[0,2] = rect[0]
        self.H0[1,2] = rect[1]
        self.H = np.eye(3)
        self.last_err = np.inf
    
        
    def show_process(self):
        plt.text(-300, -200, 'first image:\n', size = 10, color = "blue")
        plt.text(-300, 200, 'second image:\n', size = 10, color = "blue")
        self.ax1.imshow(self.ref_img_full,'gray',vmin=0,vmax=1)
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
            fill=False,color='lime',linewidth=1)
        self.ax2.add_patch(poly)
        

        self.ax2.imshow(self.cur_img,'gray',vmin=0,vmax=1)
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
            fill=False,color='yellow',linewidth=2)
        self.ax2.add_patch(poly)



        plt.pause(0.001)
        

    def track(self, img, show=False):
        self.cur_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)/255.
        itr = 0
        if(show):
            fig = plt.figure()
            self.ax1 = fig.add_subplot(211)
            self.ax2 = fig.add_subplot(212)
            

        while(True):
            if(show):
                self.show_process()

            cur_img = self.get_cur_image()
            err,residuals = self.residuals(cur_img, self.ref_img)
            if self.last_err - err < 0.0000001:
                print("OK!")
                break
            else:
                if(show):
                    self.ax2.cla()
            self.last_err = err
            print('itr %d, err:%f'%(itr,err))
            Ji = (self.image_gradient(cur_img) + self.ref_dxdy)/2.
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
            itr+=1

    def exp(self,A):
        G = np.zeros([3,3])
        A_factor = np.eye(3)
        i_factor = 1.
        for i in range(9):
            G += A_factor/i_factor
            A_factor = np.dot(A_factor, A)
            i_factor*= float(i+1) 
        return G

    def residuals(self, img1, img2):
        residuals = img1 - img2
        m = np.sum(residuals*residuals)
        return np.sqrt(m/(self.rect[2]*self.rect[3])), residuals.reshape(-1)


    def get_cur_image(self):
        H = np.dot(self.H0, self.H)
        return cv2.warpPerspective(self.cur_img, H,(self.rect[2],self.rect[3]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    def image_gradient(self, img):
        dx = np.roll(img,-1,axis=1) - img
        dy = np.roll(img,-1,axis=0) - img
        dx[:,-1] = 0.
        dx[-1,:] = 0.
        dy[:,-1] = 0.
        dy[-1,:] = 0.
        return np.dstack([dx,dy])




    def precompute(self):
        A1 = np.array([0,0,1,0,0,0,0,0,0.]).reshape([3,3])
        A2 = np.array([0,0,0,0,0,1,0,0,0.]).reshape([3,3])
        A3 = np.array([0,1,0,0,0,0,0,0,0.]).reshape([3,3])
        A4 = np.array([0,0,0,1,0,0,0,0,0.]).reshape([3,3])
        A5 = np.array([1,0,0,0,-1,0,0,0,0.]).reshape([3,3])
        A6 = np.array([0,0,0,0,-1,0,0,0,1.]).reshape([3,3])
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
    #ref_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0754.png')
    #tar_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0755.png')
    ref_img = cv2.imread('/home/liu/DP_DATA/HPatches/v_adam/2.ppm')
    tar_img = cv2.imread('/home/liu/DP_DATA/HPatches/v_adam/4.ppm')

    ref_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0260.png')
    tar_img = cv2.imread('/home/liu/bag/lookdown/gain4/frame0261.png')
    esm = esm(ref_img, [400, 100, 160, 160]) #x,y,weight,height
    esm.track(tar_img,True)
    print(esm.H)
    plt.show()