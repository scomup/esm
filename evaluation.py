#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

from esm import esm

def match(f1, f2): 
    ref_img = cv2.imread(f1)
    tar_img = cv2.imread(f2)


def hpatches_pair():
  path = "/home/liu/DP_DATA/HPatches/"
  pairs = []
  import os
  l = os.listdir(path) 
  for p in l:
    if(p[0] != 'i'):
      continue
    f1 = path + p + '/1.ppm'
    f2 = path + p + '/1.ppm'
    pairs.append([f1,f2])
  return pairs
    
def hpatches_pair16():
  path = "/home/liu/DP_DATA/HPatches/"
  pairs = []
  import os
  l = os.listdir(path) 
  for p in l:
    if(p[0] != 'i'):
      continue
    f1 = path + p + '/1.ppm'
    f2 = path + p + '/2.ppm'
    pairs.append([f1,f2])
  return pairs

def lookdown_pairs():
  path  = "/home/liu/bag/lookdown/gain4/"
  pairs = []
  import os
  l = os.listdir(path)
  l.sort()
  for idx in range(200,350):
    f1 = path+l[idx]
    f2 = path+l[idx+1]
    pairs.append([f1,f2])
  return pairs


if __name__ == '__main__':
  #pairs = hpatches_pair()
  #pairs = hpatches_pair16()
  pairs = lookdown_pairs()
  idx = 0
  for f1,f2 in pairs:
    ref_img = cv2.imread(f1)
    tar_img = cv2.imread(f2)
    ref_img = cv2.resize(ref_img, (50,50))
    tar_img = cv2.resize(tar_img, (50,50))
    ref_img = cv2.resize(ref_img, (500,500))
    tar_img = cv2.resize(tar_img, (500,500))

    #ref_img = cv2.GaussianBlur(ref_img,(15,15),0)
    #tar_img = cv2.GaussianBlur(tar_img,(15,15),0)
    #tar_img = np.roll(tar_img,-20,axis=0)
    e = esm(ref_img, [300, 100, 160, 160])
    #e = esm(ref_img, [100, 100, 160, 160])
    e.track(tar_img, True)
    fig = plt.figure()
    e.ax1 = fig.add_subplot(211)
    e.ax2 = fig.add_subplot(212)
    e.show_process()
    #plt.savefig('./img_lookdown300/%d.png'%idx)
    plt.close("all")
    idx+=1



