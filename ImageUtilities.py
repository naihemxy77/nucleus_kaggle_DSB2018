
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.morphology import (label,binary_dilation,binary_closing,binary_erosion,binary_opening,
                                  disk,convex_hull_image,skeletonize,watershed)
from skimage.feature import peak_local_max


# In[2]:

from scipy import stats
from scipy import ndimage as ndi


# In[3]:

from skimage.color import rgb2hsv


# In[4]:

def imgToMask(img): ### make image binary
    res = np.zeros_like(img)
    res[np.where(img<0)]=0
    res[np.where(img>0)]=1
    return res
def valset(labelimg): ## label values used
    vst = set(labelimg.ravel())
    vst.remove(0)
    return vst
def getLabeled(lbimg,lbval): ### extract object with specified label value
    tmp = np.zeros_like(lbimg)
    tmp[np.where(lbimg!=lbval)]=0
    tmp[np.where(lbimg==lbval)]=1
    return tmp
def tinyDottsRemove(mask,minimum=10): ### remove the tiny dotts generated during processing
    tmp = label(mask)
    m = tmp.max()
    for i in range(1,m+1):
        n = np.sum(tmp==i)
        #print(n)
        if n<minimum:
            tmp[np.where(tmp==i)]=0
            pass
        pass
    return imgToMask(tmp)
def idToImg(idstr,pth):
    return plt.imread(pth+idstr+'/images/'+idstr+'.png')
def idToMask(imgID,maskID,pth):
    return plt.imread(pth+imgID+'/masks/'+maskID+'.png')
def idToMaskAll(idstr,pth):
    img0 = plt.imread(pth+idstr+'/images/'+idstr+'.png')
    maskall = np.zeros_like(img0[:,:,0])
    allMasks = os.listdir(pth+idstr+'/masks/')
    print(len(allMasks))
    for ii in allMasks:
        maskall+=idToMask(imgID=idstr,maskID=ii.split('.')[0],pth=pth)
        pass
    return maskall



