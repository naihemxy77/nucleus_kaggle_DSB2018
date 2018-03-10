
# coding: utf-8

# In[24]:

#get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#from skimage.segmentation import morphological_chan_vese
from ShanZhai import morphological_chan_vese
from skimage.restoration import denoise_nl_means, estimate_sigma
from math import sqrt,pi


# In[5]:

from ImageUtilities import *


# In[6]:

def immaturalProcess(img):
    tmp = rgb2hsv(img[:,:,:3])
    ch0 = tmp[:,:,1]-tmp[:,:,2]
    lb = label(tinyDottsRemove(ChanVese(ch0,lambda1=1,lambda2=2),minimum=60))
    if lb.max()==1:
        ch0 = -ch0
    ch0 = (ch0-ch0.min())/(ch0.max()-ch0.min())
    sigma_est = np.mean(estimate_sigma(ch0, multichannel=False))
    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=False)
    denoise = denoise_nl_means(ch0, h=1.15 * sigma_est, fast_mode=False,**patch_kw)
    #m,s = stats.norm.fit(ch0.ravel())
    #ch0[np.where(ch0<m)]=m
    return denoise


# In[7]:

def ChanVese(img,lambda1,lambda2):
    mm,ss = stats.norm.fit(img.ravel())
    #thr_otsu = threshold_li(img.ravel())
    lb = np.zeros_like(img)
    lb[np.where(img<mm)]=1
    #lb[np.where(img<thr_otsu)]=1
    ls = morphological_chan_vese(-img, 50, init_level_set=lb, lambda1=lambda1,lambda2=lambda2,smoothing=0)
    mask = 1-ls
    return mask


# In[8]:

def RadiusFind(mask):
    area = np.sum(mask)
    rad = int(sqrt(area/2/pi))
    tmp = binary_opening(mask,selem=disk(rad))
    while np.sum(tmp)>0:
        tmp = binary_opening(mask,selem=disk(rad))
        rad+=5
        pass
    while np.sum(tmp)==0:
        tmp = binary_opening(mask,selem=disk(rad))
        rad-=1
        pass
    return rad+1


# In[146]:

def pickLargestLB(lb):
    tmpdict = {}
    vlst = list(valset(lb))
    for ii in vlst:
        tmpdict[ii]=np.sum(getLabeled(lb,ii))
        pass
    vlst.sort(key=lambda x: tmpdict[x],reverse=True)
    return vlst[0]


# In[245]:

def easyBinarySplit(img,mask):
    mask00 = ndi.binary_fill_holes(mask)
    maskConv = convex_hull_image(mask00)
    maskB = tinyDottsRemove(maskConv-mask,minimum=5)
    notConv = np.sum(maskB)/np.sum(mask00)
    if(notConv<0.05):
        return mask
    markers = np.zeros_like(mask)
    rad1 = RadiusFind(mask)
    tmp1 = binary_opening(mask,selem=disk(rad1))
    #tmp2 = mask - binary_erosion(tmp1,selem=disk(1))
    tmp2 = tinyDottsRemove(mask - tmp1,minimum=5)
    if np.sum(tmp2)==0:
        return tmp1
    rad2 = RadiusFind(tmp2)
    tmp3 = binary_opening(tmp2,selem=disk(rad2))
    if (np.max(label(tmp3)>1)or(np.sum(tmp3)<5)):
        markers = tmp3+tmp1
        pass
    else:
        markers = 2*tmp3+tmp1
    labels = watershed(-img, markers, mask=mask,connectivity=2)
    return labels


# In[22]:

def easyLabel(img,labelAll,minimum=5):
    tmpMask = np.zeros_like(labelAll)
    #lbs = label(maskAll)
    for ii in valset(labelAll):
        recursiveLabel(img=img,mask=getLabeled(labelAll,ii),tmpMask=tmpMask,minimum = minimum)
        pass
    return tmpMask


# In[14]:

def recursiveLabel(img,mask,tmpMask,minimum):
    k = tmpMask.max()
    lbs = easyBinarySplit(img=img,mask=mask)
    vlst = valset(lbs)
    vlst_del = []
    for ii in vlst:
        if np.sum(getLabeled(lbs,ii))<minimum:
            vlst_del.append(ii)
            pass
        pass
    for ii in vlst_del:
        vlst.remove(ii)
        pass
    if len(vlst)<2:
        tmpMask[np.where(mask>0)]=k+1
        return
    else:
        for ii in vlst:
            recursiveLabel(img,getLabeled(lbs,ii),tmpMask,minimum)
            pass
        return
    pass
