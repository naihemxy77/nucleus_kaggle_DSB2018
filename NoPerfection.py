
# coding: utf-8

# In[1]:

#get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import img_as_float,restoration
from skimage.segmentation import morphological_chan_vese
from skimage.restoration import denoise_nl_means, estimate_sigma
from ShanZhai import morphological_chan_vese
from scipy.ndimage.measurements import center_of_mass
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_li
from skimage.filters import threshold_isodata
from skimage.filters import threshold_adaptive
from skimage.filters import threshold_local
from AggressiveSplit import nucleiBinarySplit


# In[2]:

from skimage import exposure
from ImageUtilities import *
from EasySplit import ChanVese


# In[3]:

def Preparation(img):
    wtf = img.shape
    if(len(wtf)>2)and(wtf[2]>=3):
        chs = rgb2hsv(img[:,:,:3])
        ch0 = chs[:,:,2]+chs[:,:,1]
    else:
        ch0 = img
    vmin, vmax = stats.scoreatpercentile(ch0, (0.0, 100))
    dat = np.clip(ch0, vmin, vmax)
    #dat = ch0
    dat = (dat - vmin) / (vmax - vmin)
    bilateral = restoration.denoise_bilateral(dat,multichannel=False)
    m,s = stats.norm.fit(dat.ravel())
    dat[np.where(dat<m)]=m
    #img_eq = exposure.equalize_hist(bilateral)
    return bilateral


# In[4]:

def thresholdMask(ch,thresholdFunc):
    tmp = np.zeros_like(ch)
    thr = thresholdFunc(ch)
    tmp[np.where(ch>=thr)]=1
    return tmp
def flipMask(img,mask):
    res = np.copy(img)
    #valsum = np.sum(res[np.where(mask==1)])
    #cnt = np.sum(mask)
    res[np.where(mask == 0)]=-1
    res[np.where(mask == 1)]*=-1
    res +=1
    #res[np.where(mask == 0)]=valsum/cnt
    return res
def subtractMask(img,thrFunc1,thrFunc2):
    mask1 = thresholdMask(ch=img,thresholdFunc=thrFunc1)
    flipped = flipMask(img=img,mask=mask1)
    mask2 = thresholdMask(ch=flipped,thresholdFunc=thrFunc2)
    return binary_opening(ndi.binary_fill_holes(mask1 - mask2),selem=disk(2))


# In[5]:

def genChanVese(img,target):
    sk = stats.skew(img.ravel())
    i = max(0.9*sk-2.6,1)
    new = ChanVese(img=img,lambda1=1,lambda2=i)
    while(new[np.where(target>0)].min()==0)and(i>=0.01):
        new = ChanVese(img=img,lambda1=1,lambda2=i)
        new = tinyDottsRemove(new,minimum=5)
        i-=0.05
        pass
    print(i+0.05)
    new = ChanVese(img=img,lambda1=1,lambda2=i+0.05)
    new = tinyDottsRemove(new,minimum=5)
    return new


# In[6]:

def local1_0(img):
    return threshold_local(image=img,block_size=11)


# In[68]:

def markersGen(ch,thrFunc=local1_0):
    firstMask = thresholdMask(ch,threshold_li)
    secondMask = genChanVese(img=ch,target=firstMask)
    step0 = secondMask*thresholdMask(ch,thrFunc)
    step0 = tinyDottsRemove(step0,minimum=5)
    tmpMask = np.zeros_like(ch)
    step1 = label(step0)
    for ii in valset(step1):
        tmpMask += binary_erosion(ndi.binary_fill_holes(getLabeled(step1,ii)),selem=disk(1))
        pass
    tmpMask = binary_opening(tinyDottsRemove(tmpMask,minimum=5),selem=disk(1))
    tmpMask = tinyDottsRemove(tmpMask,minimum=12)
    return tmpMask


# In[25]:

def mainProcess(idstr,pth):
    tmpIm = idToImg(idstr,pth)
    ch0 = Preparation(img=tmpIm)
    mask0 = thresholdMask(ch=ch0,thresholdFunc=threshold_isodata)
    tmpLB = label(markersGen(ch0))
    labels = watershed(-ch0,markers=tmpLB,mask=mask0,connectivity=2)
    #vlst = valset(labels)
    #tmp = np.zeros_like(ch0)
    #for ii in vlst:
    #    tmpMask = getLabeled(labels,ii)
    #    wtf = nucleiBinarySplit(mask=tmpMask,thr=0.05)
    #    tmpvlst = valset(wtf)
    #    for jj in tmpvlst:
    #        tmpwtf = getLabeled(wtf,jj)
    #        tmp[np.where(tmpwtf>0)]=tmp.max()+1
    #        pass
    #    pass
    return labels#tmp


# In[22]:

#trainPth = "../input/stage1_train/"
#testPth = "../input/stage1_test/"


# In[27]:

#smpID = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e'
#tmpIm = idToImg(idstr=smpID,pth=trainPth)
#chs = rgb2hsv(tmpIm[:,:,:3])
#ch0 = Preparation(tmpIm)
#plt.imshow(ch0)


#plt.imshow(mainProcess(idstr=smpID,pth=trainPth),cmap="Vega20c")
