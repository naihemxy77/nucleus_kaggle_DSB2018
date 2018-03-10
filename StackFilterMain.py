
# coding: utf-8

# In[1]:

#get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import stat_models2 as sm
from skimage import img_as_float
#from skimage.segmentation import morphological_chan_vese
from ShanZhai import morphological_chan_vese
from scipy.ndimage.measurements import center_of_mass
from skimage.filters import threshold_otsu


# In[2]:

from ImageUtilities import *


# In[3]:

from EasySplit import immaturalProcess,easyLabel,ChanVese


# In[4]:

def labelToDict(lb):
    #lb = label(mask)
    vlst = valset(lb)
    tmp = {}
    for ii in vlst:
        tmpMask = getLabeled(lb,ii)
        x,y = center_of_mass(tmpMask)
        x = round(x)
        y = round(y)
        tmp[round(x),round(y)]=ii
        pass
    return tmp


# In[5]:

def mergeDetection(dict1):
    Points = list(dict1.keys())
    Points.sort(key=lambda x:dict1[x])
    #PrevPointLb = 1
    PointsStack = []#[Points[0]]
    #lbStack = [dict1[Points[0]]]
    n = len(Points)
    i=0
    j=1
    while((i<n)and(j<n)):
        if((dict1[Points[i]]==dict1[Points[j]])and(dict1[Points[i]]!=0)):
            PointsStack.append(Points[i])
            PointsStack.append(Points[j])
            j+=1
        else:
            i=j
            j=i+1
            pass
        pass
    return set(PointsStack)


# In[6]:

def addToFinal(tmplb,res):
    k = np.max(res)
    if np.max(imgToMask(tmplb)+imgToMask(res))==1:
        res[np.where(tmplb>0)]=k+1
        pass
    else:
        #print(center_of_mass(tmplb)," rejected.")
        pass


# In[7]:

def trial(lastLB,currLB,tmpMask):
    lastDict = labelToDict(lastLB)
    lastPoints = list(lastDict.keys())
    currDict = {}
    for p in lastPoints:
        p_new = (int(p[0]),int(p[1]))
        #print(p_new)
        currDict[p_new]=currLB[p_new]
        pass
    readyPoints = mergeDetection(dict1=currDict)
    for pp in readyPoints:
        if((lastLB[pp]!=0)and(currLB[pp]!=0)and(currDict[pp]!=0)):
            addToFinal(binary_closing(getLabeled(lastLB,lastLB[pp]),selem=disk(1)),res=tmpMask)
            pass
        pass


# In[8]:

def FilterStack(im,tmpMask,nStep = 20):
    img = np.copy(im)
    sk = stats.skew(img.ravel())
    #mask0 = imgToMask(ChanVese(img=im,lambda1=1,lambda2=100)+ChanVese(img=im,lambda1=1,lambda2=200)+ChanVese(img=im,lambda1=1,lambda2=300)-2)
    #img = img - img*mask0
    lambda2start = max(1,round(0.9*sk-2.6))
    #i=1
    stride = lambda2start/nStep
    for i in range(0,nStep):
    #while(i>0.05):
        #print(i)
        mask1 = tinyDottsRemove(ChanVese(img=img,lambda1=1,lambda2=lambda2start-stride*i),minimum=3)
        mask2 = tinyDottsRemove(ChanVese(img=img,lambda1=1,lambda2=lambda2start-stride*(i+1)),minimum=3)
        lb1 = label(mask1)
        lb2 = label(mask2)
        trial(currLB=lb2,lastLB=lb1,tmpMask=tmpMask)
        i=i+1
        pass
    #print(lambda2start-stride*(i+1))
    pass


# In[9]:

def modifyLabel(lb):
    vlst = valset(lb)
    for ii in vlst:
        tmp = getLabeled(lb,ii)
        if np.sum(tmp)<10:
            tmp = binary_dilation(tmp,selem=disk(1))
            lb[np.where(tmp>0)]=ii
            pass
        else:
            tmp = binary_erosion(tmp,selem=disk(1))
            lb[np.where(tmp>0)]=ii
        pass
    pass


# In[10]:

def StackFilter(imgID,pth):
    tmpIm = idToImg(idstr=imgID,pth=pth)
    chs = rgb2hsv(tmpIm[:,:,:3])
    ch0 = immaturalProcess(tmpIm)
    tmpMask = np.zeros_like(ch0)
    if np.sum(chs[:,:,0])==0:
        FilterStack(im=ch0,tmpMask=tmpMask)
        modifyLabel(tmpMask)
        pass
    else:
        #thr = threshold_otsu(ch0.ravel())
        #tmpMask[np.where(ch0>=thr)]=1
        tmpMask = sm.Thr_combined_model2(img,[0.504,0.496,0.454])
        #tmpMask = ndi.binary_fill_holes(tmpMask)
        #tmpMask = label(tmpMask)
    #alb = aggressiveLabel(tmpMask)
    elb = easyLabel(ch0,tmpMask)
    return elb


# In[11]:

def blobCoef(mask):
    tmp1 = convex_hull_image(mask)
    tmp2 = binary_closing(mask,selem=disk(1))
    return np.sum(tmp2)/np.sum(tmp1)

def tinyLabelRemove(lb,minimum=5):
    nlb = np.copy(lb)
    vlst = valset(nlb)
    for ii in vlst:
        tmp = getLabeled(nlb,ii)
        if ((np.sum(tmp)<minimum)):#or(blobCoef(tmp)<0.8)):
            nlb[np.where(nlb==ii)]=0
            pass
        pass
    return nlb
    


# In[12]:

def idToResult(idstr,pth,remove_size=12): # main function. The best remove_size is somewhere between 10 and 35, I'm not sure.
    tmpIm = idToImg(idstr=smpID,pth=trainPth)
    tmp = StackFilter(smpID,trainPth)
    tmpLB = tinyLabelRemove(tmp,remove_size)
    return tmpLB


# In[13]:

trainPth = "../input/stage1_train/"
testPth = "../input/stage1_test/"
#smpID = '00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552'
smpID = 'c96109cbebcf206f20035cbde414e43872074eee8d839ba214feed9cd36277a1' #535
#smpID = 'e7a3a7c99483c243742b6cfa74e81cd48f126dcef004016ad0151df6c16a6243' #606
#smpID = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e' #hist
#smpID = 'c322c72b9d411e631580fee9312885088b4bb14ed297aa4b246ec943533b3ffb' #bright


# In[14]:

mmk = idToMaskAll(smpID,pth=trainPth)
plt.imshow(mmk)


# In[16]:

res = idToResult(idstr=smpID,pth=trainPth)


# In[17]:

plt.imshow(res,cmap="Vega20c")


# In[ ]:



