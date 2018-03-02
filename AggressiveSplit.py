
# coding: utf-8

# In[1]:

######## It separate everything well, but it runs slower, especially when the number of nuclei is large. 
# One more parameter is used, thr. It seems the threshold should be between 0.036 and 0.088
# larger thr means more nuclei merged, namely less split

get_ipython().magic('matplotlib inline')
import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt

from skimage.morphology import label,binary_dilation,binary_closing,binary_erosion,binary_opening,disk,convex_hull_image,skeletonize,watershed,remove_small_objects
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def imgToMask(img): ### make image binary
    res = np.zeros_like(img)
    res[np.where(img<0)]=0
    res[np.where(img>0)]=1
    return res
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

def getLabeled(lbimg,lbval): ### extract object with specified label value
    tmp = np.zeros_like(lbimg)
    tmp[np.where(lbimg!=lbval)]=0
    tmp[np.where(lbimg==lbval)]=1
    return tmp

def valset(labelimg): ## label values used
    vst = set(labelimg.ravel())
    vst.remove(0)
    return vst


# In[2]:

def aggressiveLabel(mask,minimum=5,thr = 0.036):
    mask0 = label(mask)
    vst = valset(mask0)
    vlstToDel = []
    for val in vst:
        if np.sum(mask0==val)<minimum:
            vlstToDel.append(val)
            mask0[np.where(mask0==val)]=0
            pass
        pass
    for val in vlstToDel:
        vst.remove(val)
        pass

    tmpMask = np.zeros_like(mask)
    k = [max(vst)+1]
    for i in vst:
        lb = getLabeled(lbimg=mask0,lbval=i)
        reLabel(tmp=tmpMask,maskLabeled=lb,kk=k,thr = thr,minimum=5)
        pass
    return tmpMask

def reLabel(tmp,maskLabeled,kk,thr,minimum=5):
    newLabeled = nucleiBinarySplit(maskLabeled,thr=thr)
    vstTmp = valset(newLabeled)
    if len(vstTmp)==1:
        tmp[np.where(newLabeled!=0)] = kk[0]
        kk[0]+=1
        return
    for ii in vstTmp:
        mk = getLabeled(newLabeled,ii)
        #wtf.append(deepcopy(mk))
        reLabel(tmp = tmp,maskLabeled = mk,kk=kk,thr = thr,minimum=minimum)
        kk[0]+=1
        pass
    return


# In[3]:

def nucleiBinarySplit(mask,thr):
    mask00 = ndi.binary_fill_holes(mask)
    maskConv = convex_hull_image(mask00)
    maskB = tinyDottsRemove(maskConv-mask)
    notConv = np.sum(maskB)/np.sum(mask00)
    if(notConv<thr):
        return mask00
    nn = np.max(label(maskB))
    if nn>1:
        while(np.max(label(maskB))==nn):
            maskB = binary_closing(binary_dilation(maskB,selem=disk(1)),selem=disk(2))
            pass
    maskC = binary_dilation(skeletonize(maskB),selem=disk(1))
    maskD = tinyDottsRemove(binary_erosion(imgToMask(binary_dilation(maskConv,selem=disk(1))+(-1)*maskC),selem=disk(1)),minimum=10)
    maskE = binary_opening(maskD,selem=disk(1))
    #maskE = imgToMask(maskD - maskConv)
    #maskF = ndi.binary_fill_holes(maskE)
    markers = ndi.label(maskE)[0]
    markers[np.where(mask00==0)]=0
    labels = watershed(-mask, markers, mask=mask)
    return labels


# In[4]:

#smpID = '00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552' #original
#smpID = 'c96109cbebcf206f20035cbde414e43872074eee8d839ba214feed9cd36277a1' #535
smpID = 'e7a3a7c99483c243742b6cfa74e81cd48f126dcef004016ad0151df6c16a6243' #606
allMask = os.listdir("../input/stage1_train/"+smpID+'/masks')
#mmk = np.zeros((256,256)) 
#mmk = np.zeros((360,360)) 
mmk = np.zeros((520,696))
for ii in allMask:
    mmk+=np.asarray(plt.imread("../input/stage1_train/"+smpID+"/masks/"+ii),dtype=np.int32)
    pass
rMsk = aggressiveLabel(mmk,minimum=5,thr=0.05) # split nuclei. It seems the threshold should be between 0.036 and 0.088
# larger thr means more nuclei merged, namely less split
plt.imshow(rMsk, cmap = "Vega20c")
#vst = valset(rMsk)
#print(len(vst))


# In[ ]:



