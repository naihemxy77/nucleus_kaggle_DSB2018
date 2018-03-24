
# coding: utf-8

# In[1]:


##reference: https://www.kaggle.com/voglinio/separating-nuclei-masks-using-convexity-defects
get_ipython().magic('matplotlib inline')
import numpy as np 
import pandas as pd 
import os
import cv2
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage.morphology import label,binary_dilation,binary_closing,binary_erosion,binary_opening,disk,convex_hull_image,skeletonize,watershed,remove_small_objects
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


# In[2]:


from skimage.morphology import square


# In[3]:


from ImageUtilities import *


# In[4]:


def split_mask_v1(mask):
    thresh = mask.copy().astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    i = 0 
    for contour in contours:
        if  cv2.contourArea(contour) > 20:
            hull = cv2.convexHull(contour, returnPoints = False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue
            points = []
            dd = []

            #
            # In this loop we gather all defect points 
            # so that they can be filtered later on.
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                d = d / 256
                dd.append(d)

            for i in range(len(dd)):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                if dd[i] > 1.0 and dd[i]/np.max(dd) > 0.2:
                    points.append(f)

            i = i + 1
            mydict = {}
            if len(points) >= 2:
                for i in range(len(points)):
                    f1 = points[i]
                    p1 = tuple(contour[f1][0])
                    nearest = None
                    min_dist = np.inf
                    for j in range(len(points)):
                        if i != j:
                            f2 = points[j]                   
                            p2 = tuple(contour[f2][0])
                            dist = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) 
                            if dist < min_dist:
                                min_dist = dist
                                nearest = p2
                                mydict[tuple([p1,p2])]=min_dist
                                pass
                            pass
                        pass
                pointsTuples = list(mydict.keys())
                pointsTuples.sort(key = lambda x:mydict[x])
                p1,p2 = pointsTuples[0]
                cv2.line(thresh,p1,p2, [0, 0, 0], 2)
    return thresh 

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
    newLabeled = newNucleiBinarySplit(maskLabeled,thr=thr)
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


# In[5]:


def skeletonToConv(mask):
    tmpMask = np.zeros_like(mask)
    lb = label(mask)
    vlst = valset(lb)
    for ii in vlst:
        tmpMask += convex_hull_image(getLabeled(lb,ii))
        pass
    return tmpMask


# In[6]:


def newNucleiBinarySplit(mask,thr):
    props = regionprops(mask)
    prop = props[0]
    mask00 = ndi.binary_fill_holes(mask)
    maskConv = convex_hull_image(mask00)
    maskB = tinyDottsRemove(maskConv-mask)
    notConv = np.sum(maskB)/np.sum(mask00)
    if(notConv<thr):
        return mask00
    if prop.convex_area/prop.filled_area < 1.1:
        return mask00
    maskC = skeletonize(maskB)
    maskD = skeletonToConv(maskC)
    maskE = imgToMask(np.asarray(maskConv,dtype=np.int32) - maskD)
    maskF = split_mask_v1(maskE)
    maskF[np.where(mask00==0)]=0
    markers = ndi.label(maskF)[0]
    markers[np.where(mask00==0)]=0
    labels = watershed(-mask, markers, mask=mask)
    return labels




