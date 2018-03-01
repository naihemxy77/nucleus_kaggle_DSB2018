### %matplotlib inline ### add this line if you are using a notebook
import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
from skimage.morphology import label,binary_dilation,binary_closing,binary_erosion,binary_opening,\
                                disk,convex_hull_image,skeletonize,watershed,remove_small_objects
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


### Here are essential functions

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
#def splitDotts(mask): ### split the dotts clinging together
#    if np.sum(mask)<40:
#        return mask
#    maskConv = convex_hull_image(mask)
#    maskA = maskConv-mask
#    maskA1 = tinyDottsRemove(maskA)
#    maskA2 = binary_closing(binary_dilation(skeletonize(maskA1),selem=disk(4)),selem=disk(3))
#    maskA3 = binary_dilation(imgToMask(skeletonize(maskA2)+maskA1),selem=disk(1))
#    maskA4 = imgToMask(mask-maskA3)
#    maskA5 = tinyDottsRemove(maskA4)
#    #local_maxi = peak_local_max(maskA5, indices=False, footprint=np.ones((10, 10)),labels=mask)
#    markers = ndi.label(maskA5)[0]
#    labels = watershed(-mask, markers, mask=mask)
#    return labels

def nucleiSplit(mask): ## a much better version, and it's able to split nuclei that have long attach edge
    mask00 = ndi.binary_fill_holes(mask)
    maskConv = convex_hull_image(mask00)
    maskB = tinyDottsRemove(maskConv-mask)
    while(np.max(label(maskB))>1):
        maskB = binary_closing(binary_dilation(maskB,selem=disk(1)),selem=disk(2))
        pass
    maskC = binary_dilation(skeletonize(maskB),selem=disk(1))
    maskD = tinyDottsRemove(binary_erosion(imgToMask(binary_dilation(maskConv,selem=disk(1))+(-1)*maskC),selem=disk(1)))
    maskE = binary_opening(maskD,selem=disk(1))
    markers = ndi.label(maskE)[0]
    markers[np.where(mask00==0)]=0
    labels = watershed(-mask, markers, mask=mask)
    return labels  

def valset(labelimg): ## label values used
    vst = set(labelimg.ravel())
    vst.remove(0)
    return vst
def reLabel(maskAll,minimum=5): ### the main dish
    tmp = np.zeros_like(maskAll)
    mask0 = label(maskAll)
    vst = valset(mask0)
    vlstToDel = []
    for val in vst:
        if np.sum(mask0==val)<minimum: # The mask loaded may contain multiple dotts. Seems like label error. 
            vlstToDel.append(val)
            mask0[np.where(mask0==val)]=0
            pass
        pass
    for val in vlstToDel:
        vst.remove(val)
        pass
    #maskAll = tinyDottsRemove(maskAll,minimum=5) # the area is at least 5
    #mask0 = label(maskAll)
    k = 1
    for i in vst:
        lb = getLabeled(lbimg=mask0,lbval=i)
        ##tmpMask = splitDotts(lb)
        tmpMask = nucleiSplit(lb)
        for j in valset(tmpMask):
            tmp[np.where(tmpMask==j)]=k
            k+=1
            pass
        pass
    return tmp

### Here is one example

# all masks loaded into one image
smpID = '00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552'
allMask = os.listdir("../input/stage1_train/"+smpID+'/masks')
mmk = np.zeros((256,256)) 
for ii in allMask:
    mmk+=np.asarray(plt.imread("../input/stage1_train/"+smpID+"/masks/"+ii),dtype=np.int32)
    pass
rMsk = reLabel(mmk) # split nuclei
plt.imshow(rMsk, cmap = "Vega20c")
