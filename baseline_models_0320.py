import numpy as np
import AggressiveSplit as sn
#import PerfectSplit as sn
from skimage.filters import threshold_otsu,threshold_minimum
from skimage.filters import threshold_yen
from skimage.filters import threshold_isodata
from skimage.filters import threshold_li
from skimage.morphology import closing,square
from skimage.color import rgb2hed,rgb2gray
from scipy.ndimage import binary_fill_holes
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_opening,square,disk,diamond

def Thr_combined_model2(img,coef=[0.504,0.496,0.454]):
    coef = coef/np.sum(coef)
    #otsu
    threshold1 = threshold_otsu(img)#[:,:,2])
    #isodata
    threshold2 = threshold_isodata(img)#[:,:,2])
    #li
    threshold3 = threshold_li(img)#[:,:,2])
    
    threshold = threshold1*coef[0]+threshold2*coef[1]+threshold3*coef[2]
    label = img>threshold#[:,:,2]>threshold
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_combined_model3(img,coef=[0.25,0.25,0.25,0.25]):
    coef = coef/np.sum(coef)
    #otsu
    threshold1 = threshold_otsu(img)
    #isodata
    threshold2 = threshold_isodata(img)
    #li
    threshold3 = threshold_li(img)
    #yen
    threshold4 = threshold_yen(img)
    
    threshold = threshold1*coef[0]+threshold2*coef[1]+threshold3*coef[2]+threshold4*coef[3]
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def color_deconv_histo(img):
    hed = rgb2hed(img)
    markers = np.where(binary_fill_holes(hed[:,:,0]>threshold_otsu(hed[:,:,0])),1,0)
#    hed_res = sn.aggressiveLabel(markers)
#    max_label = 0
#    for a in set(hed_res.ravel()):
#        if np.sum(hed_res==a)>max_label and a != 0:
#            max_label = np.sum(hed_res==a)
    return markers#,minimum=0.05*max_label)

def getMinimum(im):
    img = rgb2gray(im[:,:,:3])
    p1, p99 = np.percentile(img, (1, 99.5))
    if p1==p99:
        return np.asarray(img>0)
    img_rescale = rescale_intensity(img, in_range=(p1, p99))
    try:
        mask0 = np.where(binary_fill_holes(img_rescale>threshold_minimum(img_rescale)),1,0)
    except:
        print("Threshold_minimum failed! Use threshold_otsu instead.")
        mask0 = np.where(binary_fill_holes(img_rescale>threshold_otsu(img_rescale)),1,0)
    #mask1 = binary_opening(mask0,selem=diamond(3))
    return mask0

def color_deconv_fluo(img):
    hed = rgb2hed(img)
    markers = np.where(hed[:,:,1]>threshold_otsu(hed[:,:,1]),1,0)
    #return sn.aggressiveLabel(markers) ##2
    hed_res=sn.aggressiveLabel(markers)
    max_label = 0
    for a in set(hed_res.ravel()):
        if np.sum(hed_res==a)>max_label and a != 0:
            max_label = np.sum(hed_res==a)
    return sn.aggressiveLabel(markers,minimum=0.05*max_label)

def Thr_min(img):
    tmp = rgb2gray(img)
    mask = np.where(tmp>threshold_minimum(tmp),1,0)
    return sn.aggressiveLabel(mask)

def Thr_avg(img):
    img = rgb2gray(img)
    p1, p99 = np.percentile(img, (1, 99.5))
    if p1==p99:
        return np.asarray(img>0)
    tmp = rescale_intensity(img, in_range=(p1, p99))
    try:
        minimum = np.where(binary_fill_holes(tmp>threshold_minimum(tmp)),1,0)
        otsu = np.where(binary_fill_holes(tmp>threshold_otsu(tmp)),1,0)
        yen = np.where(binary_fill_holes(tmp>threshold_yen(tmp)),1,0)
        mask = (minimum+otsu+yen)/3
    except:
        print("Threshold_minimum failed! Use threshold_yen instead.")
        mask = np.where(binary_fill_holes(tmp>threshold_yen(tmp)),1,0)
    return mask

def Thr_yen(img):
    tmp = rgb2gray(img)
    yen = np.where(tmp>threshold_yen(tmp),1,0)
    return yen