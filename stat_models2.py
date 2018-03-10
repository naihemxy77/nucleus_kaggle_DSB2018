import numpy as np
import pandas as pd
import AggressiveSplit as sn
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_isodata
from skimage.filters import threshold_li
from skimage.filters import threshold_minimum
from skimage.filters import threshold_mean
from skimage.filters import threshold_triangle
from skimage.morphology import closing,square
from skimage.segmentation import clear_border

def GM_1dim_model(img):
    SEED = 29480
    height,width,_ = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    img_v = (img_v-img_v.min())/(img_v.max()-img_v.min())
    gm = GaussianMixture(n_components=2,random_state=SEED)
    gm.fit(img_v)
    label_v = img_v>gm.means_[1]+gm.covariances_[1]
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label_v = np.where(label_v,1,0)
    label = np.reshape(label_v,(height,width))
    label = closing(label,square(3))
    return sn.aggressiveLabel(label)
def Thr_otsu_model(img):
    threshold = threshold_otsu(img[:,:,2])
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    if label.max()==0:
        label[:5,:5]=1
    label = closing(label,square(3))
    return sn.aggressiveLabel(label)
def Thr_combined_model(img,coef):
    coef = coef/np.sum(coef)
    #otsu
    threshold1 = threshold_otsu(img[:,:,2])
    label1 = img[:,:,2]>threshold1
    if np.sum(label1==False)<np.sum(label1==True):
        label1 = np.where(label1,0,1)
    label1 = np.where(label1,1,0)
    #isodata
    threshold2 = threshold_isodata(img[:,:,2])
    label2 = img[:,:,2]>threshold2
    if np.sum(label2==False)<np.sum(label2==True):
        label2 = np.where(label2,0,1)
    label2 = np.where(label2,1,0)
    #li
    threshold3 = threshold_li(img[:,:,2])
    label3 = img[:,:,2]>threshold3
    if np.sum(label3==False)<np.sum(label3==True):
        label3 = np.where(label3,0,1)
    label3 = np.where(label3,1,0)
    label = coef[0]*label1+coef[1]*label2+coef[2]*label3
    label = np.where(label>0.5,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_combined_model2(img,coef):
    coef = coef/np.sum(coef)
    #otsu
    threshold1 = threshold_otsu(img[:,:,2])
    #isodata
    threshold2 = threshold_isodata(img[:,:,2])
    #li
    threshold3 = threshold_li(img[:,:,2])
    
    threshold = threshold1*coef[0]+threshold2*coef[1]+threshold3*coef[2]
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_combined_model3(img,coef):
    coef = coef/np.sum(coef)
    #otsu
    threshold1 = threshold_otsu(img[:,:,2])
    #isodata
    threshold2 = threshold_isodata(img[:,:,2])
    #li
    threshold3 = threshold_li(img[:,:,2])
    #yen
    threshold4 = threshold_yen(img[:,:,2])
    
    threshold = threshold1*coef[0]+threshold2*coef[1]+threshold3*coef[2]+threshold4*coef[3]
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_yen_model(img):
    threshold = threshold_yen(img[:,:,2])
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_isodata_model(img):
    threshold = threshold_isodata(img[:,:,2])
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_li_model(img):
    threshold = threshold_li(img[:,:,2])
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_mean_model(img):
    threshold = threshold_mean(img[:,:,2])
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_triangle_model(img):
    threshold = threshold_triangle(img[:,:,2])
    label = img[:,:,2]>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)
def Thr_hard_model(img,thr):
    img_t = img[:,:,2]
    img_t = (img_t-img_t.min())/(img_t.max()-img_t.min())
    label = img_t<thr
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    label = closing(label,square(3))
    if label.max()==0:
        label[:5,:5]=1
    return sn.aggressiveLabel(label)