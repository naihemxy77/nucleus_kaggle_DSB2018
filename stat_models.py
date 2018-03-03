import numpy as np
import pandas as pd
import AggressiveSplit as sn
from sklearn.mixture import GaussianMixture
import submission_encoding
import matplotlib.pyplot as plt
import pickle
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.filters import threshold_isodata
from skimage.filters import threshold_li
from skimage.filters import threshold_minimum
from skimage.filters import threshold_mean
from skimage.filters import threshold_triangle

def GM_1dim_model(img,k):
    SEED = 29480
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    img_v = (img_v-img_v.min())/(img_v.max()-img_v.min()) 
    gm = GaussianMixture(n_components=2,random_state=SEED)
    gm.fit(img_v)
    label_v = img_v>gm.means_[1]+k*gm.covariances_[1]
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    return sn.aggressiveLabel(label)
def Thr_1dim_model(img,k):
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    #threshold = threshold_otsu(img[:,:,2])
    label_v = img_v>k
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    return sn.aggressiveLabel(label)
def Thr_otsu_model(img):
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    threshold = threshold_otsu(img[:,:,2])
    label_v = img_v>threshold
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    return label
def Thr_yen_model(img): #bad
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    threshold = threshold_yen(img[:,:,2])
    label_v = img_v>threshold
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    label = np.where(label,1,0)
    return label
def Thr_isodata_model(img):
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    threshold = threshold_isodata(img[:,:,2])
    label_v = img_v>threshold
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    label = np.where(label,1,0)
    return label
def Thr_li_model(img):
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    threshold = threshold_li(img[:,:,2])
    label_v = img_v>threshold
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    label = np.where(label,1,0)
    return label
def Thr_minimum_model(img):
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    threshold = threshold_minimum(img[:,:,2])
    label_v = img_v>threshold
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    label = np.where(label,1,0)
    return label
def Thr_mean_model(img): #bad
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    threshold = threshold_mean(img[:,:,2])
    label_v = img_v>threshold
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    label = np.where(label,1,0)
    return label
def Thr_triangle_model(img): #bad
    height,width,layer = img.shape
    img_v = np.reshape(img[:,:,2],(height*width,1))
    threshold = threshold_triangle(img[:,:,2])
    label_v = img_v>threshold
    if np.sum(label_v==False)<np.sum(label_v==True):
        label_v = np.where(label_v,0,1)
    label = np.reshape(label_v,(height,width))
    label = np.where(label,1,0)
    return label