#import pandas as pd
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel

def minmax_norm(img):
    img=img[:,:,0:6]
    img_bound= sobel(rgb2gray(img[:,:,0:3]))
    img_bound = np.reshape(img_bound,(img_bound.shape[0],img_bound.shape[1],1))
    img_norm= (img-img.min())/(img.max()-img.min())
    return np.concatenate((img_norm,img_bound),axis=2)

def rgb_norm(img):
    img=img[:,:,0:6]
    img_bound= sobel(rgb2gray(img[:,:,0:3]))
    img_bound = np.reshape(img_bound,(img_bound.shape[0],img_bound.shape[1],1))
    img_norm= img/255
    return np.concatenate((img_norm,img_bound),axis=2)
    
    
def invert_norm(img):
    img=img[:,:,0:6]
    img_norm = (img-img.min())/(img.max()-img.min())
    img_norm = 1-img_norm
    img_bound= sobel(rgb2gray(img[:,:,0:3]))
    img_bound = np.reshape(img_bound,(img_bound.shape[0],img_bound.shape[1],1))
    return np.concatenate((img_norm,img_bound),axis=2)
