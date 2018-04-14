# Actually I believe it works for all fluo image with hsv_dominant[2]<0.1
import numpy as np
import pandas as pd
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_opening,square,disk,diamond
from skimage.color import rgb2gray
from skimage.filters import threshold_minimum,threshold_otsu
from scipy.ndimage import binary_fill_holes

def getMinimum(im):
    img = rgb2gray(im[:,:,:3])
    p1, p99 = np.percentile(img, (1, 99.5))
    if p1==p99:
        return np.asarray(img>0)
    img_rescale = rescale_intensity(img, in_range=(p1, p99))
    try:
        mask0 = binary_fill_holes(img_rescale>threshold_minimum(img_rescale))
    except:
        print("Threshold_minimum failed! Use threshold_otsu instead.")
        mask0 = binary_fill_holes(img_rescale>threshold_otsu(img_rescale))
    #mask1 = binary_opening(mask0,selem=diamond(3))
    return mask0
