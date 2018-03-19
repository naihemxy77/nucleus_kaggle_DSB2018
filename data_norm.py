#import pandas as pd
#import numpy as np

def minmax_norm(img):
    return (img-img.min())/(img.max()-img.min())

def rgb_norm(img):
    return img/255