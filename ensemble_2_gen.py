import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages

from skimage.measure import label
from ImageUtilities import *
from skimage.exposure import equalize_adapthist,rescale_intensity
from skimage.transform import pyramid_reduce,pyramid_expand
from skimage.morphology import binary_closing,binary_dilation,binary_erosion,binary_opening,square,disk,diamond
from skimage.color import rgb2gray
from skimage.filters import threshold_local,threshold_minimum,threshold_otsu
from scipy.ndimage import binary_fill_holes
from data_norm import minmax_norm,invert_norm

from skimage.filters import gaussian

def getLocal(img):
    ch = rgb2gray(img[:,:,:3])
    s0 = np.asarray(ch>ch.mean(),dtype=np.int32)
    s1 = np.asarray(ch>threshold_local(ch,block_size=23),dtype=np.int32)
    s2 = np.asarray(ch>threshold_local(ch,block_size=51),dtype=np.int32)
    s3 = np.asarray(ch>threshold_local(ch,block_size=75),dtype=np.int32)
    s4 = np.asarray(ch>threshold_local(ch,block_size=101),dtype=np.int32)
    s5 = np.asarray(ch>threshold_local(ch,block_size=169),dtype=np.int32)
    tmp = gaussian(s1*s2*s3*s4*s5*s0,sigma=1.5)
#    tmp = tmp>threshold_otsu(tmp)
    return tmp

print('load test files...')
test_df2 = pickle.load(open('./inputs/test_df2.p','rb'))
test_stage2_ensemble2 = pickle.load(open('./inputs/test_stage2_ensemble2.p','rb'))

print('load base...')
baseres = pd.read_pickle('./inputs/base_pred_stage2.p')#baseres
baseres = pd.DataFrame(baseres, columns=['ImageId','ImageLabel'])
print('load unet...')
unetres = pd.read_pickle('./inputs/UnetRes.p') #unetres
unetres = pd.DataFrame(unetres, columns=['ImageId','ImageLabel'])
print('load zoom...')
zoomallres = pd.read_pickle('./inputs/ZoomNetAllRes.p') #zoomallres
zoomallres = pd.DataFrame(zoomallres, columns=['ImageId','ImageLabel'])
print('load hollow...')
hollowres = pd.read_pickle('./inputs/ZoomNetHollowRes.p')#hollowres
hollowres = pd.DataFrame(hollowres, columns=['ImageId','ImageLabel'])
print('load new...')
newunetres = pd.read_pickle('./inputs/NewUnetRes.p')#newunet
newunetres = pd.DataFrame(newunetres, columns=['ImageId','ImageLabel'])


final_label = []
for img_index in range(test_df2.shape[0]):          
    img = test_df2.loc[img_index,'Image'][:,:,:3]
    img_id = test_df2.loc[img_index,'ImageId']

    if test_df2.loc[test_df2.ImageId==img_id,'hsv_cluster'].item()==0:
        img = minmax_norm(img)
    else:
        img = invert_norm(img)
    
    #Predictions from different models
    mask0 = baseres.loc[baseres.ImageId==img_id,'ImageLabel'].item().squeeze()
    mask1 = unetres.loc[unetres.ImageId==img_id,'ImageLabel'].item().squeeze()
    mask2 = zoomallres.loc[zoomallres.ImageId==img_id,'ImageLabel'].item().squeeze()
    mask3 = hollowres.loc[hollowres.ImageId==img_id,'ImageLabel'].item().squeeze()
    mask4 = newunetres.loc[newunetres.ImageId==img_id,'ImageLabel'].item().squeeze()
    mask5 = getLocal(img)
    #Weights for different models
    w0 = int(test_stage2_ensemble2.loc[test_stage2_ensemble2.ImageId==img_id,'baseline'].item())
    w1 = int(test_stage2_ensemble2.loc[test_stage2_ensemble2.ImageId==img_id,'Unet'].item())
    w2 = int(test_stage2_ensemble2.loc[test_stage2_ensemble2.ImageId==img_id,'ZoomNetAll'].item())
    w3 = int(test_stage2_ensemble2.loc[test_stage2_ensemble2.ImageId==img_id,'Hollow'].item())
    w4 = int(test_stage2_ensemble2.loc[test_stage2_ensemble2.ImageId==img_id,'NewUnet'].item())
    w5 = int(test_stage2_ensemble2.loc[test_stage2_ensemble2.ImageId==img_id,'getLocal'].item())
    denominator = w0+w1+w2+w3+w4+w5
    
    new_mask = ( mask0*w0 + mask1*w1 + mask2*w2 +mask3*w3 + mask4*w4 +mask5*w5 ) / denominator
    
    if np.max(new_mask) > 1:
        print(str(img_index)+' has pixels more than 1 intensity....')
        break
    if np.min(new_mask) < 0:
        print(str(img_index)+' has pixels less than 0 intensity....')
        break
    
    new_mask = np.where(new_mask>0.5,1,0)
    final_label.append((img_id,new_mask))

test_pred_stage2_ensemble2 = pd.DataFrame(final_label, columns=['ImageId','ImageLabel'])
pickle.dump( test_pred_stage2_ensemble2, open( "test_pred_stage2_ensemble2.p", "wb" ) )