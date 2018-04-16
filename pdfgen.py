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

Need_NPP = pd.read_pickle('./inputs/fuck.p')
predList = [0,0,0,0,0]

def partialReaderFull(start,end):
    test_df = pd.read_pickle('./inputs/test_df2.p')
    test_df_tmp = test_df.loc[start:end-1,:]
    test_df = test_df.drop(columns=test_df.columns)
    del test_df
    return test_df_tmp

def partialReader(pfile,start,end):
    tmp = pd.read_pickle(pfile)
    try:
        #print(tmp.columns)
        tmp.drop(columns=['index'],inplace=True)
        tmp.rename(columns={'ImageOutput':'ImageLabel'},inplace=True)
    except:
        tmp = pd.DataFrame(tmp,columns=['ImageId','ImageLabel'])
    tmp0 = tmp.loc[start:end,:]
    tmp = tmp.drop(columns=tmp.columns)
    del tmp
    return tmp0

def readAll(start,end):
    predList[0] = partialReader('./inputs/base_pred_stage2.p',start=start,end=end)#baseres
    predList[1] = partialReader('./inputs/UnetRes.p',start=start,end=end) #unetres
    predList[2] = partialReader('./inputs/ZoomNetAllRes.p',start=start,end=end) #zoomallres
    predList[3] = partialReader('./inputs/ZoomNetHollowRes.p',start=start,end=end)#hollowres
    predList[4] = partialReader('./inputs/NewUnetRes.p',start=start,end=end)#newunet

plots_name = ['baseline','Unet','ZoomNetAll','Hollow','NewUnet','getLocal','Image'] 
test_df2 = partialReaderFull(3010,3019)
tot_num = test_df2.shape[0]

r_n = 10
c_n = len(plots_name)
img_index = 3010

start = 3010
end = 3018

readAll(start,end)

baseres,unetres,zoomallres,hollowres,newunetres = predList

with PdfPages('stage2_pred_comparison_all'+str(start)+'.pdf') as pdf:
    for NN in range(tot_num//r_n+1):
        print('Page '+str(NN)+' is generating...')
        fig,ax = plt.subplots(r_n,c_n,figsize=(10,15))
        for r in range(r_n):
            if img_index < tot_num:
                #Image Information               
                img = test_df2.loc[img_index,'Image'][:,:,:3]
                img_id = test_df2.loc[img_index,'ImageId']
                npp = Need_NPP.loc[Need_NPP.ImageId==img_id,'NPP'].item()
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
                #Image iter
                img_index = img_index+1
                
                #Show Ma Ge's Pan Duan
                ax[r,0].set_ylabel(str(npp))
                #Show all modeling results
                ax[r,0].imshow(mask0)
                ax[r,1].imshow(mask1)
                ax[r,2].imshow(mask2)
                ax[r,3].imshow(mask3)
                ax[r,4].imshow(mask4)
                ax[r,5].imshow(mask5)
                #Show original images (in the last column)
                ax[r,6].imshow(img)
                ax[r,6].set_ylabel('Index: '+str(img_index-1))
                ax[r,6].yaxis.set_label_position("right")
                #Axis/title settings
                for c in range(c_n):
                    ax[r,c].get_xaxis().set_ticks([])
                    ax[r,c].get_yaxis().set_ticks([])
            if r == 0:
                for kk in range(len(plots_name)):
                    ax[r,kk].set_title(plots_name[kk])
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
