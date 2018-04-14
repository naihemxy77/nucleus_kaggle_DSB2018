import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import baseline_models_0320 as sm
import random

test_df2 = pickle.load(open("./inputs/test_df2.p","rb"))

label_pred = []
for i in np.arange(0,test_df2.shape[0]):
    img = test_df2.Image[i]
    hsv_dominants = test_df2.hsv_dominant[i]
    print(str(i)+"th image is being processed...")
    if test_df2.hsv_cluster[i] == 0:
        if hsv_dominants[2]<0.1:
            label = sm.getMinimum(img)
        else:
            label = sm.Thr_avg(img)
    elif test_df2.hsv_cluster[i] == 4 and hsv_dominants[0]==0 and hsv_dominants[1]==0:
        if hsv_dominants[2]<0.1:
            label = sm.getMinimum(img)
        else:
            label = sm.Thr_avg(img)
    else:
        label = sm.color_deconv_histo(img)
    label_pred.append((test_df2.ImageId[i],label))

pickle.dump(label_pred,open( "basemodel_pred_0414.p","wb" ))
#test_df2['ImageLabel'] = label_pred

#r_n = sample_num
#c_n = 2
#fig, ax = plt.subplots(r_n,c_n,figsize=(10,10))
#ax = ax.ravel()
#i = 0
#for ii,row in trial.iterrows():
#    ax[i].imshow(row['Image'])
#    ax[i].axis('off')
#    i = i+1
#    ax[i].imshow(row['ImageLabel'])
#    ax[i].axis('off')
#    i = i+1