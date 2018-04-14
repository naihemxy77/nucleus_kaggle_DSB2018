import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import baseline_models_0320 as sm
import random

def baseline_model(df):
    label_pred = []
    for i,row in df.iterrows():
        img = row['Image']
        #print(str(i)+"th image is being processed...")
        if row['hsv_cluster'] == 0:
            #label = sm.Thr_combined_model2(img[:,:,2])
            label = sm.Thr_avg(img)
        else:
            label = sm.color_deconv_histo(img)
        label_pred.append(label)
        
    return label_pred

#test_df2 = pickle.load(open("./inputs/test_df2.p","rb"))
sample_num = 5
WholeIndex = np.arange(0,test_df2.shape[0])
Index = random.sample(set(WholeIndex), sample_num)
trial = test_df2.loc[Index,:]
label_pred = baseline_model(trial)
trial['ImageLabel'] = label_pred

r_n = sample_num
c_n = 2
fig, ax = plt.subplots(r_n,c_n,figsize=(10,10))
ax = ax.ravel()
i = 0
for ii,row in trial.iterrows():
    ax[i].imshow(row['Image'])
    ax[i].axis('off')
    i = i+1
    ax[i].imshow(row['ImageLabel'])
    ax[i].axis('off')
    i = i+1