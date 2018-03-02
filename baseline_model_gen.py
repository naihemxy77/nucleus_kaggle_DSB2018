import numpy as np
import pandas as pd
import SplitNuclei
from sklearn.mixture import GaussianMixture
import submission_encoding
import matplotlib.pyplot as plt
import pickle

def baseline_model(df):
    SEED = 2948
    label_pred = []
    for i in range(df.shape[0]):
        print(str(i)+"th image is being processed...")
        if df.loc[i,'hsv_cluster'] == 0:
            img = df.loc[i,'Image']
            height,width,layer = img.shape
            img_v = np.reshape(img[:,:,2],(height*width,1))
            img_v = (img_v-img_v.min())/(img_v.max()-img_v.min()) 
            gm = GaussianMixture(n_components=2,random_state=SEED)
            gm.fit(img_v)
            label_v = img_v>gm.means_[1]+gm.covariances_[1]
            if np.sum(label_v==False)<np.sum(label_v==True):
                label_v = np.where(label_v,0,1)
            label = np.reshape(label_v,(height,width))
            label = SplitNuclei.reLabel(label)
            label_pred.append(label)
        elif df.loc[i,'hsv_cluster'] == 2:
            img = df.loc[i,'Image']
            height,width,layer = img.shape
            img_v = np.reshape(img[:,:,2],(height*width,1))
            img_v = (img_v-img_v.min())/(img_v.max()-img_v.min()) 
            label_v = img_v<0.4
            if np.sum(label_v==False)<np.sum(label_v==True):
                label_v = np.where(label_v,0,1)
            label = np.reshape(label_v,(height,width))
            label_pred.append(label)
        else:
            img = df.loc[i,'Image']
            height,width,layer = img.shape
            img_v = np.reshape(img,(height*width,layer))
            img_v = (img_v-img_v.min())/(img_v.max()-img_v.min()) 
            gm = GaussianMixture(n_components=2,random_state=SEED)
            gm.fit(img_v)
            label_v = gm.predict(img_v)
            if np.sum(label_v==False)<np.sum(label_v==True):
                label_v = np.where(label_v,0,1)
            label = np.reshape(label_v,(height,width))
            label = SplitNuclei.reLabel(label)
            label_pred.append(label)
    return label_pred

test_df = pickle.load(open("test_df.p","rb"))
label_pred = baseline_model(test_df)
test_df['ImageLabel'] = label_pred

r_n = 10
c_n = 7
fig, m_axs = plt.subplots(r_n, c_n,figsize=(10,15))
for i in range(r_n):
    for j in range(c_n):
        id_pointer = c_n*i+j
        if id_pointer<test_df.shape[0]:
            m_axs[i][j].imshow(test_df.loc[id_pointer,'ImageLabel'])
            m_axs[i][j].axis('off')
            m_axs[i][j].set_title(str(id_pointer),fontsize=7)

submission_encoding.submission_gen(test_df, 'baseline_submission')