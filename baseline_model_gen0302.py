import numpy as np
import pandas as pd
import AggressiveSplit as sn
from sklearn.mixture import GaussianMixture
import submission_encoding
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import stat_models as sm

def baseline_model(df):
    label_pred = []
    for i in range(df.shape[0]):
        img = df.loc[i,'Image']
        print(str(i)+"th image is being processed...")
        label_otsu = sm.Thr_otsu_model(img)
        label_isodata = sm.Thr_isodata_model(img)
        label_li = sm.Thr_li_model(img)
        #label_min = sm.Thr_minimum_model(img)
        label = (label_otsu+label_isodata+label_li)/3
        label = np.where(label>0.5,1,0)
        label_pred.append(sn.aggressiveLabel(label))
        
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
pp = PdfPages('baseline_pred0302.pdf')
pp.savefig(fig)
pp.close()

submission_encoding.submission_gen(test_df, 'baseline_submission0302')