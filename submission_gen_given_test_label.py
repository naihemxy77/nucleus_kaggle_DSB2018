import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import submission_encoding
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import AggressiveSplit as sn
from skimage.morphology import square,binary_closing

cutoff = 0.5
##Generate Test Masks and Submission files given pickle outputs from models
##In current file, if image type is not histological, then test masks will be
##predicted by the linear combination of otsu,iso and li.
submissionfilename = 'Zoom_invert_jac'
test_label = pickle.load(open( "Test_Label.p","rb" ))
test_label = pd.DataFrame(test_label, columns=['ImageId','ImageOutput'])

test_label_rot = pickle.load(open( "Test_Label_rot.p","rb" ))
test_label_rot = pd.DataFrame(test_label_rot, columns=['ImageId','ImageOutput'])

selected_id = [12,19,22,44,49,51,56,62]#[5,7,12,14,19,22,24,44,49,51,56,62]
final_label = []
for i in range(test_label.shape[0]):
    #if i in selected_id:
    print(str(i)+"th image is being processed...")
    img = test_label.loc[i,'ImageOutput']
    img_rot = test_label_rot.loc[i,'ImageOutput']
    if test_label.loc[i,'ImageId'] == test_label_rot.loc[i,'ImageId']:
        tmp = np.rot90(img_rot,k=1,axes=(1,0))
        img_combined = (img+tmp)/2
    else:
        img_combined = img
        print(str(i)+'th labels do not match with each other!!!')
    img_combined = np.where(img_combined>cutoff,1,0)
    
    if np.sum(img_combined)>0.9*img_combined.shape[0]*img_combined.shape[1]:
        img_combined = np.where(img_combined>0,0,1)
        img_combined[:5,:5] = 1
    if img_combined.max()==0:
        img_combined[:5,:5] = 1
    label_i = sn.aggressiveLabel(img_combined.squeeze())
    final_label.append(label_i)
#test_label = test_label.loc[selected_id,:]
#test_label = test_label.reset_index(drop=True)
test_label['ImageLabel'] = final_label
del final_label

r_n = 10
c_n = 7
fig, m_axs = plt.subplots(r_n, c_n,figsize=(10,15))
for i in range(r_n):
    for j in range(c_n):
        id_pointer = c_n*i+j
        if id_pointer<test_label.shape[0]:
            m_axs[i][j].imshow(test_label.loc[id_pointer,'ImageLabel'].squeeze())
            m_axs[i][j].axis('off')
            m_axs[i][j].set_title(str(id_pointer),fontsize=7)
pp = PdfPages(submissionfilename+'.pdf')
pp.savefig(fig)
pp.close()

submission_encoding.submission_gen(test_label, submissionfilename)
