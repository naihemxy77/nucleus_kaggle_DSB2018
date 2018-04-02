from RandomGenClass import DataGenerator
import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import SillyNet as nn_model
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import pickle

#Train Test Split parameters
id_num = 'trial'
r = 0.1
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5
random.seed(124335)
#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]

#Import test data pieces given image type: 'all','fluo','histo' or 'bright'
Test_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='histo',train=False,reflection=False)
print('Start to predict...')
model = nn_model.model_gen(InputDim)
model.load_weights(filepath = './trial/model_'+str(id_num)+'.hdf5')
pred_outputs = []
for t in range(Test_data.shape[0]):
    print(str(t)+'th image is being predicted...')
    test_x = Test_data.loc[t,'X']
    pred_test = model.predict(test_x)
    pred_outputs.append(pred_test)
print('Preparing test labels...')
Test_Label = []
for i in range(Test_data.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs[i]
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label,open( "./trial/Test_Label.p","wb" ))

del Test_data
del Test_Label
del pred_outputs
#Import test data pieces (rotated) given image type: 'all','fluo','histo' or 'bright'
Test_data_rot = ionn.sub_fragments_extract_rot(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='histo',train=False,reflection=False)
print('Start to predict...')
model.load_weights(filepath = './trial/model_'+str(id_num)+'.hdf5')
pred_outputs = []
for t in range(Test_data_rot.shape[0]):
    print(str(t)+'th image is being predicted...')
    test_x = Test_data_rot.loc[t,'X']
    pred_test = model.predict(test_x)
    pred_outputs.append(pred_test)
print('Preparing test labels...')
Test_Label_rot = []
for i in range(Test_data_rot.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs[i]
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data_rot.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label_rot.append((Test_data_rot.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label_rot,open( "./trial/Test_Label_rot.p","wb" ))

import matplotlib
matplotlib.use('Agg')

import submission_encoding
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import AggressiveSplit as sn

##Generate Test Masks and Submission files given pickle outputs from models
##In current file, if image type is not histological, then test masks will be
##predicted by the linear combination of otsu,iso and li.
submissionfilename = './trial/histo_trial'
test_label = pickle.load(open( "./trial/Test_Label.p","rb" ))
test_label = pd.DataFrame(test_label, columns=['ImageId','ImageOutput'])

test_label_rot = pickle.load(open( "./trial/Test_Label_rot.p","rb" ))
test_label_rot = pd.DataFrame(test_label_rot, columns=['ImageId','ImageOutput'])

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
test_label['ImageLabel'] = final_label
del final_label

r_n = 2
c_n = 2
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