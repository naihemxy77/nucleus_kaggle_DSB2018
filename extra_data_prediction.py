from RandomGenClass import DataGenerator
import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import ZoomNet_0320 as nn_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
import h5py
import pickle
from sklearn.model_selection import KFold
import random

#Train Test Split parameters
n = 5
id_num = 'Guo_0321_ZoomNet_invert_norm_'+str(n)+'fold'
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5

random.seed(124335)
#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]

def model_predict(I,Test_data):
    model = nn_model.model_gen(InputDim)
    #test data prediction
    print(str(I)+'th cv model to predict...')
    model.load_weights(filepath = './model/model_'+str(id_num)+'_'+str(I)+'.hdf5')
    Test_Label_I = []
    for t in range(Test_data.shape[0]):
        print(str(t)+'th image is being predicted...')
        test_x = Test_data.loc[t,'X']
        pred_test = model.predict(test_x)
        Test_Label_I.append(pred_test)
    ##To release GPU memory by deleting model
    del model
    return Test_Label_I

#Import test data pieces given image type: 'all','fluo','histo' or 'bright'
Extra_data = ionn.sub_fragments_extract_extra(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,reflection=False)
print('Start to predict...')
#pred_outputs_kfold = []
for i in range(n):
    if i == 0:
        pred_outputs_kfold=np.array(model_predict(i,Extra_data))
    else:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,Extra_data))
print(pred_outputs_kfold.shape)
pred_outputs_kfold = pred_outputs_kfold/n
print('Preparing extra labels...')
Extra_Label = []
for i in range(Extra_data.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs_kfold[i]
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Extra_Label.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    OutputImage = np.where(OutputImage>cutoff,1,0)
    Extra_Label.append((Extra_Label.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Extra_Label,open( "Extra_Label.p","wb" ))