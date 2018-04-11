import sys
import numpy as np
import pandas as pd
import pickle
import InputOutputForNN as ionn
#from GetInput import *

NNN = sys.argv[1]

print(NNN)

n = 5
id_num = 'Ma_0406_MoGaiNet_hollow_'+str(n)+'_fold'

InputDim = [128,128]
OutputDim = [68,68]
Stride = [34,34]

import MoGaiNet as nn_model
#import BigNet as nn_model

def model_predict(I,Test_data):
    model = nn_model.model_gen()
    #test data prediction
    print(str(I)+'th cv model to predict...')
    model.load_weights(filepath = '../model_'+str(id_num)+'_'+str(I)+'.hdf5') ####modify path
    Test_Label_I = []
    for t in range(Test_data.shape[0]):
        print(str(t)+'th image is being predicted...')
        test_x = Test_data.loc[t,'X']
        pred_test = model.predict(test_x)
        Test_Label_I.append(pred_test)
    ##To release GPU memory by deleting model
    del model
    return Test_Label_I

Test_data = ionn.sub_fragments_extract(NNN=NNN,InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=True,reflection=True)
print('Start to predict...')
for i in range(n):
    if i == 0:
        pred_outputs_kfold=np.array(model_predict(i,Test_data))
    else:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,Test_data))
print(pred_outputs_kfold.shape)
pred_outputs_kfold = pred_outputs_kfold/n
print('Preparing test labels...')
Test_Label = []
for i in range(Test_data.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs_kfold[i]
    pred_label = pred_test#ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])#pred_test
    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label,open( "Train_Label_hollow_"+str(NNN)+".p","wb" ))

del Test_data
del Test_Label
del pred_outputs_kfold
Import test data pieces (rotated) given image type: 'all','fluo','histo' or 'bright'
Test_data_rot = ionn.sub_fragments_extract_rot(NNN=NNN,InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=True,reflection=False)
print('Start to predict...')
#pred_outputs_kfold = []
for i in range(n):
    if i == 0:
        pred_outputs_kfold=np.array(model_predict(i,Test_data_rot))
    else:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,Test_data_rot))
print(pred_outputs_kfold.shape)
pred_outputs_kfold = pred_outputs_kfold/n
print('Preparing test labels...')
Test_Label_rot = []
for i in range(Test_data_rot.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs_kfold[i]
    pred_label = pred_test#ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data_rot.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label_rot.append((Test_data_rot.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label_rot,open( "Train_Label_rot_hollow_"+str(NNN)+".p","wb" ))
