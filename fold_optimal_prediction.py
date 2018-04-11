import InputOutputForNN as ionn
import InputOutputForNN_unet as ionn_unet
import pandas as pd
import numpy as np
import ZoomNet_0320 as zoom_model
import model_unet_compile_iou_loss20180403 as unet_model
import h5py
import pickle
import random

#Train Test Split parameters
n = 8
zoom_id = 'Guo_0409_Zoom_invert_'+str(n)+'fold'
unet_id = 'Mao_0407_unet_all_iou_300epoch_'+str(n)+'fold'
SEED = 932894

random.seed(124335)
#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]
from keras import backend as K
# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
def model_predict(I,model,Test_data):
    print(str(I)+'th cv model to predict...')
    if model == 'unet':
        model = unet_model.get_unet(InputDim)
        model.load_weights(filepath = './inputs/model_'+str(unet_id)+'_'+str(I)+'.hdf5')
    else:
        model = zoom_model.model_gen(InputDim)
        model.load_weights(filepath = './inputs/model_'+str(zoom_id)+'_'+str(I)+'.hdf5')
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
Test_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
Test_data_unet = ionn_unet.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
print('Start to predict...')
zoom_set = [2,3,4]
for i in range(n):
    if i == 0:
        pred_outputs_kfold=np.array(model_predict(i,'unet',Test_data_unet))
    elif i in zoom_set:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,'zoom',Test_data))
    else:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,'unet',Test_data_unet))
print(pred_outputs_kfold.shape)
pred_outputs_kfold = pred_outputs_kfold/n
print('Preparing test labels...')
Test_Label = []
for i in range(Test_data.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs_kfold[i]
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label,open( "Test_Label_fold_combined.p","wb" ))

del Test_data
del Test_data_unet
del Test_Label
del pred_outputs_kfold
#Import test data pieces (rotated) given image type: 'all','fluo','histo' or 'bright'
Test_data_rot = ionn.sub_fragments_extract_rot(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
Test_data_rot_unet = ionn_unet.sub_fragments_extract_rot(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
print('Start to predict...')
zoom_set = [2,3,4]
for i in range(n):
    if i == 0:
        pred_outputs_kfold=np.array(model_predict(i,'unet',Test_data_rot_unet))
    elif i in zoom_set:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,'zoom',Test_data_rot))
    else:
        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,'unet',Test_data_rot_unet))
print(pred_outputs_kfold.shape)
pred_outputs_kfold = pred_outputs_kfold/n
print('Preparing test labels...')
Test_Label_rot = []
for i in range(Test_data_rot.shape[0]):
    print(str(i)+'th model is processing...')
    pred_test = pred_outputs_kfold[i]
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data_rot.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    Test_Label_rot.append((Test_data_rot.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label_rot,open( "Test_Label_rot_fold_combined.p","wb" ))