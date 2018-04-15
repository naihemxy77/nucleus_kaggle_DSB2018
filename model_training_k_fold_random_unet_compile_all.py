from RandomGenClass_cluster_input import DataGenerator
import InputOutputForNN_cluster_input as ionn
import pandas as pd
import numpy as np
import model_unet_extract_more_11layer_20180402 as nn_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
import h5py
import pickle
from sklearn.model_selection import KFold
import random

#Train Test Split parameters
n = 8
id_num = 'Mao_0409_unet_all_7layer_'+str(n)+'fold'
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5

train_df = pickle.load(open("/home/mao47/Kaggle_nuclei/inputs/train_df.p","rb"))
random.seed(124335)
#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]
#Extract train data imageids
train_df = train_df
total_ids = list(train_df['ImageId'].values)
#If just want to train fluorescent data (similarly, 1 for histo and 2 for bright)
#train_df = train_df[train_df['hsv_cluster']==0]
#total_ids = list(train_df.loc[train_df['hsv_cluster']==0,'ImageId'].values)

#Split images into cross-fold sets (note that pieces for one image always together belong to train/val set)
kf = KFold(n_splits=n, shuffle=True, random_state=SEED)
ids=list(kf.split(total_ids))
loc_i = np.arange(n)

def model_fitting(ids,I,train_df):
    #concatenate all pieces into one dataset
    train_ids = [total_ids[i] for i in ids[0]]
    val_ids = [total_ids[i] for i in ids[1]]
    #model fitting
    model = nn_model.get_unet(InputDim)
    #model.load_weights(filepath = '/home/mao47/Kaggle_nuclei/model_'+str(id_num)+'_'+str(I)+'.hdf5')
    epochs_number = 300
    batch_size = 20
    #earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('/home/mao47/Kaggle_nuclei/model_'+str(id_num)+'_'+str(I)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = History()
    params ={'dim_x': InputDim[0],
             'dim_y': InputDim[1],
             'dim_z': 7,
             'batch_size': batch_size,
             'shuffle': True}
    training_generator = DataGenerator(**params).generate(train_ids,train_df)
    validation_generator = DataGenerator(**params).generate(val_ids,train_df)
    output_history = model.fit_generator(generator=training_generator, steps_per_epoch=len(train_ids)//batch_size, epochs=epochs_number, validation_data=validation_generator,validation_steps=len(val_ids)//batch_size, callbacks=[mcp_save,history])
    print('done.')
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('/home/mao47/Kaggle_nuclei/history_'+str(id_num)+'_'+str(I)+'.csv', sep='\t', index=True, float_format='%.4f')
    ##To release GPU memory by deleting model
    del output_history
    del model

def model_predict(I,Test_data):
    model = nn_model.get_unet(InputDim)
    #test data prediction
    print(str(I)+'th cv model to predict...')
    model.load_weights(filepath = '/home/mao47/Kaggle_nuclei/model_'+str(id_num)+'_'+str(I)+'.hdf5')
    Test_Label_I = []
    for t in range(Test_data.shape[0]):
        print(str(t)+'th image is being predicted...')
        test_x = Test_data.loc[t,'X']
        pred_test = model.predict(test_x)
        Test_Label_I.append(pred_test)
    ##To release GPU memory by deleting model
    del model
    return Test_Label_I

for i in range(1):
    print(str(i)+'th run is starting...')
    model_fitting(ids[i],i,train_df)

#Import test data pieces given image type: 'all','fluo','histo' or 'bright'
#Test_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
#print('Start to predict...')

#pred_outputs_kfold = []
#for i in range(n):
#    if i == 0:
#        pred_outputs_kfold=np.array(model_predict(i,Test_data))
#    else:
#        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,Test_data))
#print(pred_outputs_kfold.shape)
#pred_outputs_kfold = pred_outputs_kfold/n
#print('Preparing test labels...')
#Test_Label = []
#for i in range(Test_data.shape[0]):
#    print(str(i)+'th model is processing...')
#    pred_test = pred_outputs_kfold[i]
#    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
#    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
#    #OutputImage = np.where(OutputImage>cutoff,1,0)
#    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
#print('Saving results...')
#pickle.dump(Test_Label,open( "/home/mao47/Kaggle_nuclei/Test_Label_all_7layer.p","wb" ))
#
#del Test_data
#del Test_Label
#del pred_outputs_kfold
##Import test data pieces (rotated) given image type: 'all','fluo','histo' or 'bright'
#Test_data_rot = ionn.sub_fragments_extract_rot(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
#print('Start to predict...')
##pred_outputs_kfold = []
#for i in range(n):
#    if i == 0:
#        pred_outputs_kfold=np.array(model_predict(i,Test_data_rot))
#    else:
#        pred_outputs_kfold=pred_outputs_kfold+np.array(model_predict(i,Test_data_rot))
#print(pred_outputs_kfold.shape)
#pred_outputs_kfold = pred_outputs_kfold/n
#print('Preparing test labels...')
#Test_Label_rot = []
#for i in range(Test_data_rot.shape[0]):
#    print(str(i)+'th model is processing...')
#    pred_test = pred_outputs_kfold[i]
#    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
#    OutputImage = ionn.OutputStitch(img_shape=Test_data_rot.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
#    #OutputImage = np.where(OutputImage>cutoff,1,0)
#    Test_Label_rot.append((Test_data_rot.loc[i,'ImageId'],OutputImage))
#print('Saving results...')
#pickle.dump(Test_Label_rot,open( "/home/mao47/Kaggle_nuclei/Test_Label_all_7layer_rot.p","wb" ))