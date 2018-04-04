from RandomGenClass_nuclei import DataGenerator
import InputOutputForNN as ionn
import pandas as pd
import numpy as np
import ZoomNet_jac_0402 as nn_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
import h5py
import pickle
from sklearn.model_selection import KFold
import random

#Train Test Split parameters
n = 5
id_num = 'Guo_0404_Zoom_nuclei_'+str(n)+'fold'
SEED = 932894
#Confidence threshold for nuclei identification
cutoff = 0.5

train_df = pickle.load(open("./inputs/train_df2.p","rb"))
recluster = pickle.load(open('./inputs/recluster.p','rb'))
train_df = pd.merge(train_df, recluster, left_on=['ImageId'],
              right_on=['ImageId'],
              how='inner')

from keras import backend as K
# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
def getSubinfo(img_id,df,output_shape):
    img = df.loc[df['ImageId']==img_id,'Image'].item()
    masks = df.loc[df['ImageId']==img_id,'ImageLabel'].item()
    hsv_cluster = df.loc[df['ImageId']==img_id,'hsv_cluster_x'].item()
    hsv_dominant = df.loc[df['ImageId']==img_id,'hsv_dominant'].item()
    hollow_cluster = df.loc[df['ImageId']==img_id,'KmeansClassified'].item()
    dim_x,dim_y,_ = img.shape
    num_mask = len(np.unique(masks))-1
    mask = np.where(masks>0,1,0)
    result = []
    for index in range(num_mask):
        labels = np.where(masks==index+1)
        if np.sum(labels) > 0:
            x_start = max(0,min(labels[0])-5)
            x_end = min(masks.shape[0],max(labels[0])+5)
            y_start = max(0,min(labels[1])-5)
            y_end = min(masks.shape[1],max(labels[1])+5)
            img_i = img[x_start:x_end,y_start:y_end,:]
            #img_i = resize(img_i,output_shape,mode='reflect',preserve_range=True)
            mask_i = mask[x_start:x_end,y_start:y_end]
            #mask_i = resize(mask_i,output_shape,mode='reflect',preserve_range=True)
            mask_i = np.where(mask_i>0.99,1,0)
            result.append((img_id, img_i, mask_i, hsv_dominant, hsv_cluster, hollow_cluster))
        starth = random.randint(0, dim_x-output_shape[0])
        startw = random.randint(0, dim_y-output_shape[1])
        img_i = img[starth:starth+output_shape[0],startw:startw+output_shape[1],:]
        mask_i = mask[starth:starth+output_shape[0],startw:startw+output_shape[1]]
        result.append((img_id, img_i, mask_i, hsv_dominant, hsv_cluster, hollow_cluster))
    return result

output_shape = (128,128)
for i in range(train_df.shape[0]):
    #print(str(i)+'th image is processing...')
    img_id = train_df.loc[i,'ImageId']
    tmp = getSubinfo(img_id,train_df,output_shape)
    if i == 0:
        train_df_nuclei = tmp
    else:
        train_df_nuclei = train_df_nuclei + tmp
train_df_nuclei = pd.DataFrame(train_df_nuclei,columns=(['ImageId','Image','ImageLabel','hsv_dominant','hsv_cluster','hollow_cluster']))
del train_df

random.seed(124335)
#Fragment parameters
InputDim = [128,128]
OutputDim = [100,100]
Stride = [50,50]
#Extract train data imageids
total_ids = list(train_df_nuclei.index)
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
    model = nn_model.model_gen(InputDim)
    epochs_number = 30
    batch_size = 32
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('model_'+str(id_num)+'_'+str(I)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = History()
    params ={'dim_x': InputDim[0],
             'dim_y': InputDim[1],
             'dim_z': 3,
             'batch_size': batch_size,
             'shuffle': True}
    training_generator = DataGenerator(**params).generate(train_ids,train_df)
    validation_generator = DataGenerator(**params).generate(val_ids,train_df)
    output_history = model.fit_generator(generator=training_generator, steps_per_epoch=len(train_ids)//batch_size, epochs=epochs_number, validation_data=validation_generator,validation_steps=len(val_ids)//batch_size, callbacks=[earlyStopping,mcp_save,history])
    print('done.')
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('history_'+str(id_num)+'_'+str(I)+'.csv', sep='\t', index=True, float_format='%.4f')
    ##To release GPU memory by deleting model
    del output_history
    del model

def model_predict(I,Test_data):
    model = nn_model.model_gen(InputDim)
    #test data prediction
    print(str(I)+'th cv model to predict...')
    model.load_weights(filepath = 'model_'+str(id_num)+'_'+str(I)+'.hdf5')
    Test_Label_I = []
    for t in range(Test_data.shape[0]):
        print(str(t)+'th image is being predicted...')
        test_x = Test_data.loc[t,'X']
        pred_test = model.predict(test_x)
        Test_Label_I.append(pred_test)
    ##To release GPU memory by deleting model
    del model
    return Test_Label_I

for i in range(n):
    print(str(i)+'th run is starting...')
    model_fitting(ids[i],i,train_df_nuclei)

#Import test data pieces given image type: 'all','fluo','histo' or 'bright'
Test_data = ionn.sub_fragments_extract(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
print('Start to predict...')
#pred_outputs_kfold = []
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
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label.append((Test_data.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label,open( "Test_Label.p","wb" ))

del Test_data
del Test_Label
del pred_outputs_kfold
#Import test data pieces (rotated) given image type: 'all','fluo','histo' or 'bright'
Test_data_rot = ionn.sub_fragments_extract_rot(InputDim=InputDim,OutputDim=OutputDim,Stride=Stride,image_type='all',train=False,reflection=False)
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
    pred_label = ionn.MidExtractProcess(pred_test,InputDim[0],InputDim[1],OutputDim[0],OutputDim[1])
    OutputImage = ionn.OutputStitch(img_shape=Test_data_rot.loc[i,'ImageShape'],output=pred_label,strideX=Stride[0],strideY=Stride[1])
    #OutputImage = np.where(OutputImage>cutoff,1,0)
    Test_Label_rot.append((Test_data_rot.loc[i,'ImageId'],OutputImage))
print('Saving results...')
pickle.dump(Test_Label_rot,open( "Test_Label_rot.p","wb" ))
