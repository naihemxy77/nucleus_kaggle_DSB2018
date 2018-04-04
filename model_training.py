from GetInput import *
from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator

n = 5
id_num = 'Ma_0403_MoGaiNet_solid_Punch_'+str(n)+'_fold'
epochs_number = 35
batch_size = 32

X,Y = getDataSet('solid')
Y = Y[:,30:98,30:98,:]
InputDim = X.shape[1:]

indexes = np.arange(len(X))
np.random.shuffle(indexes)
kf = KFold(n_splits=n, shuffle=True)
idx=list(kf.split(indexes))
loc_i = np.arange(n)
train_idx = [indexes[idxes[0]] for idxes in idx]
val_idx = [indexes[idxes[1]] for idxes in idx]

import MoGaiNet as nn_model

datagen = ImageDataGenerator(vertical_flip=False,horizontal_flip=False,rotation_range=0,fill_mode="reflect")

def model_fitting(I):
    model = nn_model.model_gen(InputDim)
    mcp_save = ModelCheckpoint('model_'+str(id_num)+'_'+str(I)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = History()
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, cooldown=0)    
    X_train = X[indexes[train_idx[I]],:,:,:]
    Y_train = Y[indexes[train_idx[I]],:,:,:]
    X_val = X[indexes[val_idx[I]],:,:,:]
    Y_val = Y[indexes[val_idx[I]],:,:,:]
    datagen.fit(X_train)
    output_history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                         steps_per_epoch=len(X_train)//batch_size, 
                                         epochs=epochs_number, validation_data=(X_val,Y_val),
                                         callbacks=[mcp_save,history])#,reduce_lr])
    print('done.')
    df = pd.DataFrame.from_dict(history.history,orient="index")
    df = df.transpose()
    df.to_csv('history_'+str(id_num)+'_'+str(I)+'.csv', sep='\t', index=True, float_format='%.4f')
    del model
    del output_history
    del df
    clear_session()
    pass

for i in range(n):
    print(str(i)+'th run is starting...')
    model_fitting(i)
