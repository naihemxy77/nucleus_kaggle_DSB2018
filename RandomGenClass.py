###https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import numpy as np
import pickle
import random
import data_norm
from skimage.transform import resize

#random.seed(124335)
#train_df = pickle.load(open("../input/train_df.p","rb"))

def data_augmentation(x_img,y_label,dim_x,dim_y):
    p = random.uniform(0,1)
    if p < 0.34:
        x_img_a = x_img
        y_label_a = y_label
    elif p >= 0.34 and p < 0.67: #horizontal flip
        band_1_mirror_lr = np.flip(x_img[:,:,0], 0)
        band_2_mirror_lr = np.flip(x_img[:,:,1], 0)
        band_3_mirror_lr = np.flip(x_img[:,:,2], 0)
        x_img_a = np.dstack((band_1_mirror_lr, band_2_mirror_lr,band_3_mirror_lr))
        y_label_a = np.flip(y_label,0)
    else:#elif p >= 0.5 and p < 0.75: #vertical flip
        band_1_mirror_ud = np.flip(x_img[:,:,0], 1)
        band_2_mirror_ud = np.flip(x_img[:,:,1], 1)
        band_3_mirror_ud = np.flip(x_img[:,:,2], 1)
        x_img_a = np.dstack((band_1_mirror_ud, band_2_mirror_ud,band_3_mirror_ud))
        y_label_a = np.flip(y_label,1)
#    else: #0.5 zoom in
#        starth = random.randint(0, dim_x//2)
#        startw = random.randint(0, dim_y//2)
#        x_img_a = x_img[starth:starth+dim_x//2,startw:startw+dim_y//2,:]
#        y_label_a = y_label[starth:starth+dim_x//2,startw:startw+dim_y//2]
#        x_img_a = resize(x_img_a,(dim_x,dim_y), mode='constant',preserve_range=True)
#        y_label_a = resize(y_label_a,(dim_x,dim_y), mode='constant',preserve_range=True)
    return x_img_a,y_label_a

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 128, dim_y = 128, dim_z = 3, batch_size = 32, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, list_IDs,train_df):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(list_IDs_temp,train_df)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, list_IDs_temp,train_df):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
      y = np.empty((self.batch_size, self.dim_x, self.dim_y, 1), dtype = int)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          whole_img = train_df.loc[train_df['ImageId']==ID,'Image'].item()
          whole_label = train_df.loc[train_df['ImageId']==ID,'ImageLabel'].item()
          #cluster = train_df.loc[train_df['ImageId']==ID,'hsv_cluster'].item()
          #if cluster == 0:
          whole_img = data_norm.minmax_norm(whole_img)
          #else:
          #    whole_img = data_norm.invert_norm(whole_img)
          #
          height,width,_ = whole_img.shape
          starth = random.randint(0, height-self.dim_x)
          startw = random.randint(0, width-self.dim_y)
          x_img = whole_img[starth:starth+self.dim_x,startw:startw+self.dim_y,:]
          y_label = whole_label[starth:starth+self.dim_x,startw:startw+self.dim_y]
          #x_img_a,y_label_a = data_augmentation(x_img,y_label,self.dim_x,self.dim_y)
          # Store volume
          X[i, :, :, :] = x_img
          # Store class
          y[i,:,:,0] = y_label

      return X, y
