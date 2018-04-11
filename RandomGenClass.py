###https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import numpy as np
import pickle
import random
import data_norm
from skimage.transform import resize

#random.seed(124335)
#train_df = pickle.load(open("../input/train_df.p","rb"))

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
          imax = int(len(indexes))
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = indexes[i]#[list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

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
      whole_img = train_df.loc[train_df['ImageId']==list_IDs_temp,'Image'].item()
      whole_label = train_df.loc[train_df['ImageId']==list_IDs_temp,'ImageLabel'].item()
      whole_img = data_norm.invert_norm(whole_img)
      height,width,_ = whole_img.shape
      for i in range(self.batch_size):
          starth = random.randint(0, height-self.dim_x)
          startw = random.randint(0, width-self.dim_y)
          x_img = whole_img[starth:starth+self.dim_x,startw:startw+self.dim_y,:]
          y_label = whole_label[starth:starth+self.dim_x,startw:startw+self.dim_y]
          # Store volume
          X[i, :, :, :] = x_img
          # Store class
          y[i,:,:,0] = y_label

      return X, y
