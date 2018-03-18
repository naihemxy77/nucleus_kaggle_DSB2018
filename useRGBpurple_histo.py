import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io
import os
from sklearn.cluster import KMeans
import pickle

def image_ids_in(root_dir, ignore=[]):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids
def read_image(image_id, pattern, space="rgb"):
    image_file = pattern.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    # Drop alpha which is not used
    image = image[:, :, :3]
    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    return image
def get_images_details(image_ids,img_pattern):
    details = []
    for image_id in image_ids:
        image_rgb = read_image(image_id, img_pattern, space="rgb")
        info = (image_id, image_rgb)
        details.append(info)
    return details

root_dir = "../input/stage1_test"
img_pattern = "%s/{}/images/{}.png" % root_dir
IMAGE_ID = "ImageId"
IMAGE = 'Image'

test_image_ids = image_ids_in(root_dir)
details = get_images_details(test_image_ids, img_pattern)
COLS = ['ImageId', 'Image']
test_histo_df = pd.DataFrame(details, columns=COLS)

def guess_purple(image):
    guess_matrix=np.zeros([image.shape[0],image.shape[1]])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j,2]<255/2 and image[i,j,0] < image[i,j,2] and image[i,j,1] < image[i,j,0]:
                guess_matrix[i,j]=1
    return guess_matrix

test_label_guess = []
for k in range(test_histo_df.shape[0]):
    tmp=guess_purple(test_histo_df['Image'][k])
    test_label_guess.append(tmp)
    
test_histo_df['ImageOutput']=test_label_guess

pickle.dump(test_histo_df,open( "Test_Label_histo_color.p","wb" ))
