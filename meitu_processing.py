##from: https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering

##### Generate training or test data using data_process

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing,square
import AggressiveSplit as sn
import submission_encoding
from matplotlib.backends.backend_pdf import PdfPages

def image_ids_in(root_dir, ignore=[]):
    ids_list = []
    for id_i in os.listdir(root_dir):
        id_i = id_i.split('.')[0]
        ids_list.append(id_i)
    return ids_list
def read_image(image_id, pattern):
    image_file = pattern.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    # Drop alpha which is not used
    image = image[:, :, :3]
    image = rgb2gray(image)
    return image

def make_label(image):
    threshold = threshold_otsu(image)
    label = image>threshold
    if np.sum(label==False)<np.sum(label==True):
        label = np.where(label,0,1)
    label = np.where(label,1,0)
    if label.max()==0:
        label[:5,:5]=1
    label = closing(label,square(3))
    return sn.aggressiveLabel(label)

def get_images_details(image_ids,img_pattern):
    details = []
    i=1
    for image_id in image_ids:
        print(i)
        i=i+1
        image = read_image(image_id, img_pattern)
        image_label = make_label(image)
        info = (image_id, image, image_label)
        details.append(info)
    return details

def data_process():
    root_dir = "../meitu/"
    img_pattern = "%s/{}.png" % root_dir
    IMAGE_ID = "ImageId"
    IMAGE = 'Image'
    IMG_LABEL = 'ImageLabel'
    # Load stage 1 image identifiers.
    print('Getting MeiTu images ...')
    image_ids = image_ids_in(root_dir)
    details = get_images_details(image_ids, img_pattern)
    COLS = [IMAGE_ID, IMAGE, IMG_LABEL]
    df = pd.DataFrame(details, columns=COLS)
    return df

test_meitu = data_process()

r = 2
c = 2
fig,ax = plt.subplots(r,c)
for i in range(r):
    for j in range(c):
        im_index = i*c+j
        img_meitu = test_meitu.loc[im_index,'Image']
        ax[i,j].imshow(img_meitu)
plt.show()
fig,ax = plt.subplots(r,c)
for i in range(r):
    for j in range(c):
        im_index = i*c+j
        img_meitu = test_meitu.loc[im_index,'ImageLabel']
        ax[i,j].imshow(img_meitu)
plt.show()
pp = PdfPages('meitu_pred0311.pdf')
pp.savefig(fig)
pp.close()

submission_encoding.submission_gen(test_meitu, 'meitu_pred0311')