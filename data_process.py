##from: https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering

##### Generate training or test data using data_process

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io
import os
from sklearn.cluster import KMeans

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

def read_image_labels(image_id, pattern):
    mask_file = pattern.format(image_id)
    masks = skimage.io.imread_collection(mask_file).concatenate()    
    num_masks = masks.shape[0]
    labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = 1 #index + 1
    return labels

def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist

def get_images_details(image_ids,label,img_pattern,mask_pattern):
    details = []
    for image_id in image_ids:
        image_hsv = read_image(image_id, img_pattern, space="hsv")
        dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(image_hsv, top_colors=1)
        dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] * dominant_colors_hsv.shape[1])
        if label:
            labels = read_image_labels(image_id, mask_pattern)
            info = (image_id, image_hsv, labels, dominant_colors_hsv.squeeze())
        else:
            info = (image_id, image_hsv, dominant_colors_hsv.squeeze())
        details.append(info)
    return details

def data_process(datatype='train',label=True,cluster=True,stage='stage1'):
    root_dir = "../input/"+stage+"_"+datatype
    img_pattern = "%s/{}/images/{}.png" % root_dir
    mask_pattern = "%s/{}/masks/*.png" % root_dir
    IMAGE_ID = "ImageId"
    IMAGE = 'Image'
    LABEL = 'ImageLabel'
    HSV_CLUSTER = "hsv_cluster"
    HSV_DOMINANT = "hsv_dominant"
    # Load stage 1 image identifiers.
    train_image_ids = image_ids_in(root_dir)
    details = get_images_details(train_image_ids, label, img_pattern, mask_pattern)
    if label:
        COLS = [IMAGE_ID, IMAGE, LABEL, HSV_DOMINANT]
    else:
        COLS = [IMAGE_ID, IMAGE, HSV_DOMINANT]
    df = pd.DataFrame(details, columns=COLS)
    if cluster:
        X = (pd.DataFrame(df[HSV_DOMINANT].values.tolist())).as_matrix()
        kmeans = KMeans(n_clusters=3).fit(X)
        clusters = kmeans.predict(X)
        centroids = kmeans.cluster_centers_
        df[HSV_CLUSTER] = clusters
        return df, centroids
    else: return df
