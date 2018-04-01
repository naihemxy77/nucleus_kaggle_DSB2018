import numpy as np
from sklearn.cluster import KMeans

def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    centers = clt.cluster_centers_
    results = []
    for i in range(len(hist)):
        results.append((hist[i],centers[i]))
    return results

def minmax_norm(img):
    return (img-img.min())/(img.max()-img.min())

def rgb_norm(img):
    return img/255

def invert_norm(img):
    img = (img-img.min())/(img.max()-img.min())
    img = 1-img
    return img

def minmax_norm_extend(img,color_num=2):
    colors = get_domimant_colors(img,top_colors=color_num)
    colors=sorted(colors, key=lambda x: (-x[0], x[1]))
    domi = np.zeros((img.shape[0],img.shape[1],img.shape[2]*color_num))
    for i in range(color_num):
        domi[:,:,i*img.shape[2]:(i+1)*img.shape[2]] = colors[i][1][:3]
    img = np.concatenate((img,domi),axis=-1)
    img_norm = (img-img.min())/(img.max()-img.min())
    return img_norm