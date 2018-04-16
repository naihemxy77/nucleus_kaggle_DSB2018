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

def minmax_per_layer(img):
    _,_,n = img.shape
    img_new = np.zeros_like(img)
    for i in range(n):
        img_new[:,:,i] = minmax_norm(img[:,:,i])
    return img_new

def bg_extend(img,color_num=2):
    colors = get_domimant_colors(img,top_colors=color_num)
    colors=sorted(colors, key=lambda x: (-x[0], x[1]))
    domi = np.zeros((img.shape[0],img.shape[1],img.shape[2]*color_num))
    for i in range(color_num):
        tmp = colors[i][1][:3]
        tmp2 = []
        for k in range(len(tmp)):
            tmp2.append(round(tmp[k]))
        domi[:,:,i*img.shape[2]:(i+1)*img.shape[2]] = np.array(tmp2).astype('int')
    img_extend = np.concatenate((img,domi),axis=-1)
    return img_extend

def minmax_norm_extend(img,color_num=2):
    img = bg_extend(img,color_num=color_num)
    img_norm = (img-img.min())/(img.max()-img.min())
    return img_norm

def img_extend(df,color_num=2):
    df_extend = df.copy(deep=True)
    img_list = []
    i = 0
    for index,row in df.iterrows():
        i = i+1
        print(str(i)+'th image is processing for image extension...')
        img = row['Image']
        img_extend = bg_extend(img,color_num=color_num)
        img_list.append(img_extend)
    df_extend['Image'] = img_list
    return df_extend