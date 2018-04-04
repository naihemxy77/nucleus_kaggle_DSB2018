
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt


# In[24]:


from ImageUtilities import *
from skimage.filters import threshold_local,threshold_otsu
import data_norm
from math import sqrt
from sklearn.cluster import KMeans


# In[3]:


trainPath = '../input/stage1_train/'


# In[4]:


train_df = pd.read_pickle("../input/train_df.p")


# In[5]:


def idToBlocksize(idstr,pth):
    ids = os.listdir(pth+idstr+'/masks/')
    if '.DS_Store' in ids: # bug in mac os
        ids.remove('.DS_Store')
    nMask = len(ids)
    meanArea = np.sum(idToMaskAll(idstr=idstr,pth=pth))/nMask # The average area of each nuclei
    return int(sqrt(meanArea))//2*4+1 # The block_size used for threshold_local. Must be odd.


# In[7]:


result0 = []


# In[8]:


for i in range(0,len(train_df)):
    imgID = train_df.ImageId[i]
    result0.append((imgID,train_df.hsv_cluster[i],idToBlocksize(imgID,trainPath)))
    pass


# In[9]:


meta0 = pd.DataFrame(result0,columns=(['ImageId','hsv_cluster','block_size']))


# In[10]:


_ = plt.hist(meta0.block_size,bins=35)


# In[11]:


def getLocal(ch):
    s0 = np.asarray(ch>ch.mean(),dtype=np.int32)
    s1 = np.asarray(ch>threshold_local(ch,block_size=23),dtype=np.int32)
    s2 = np.asarray(ch>threshold_local(ch,block_size=51),dtype=np.int32)
    s3 = np.asarray(ch>threshold_local(ch,block_size=75),dtype=np.int32)
    s4 = np.asarray(ch>threshold_local(ch,block_size=101),dtype=np.int32)
    s5 = np.asarray(ch>threshold_local(ch,block_size=169),dtype=np.int32)
    return s1*s2*s3*s4*s5*s0


# In[15]:


fluoExample = data_norm.minmax_norm(train_df.Image[53])


# In[16]:


plt.imshow(getLocal(fluoExample[:,:,0]))


# In[17]:


histExample = data_norm.invert_norm(train_df.Image[2])


# In[18]:


plt.imshow(getLocal(histExample[:,:,0]))


# In[19]:


brightExample = data_norm.invert_norm(train_df.Image[23])


# In[20]:


plt.imshow(getLocal(brightExample[:,:,0]))


# In[21]:


result = []


# In[22]:


for i in range(0,len(train_df)):
    imgID = train_df.ImageId[i]
    img = train_df.Image[i]
    mask = train_df.ImageLabel[i]                   
    
    xlen1,ylen1 = mask.shape # just to solve bugs, sorry
    xlen2,ylen2,_ = img.shape
    xlen = min(xlen1,xlen2)
    ylen = min(ylen1,ylen2)
    img = img[:xlen,:ylen,:]
    mask = mask[:xlen,:ylen]
    
    
    if train_df.hsv_cluster[i] == 0:
        img = data_norm.minmax_norm(img)
    else:
        img = data_norm.invert_norm(img)
        pass
    
    localMask = np.zeros_like(img[:,:,0],dtype=np.int32)
    for j in range(0,3):
        ch = img[:,:,j]
        tmpMask = getLocal(ch) 
        localMask += tmpMask
        pass
    localMask = np.asarray(localMask>0,dtype=np.int32) # Just 1 and 0. Predicted by getLocal
    
    noise_ratio = 1-np.sum(localMask*mask)/np.sum(localMask)
    hollow_ratio = 1-np.sum(localMask*mask)/np.sum(mask)
    result.append((imgID,noise_ratio,hollow_ratio,train_df.hsv_cluster[i],idToBlocksize(imgID,trainPath)))
    pass


# In[23]:


hollow_ratio = pd.DataFrame(result,columns=(['ImageId','Noise_Ratio','Hollow_Ratio','hsv_cluster','block_size']))


# In[38]:


plt.scatter(hollow_ratio.Noise_Ratio,hollow_ratio.Hollow_Ratio,c=hollow_ratio.hsv_cluster) 


# In[26]:


plt.scatter(hollow_ratio.block_size,hollow_ratio.Hollow_Ratio,c=hollow_ratio.hsv_cluster) 


# In[27]:


plt.scatter(hollow_ratio.block_size,hollow_ratio.Noise_Ratio,c=hollow_ratio.hsv_cluster) 


# In[41]:


meta_data = np.zeros((670,3))
meta_data[:,0] = data_norm.minmax_norm(hollow_ratio.block_size)
meta_data[:,1] = hollow_ratio.Noise_Ratio
meta_data[:,2] = hollow_ratio.Hollow_Ratio
kmeans = KMeans(n_clusters=3, random_state=0).fit(meta_data)


# In[42]:


plt.scatter(meta_data[:,0],meta_data[:,1],c=kmeans.labels_) # X vs Y: block_size vs Noise_Ratio


# In[43]:


plt.scatter(meta_data[:,0],meta_data[:,2],c=kmeans.labels_) # X vs Y: block_size vs Hollow_Ratio


# In[44]:


plt.scatter(meta_data[:,1],meta_data[:,2],c=kmeans.labels_) # X vs Y: Noise_Ratio vs Hollow_Ratio


# In[45]:


hollow_ratio['KmeansClassified'] = kmeans.labels_


# In[36]:


hollow_ratio.to_pickle('recluster.p')


# In[47]:


shapeCluster0 = hollow_ratio[hollow_ratio.KmeansClassified==0]


# In[48]:


shapeCluster1 = hollow_ratio[hollow_ratio.KmeansClassified==1]


# In[49]:


shapeCluster2 = hollow_ratio[hollow_ratio.KmeansClassified==2]


# In[50]:


shapeCluster0.head(n=10)


# In[91]:


shapeCluster1.head(n=20)


# In[52]:


shapeCluster2.head(n=10)


# In[53]:


# shapeCluster0 contains only fluorescent and histochemical samples.
# fluorescent example
fluo = data_norm.minmax_norm(train_df.Image[36])


# In[54]:


plt.imshow(fluo[:,:,0])


# In[55]:


plt.imshow(getLocal(fluo[:,:,0]))


# In[56]:


set(shapeCluster0.hsv_cluster)


# In[59]:


# shapeCluster0 contains only fluorescent and histochemical samples.
# histochemical example
hist = data_norm.invert_norm(train_df.Image[37])


# In[60]:


plt.imshow(hist[:,:,0])


# In[61]:


plt.imshow(getLocal(hist[:,:,0]))


# In[88]:


# shapeCluster1 contains only fluorescent and histochemical samples.
# Here is a fluorescent example
fluo = data_norm.minmax_norm(train_df.Image[5])


# In[89]:


plt.imshow(fluo[:,:,0])


# In[90]:


plt.imshow(getLocal(fluo[:,:,0]))


# In[95]:


# shapeCluster1 contains only fluorescent and histochemical samples.
# histochemical example
hist = data_norm.invert_norm(train_df.Image[40])


# In[96]:


plt.imshow(hist[:,:,0])


# In[97]:


plt.imshow(getLocal(hist[:,:,0]))


# In[69]:


# shapeCluster2 contains all 3 samples.
# Here is a fluorescent example
fluo = data_norm.minmax_norm(train_df.Image[18])


# In[70]:


plt.imshow(fluo[:,:,0])


# In[72]:


plt.imshow(getLocal(fluo[:,:,0]))


# In[76]:


# shapeCluster2 contains all 3 samples.
# Here is a histochemical example
hist = data_norm.invert_norm(train_df.Image[4])


# In[77]:


plt.imshow(hist[:,:,0])


# In[78]:


plt.imshow(getLocal(hist[:,:,0]))


# In[79]:


# shapeCluster2 contains all 3 samples.
# Here is a bright example
bright = data_norm.invert_norm(train_df.Image[25])


# In[80]:


plt.imshow(bright[:,:,0])


# In[81]:


plt.imshow(getLocal(bright[:,:,0]))


# In[82]:


len(shapeCluster0)


# In[83]:


len(shapeCluster1)


# In[84]:


len(shapeCluster2)


# In[3]:


(539+484*19)*5/3600

