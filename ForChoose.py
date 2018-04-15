
# coding: utf-8

# In[1]:


import pandas as pd
test_df2 = pd.read_pickle("../inputs/test_df2.p")
res = test_df2.ImageId
res = pd.DataFrame(res,columns = ['ImageId'])
baseline = []
Unet = []
ZoomNetAll = [] 
Hollow = []
NewUnet = []
getLocal = []


# In[30]:


def notvalid(dd):
    if dd=='':
        return 1
    if len(dd)!=6:
        print('Invalid input. Retype it please...')
        return 1
    s = 0
    for ii in dd:
        s+=int(ii)
        if int(ii)==0 or int(ii)==1:
            pass
        else:
            print('Invalid input. Retype it please...')
            return 1
        pass
    if s==0:
        print('You have to choose at least 1. Retype it please...')
        return 1
    return 0


# In[27]:


def getRes(i):
    dd = ''
    while(notvalid(dd)):
        dd = list(input(str(i)+': (for example: 111000)'))
    return dd


# In[12]:


for i in range(0,3019):
    dd = getRes(i)
    ll = list(dd)
    baseline.append(ll[0])
    Unet.append(ll[1])
    ZoomNetAll.append(ll[2])
    Hollow.append(ll[3])
    NewUnet.append(ll[4])
    getLocal.append(ll[5])


# In[13]:


res['baseline']=baseline
res['Unet']=Unet
res['ZoomNetAll']=ZoomNetAll
res['Hollow']=Hollow
res['NewUnet']=NewUnet
res['getLocal']=getLocal
res.to_pickle('resultOfChoices.p')


