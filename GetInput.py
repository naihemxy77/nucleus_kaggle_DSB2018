from data_norm import minmax_norm,invert_norm
import numpy as np
import pandas as pd
from InputOutputForNN import InputGeneration
from skimage.transform import pyramid_reduce

def renorm_minmax(train_df):
    train_df.rename(columns={'ImageInput':'OldInput','ImageLabel':'OldImageLabel'},inplace=True)
    newInput = []
    for i in range(0,len(train_df)):
        ids = train_df.ImageId[i]
        tmpimg = train_df.OldInput[i]
        newimg = np.zeros_like(tmpimg)
        newlabel = train_df.OldImageLabel[i]
        local = tmpimg[:,:,3]
        img = tmpimg[:,:,:3]
        newimg[:,:,:3] += minmax_norm(img)
        newimg[:,:,3] += local
        xlen,ylen = newimg.shape[:2]
        while min(xlen,ylen)>360:
            newimg = pyramid_reduce(image=newimg,sigma=0)
            newlabel = pyramid_reduce(image=newlabel,sigma=0)
            xlen,ylen = newimg.shape[:2]
            pass
        newInput.append((ids,newimg,newlabel))
        pass
    newDF = pd.DataFrame(newInput,columns=(['ImageId','ImageInput','ImageLabel']))
    train_df = train_df.merge(newDF,on=(['ImageId']))
    train_df = train_df.drop(columns=(['OldInput','OldImageLabel']))
    return train_df

def getInput():
    print("Generating input data ..........")
    train_df = pd.read_pickle("../inputs/train_df.p")
    newInput = []
    for i in range(0,len(train_df)):
        ids = train_df.ImageId[i]
        img = train_df.Image[i]
        if train_df.hsv_cluster[i] == 0:
            imgInput = minmax_norm(img)
        else:
            imgInput = invert_norm(img)
            pass
        newInput.append((ids,imgInput))
        pass
    imgInput = pd.DataFrame(newInput,columns=(['ImageId','ImageInput']))
    train_df = train_df.merge(imgInput,on=(['ImageId']))
    #train_df = train_df.drop(['Image'],axis=1)
    #train_local = pd.read_pickle('../inputs/train_local1.p')
    #train_tmp = pd.read_pickle('../inputs/train_local2.p')
    #train_local = pd.concat([train_local,train_tmp])
    #train_tmp = pd.read_pickle('../inputs/train_local3.p')
    #train_local = pd.concat([train_local,train_tmp])
    #train_df_local = train_df.merge(train_local,on=(['ImageId']))
    #train_df_local = renorm_minmax(train_df_local)
    #train_df2 = pd.read_pickle("../inputs/multiLayerLabels.p")
    #train_df0 = train_df_local.merge(train_df2,on=(['ImageId']))
    train_hollow = pd.read_pickle("../inputs/recluster.p")
    train_df0 = train_df.merge(train_hollow,on=(['ImageId']))
    return train_df0

def getSolid():
    train_df = getInput()
    train_solid = train_df[train_df.KmeansClassified==1]
    return train_solid

def getMild():
    train_df = getInput()
    train_mild = train_df[train_df.KmeansClassified==2]
    return train_mild

def getHollow():
    train_df = getInput()
    train_hollow = train_df[train_df.KmeansClassified==0]
    return train_hollow

def dataSetGeneration(dataFrame):
    X = dataFrame['ImageInput']
    Y = dataFrame['ImageLabel']
    X_ndarray = np.zeros((20000,128,128,3))
    Y_ndarray = np.zeros((20000,128,128,1))
    i=0
    for ii in X.keys():
        tmpImg = X[ii].data
        tmpLabel = np.atleast_3d(Y[ii].data)
        #print(tmpImg.shape,tmpLabel.shape)
        xlen1,ylen1 = tmpImg.shape[0:2]
        xlen2,ylen2 = tmpLabel.shape[0:2]
        xlen = min(xlen1,xlen2)
        ylen = min(ylen1,ylen2)
        tmpImg = np.asarray(tmpImg)[:xlen,:ylen,:]
        tmpLabel = np.asarray(tmpLabel)[:xlen,:ylen,:]
        extractX = InputGeneration(img=tmpImg,inputX=128,inputY=128,outputX=68,outputY=68,reflection=True,strideX=68,strideY=68)
        extractY = InputGeneration(img=tmpLabel,inputX=128,inputY=128,outputX=68,outputY=68,reflection=True,strideX=68,strideY=68)
        a,b,c,d = extractX.shape
        X_ndarray[i:i+a,:,:,:] = extractX
        Y_ndarray[i:i+a,:,:,:] = extractY
        i+=a
        if i>20000:
            print("Dataset larger than 5000 image fragments.Please adjust the GetInput module.")
            pass
        pass
    print('Dataset generated.')
    return (X_ndarray[0:i,:,:,:],Y_ndarray[0:i,:,:,:])

def getDataSet(option='all'):
    if option=='all':
        return dataSetGeneration(getInput())
    elif option=='solid':
        return dataSetGeneration(getSolid())
    elif option=='mild':
        return dataSetGeneration(getMild())
    elif option=='hollow':
        return dataSetGeneration(getHollow())
    else:
        raise ValueError('Choose from all,solid,mild or hollow. No other options.')
    return None
