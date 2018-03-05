#Here are the essential functions

def tileExtract(bigPic,x,y,tileSideLengthX,tileSideLengthY):
    # cut the image with specified size and origin at x,y
    return bigPic[x:x+tileSideLengthX,y:y+tileSideLengthY,:]
def tileStick(bigPic,tile,x,y,tileSideLengthX,tileSideLengthY):
    # stack the values at specified position. Reverse function of tileExtract
    bigPic[x:x+tileSideLengthX,y:y+tileSideLengthY,:] += tile[:,:,:]
    
def paddingPeriod(img):
    tmpimg = np.atleast_3d(img)
    xlen,ylen,nlayer = tmpimg.shape
    bigPic = np.zeros((3*xlen,3*ylen,nlayer))
    for i in range(0,3):
        for j in range(0,3):
            bigPic[i*xlen:(i+1)*xlen,j*ylen:(j+1)*ylen,:] = tmpimg
            pass
        pass
    return bigPic
def paddingReflection(img):
    tmpimg = np.atleast_3d(img)
    xlen,ylen,nlayer = tmpimg.shape
    bigPic = np.zeros((3*xlen,3*ylen,nlayer))
    bigPic[xlen:2*xlen,ylen:2*ylen,:] = tmpimg
    bigPic[xlen:2*xlen,0:ylen,:] = tmpimg[:,::-1,:]
    bigPic[xlen:2*xlen,2*ylen:3*ylen,:] = tmpimg[:,::-1,:]
    bigPic[0:xlen,:,:] = bigPic[2*xlen-1:xlen-1:-1,:,:]
    bigPic[2*xlen:3*xlen,:,:] = bigPic[2*xlen-1:xlen-1:-1,:,:]
    return bigPic

def InputGeneration(img,inputX,inputY,outputX,outputY,strideX,strideY,reflection=True):
    # Matrix returned is 4-d, [numberOfTiles,inputX,inputY,image_layers]
    if outputX%strideX != 0:
        print("Invalid strideX. OutputX/strideX must be an integer.")
        return
    if outputY%strideY != 0:
        print("Invalid strideY. OutputY/strideY must be an integer.")
        return
    if reflection==True:
        bigPic = paddingReflection(img=img)
    else:
        bigPic = paddingPeriod(img=img)
        pass
    img = np.atleast_3d(img)
    xlen,ylen,nlayers = img.shape
    startPointX = xlen-outputX//2
    
    estimate = int((xlen+outputX)/strideX)*int((ylen+outputY)/strideY)*2
    tmp = np.zeros((estimate,inputX,inputY,nlayers))
    
    k = 0
    while(startPointX<xlen*2):
        startPointY = ylen-outputY//2
        while(startPointY<ylen*2):
            tmp[k,:,:,:] = tileExtract(bigPic=bigPic, x=startPointX-int((inputX-outputX)/2),y=startPointY-int((inputY-outputY)/2),tileSideLengthX=inputX,tileSideLengthY=inputY)
            k+=1
            startPointY+=strideY
            pass
        startPointX+=strideX
        pass
    result = np.zeros((k,inputX,inputY,nlayers))
    result[:,:,:,:] = tmp[:k,:,:,:]
    return result
    
def fakeProcess(inputData,inputX,inputY,outputX,outputY):
    # Just extract the middle part from input. 
    nn,_,_,nlayers = inputData.shape
    res = np.zeros((nn,outputX,outputY,nlayers))
    res[:,:,:,:] = inputData[:,int((inputX-outputX)/2):int((inputX+outputX)/2),int((inputY-outputY)/2):int((inputY+outputY)/2),:]
    return res

def OutputStitch(img,output,strideX,strideY):
    # Each pixel returned is stacked (outputX/strideX)*(outputY/strideY) times. 
    img = np.atleast_3d(img)
    xlen,ylen,nlayers = img.shape
    nn,outputX,outputY,_ = output.shape
    bigPic = np.zeros((3*xlen,3*ylen,nlayers))
    startPointX = xlen-outputX//2
    k = 0
    while(startPointX<xlen*2):
        startPointY = ylen-outputY//2
        while(startPointY<ylen*2):
            tileStick(bigPic=bigPic,tile=output[k,:,:,:],tileSideLengthX=outputX,tileSideLengthY=outputY,x=startPointX,y=startPointY)
            k+=1
            startPointY+=strideY
            pass
        startPointX+=strideX
        pass
    result = bigPic[xlen:2*xlen,ylen:2*ylen,:]/((outputX/strideX)*(outputY/strideY))
    return result

###Here is one example
### %matplotlib inline # add this line if you are using a notebook
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

#smpID = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e'
#tmpIm = plt.imread("../input/stage1_train/"+smpID+"/images/"+smpID+'.png')

#plt.imshow(tmpIm) # add this line if you are using a notebook. It's just a preview of the original image.

import pickle
train_df = pickle.load(open("train_df.p","rb"))
test_df = pickle.load(open("test_df.p","rb"))

train_f_df = train_df[train_df['hsv_cluster']==0]
train_h_df = train_df[train_df['hsv_cluster']==1]
train_b_df = train_df[train_df['hsv_cluster']==2]
test_f_df = test_df[test_df['hsv_cluster']==0]
test_h_df = test_df[test_df['hsv_cluster']==1]
test_b_df = test_df[test_df['hsv_cluster']==2]

def X_gen(img_series):
    for i in range(len(img_series)):
        if i == 0:
            InputDataset = InputGeneration(img=img_series[i],inputX=128,inputY=128,outputX=100,outputY=100,strideX=50,strideY=50,reflection=False)
        else:
            tmp = InputGeneration(img=img_series[i],inputX=128,inputY=128,outputX=100,outputY=100,strideX=50,strideY=50,reflection=False)
            InputDataset = np.concatenate((InputDataset,tmp),axis=0)
    return InputDataset
def y_gen(label_series):
    for i in range(len(label_series)):
        if i == 0:
            InputDataset = InputGeneration(img=label_series[i],inputX=100,inputY=100,outputX=100,outputY=100,strideX=50,strideY=50,reflection=False)
        else:
            tmp = InputGeneration(img=label_series[i],inputX=100,inputY=100,outputX=100,outputY=100,strideX=50,strideY=50,reflection=False)
            InputDataset = np.concatenate((InputDataset,tmp),axis=0)
    InputDataset = np.reshape(InputDataset,(InputDataset.shape[:3]))
    return InputDataset

img_series = list(train_f_df.loc[:,'Image'])
label_series = list(train_f_df.loc[:,'ImageLabel'])

trainX_fluore = X_gen(img_series)
trainy_fluore = y_gen(label_series)

#I = 289
#img = train_df.loc[I,'Image']
#label = train_df.loc[I,'ImageLabel']
#answer = np.zeros_like(img)
#
#def ExampleView(img,answer,y=False):
#    InputDataset = InputGeneration(img=img,inputX=128,inputY=128,outputX=100,outputY=100,strideX=50,strideY=50,reflection=False)
#    print(InputDataset.shape)
#    OutputDataset = fakeProcess(InputDataset,inputX=128,inputY=128,outputX=100,outputY=100)
#    print(OutputDataset.shape)
#    final=OutputStitch(answer,OutputDataset,strideX=50,strideY=50)
#    plt.figure()
#    plt.imshow(final)
#    if y: InputDataset = np.reshape(InputDataset,(InputDataset.shape[:3]))
#    
#    fig,ax = plt.subplots(3,3)
#    for i in range(3):
#        for j in range(3):
#            ax[i,j].imshow(InputDataset[i*3+j])
#            ax[i,j].axis('off')
#    plt.show()
#
#ExampleView(img,answer)
#ExampleView(label,answer,y=True)