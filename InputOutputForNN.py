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
    startPointX = xlen-outputX
    
    estimate = int((xlen+2*outputX)/strideX)*int((ylen+2*outputY)/strideY)*2
    tmp = np.zeros((estimate,inputX,inputY,nlayers))
    
    k = 0
    while(startPointX<xlen*2+outputX):
        startPointY = ylen-outputY
        while(startPointY<ylen*2+outputY):
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
    startPointX = xlen-outputX
    k = 0
    while(startPointX<xlen*2+outputX):
        startPointY = ylen-outputY
        while(startPointY<ylen*2+outputY):
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

smpID = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e'
tmpIm = plt.imread("../input/stage1_train/"+smpID+"/images/"+smpID+'.png')

#plt.imshow(tmpIm) # add this line if you are using a notebook. It's just a preview of the original image.

InputDataset = InputGeneration(img=ch0,inputX=100,inputY=100,outputX=80,outputY=80,strideX=40,strideY=40)
print(InputDataset.shape)

OutputDataset = fakeProcess(wtf3,inputX=100,inputY=100,outputX=80,outputY=80)
print(OutputDataset.shape)

OutputImage = OutputStitch(img=ch0,output=wtf4,strideX=40,strideY=40)
#plt.imshow(OutputImage)
