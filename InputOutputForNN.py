#Here are the essential functions
import numpy as np
import pandas as pd
import pickle
import data_norm

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
    
def MidExtractProcess(inputData,inputX,inputY,outputX,outputY):
    # Just extract the middle part from input. 
    nn,_,_,nlayers = inputData.shape
    res = np.zeros((nn,outputX,outputY,nlayers))
    res[:,:,:,:] = inputData[:,int((inputX-outputX)/2):int((inputX+outputX)/2),int((inputY-outputY)/2):int((inputY+outputY)/2),:]
    return res

def OutputStitch(img_shape,output,strideX,strideY):
    # Each pixel returned is stacked (outputX/strideX)*(outputY/strideY) times. 
    #img = np.atleast_3d(img)
    xlen,ylen = img_shape
    nn,outputX,outputY,nlayers = output.shape
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

def sub_fragments_extract(InputDim=(128,128),OutputDim=(100,100),Stride=(50,50),image_type='all',train=True,reflection=False):
    print('Loading pickle file ...')
    if train:
        df_all = pickle.load(open("./inputs/train_df.p","rb"))
    else:
        df_all = pickle.load(open("./inputs/test_df.p","rb"))
    if image_type == 'all':
        df = df_all
    elif image_type == 'fluo':
        df = df_all[df_all['hsv_cluster']==0]
    elif image_type == 'histo':
        df = df_all[df_all['hsv_cluster']==1]
    elif image_type == 'bright':
        df = df_all[df_all['hsv_cluster']==2]
    elif image_type == 'others':
        df = df_all[df_all['hsv_cluster']==4]
    else:
        raise ValueError('image_type has to be all, fluo, histo or bright ...')
    inputX,inputY = InputDim
    outputX,outputY = OutputDim
    strideX,strideY = Stride
    details = []
    print('Start to generate fragment datasets ...')
    i = 1
    for index,row in df.iterrows():
        print('{:d}th image is processing ... ({:d}/{:d})'.format(index,i,df.shape[0]))
        img = row['Image']
        if row['hsv_cluster'] == 1:
            img = data_norm.rgb_norm(img)
        else:
            img = data_norm.minmax_norm(img)
        ImageId = row['ImageId']
        ImageShape = row['Image'].shape[:2]
        X = InputGeneration(img=img,inputX=inputX,inputY=inputY,outputX=outputX,outputY=outputY,strideX=strideX,strideY=strideY,reflection=reflection)
        if train:
            y = InputGeneration(img=row['ImageLabel'],inputX=inputX,inputY=inputY,outputX=outputX,outputY=outputY,strideX=strideX,strideY=strideY,reflection=reflection)
            #n,h,w,l = y.shape
            #y = np.reshape(y,(n,h,w))
            info = (ImageId,ImageShape,X,y)
        else: info = (ImageId,ImageShape,X)
        details.append(info)
        i = i+1
    if train:
        COL = ['ImageId','ImageShape','X','y']
    else: COL = ['ImageId','ImageShape','X']
    Piece_data = pd.DataFrame(details, columns=COL)
    return Piece_data

####Here is one example

#### %matplotlib inline # add this line if you are using a notebook
#import numpy as np
#import pandas as pd
#import os
#from matplotlib import pyplot as plt
#
#smpID = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e'
#tmpIm = plt.imread("../input/stage1_train/"+smpID+"/images/"+smpID+'.png')
#
##plt.imshow(tmpIm) # add this line if you are using a notebook. It's just a preview of the original image.
#
#InputDataset = InputGeneration(img=ch0,inputX=100,inputY=100,outputX=80,outputY=80,strideX=40,strideY=40)
#print(InputDataset.shape)
#
#OutputDataset = fakeProcess(wtf3,inputX=100,inputY=100,outputX=80,outputY=80)
#print(OutputDataset.shape)
#
#OutputImage = OutputStitch(img=ch0,output=wtf4,strideX=40,strideY=40)
##plt.imshow(OutputImage)

##Further illustrations
#Piece_data = sub_pieces_extract(InputDim=(128,128),OutputDim=(100,100),Stride=(50,50),image_type='histo',train=True,reflection=False)
#h = 7
#w = 8
#I = 10
#x = Piece_data.loc[I,'X']
#y = Piece_data.loc[I,'y']
#image_shape = Piece_data.loc[I,'ImageShape']
#fig,ax = plt.subplots(h,w)
#for i in range(h):
#    for j in range(w):
#        ax[i,j].imshow(x[w*i+j])
#        ax[i,j].axis('off')
#plt.show()
#y_show = y.reshape((h*w,128,128))
#fig,ax = plt.subplots(h,w)
#for i in range(h):
#    for j in range(w):
#        ax[i,j].imshow(y_show[w*i+j])
#        ax[i,j].axis('off')
#plt.show()
#
#y_pred=ionn.MidExtractProcess(inputData=y,inputX=128,inputY=128,outputX=100,outputY=100)
#OutputImage = ionn.OutputStitch(img_shape=image_shape,output=y_pred,strideX=50,strideY=50)
#O_Image_show = OutputImage.reshape(OutputImage.shape[0],OutputImage.shape[1])
#plt.imshow(O_Image_show)