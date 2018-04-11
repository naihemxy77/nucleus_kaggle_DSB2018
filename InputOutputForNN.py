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
        #if row['hsv_cluster'] == 0:
        #    img = data_norm.minmax_norm(img)
        #else:
        img = data_norm.invert_norm(img)
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

def sub_fragments_extract_rot(InputDim=(128,128),OutputDim=(100,100),Stride=(50,50),image_type='all',train=True,reflection=False):
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
        img = np.dstack((np.rot90(img[:,:,0]), np.rot90(img[:,:,1]),np.rot90(img[:,:,2])))
        #if row['hsv_cluster'] == 0:
        #    img = data_norm.minmax_norm(img)
        #else:
        img = data_norm.invert_norm(img)
        ImageId = row['ImageId']
        ImageShape = img.shape[:2]
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
