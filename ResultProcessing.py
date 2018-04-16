import pandas as pd
import numpy as np
import os

def resultCombination(fid,i):
    i = int(i)
    tmpOutput = pd.read_pickle("Average_Three_"+fid+'_'+str(i)+'.p')
    #tmpOutput = pd.DataFrame(tmpOutput,columns=['ImageId','ImageOutput'])
    return tmpOutput

def resultConcate(fid,start,end):
    firstFrame = resultCombination(fid,start)
    i=start+1
    while(i<end):
        tmpFrame = resultCombination(fid,i)
        firstFrame = pd.concat([firstFrame,tmpFrame],ignore_index=True)
        i+=1
        pass
    firstFrame.drop_duplicates(subset=['ImageId'],inplace=True,keep='first')
    firstFrame = firstFrame.reset_index()
    print('After concatenation: ',firstFrame.shape)
    return firstFrame
