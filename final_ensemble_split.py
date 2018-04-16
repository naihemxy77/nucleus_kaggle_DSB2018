
# coding: utf-8

# In[ ]:


import sys
import multiprocessing
import time
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
#from matplotlib.backends.backend_pdf import PdfPages
import PerfectSplit as sn
#from skimage.morphology import square,binary_closing
from pathlib import Path
home = str(Path.home())
#from GetInput import *
global unfinished
unfinished=0
from skimage.measure import label


# In[ ]:


NNN = sys.argv[1]

print(NNN)

cutoff = 0.5

submissionfilename = 'Redo_stage2_submission2'

start = int(NNN)*100
end = min((int(NNN)+1)*100,3019)

result = pd.read_pickle('../inputs/redo.p')

final_label = []
for i in list(result.index[start:end]):
    unfinished = 0 ####
    print(str(i)+"th image is being processed...")
    img_id = result.ImageId[i]
    img_combined = result.ImageLabel[i]
    
    if np.sum(img_combined)>0.95*img_combined.shape[0]*img_combined.shape[1]:
        img_combined = np.where(img_combined>0,0,1)
        img_combined[:5,:5] = 1
    if img_combined.max()==0:
        img_combined[:5,:5] = 1
    
    try:
        del manager
    except NameError:
        pass
    try:
        del p
    except NameError:
        pass
    try:
        del relist
    except NameError:
        pass
    
    manager = multiprocessing.Manager()
    relist = manager.list()
    
    p = multiprocessing.Process(target=sn.aggressiveLabel,args=(img_combined.squeeze(),relist))
    p.start()
    
    p.join(60)
    if p.is_alive():
        print("running... let's kill it...")
        # Terminate
        p.terminate()
        p.join()
        unfinished = 1
        pass
    
    if unfinished:
        valFake = sn.avgArea(img_combined.squeeze())
        if valFake > 300:
            label_i = np.zeros_like(img_combined.squeeze())
        else:
            label_i = label(img_combined.squeeze())
    else:
        label_i = relist[0]
    #label_i = sn.aggressiveLabel(img_combined.squeeze()) 
    final_label.append((img_id,label_i))

final_label = pd.DataFrame(final_label, columns=['ImageId','ImageLabel'])
final_label.to_pickle('Redo_Result_'+str(NNN)+".p")



