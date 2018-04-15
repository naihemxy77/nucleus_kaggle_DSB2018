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

NNN = sys.argv[1]

print(NNN)

cutoff = 0.5

submissionfilename = 'Ensemble_stage2_submission1'

start = int(NNN)*50
end = min((int(NNN)+1)*50,3019)
#Best unet
print('read in best unet results...')
test_label_unet_tot = pickle.load(open( "../inputs/UnetRes.p","rb" ))
test_label_unet_tot = pd.DataFrame(test_label_unet_tot, columns=['ImageId','ImageOutput'])
test_label_unet = test_label_unet_tot.loc[start:end,:]
del test_label_unet_tot
#Best hollow
print('read in best zoomnet results...')
test_label_zoom_tot = pickle.load(open( "../inputs/ZoomNetAllRes.p","rb" ))
test_label_zoom_tot = pd.DataFrame(test_label_zoom_tot, columns=['ImageId','ImageOutput'])
test_label_zoom = test_label_zoom_tot.loc[start:end,:]
del test_label_zoom_tot
#Best hollow
print('read in best hollow results...')
test_label_hollow_tot = pickle.load(open( "../inputs/ZoomNetHollowRes.p","rb" ))
test_label_hollow_tot = pd.DataFrame(test_label_hollow_tot, columns=['ImageId','ImageOutput'])
test_label_hollow = test_label_hollow_tot.loc[start:end,:]
del test_label_hollow_tot

final_label = []
for i in list(test_label_unet.index):
    unfinished = 0 ####
    print(str(i)+"th image is being processed...")
    unet_id = test_label_unet.ImageId[i]
#    zoom_id = test_label_zoom.ImageId[i]
#    hollow_id = test_label_hollow.ImageId[i]
#    if unet_id == zoom_id and unet_id == hollow_id:
    img_unet = test_label_unet.loc[test_label_unet['ImageId']==unet_id,'ImageOutput'].item()
    img_zoom = test_label_zoom.loc[test_label_zoom['ImageId']==unet_id,'ImageOutput'].item()
    img_hollow = test_label_hollow.loc[test_label_hollow['ImageId']==unet_id,'ImageOutput'].item()

    img_combined = (img_unet+img_zoom+img_hollow)/3
    img_combined = np.where(img_combined>cutoff,1,0)
    
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
    
    p = multiprocessing.Process(target=sn.aggressiveLabel,args=(img_combined,relist))
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
        label_i = label(img_combined)
    else:
        label_i = relist[0]
    #label_i = sn.aggressiveLabel(img_combined.squeeze()) 
    final_label.append((unet_id,label_i))
    test_label_unet = test_label_unet.drop(test_label_unet[test_label_unet.ImageId==unet_id].index)
    test_label_zoom = test_label_zoom.drop(test_label_zoom[test_label_zoom.ImageId==unet_id].index)
    test_label_hollow = test_label_hollow.drop(test_label_hollow[test_label_hollow.ImageId==unet_id].index)

final_label = pd.DataFrame(final_label, columns=['ImageId','ImageLabel'])
pickle.dump(final_label,open( "Average_Three_0414"+str(NNN)+".p","wb" ))


