#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import pandas as pd
import submission_encoding
#import matplotlib.pyplot as plt
import pickle
#from matplotlib.backends.backend_pdf import PdfPages
import PerfectSplit as sn
#from skimage.morphology import square,binary_closing

cutoff = 0.5

submissionfilename = 'Ensemble_stage2_submission1'
#Best unet
print('read in best unet results...')
test_label_unet = pickle.load(open( "./inputs/UnetRes.p","rb" ))
test_label_unet = pd.DataFrame(test_label_unet, columns=['ImageId','ImageOutput'])
#Best hollow
print('read in best zoomnet results...')
test_label_zoom = pickle.load(open( "./inputs/ZoomNetAllRes.p","rb" ))
test_label_zoom = pd.DataFrame(test_label_zoom, columns=['ImageId','ImageOutput'])
#Best hollow
print('read in best hollow results...')
test_label_hollow = pickle.load(open( "./inputs/ZoomNetHollowRes.p","rb" ))
test_label_hollow = pd.DataFrame(test_label_hollow, columns=['ImageId','ImageOutput'])

final_label = []
for i in range(test_label_unet.shape[0]):
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
        
    label_i = sn.aggressiveLabel(img_combined.squeeze())
    final_label.append((unet_id,label_i))

final_label = pd.DataFrame(final_label, columns=['ImageId','ImageLabel'])
pickle.dump(final_label,open( "Average_Three_0414.p","wb" ))

submission_encoding.submission_gen(final_label, submissionfilename)