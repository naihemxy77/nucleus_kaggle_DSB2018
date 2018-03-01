import numpy as np
import pandas as pd
import iou_loss
import SplitNuclei as sn
from skimage.morphology import label # label regions
import skimage
import matplotlib.pyplot as plt
import data_process
import random
import pickle

def SplitNucleiEval(df,Index):
    Index = np.array(Index)
    masks_list = []
    masks_splitted_list = []
    for i in range(len(Index)):
        mask = df.loc[Index[i],'ImageLabel']
        mask_splitted = sn.reLabel(mask)
        masks_list.append(mask)
        masks_splitted_list.append(mask_splitted)
    res,p_list=iou_loss.IoU_Loss(masks_list,Index,df)
    res_splitted,p_list_splitted=iou_loss.IoU_Loss(masks_splitted_list,Index,df)
    
    print('Overall IoU Score of original masks:{:1.3f}'.format(res))
    print('Overall IoU Score of splitted masks:{:1.3f}'.format(res_splitted))
    
    match_ori = np.sum(p_list,axis=1)/10
    match_spl = np.sum(p_list_splitted,axis=1)/10
    details = []
    for i in range(len(Index)):
        info = (Index[i],match_ori[i],match_spl[i],match_spl[i]-match_ori[i],df.loc[Index[i],'ImageId'])
        details.append(info)
    res_df = pd.DataFrame(details, columns=['Index','IoU_Score_before_split','IoU_Score_after_split','IoU_improvement_after_split','ImageId'])
    res_df=res_df.sort_values(by=['IoU_improvement_after_split','IoU_Score_before_split'],ascending=False)
    res_df.to_csv('SplitNucleiResults.csv', index = False)
    id_sim = np.where(np.abs(match_ori-match_spl)<=0.01)[0]
    id_bet = np.where(match_spl-match_ori>0.01)[0]
    id_wor = np.where(match_ori-match_spl>0.01)[0]
    print('{:d} Images similar in both cases.'.format(len(id_sim)))
    print('{:d} Images better in original case.'.format(len(id_wor)))
    print('{:d} Images better in splitted case.'.format(len(id_bet)))
    
    return res_df

def ExampleView(df,Index):
    mask_pred = df.loc[Index,'ImageLabel']
    mask_splitted = sn.reLabel(mask_pred)
    
    mfile = "../input/stage1_train/{}/masks/*.png".format(df.loc[Index,'ImageId'])
    masks = skimage.io.imread_collection(mfile).concatenate()
    
    num_masks, height, width = masks.shape
    
    # Make a ground truth array and summary label image
    y_true = np.zeros((num_masks, height, width), np.uint16)
    y_true[:,:,:] = masks[:,:,:] // 255  # Change ground truth mask to zeros and ones
    
    true_split = np.zeros((height, width), np.uint16)
    for i in range(num_masks):
        true_split[y_true[i]==1] = i+1
    
    lab_img = label(mask_pred>0)
    if lab_img.max()<1:
            lab_img[0,0] = 1 # ensure at least one prediction per image
    y_pred = np.zeros((lab_img.max(), height, width), np.uint16)
    for i in range(lab_img.max()):
        y_pred[i] = lab_img==i+1
    
    y_pred_splitted = np.zeros((mask_splitted.max(), height, width), np.uint16)
    for i in range(mask_splitted.max()):
        y_pred_splitted[i] = mask_splitted==i+1
    
    # Show simulated predictions
    fig,ax = plt.subplots(1,3,figsize=(10,30))
    ax[0].imshow(true_split)
    ax[0].set_title("True Label "+str(Index))
    ax[0].axis('off')
    ax[1].imshow(lab_img)
    ax[1].set_title("Original Label "+str(Index))
    ax[1].axis('off')
    ax[2].imshow(mask_splitted)
    ax[2].set_title("Splitted Label "+str(Index))
    ax[2].axis('off')
    plt.show()
    
    # Compute number of objects
    num_true = len(y_true)
    num_pred = len(y_pred)
    num_pred_splitted = len(y_pred_splitted)
    print("Number of true objects:", num_true)
    print("Number of predicted objects:", num_pred)
    print("Number of predicted objects (after split):", num_pred_splitted)
    
    # Compute iou score for each prediction
    iou = []
    for pr in range(num_pred):
        bol = 0  # best overlap
        bun = 1e-9  # corresponding best union
        for tr in range(num_true):
            olap = y_pred[pr] * y_true[tr]  # Intersection points
            osz = np.sum(olap)  # Add the intersection points to see size of overlap
            if osz > bol:  # Choose the match with the biggest overlap
                bol = osz
                bun = np.sum(np.maximum(y_pred[pr], y_true[tr]))  # Union formed with sum of maxima
        iou.append(bol / bun)
    iou2 = []
    for pr in range(num_pred_splitted):
        bol = 0  # best overlap
        bun = 1e-9  # corresponding best union
        for tr in range(num_true):
            olap = y_pred_splitted[pr] * y_true[tr]  # Intersection points
            osz = np.sum(olap)  # Add the intersection points to see size of overlap
            if osz > bol:  # Choose the match with the biggest overlap
                bol = osz
                bun = np.sum(np.maximum(y_pred_splitted[pr], y_true[tr]))  # Union formed with sum of maxima
        iou2.append(bol / bun)
    # Loop over IoU thresholds
    p = 0
    p2 = 0
    print("Original\t\t\t\tSplitted")
    print("Thresh\tTP\tFP\tFN\tPrec.\tTP\tFP\tFN\tPrec_splitted.")
    for t in np.arange(0.5, 1.0, 0.05):
        matches = iou > t
        matches2 = iou2 > t
        tp = np.count_nonzero(matches)  # True positives
        fp = num_pred - tp  # False positives
        fn = num_true - tp  # False negatives
        tp2 = np.count_nonzero(matches2)  # True positives
        fp2 = num_pred_splitted - tp2  # False positives
        fn2 = num_true - tp2  # False negatives
        p += tp / (tp + fp + fn)
        p2 += tp2 / (tp2 + fp2 + fn2)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, tp / (tp + fp + fn),tp2, fp2, fn2, tp2 / (tp2 + fp2 + fn2)))
    
    print("AP\t-\t-\t-\t{:1.3f}\t-\t-\t-\t{:1.3f}".format(p / 10, p2/10))
    print('One worst matched nuclei in orignal mask file ...')
    # Show one example
    I = np.argmin(iou)
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(y_pred[I])
    ax[0].set_title("Original Label "+str(I))
    ax[0].axis('off')
    ax[1].imshow(y_pred_splitted[I])
    ax[1].set_title("Splitted Label "+str(I))
    ax[1].axis('off')

###################MAIN#######################################################

#Some problematic image index to see: 535, 606

##Read in data
train_df = pickle.load(open("train_df.p","rb"))
centroids = pickle.load(open("centroids.p","rb"))
test_df = pickle.load(open("test_df.p","rb"))

##Randomly sample a few images to evaluate; each index is row index of dataframe
#WholeIndex = np.arange(0,train_df.shape[0])
#Index = random.sample(set(WholeIndex), 5)
#Eval_Res = SplitNucleiEval(train_df,Index)

#Read in whole dataset for a thorough evaluation (take hours)
WholeIndex = np.arange(0,train_df.shape[0])
Eval_Res = SplitNucleiEval(train_df,WholeIndex)
pickle.dump(Eval_Res, open("Eval_Res.p","wb"))

##Show a particular example with one specified row index
#ExampleView(train_df,601)