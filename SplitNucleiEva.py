import numpy as np
import iou_loss
import SplitNuclei as sn
from skimage.morphology import label # label regions
import skimage
import matplotlib.pyplot as plt
import data_process

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
    
    print('Overall IoU loss of original masks:{:1.3f}'.format(res))
    print('Overall IoU loss of splitted masks:{:1.3f}'.format(res_splitted))
    
    match_ori = np.sum(p_list,axis=1)/10
    match_spl = np.sum(p_list_splitted,axis=1)/10
    id_sim = np.where(np.abs(match_ori-match_spl)<=0.01)[0]
    id_bet = np.where(match_spl-match_ori>0.01)[0]
    id_wor = np.where(match_ori-match_spl>0.01)[0]
    print('{:d} Images similar in both cases.'.format(len(id_sim)))
    print('{:d} Images better in original case.'.format(len(id_wor)))
    print('{:d} Images better in splitted case.'.format(len(id_bet)))
    
    print('Images similar in both cases: ',Index[id_sim])
    print('Images better in original case: ',Index[id_wor])
    print('Images better in splitted case: ',Index[id_bet])
    
    return Index[id_sim],Index[id_wor],Index[id_bet]

def ExampleView(df,Index):
    mask_pred = df.loc[Index,'ImageLabel']
    mask_splitted = sn.reLabel(mask_pred)
    
    mfile = "../input/stage1_train/{}/masks/*.png".format(df.loc[Index,'ImageId'])
    masks = skimage.io.imread_collection(mfile).concatenate()
    
    num_masks, height, width = masks.shape
    
    # Make a ground truth array and summary label image
    y_true = np.zeros((num_masks, height, width), np.uint16)
    y_true[:,:,:] = masks[:,:,:] // 255  # Change ground truth mask to zeros and ones
    
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
    fig,ax = plt.subplots(1,2,figsize=(15,30))
    ax[0].imshow(lab_img)
    ax[0].set_title("Original Label "+str(Index))
    ax[0].axis('off')
    ax[1].imshow(mask_splitted)
    ax[1].set_title("Splitted Label "+str(Index))
    ax[1].axis('off')
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
    print("Thresh\tTP\tFP\tFN\tPrec.\tTP\tFP\tFN\tPrec_clean.")
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
df = data_process.data_process(datatype='train',label=True,cluster=False)

##Randomly sample a few images to evaluate; each index is row index of dataframe
#WholeIndex = np.arange(0,df.shape[0])
#Index = random.sample(set(WholeIndex), 5)
#id_sim,id_wor,id_bet = SplitNucleiEval(df,Index)

#Read in whole dataset for a thorough evaluation (take hours)
WholeIndex = np.arange(0,df.shape[0])
id_sim,id_wor,id_bet = SplitNucleiEval(df,WholeIndex)

##Show a particular example with one specified row index
##One example better in splitted case
#ExampleView(df,id_bet[random.randint(0,len(id_bet)-1)])
##One example worse in splitted case
#ExampleView(df,id_wor[random.randint(0,len(id_wor)-1)])

ExampleView(df,18)