##from: https://www.kaggle.com/glenslade/alternative-metrics-kernel

##### Call IoU_Loss(input_mask_list, index, df) to calculate IoU losses
##### given predicted masks list and corresponding imageIds. df should 
##### at least contains the column of 'ImageId'; with index to be a 
##### a list of df indexes corresponding to predicted masks

import skimage
from skimage.morphology import label # label regions
import numpy as np

def IoU_Loss(input_mask_list,index,df):
    assert len(input_mask_list) == len(list(index)), "Length of input_mask and id_list should be the same."
    p_list = []
    for i in range(len(input_mask_list)):
        input_mask = input_mask_list[i]
        mid = df.loc[index[i],'ImageId']
        mfile = "../input/stage1_train/{}/masks/*.png".format(mid)
        masks = skimage.io.imread_collection(mfile).concatenate()
        num_masks, height, width = masks.shape
        # Make a ground truth array and summary label image
        y_true = np.zeros((num_masks, height, width), np.uint16)
        y_true[:,:,:] = masks[:,:,:] // 255  # Change ground truth mask to zeros and ones
        labels = np.zeros((height, width), np.uint16)
        labels[:,:] = np.sum(y_true, axis=0)  # Add up to plot all masks
        if (len(input_mask.shape)==3 and input_mask.shape[0] == 1) or len(input_mask.shape)==2:
            if np.unique(input_mask).shape[0] < 3:
                lab_img = label(input_mask>0)
                if lab_img.max()<1:
                    lab_img[0,0] = 1 # ensure at least one prediction per image
            else: lab_img = input_mask
            label_set = np.unique(lab_img)
            label_set = np.delete(label_set,0)
            y_pred = np.zeros((len(label_set), height, width), np.uint16)
            for i in range(len(label_set)):
                y_pred[i] = lab_img==label_set[i]
        else: y_pred = input_mask
        # Compute number of objects
        num_true = len(y_true)
        num_pred = len(y_pred)
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
        # Loop over IoU thresholds
        p = []
        for t in np.arange(0.5, 1.0, 0.05):
            matches = iou > t
            tp = np.count_nonzero(matches)  # True positives
            fp = num_pred - tp  # False positives
            fn = num_true - tp  # False negatives
            p.append(tp / (tp + fp + fn))
        p_list.append(p)
    p_list = np.array(p_list)
    return np.mean(p_list),p_list