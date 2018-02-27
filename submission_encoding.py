##from: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
##and   https://www.kaggle.com/kmader/nuclei-overview-to-submission

##### Call submission_gen(df, filename, cut_off = 0.5) to generate submission
##### file from a dataframe with ImageId and ImageLabel columns

from skimage.morphology import label # label regions
import numpy as np
import pandas as pd
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list (for one nuclei label)
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cut_off = 0.5):
    '''
    Apply skimage.morphology.label so that all connected regions are 
    assigned the same values. Then rle_encoding for each region.
    '''
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)

def submission_gen(df, filename, cut_off = 0.5):
    '''
    Generate a submission csv file from input dataframe df
    df should at least contains column ImageID and ImageLabel
    '''
    out_pred_list = []
    for _,row in df.iterrows():
        pred_encode = list(prob_to_rles(row['ImageLabel'], cut_off))
        for nuclei_code in pred_encode:
            out_pred_list+=[dict(ImageId=row['ImageId'], 
                             EncodedPixels = ' '.join(np.array(nuclei_code).astype(str)))]
    out_pred_df = pd.DataFrame(out_pred_list)
    out_pred_df[['ImageId', 'EncodedPixels']].to_csv(filename+'.csv', index = False)

##A simple test, uncomment if to try
#a1=np.array([[0,0,1],[1,0,1],[0,0,1]])
#a2=np.array([[1,0,0],[0,0,1],[0,1,0]])
#a3=np.array([[0,0,1],[1,0,0],[1,1,0]])
#x=[{'ImageId':'a1','ImageLabel':a1},{'ImageId':'a2','ImageLabel':a2},{'ImageId':'a3','ImageLabel':a3}]
#x=pd.DataFrame.from_dict(x)
#submission_gen(x, 'small_example')