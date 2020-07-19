import random
import numpy as np

def train_or_val(ratio_train):
    '''
    This function returns "train" or "val" depending on ratio_train
    Argument: training dataset ratio (float) 0.0~1.0
    Usage:
    import mylib.utils as myutils
    if train_or_val() == "val":
        shutil.move(imgfile, dirname_dataset_val+"\\" + str(j) + "/")
    '''
    s_train_or_val = "train"
    rn = np.random.rand()
    if rn <= ratio_train:s_train_or_val = "train"
    elif rn > ratio_train:s_train_or_val = "val"
    return s_train_or_val