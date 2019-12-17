import numpy as np
import cv2
from scipy import stats


def vertical_Proj(img_binary):
    
    #img_binary = cv2.flip(img_binary, 1)
    img_binary = np.array(img_binary)
    img = np.copy(img_binary)
    VP = (np.sum(img, axis = 0))//255
    thresh = stats.mode(VP[VP != 0])[0][0]
    VP2 = VP[VP != 0]
    VP2 = VP2.tolist()
    VP = VP.tolist()
    max_val = max(VP)
    min_val = min(VP2)
    avg_val = (max_val + min_val)//2
    actual_min = min_val
    VP = (np.sum(img, axis = 0))//255
    VP2 = VP[VP != 0]
    
    img[:,VP<=min_val] = 0
    while min_val < thresh:
        VP = (np.sum(img, axis = 0))//255
        VP2 = VP[VP != 0]
        VP2 = VP2.tolist()
        min_val = min(VP2)
        img[:,VP<=min_val] = 0
            
    cv2.imshow('Trial', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    SI = 0
    for i in range(len(VP)-1):
        
        if ((VP[i+1] - VP[i] > 0) and (VP[i+1] - VP[i] <= avg_val)) and VP[i+1] > thresh and VP[i] > thresh:
            
            cv2.imshow('Trial', img[:,:i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            SI = i
    '''                    
    return img