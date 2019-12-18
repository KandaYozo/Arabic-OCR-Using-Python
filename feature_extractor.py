import numpy as np
import cv2
from scipy import stats
from heapq import nsmallest

def vertical_Proj(img_binary):
    
    #img_binary = cv2.flip(img_binary, 1)
    img_binary = np.array(img_binary)
    img = np.copy(img_binary)
    VP = (np.sum(img, axis = 0))//255
    
    thresh = stats.mode(VP[VP != 0])[0][0]
    VP2 = VP[VP != 0]
    max_val = max(VP.tolist())
    min_val = min(VP2.tolist())
    avg_val = (max_val + min_val)//2
    second_min = nsmallest(2, VP2.tolist())[-1]
    img[:,VP<=second_min] = 0
        
    #resized = cv2.resize(img, (600,250), interpolation = cv2.INTER_NEAREST)        
    cv2.imshow('Trial', img)
    #cv2.imshow('Trial', resized)
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