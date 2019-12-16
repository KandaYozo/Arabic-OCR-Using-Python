import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

def vertical_Proj(img_binary):
    
    img_binary = cv2.flip(img_binary, 1)
    
    img_binary = np.array(img_binary)
    img = np.copy(img_binary)
    img = img.astype(int)
    img = img//255
    
    img = np.sum(img, axis = 0)
    print(img)
    
    thresh = stats.mode(img[img != 0])[0][0]
    #print("Thresh is: {}".format(thresh))
    #img_binary[: , img <= thresh] = 0
    #cv2.imshow('T', img_binary)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    '''
    for i in range(d.shape[0]):
        if d[i] > 10000:
            cv2.imshow('T', img_binary[:,:i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    '''
    
    return img