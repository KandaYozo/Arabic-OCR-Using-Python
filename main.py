import numpy as np
import glob
from feature_extractor import extract
import cv2

'''
Images_Path = './Test Data Set/'
Text_Path = './Test Data Set/'
'''

Images_Path = './Pattern Data Set/scanned/'
Images_Path = './Test Data Set/'

Number_Of_Files = 1 #Sample of Files to check on

gen =  glob.iglob(Images_Path + "*.png")
for i in range(Number_Of_Files):
    py = next(gen)
    input_image = cv2.imread(py)
    
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    ret,thresh1 = cv2.threshold(input_image, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    #cv2.imshow('Thresholded Image', thresh1)
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    #cv2.imshow('After Dilation', dilation)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = input_image.copy()
    
    for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            fx = x+w
            fy = y+h
            
            cv2.rectangle(im2, (x, y), (fx, fy), (0, 255, 0), 2)
            trial_image = np.array(input_image)
            
            scale_percent = 100 # percent of original size
            width = int(trial_image.shape[1] * scale_percent / 100)
            height = int(trial_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(trial_image[y:fy,x:fx], dim, interpolation = cv2.INTER_AREA)
            resized[resized < 255] = 0
            
            #If I change 255 - resized then I will use Erosion inside extract
            resized = 255 - resized
            
            #cv2.imshow('Single Word', resized)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    
            extract(resized)
    
            
    #cv2.imshow('final', im2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
