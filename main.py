import numpy as np
import glob
import cv2
import imutils
import matplotlib.pyplot as plt

from imutils import contours
from feature_extractor import vertical_Proj
from helper_functions import correct_skew

predefined_word_width = 300
predefined_word_height = 800

#Images_Path = './Pattern Data Set/scanned/'
#Text_Path = './Pattern Data Set/text/'
Images_Path = './Test Data Set/'

Number_Of_Files = 1 #Sample of Files to check on           
gen =  glob.iglob(Images_Path + "*.png")
for i in range(Number_Of_Files):
    py = next(gen)
    input_image = cv2.imread(py)
    
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #Correct image if it is rotated:
    input_image = correct_skew(input_image)
    
    ret,thresh1 = cv2.threshold(input_image, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    contours_initial, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = input_image.copy()
    
    # sort the contours
    for method in ("top-to-bottom","right-to-left"):
	    contours_initial, boundingBoxes = contours.sort_contours(contours_initial, method=method)
    Total_Projections = []
    for cnt in contours_initial:
            x, y, w, h = cv2.boundingRect(cnt)
            fx = x+w
            fy = y+h
             
            cv2.rectangle(im2, (x, y), (fx, fy), (0, 255, 0), 2)
            trial_image = np.array(input_image)
            
            resized = trial_image[y:fy,x:fx]
            # resize image
            dim = (predefined_word_height,predefined_word_width)
            trial_image[trial_image < 255] = 0
            trial_image = 255 - trial_image
            resized = cv2.resize(trial_image[y:fy,x:fx], dim, interpolation = cv2.INTER_AREA)
            
            cv2.imshow('T', resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            VP = vertical_Proj(resized)
            #Total_Projections.append(VP)
    

