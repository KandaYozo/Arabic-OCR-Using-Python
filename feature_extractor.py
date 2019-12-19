import cv2
import imutils
import numpy as np
from scipy import stats
from imutils import contours

####################################################

def correct_skew(thresh):
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
	    angle = -(90 + angle)
    else:
	    angle = -angle
    
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def getVerticalProjection(img):
    return (np.sum(img, axis = 0))//255

def getHorizontalProjection(img):
    return (np.sum(img, axis = 1))//255

def DetectLines(line_img):
    input_image = correct_skew(line_img) 
    _,thresh = cv2.threshold(input_image, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,5))
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
    
    contours_initial, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    im2 = input_image.copy()
    
    Detected_Lines = []
    # sort the contours
    for method in ("right-to-left","top-to-bottom"):
	    contours_initial, boundingBoxes = contours.sort_contours(contours_initial, method=method)
    for cnt in contours_initial:
            x, y, w, h = cv2.boundingRect(cnt)
            fx = x+w
            fy = y+h
            cv2.rectangle(im2, (x, y), (fx, fy), (100, 100, 100), 2)
            trial_image = input_image[y:fy,x:fx]
            #trial_image[trial_image < 255] = 0
            #trial_image = 255 - trial_image
            Detected_Lines.append(trial_image)
    #cv2.imshow("Line",im2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return Detected_Lines

def DetectWords(line_img):
    input_image = correct_skew(line_img) 
    _,thresh = cv2.threshold(input_image, 127, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)
    contours_initial, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    Detected_Words = []
    # sort the contours
    for method in ("right-to-left"):
	    contours_initial, boundingBoxes = contours.sort_contours(contours_initial, method=method)
    for cnt in contours_initial:
            x, y, w, h = cv2.boundingRect(cnt)
            fx = x+w
            fy = y+h
            
            trial_image = input_image[y:fy,x:fx]
            
            #cv2.imshow("Detected Word",trial_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    
            Detected_Words.append(trial_image)
    return Detected_Words

def BaselineDetection(line_img): #4
    PV = []
    BLI = 0
    _,thresh_img = cv2.threshold(line_img,127,255,cv2.THRESH_BINARY_INV)
    thresh_img = np.asarray(thresh_img)
    
    HP = getHorizontalProjection(thresh_img)
    PV_Indices = (HP > np.roll(HP,1)) & (HP > np.roll(HP,-1))
    
    for i in range(len(PV_Indices)):
        if PV_Indices[i] == True:
            PV.append(HP[i])
    MAX = max(PV)
    
    for i in range(len(HP)):
        if HP[i] == MAX:
            BLI = i
            
    return BLI

def FindingMaximumTransitions(line_img, BLI): #5
    MaxTransitions = 0
    MTI = BLI
    i = BLI
    while i < line_img.shape[0]:
        CurrentTransitions = 0
        FLAG = 0
        j = 0
        while j < line_img.shape[1]:
            if line_img[i,j]==1 and FLAG == 0:
                CurrentTransitions += 1
                FLAG = 1
            elif line_img[i,j]!=1 and FLAG == 1:
                FLAG = 0
            j += 1
        #END WHILE
        if CurrentTransitions >= MaxTransitions:
             MaxTransitions = CurrentTransitions
             MTI = i
        i += 1
    #END WHILE    
    return MTI


def returnToBGR(image):
    return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)


#WIP:
def CutPointIdentification(line_img,BLI,MTI):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    openingImage = cv2.morphologyEx(line_img, cv2.MORPH_OPEN, kernel)
    VP = getVerticalProjection(line_img)
    pass