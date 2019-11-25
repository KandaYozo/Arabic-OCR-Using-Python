import numpy as np
import cv2

#Function expects the image in binary
def extract(img):
    
    H = cv2.Sobel(img, cv2.CV_8U, 0, 2)
    V = cv2.Sobel(img, cv2.CV_8U, 2, 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)) 
    
    #If Word is 255 - then erode
    new_img = cv2.erode(img, kernel, iterations= 5)
    #else
    #new_img = cv2.dilate(img, kernel, iterations= 5)
    
    contours, hierarchy = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    
    
    #cv2.imshow('Trial For Letters',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) 
    new_img = cv2.erode(new_img, kernel, iterations= 2)
                
    #cv2.imshow('Trial After Erosion',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    