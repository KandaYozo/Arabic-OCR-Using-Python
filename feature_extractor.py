import numpy as np
import cv2

#Function expects the image in binary
def extract(img):
    
    H = cv2.Sobel(img, cv2.CV_8U, 0, 2)
    V = cv2.Sobel(img, cv2.CV_8U, 2, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 15))
    V = cv2.erode(V, kernel, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 40))
    V = cv2.dilate(V, kernel, iterations=10)
    image_Outer_Counters = H+V

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 1))
    V = cv2.dilate(V, kernel, iterations=3)

    left_parts = img - V

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 1))
    left_parts = cv2.erode(left_parts, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 50))
    left_parts = cv2.dilate(left_parts, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 40))
    left_parts = cv2.erode(left_parts, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 10))
    left_parts = cv2.dilate(left_parts, kernel, iterations=1)

    before_lineFix = left_parts

    cv2.imshow('Trial For before', before_lineFix)
    lines = cv2.HoughLinesP(left_parts,1,np.pi/180,10,150,10)
    print(len(lines))
    for line in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[line]:
            if(x2-x1) == 0:
                cv2.line(left_parts,(x1,y1),(x2,y2),(255,255,255),5) 
    cv2.imshow('Trial For Letters', left_parts)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(left_parts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours,key=lambda x: cv2.contourArea(x))
    for cnt in contours[0:10]:
        x, y, w, h = cv2.boundingRect(cnt)
        fx = x+w
        fy = y+h
        cv2.rectangle(img, (x, y), (fx, fy), (0, 255, 0), 2)
        img = 255 - img
        #cv2.imshow('Trial For Letters', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
  
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)) 
    
    #If Word is 255 - then erode
    new_img = cv2.erode(img, kernel, iterations= 5)
    #else
    #new_img = cv2.dilate(img, kernel, iterations= 1)
    
    contours, hierarchy = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    
    #cv2.imshow('Trial For Letters',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) 
    new_img = cv2.erode(new_img, kernel, iterations= 2)
    
    #cv2.imshow('Trial After erode',new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    



    
    
    