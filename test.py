import numpy as np
import glob
from feature_extractor import extract
import cv2
import matplotlib.pyplot as plt

# input_image = cv2.imread('./Test Data Set/test.png')

# input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# ret,img_binary = cv2.threshold(input_image, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
def vertical_Proj(img_binary):
    hist_list = np.zeros(img_binary.shape[1])
    print(img_binary.shape)
    for i in range(0, img_binary.shape[1]):
        for j in range(0, img_binary.shape[0]):
            if img_binary[j,i] == 255:
                hist_list[i] += 1
    print(hist_list)
    for i in range(0, img_binary.shape[1]):
        for j in range(0, img_binary.shape[0]):
            if hist_list[i] <=35:
                img_binary[j, i] = 0
    cv2.imshow('Proj', img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()