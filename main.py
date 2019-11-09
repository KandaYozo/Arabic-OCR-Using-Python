import numpy as np
import glob
import skimage.io as io
from skimage.color import rgb2gray
from feature_extractor import extract

'''
Images_Path = './Pattern Data Set/scanned/'
Text_Path = './Pattern Data Set/text/'
'''

Images_Path = './Test Data Set/'
Text_Path = './Test Data Set/'

Number_Of_Files = 1 #Sample of Files to check on

Actual = 0
Detected = 0
#gen =  glob.iglob(Images_Path + "*.png")
for i in range(Number_Of_Files):
    #py = next(gen)
    py = './Test Data Set\\test.png'
    text_name = py.split("\\")[1].split('.')[0] + '.txt'
    print(py)
    img = io.imread(py)
    img = rgb2gray(img)
    _, number = extract(img)
    file = open(Text_Path + text_name, "r",  encoding="utf-8")
    wordcount = len(file.read().split())
    #print('Actual:{}, Detected:{}'.format(wordcount, number))
    Actual += wordcount
    Detected += number

Accuracy = 100 - abs((Actual - Detected)/ Actual) * 100
print('Total Actual:{}, Total Detected:{}, Accuracy:{}%'.format(Actual, Detected, Accuracy))