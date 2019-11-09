import numpy as np
import glob
import skimage.io as io
from skimage.color import rgb2gray
from feature_extractor import extract

Number_Of_Files = 1 #Sample of Files to check on

Actual = 0
Detected = 0
gen =  glob.iglob("./Pattern Data Set/scanned/*.png")
for i in range(Number_Of_Files):
    py = next(gen)
    text_name = py.split("\\")[1].split('.')[0] + '.txt'
    img = io.imread(py)
    img = rgb2gray(img)
    _, number = extract(img)
    file = open('./Pattern Data Set/text/'+text_name, "r",  encoding="utf-8")
    wordcount = len(file.read().split())
    #print('Actual:{}, Detected:{}'.format(wordcount, number))
    Actual += wordcount
    Detected += number

Accuracy = 100 - abs((Actual - Detected)/ Actual) * 100
print('Total Actual:{}, Total Detected:{}, Accuracy:{}%'.format(Actual, Detected, Accuracy))