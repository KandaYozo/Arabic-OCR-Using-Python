import os
import cv2
import numpy as np
from io import StringIO
from read_files import read_text_file
import feature_extractor as FE
from commonfunctions import *


input_image = cv2.imread("./Test Data Set/capr1.png")

all_words = FE.extractSeparateLettersWholeImage(input_image)

path = './Test Data Set/'
fileName = 'capr1'
lis = read_text_file(path,fileName)
print(lis)