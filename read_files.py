import os
import cv2
import numpy as np
from io import StringIO



def get_words_from_file(path,fileName):
    wordList = []
    text = path+fileName
    t = np.loadtxt(text, dtype = 'str',encoding='utf-8', delimiter='\n')
    if t.shape == ():
        t = t.reshape(1,)
        for line in t[0].split(" "):
            wordList.append(line)
        return wordList
    for line in t:
        l = line.split(" ")
        wordList.extend(l)
    return wordList

def get_letters_from_word(wordList):
    lettersList = []
    for word in wordList:
        char = list(word)
        lettersList.append(char)
    return lettersList

def read_text_file(path,fileName):
    lis = get_words_from_file(path,fileName)
    lis2 = get_letters_from_word(lis)
    return lis2

path = './Test Data Set/'
fileName = 'test2.txt'
lis = read_text_file(path,fileName)
print(lis)