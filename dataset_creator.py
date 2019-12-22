import os
import cv2
import numpy as np
import feature_extractor as FE
from commonfunctions import *
from read_files import read_text_file
import csv

def ShowImageCV2(image):
    for img in image:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def createArabicDictionary():
    D = {}
    D['ا'] = 0
    D['ب'] = 1
    D['ت'] = 2
    D['ث'] = 3
    D['ج'] = 4
    D['ح'] = 5
    D['خ'] = 6
    D['د'] = 7
    D['ذ'] = 8
    D['ر'] = 9
    D['ز'] = 10
    D['س'] = 11
    D['ش'] = 12
    D['ص'] = 13
    D['ض'] = 14
    D['ط'] = 15
    D['ظ'] = 16
    D['ع'] = 17
    D['غ'] = 18
    D['ف'] = 19
    D['ق'] = 20
    D['ك'] = 21
    D['ل'] = 22
    D['م'] = 23
    D['ن'] = 24
    D['ه'] = 25
    D['و'] = 26
    D['ي'] = 27
    D['لا'] = 29
    return D

def saveLettersToImages(letter,label):
    resized = cv2.resize(letter, (28,28), interpolation = cv2.INTER_AREA)
    # VP = FE.getVerticalProjection(resized)
    # HP = FE.getHorizontalProjection(resized)
    VP_ink,HP_ink = FE.Black_ink_histogram(letter)
    Com1,Com2 = FE.Center_of_mass(letter)
    CC = FE.Connected_Component(letter)
    r1,r2,r3,r4,r5,r6,r7,r8,r9,r10 = FE.ratiosBlackWhite(letter)
    hw = FE.height_over_width(letter)
    concat = [*VP_ink, *HP_ink]
    concat.append(Com1)
    concat.append(Com2)
    concat.append(CC)
    concat.append(r1)
    concat.append(r2)
    concat.append(r3)
    concat.append(r4)
    concat.append(r5)
    concat.append(r6)
    concat.append(r7)
    concat.append(r8)
    concat.append(r9)
    concat.append(r10)
    concat.append(hw)
    # concat = np.concatenate((VP_ink, HP_ink,Com1,Com2,CC,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10, hw), axis=0)
    concat.append(label)
    print(concat)
    with open("image_label_pair.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(concat)
  
def checkNumberOfSeparations(wordSeparationList,lettersOfWordList): #Expecting an ndarray and a list
    '''
    wordSeaparationList = single word segmentation
    lettersOfWordList = letters of word from text file
    '''
    checkBool = False
    
    numberofSegments = len(wordSeparationList)
    lettersOfWordList = np.array(lettersOfWordList)
    actualNumber = len(lettersOfWordList)
    if numberofSegments == actualNumber: # No action needed as number of segments same as number of letters
        checkBool = True
        return wordSeparationList,checkBool 
    
    if numberofSegments < actualNumber: # This means that the BLI value caused an error and the word was under segmented
        return wordSeparationList,checkBool
    
    if numberofSegments > actualNumber: #Oversegmented word but may be handled
        return wordSeparationList,checkBool

def createDataSet(images,labels):
    D = createArabicDictionary()
    i = 0
    for word in images:
        j = 0
        segmented_list, no_segmentation_error = checkNumberOfSeparations(word, labels[i])
        
        if no_segmentation_error:
            for l in labels[i]:
                label = D[l]
                saveLettersToImages(word[j], label)
                j+=1
        i += 1