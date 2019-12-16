import numpy as np
import glob
import codecs

Text_Path = './Pattern Data Set/text/'
#Text_Path = './Test Data Set/'

def read_from_file(fileName):
    W = []
    f = open(fileName,"r", encoding='utf-8')
    fl =f.readlines()
    f.close()
    W = fl[0].split(' ')
    return W

def createDictionary(listOfWords,dictionaryWords):
    for word in listOfWords:
        if word in dictionaryWords:
            dictionaryWords[word] += 1
        else:
            dictionaryWords[word] = 1
        
def countWords(pathToText):
    
    wordsDictionary = {}
    
    Number_Of_Files = 20900 #Sample of Files to check on  
    gen =  glob.iglob(pathToText + "*.txt")
    for i in range(Number_Of_Files):
        py = next(gen)
        W = read_from_file(py)
        createDictionary(W,wordsDictionary)
        if i % 100 == 0:
            print("iteration.{}".format(i))
            print('Length of Dictionary:{}'.format(len(wordsDictionary)))
    
    return wordsDictionary        
        
#countWords(Text_Path)
        