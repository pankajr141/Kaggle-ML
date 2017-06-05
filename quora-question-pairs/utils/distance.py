'''
Created on 14-May-2017

@author: amuse
'''
import re
import nltk
import math
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def getJaccordScore(sentence1, sentence2, removeStopWords=False, stemming=False, lemmatize=False):    
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    lmtzr = WordNetLemmatizer()
    sent1 = nltk.word_tokenize(sentence1)
    sent2 = nltk.word_tokenize(sentence2)
    
    sent1 = [ unicode(word, errors='ignore') for word in sent1]
    sent2 = [ unicode(word, errors='ignore') for word in sent2]

    #print sent1
    if removeStopWords:
        sent1 = [word for word in sent1 if word.lower() not in stop_words]
        sent2 = [word for word in sent2 if word.lower() not in stop_words]
        #[porter.stem(i.lower()) for i in sent1 if i.lower() not in stop_words]
    
    if stemming:
        sent1 = [porter.stem(word.lower()) for word in sent1]
        sent2 = [porter.stem(word.lower()) for word in sent2]

    #print sent1
    #print sent2    
    if lemmatize:
        sent1 = [lmtzr.lemmatize(word.lower()) for word in sent1]
        sent2 = [lmtzr.lemmatize(word.lower()) for word in sent2]
    #print sent1
    #print sent2
    sent1_ = np.array(sent1)
    sent1_ = np.reshape(sent1, (len(sent1), 1))

    sent2_ = np.array(sent2)
    sent2_ = np.reshape(sent2, (len(sent2), 1))
    
    sent1, sent2 = set(sent1), set(sent2)
    jaccordScore = len(sent1.intersection(sent2))/float(len(sent1.union(sent2)))

    #print sent1_.shape
    #print sent2_.shape
    #print scipy.spatial.distance.cdist(sent1_, sent2_, 'jaccard')
    return jaccordScore
    
def getCosineScore(sentence1, sentence2):
    WORD = re.compile(r'\w+')
    vec1 = Counter(WORD.findall(sentence1))
    vec2 = Counter(WORD.findall(sentence2))
    print vec1
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
