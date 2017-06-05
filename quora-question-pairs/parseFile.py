'''
Created on 13-May-2017

@author: amuse
'''
import os
import nltk
import pandas as pd
import sklearn
import numpy as np
import sys  
import scipy
from scipy.spatial import distance as scp_dist
import re
import time
import math
from multiprocessing.pool import Pool
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from utils import distance as lc_dist, similarity
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from utils import similarity
from utils import helper

reload(sys)  
sys.setdefaultencoding('utf8')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#stNER = StanfordNERTagger('data/english.all.3class.distsim.crf.ser.gz', 'data/stanford-ner.jar', encoding='utf-8')
#stPOS = StanfordPOSTagger('data/english-bidirectional-distsim.tagger', 'data/stanford-postagger.jar', encoding='utf-8') 
mode  = 'test'

trainFile = os.path.join("data", "train.csv")
#trainFile = os.path.join("data", "train_5K.csv")
testFile = os.path.join("data", "test.csv")
#testFile = os.path.join("data", "test_20K.csv")

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
lmtzr = WordNetLemmatizer()
WORD = re.compile(r'\w+')
wvGoogle = similarity.getGoogleWord2Vec()
wvQuora = similarity.getQuoraWord2Vec()
gloveModel = similarity.getGloveModel()
 
def nltkTokenize(sentence, removeStopWords=False, stemming=False, lemmatize=False):
    if 'lemmatize' in nltkTokenize.__dict__.keys():
        lemmatize = nltkTokenize.__dict__['lemmatize']
    if 'stemming' in nltkTokenize.__dict__.keys():
        stemming = nltkTokenize.__dict__['stemming']
    if 'removeStopWords' in nltkTokenize.__dict__.keys():
        removeStopWords = nltkTokenize.__dict__['removeStopWords']    

    sent = nltk.word_tokenize(sentence)
    sent = [unicode(str(word), errors='ignore') for word in sent]

    if removeStopWords:
        sent = [word for word in sent if word.lower() not in stop_words]
    
    if stemming:
        sent = [porter.stem(word.lower()) for word in sent]
   
    if lemmatize:
        sent = [lmtzr.lemmatize(word.lower()) for word in sent]

    return sent
    
def getPosTagFeatures(sentence1, sentence2, pos_tagger=None):
    #sent1 = nltk.pos_tag(nltk.word_tokenize(sentence1))
    sent1 = [unicode(str(word), errors='ignore') for word in nltk.word_tokenize(sentence1)]
    sent2 = [unicode(str(word), errors='ignore') for word in nltk.word_tokenize(sentence2)]
    if pos_tagger:
        sent1 = pos_tagger.tag(sent1)
        sent2 = pos_tagger.tag(sent2)

    else:
        sent1 = nltk.pos_tag(sent1)
        sent2 = nltk.pos_tag(sent2)
    
    sent1_nouns = map(lambda x: lmtzr.lemmatize(x[0].lower()), filter(lambda x: x[1] in ['NN', 'NNS', 'NNP', 'NNPS'], sent1))
    sent2_nouns = map(lambda x: lmtzr.lemmatize(x[0].lower()), filter(lambda x: x[1] in ['NN', 'NNS', 'NNP', 'NNPS'], sent2))
    
    sent1_verbs = map(lambda x: lmtzr.lemmatize(x[0].lower()), filter(lambda x: x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], sent1))
    sent2_verbs = map(lambda x: lmtzr.lemmatize(x[0].lower()), filter(lambda x: x[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], sent2))

    sent1_totalNouns = len(sent1_nouns)
    sent2_totalNouns = len(sent2_nouns)

    sent1_totalVerbs = len(sent1_verbs)
    sent2_totalVerbs = len(sent2_verbs)
        
    common_nouns = len(set(sent1_nouns).intersection(set(sent2_nouns)))
    common_verbs = len(set(sent1_verbs).intersection(set(sent2_verbs)))

    return {    'sent1_totalNouns' : sent1_totalNouns,
                'sent2_totalNouns': sent2_totalNouns,
                'common_nouns': common_nouns,
                'sent1_totalVerbs': sent1_totalVerbs, 
                'sent2_totalVerbs': sent2_totalVerbs,
                'common_verbs' : common_verbs   
            }
    #print sent1

def createFeatures(df):
    data = []
    label = []
    columns = []
    df, fdist, total_words = df
    dfNo, df = df
    df = df.fillna("")
    for cntr, row in enumerate(df.iterrows()):
        sentence1 = row[1]['question1']
        sentence2 = row[1]['question2']
        sentence1 = unicode(str(sentence1), errors='ignore')
        sentence2 = unicode(str(sentence2), errors='ignore')

        sentence1 = helper.removeUnwantedWordsAndConvertToLower(sentence1)
        sentence2 = helper.removeUnwantedWordsAndConvertToLower(sentence2)

        if sentence1 == "":
            sentence1 = "filler"
        if sentence2 == "":
            sentence2 = "filler"
        if mode == "train":
            is_duplicate = row[1]['is_duplicate']

        #jaccordScore = lc_dist.getJaccordScore(sentence1, sentence2, removeStopWords=False, stemming=False, lemmatize=False)
        #cosineScore = lc_dist.getCosineScore(sentence1, sentence2)
        
        nltkTokenize.lemmatize = True
        tf_vectorizer = CountVectorizer(tokenizer=nltkTokenize)
        #print str(sentence1), str(sentence2)
        out = tf_vectorizer.fit_transform([sentence1, sentence2]).todense()

        #print tf_vectorizer.get_feature_names()

        jaccordScore = 1 - scp_dist.cdist(out[0], out[1], 'jaccard')[0][0]
        cosineScore = 1 - scp_dist.cdist(out[0], out[1], 'cosine')[0][0]
        features = getPosTagFeatures(sentence1, sentence2, pos_tagger=None)
        sentenceSemanticSimilarity, sentenceWordOrderSimilarity = similarity.getSentenceSimilarity(sentence1, sentence2, fdist, total_words)
        sentenceSemanticSimilarityGW2V, sentenceWordOrderSimilarityGW2V = similarity.getSentenceSimilarity(sentence1, sentence2, fdist, total_words, 
                                                                          wv=wvGoogle, default_type='word2vec_google_gensim')
        sentenceSemanticSimilarityQW2V, sentenceWordOrderSimilarityQW2V = similarity.getSentenceSimilarity(sentence1, sentence2, fdist, total_words,
                                                                          wv=wvQuora, default_type='word2vec_quora_gensim')
        sentenceSemanticSimilarityGlove, sentenceWordOrderSimilarityGlove = similarity.getSentenceSimilarity(sentence1, sentence2, fdist, total_words, 
                                                                          model=gloveModel, default_type='glove')
        sentenceSemanticSimilarityGW2VBuiltIn = similarity.getSentenceSimilarityBuiltIn(sentence1, sentence2, wv=wvGoogle)
        sentenceSemanticSimilarityQW2VBuiltIn = similarity.getSentenceSimilarityBuiltIn(sentence1, sentence2, wv=wvQuora)
 
        #print cntr, jaccordScore, cosineScore, is_duplicate, sentence1, '#', sentence2
        #print sentenceSemanticSimilarity, sentenceWordOrderSimilarity
        if not columns:
            columns.extend(['jaccordScore', 'cosineScore'])
            columns.extend(['sentenceSemanticSimilarity', 'sentenceWordOrderSimilarity'])
            columns.extend(['sentenceSemanticSimilarityGW2V', 'sentenceWordOrderSimilarityGW2V'])
            columns.extend(['sentenceSemanticSimilarityQW2V', 'sentenceWordOrderSimilarityQW2V'])
            columns.extend(['sentenceSemanticSimilarityGlove', 'sentenceWordOrderSimilarityGlove'])
            columns.extend(['sentenceSemanticSimilarityGW2VBuiltIn'])
            columns.extend(['sentenceSemanticSimilarityQW2VBuiltIn'])
            columns.extend(features.keys())
            if mode == "train":
                columns.extend(['label'])

        data_ = []
        data_.extend([jaccordScore, cosineScore])
        data_.extend([sentenceSemanticSimilarity, sentenceWordOrderSimilarity])
        data_.extend([sentenceSemanticSimilarityGW2V, sentenceWordOrderSimilarityGW2V])
        data_.extend([sentenceSemanticSimilarityQW2V, sentenceWordOrderSimilarityQW2V])
        data_.extend([sentenceSemanticSimilarityGlove, sentenceWordOrderSimilarityGlove])
        data_.extend([sentenceSemanticSimilarityGW2VBuiltIn])
        data_.extend([sentenceSemanticSimilarityQW2VBuiltIn])
        data_.extend(features.values())
        if mode == "train":
            data_.extend([is_duplicate])
        data.append(data_)
        #print data
        if mode == "train":
            label.append(is_duplicate)
        #if cntr == 10:
        #    break

    df = pd.DataFrame(data=data, columns=columns)
    print dfNo, df.shape
    return df


def main():
    starttime = time.time()
    dfT = None
    if mode == "test":
       print testFile
       dfTest = pd.read_csv(testFile)
       dfT = dfTest
    elif mode == "train": 
       print trainFile
       dfTrain = pd.read_csv(trainFile)
       dfT = dfTrain
    #dfTrain = dfTrain.head(143)
    print dfT.shape
    chunksList = helper.divideDataFrame(dfT, 200)
    print "Totalframe", math.ceil(dfT.shape[0]/200)
    fdist, total_words = similarity.getCorpusFreqDist()
    chunksList = zip(chunksList, [fdist] * len(chunksList), [total_words] * len(chunksList))
    pool = Pool(50)
    dfLst = pool.map(createFeatures, chunksList)
    pool.terminate()
    df = pd.concat(dfLst, axis=0)
    print df.shape
    #dfTrain = dfTrain.head(5000)    
    #dfTrain.to_csv(os.path.join("data", "train_5K.csv"))
    #print df
    outfile = "data/quora_features_train.csv" if mode == "train" else "data/quora_features_test.csv"
    df.to_csv(outfile)
    print "Total time for feature generation", str(time.time() - starttime)

if __name__ == "__main__":
    main()
