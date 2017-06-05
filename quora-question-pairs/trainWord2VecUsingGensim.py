import pandas as pd
import gensim
from gensim.models import Word2Vec
from utils import helper
import time
import sys
#sys.path.append('..')

def readSentences():
    sentences = ["i am not able to understand this in any way", "new york city live", "you are creating creaters a ambiguous nature", "hello IS IT TRUE", "DUMMUTY WORD", "DUTY OF and narci", "heaven soul ", " creaters heaven and soul"]
    
    sentences = map(lambda x: x.split(), sentences)
    return sentences

def readSentencesQuora():
    dfTrain = pd.read_csv("data/train.csv")
    dfTrain = dfTrain.fillna('')
    sentences = dfTrain['question1'].tolist()
    sentences1 = dfTrain['question2'].tolist()
    dfTest = pd.read_csv("data/test.csv")
    dfTest = dfTest.fillna('')
    sentences2 = dfTest['question1'].tolist()
    sentences3 = dfTest['question2'].tolist()
    sentences.extend(sentences1)
    sentences.extend(sentences2)
    sentences.extend(sentences3)
    sentences = map(lambda x: helper.removeUnwantedWordsAndConvertToLower(x), sentences)
    sentences = map(lambda x: str(x).split(), sentences)
    print sentences[0:10]
    print len(sentences)
    return sentences
    #print dfTrain.columns

def trainGensimModel(sentences, modelPath):
    starttime = time.time()
    bigram_transformer = gensim.models.Phrases(sentences)
    #print bigram_transformer[sentences]
    #model = Word2Vec(bigram_transformer[sentences], size=100, window=5, min_count=5, workers=4)
    model = Word2Vec(bigram_transformer[sentences], min_count=10, size=400, workers=40)
    model.save(modelPath)
    print "Total Model Training",  str(time.time() - starttime)

def readModel(filePath):
    model = Word2Vec.load(filePath)
    print model.wv.__dict__
    print model.wv['duty']

if __name__ == "__main__":
    filePath =  "word2vecQuora.model"  
    #sentences =  readSentences()
    sentences =  readSentencesQuora()
    trainGensimModel(sentences, filePath)
    #readModel(filePath)
