'''
Created on May 17, 2017

@author: 703188429
'''
import nltk
import math
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus import brown
from scipy.spatial import distance as sc_dist
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from glove import Glove

def getCorpusFreqDist():
    print ">> Reading brown corpus"
    total_corpus_words = brown.words()
    fdist = nltk.FreqDist(w.lower() for w in total_corpus_words)
    total_words = len(total_corpus_words)
    return fdist, total_words

def getGoogleWord2Vec():
    preTrainedModelFile = 'corpus/GoogleNews-vectors-negative300.bin'
    print ">> Reading", preTrainedModelFile
    wv = KeyedVectors.load_word2vec_format(preTrainedModelFile, binary=True)
    return wv 

def getQuoraWord2Vec():
    preTrainedModelFile = 'corpus/word2vecQuora.model'
    print ">> Reading", preTrainedModelFile
    model = Word2Vec.load(preTrainedModelFile)    
    return model.wv

def getGloveModel():
    preTrainedModelFile = "corpus/glove.840B.300d_500.txt"
    preTrainedModelFile = "corpus/glove.840B.300d.txt"
    preTrainedModelFile = "corpus/glove.42B.300d.txt"
    print ">> Reading", preTrainedModelFile
    model = Glove.load_stanford(preTrainedModelFile)
    return model

def getWordSimilarity(word1, word2, threshold=0.2):
    """The funtion calculate the word similarity bases of the distance b/w synsets of word in wordnet"""
    if word1 == word2:
        return 1

    syns1 = wordnet.synsets(word1)
    syns2 = wordnet.synsets(word2)

    if not all([syns1, syns2]):
        return 0

    similarityScores = syns1[0].wup_similarity(syns2[0])
    similarityScores = 0 if similarityScores < threshold else similarityScores
    return similarityScores

def getWordSimilarityW2V(word1, word2, wv, threshold=0.2):
    try:
        wv.vocab[word1]
        wv.vocab[word2]
        similarityScores = wv.similarity(word1, word2)
        similarityScores = 0 if similarityScores < threshold else similarityScores
    except Exception, err:
        return 0
    return similarityScores

def getWordSimilarityGlove(word1, word2, model, threshold=0.2):
    try:
        vec1 = model.word_vectors[model.dictionary[word1]]
        vec2 = model.word_vectors[model.dictionary[word2]]
        cosineDistance = 1 - sc_dist.cdist([vec1], [vec2], 'cosine')[0][0]
        cosineDistance = 0 if cosineDistance < threshold else cosineDistance
    except Exception, err:
        #print err
        return 0
    return cosineDistance

def getInformationScore(word, fdist=None, total_words=None):
    """
    Function give information score of that word from the whole corpus, 
    it simply log of number of time that word is present divided by the total number of words
    """
    if not all([fdist, total_words]):
        fdist, total_words = getCorpusFreqDist()

    informationScore = 1 - ( np.log(fdist.get(word, 0) + 1) / np.log(total_words + 1))
    return informationScore

def getSentenceSimilarity(sentence1, sentence2, fdist=None, total_words=None, wv=None, model=None, default_type=None):
    sent1 =  nltk.word_tokenize(sentence1)
    sent2 =  nltk.word_tokenize(sentence2)
    sent1 = [unicode(str(word), errors='ignore') for word in sent1]
    sent2 = [unicode(str(word), errors='ignore') for word in sent2]    
    # Create a combined list with all the unique tokens from both sentences
    combinedList = list(set(sent1 + sent2))
    
    """
    Create vectors v1, v2
    Such that v1 contains size of len(combinedList) and each element of this vector will be filled with this formula.
    si = sh * I(wi) * I(wih)
    where wi is the element of combinedList at that index, wih is the word of sent1 with which wi has max similarity value.
    si is the max similarity value.
    Here I(w) is a function which will tell us the importance value of that word in corpus.
    """
    
    semanticSimilarityList1 = [] 
    semanticSimilarityList2 = [] 
    wordOrderSimilarityList1 = [] 
    wordOrderSimilarityList2 = [] 

    for wi in combinedList:
        if default_type == "word2vec_google_gensim":
            similarityScores1 = map(lambda x: getWordSimilarityW2V(wi, x, wv), sent1)
        elif default_type == "word2vec_quora_gensim":
            similarityScores1 = map(lambda x: getWordSimilarityW2V(wi, x, wv), sent1)
        elif default_type == "glove":
            similarityScores1 = map(lambda x: getWordSimilarityGlove(wi, x, model), sent1)
        else:
            similarityScores1 = map(lambda x: getWordSimilarity(wi, x), sent1)
        wih1 = sent1[similarityScores1.index(max(similarityScores1))]
        sh1 = max(similarityScores1)
        Iwi1 = getInformationScore(wi, fdist=fdist, total_words=total_words)
        Iwih1 = getInformationScore(wih1, fdist=fdist, total_words=total_words)
        _semanticScore1 = sh1 * Iwi1 * Iwih1
        #print "S1 -> ", wi, wih1, sh1, Iwi1, Iwih1, ":",_semanticScore1
        semanticSimilarityList1.append(_semanticScore1)
        if default_type == "word2vec_google_gensim":
            similarityScores2 = map(lambda x: getWordSimilarityW2V(wi, x, wv), sent2)
        elif default_type == "word2vec_quora_gensim":
            similarityScores2 = map(lambda x: getWordSimilarityW2V(wi, x, wv), sent2)
        elif default_type == "glove":
            similarityScores2 = map(lambda x: getWordSimilarityGlove(wi, x, model), sent2)
        else:
            similarityScores2 = map(lambda x: getWordSimilarity(wi, x), sent2)
        wih2 = sent2[similarityScores2.index(max(similarityScores2))]
        sh2 = max(similarityScores2)
        Iwi2 = getInformationScore(wi, fdist=fdist, total_words=total_words)
        Iwih2 = getInformationScore(wih2, fdist=fdist, total_words=total_words)
        _semanticScore2 = sh2 * Iwi2 * Iwih2
        #print "S2 -> ", wi, wih2, sh2, Iwi2, Iwih2, ":", _semanticScore2
        semanticSimilarityList2.append(_semanticScore2)
        
        # Override default sentence similarity threshold to calculate wordorder similarity
        similarityScores1 = map(lambda x: 0 if x < 0.4 else x, similarityScores1)
        _wordOrderSemanticScore1 = 0 if max(similarityScores1) == 0 else similarityScores1.index(max(similarityScores1)) + 1
        wordOrderSimilarityList1.append(_wordOrderSemanticScore1)
        similarityScores2 = map(lambda x: 0 if x < 0.4 else x, similarityScores2)
        _wordOrderSemanticScore2 = 0 if max(similarityScores2) == 0 else similarityScores2.index(max(similarityScores2)) + 1
        wordOrderSimilarityList2.append(_wordOrderSemanticScore2)

    sentenceSemanticSimilarity = 1 - sc_dist.cdist([semanticSimilarityList1], [semanticSimilarityList2], 'cosine')[0][0]
    sentenceSemanticSimilarity = 0 if math.isnan(sentenceSemanticSimilarity) else sentenceSemanticSimilarity 
    _sum = np.sum(np.array(wordOrderSimilarityList1) + np.array(wordOrderSimilarityList2))
    _diff = np.sum(np.absolute(np.array(wordOrderSimilarityList1) - np.array(wordOrderSimilarityList2)))
    sentenceWordOrderSimilarity = 0 if _sum == 0 else 1 - (float(_diff)/_sum)
    return sentenceSemanticSimilarity, sentenceWordOrderSimilarity

def checkWordInWV(word, wv):
    try:
       wv.vocab[word]
    except Exception, err:
       return False
    return True

def getSentenceSimilarityBuiltIn(sentence1, sentence2, wv):
    ws1 = nltk.word_tokenize(sentence1)
    ws2 = nltk.word_tokenize(sentence2)
    ws1 = filter(lambda x: checkWordInWV(x, wv), ws1)
    ws2 = filter(lambda x: checkWordInWV(x, wv), ws2)
    if not all([ws1, ws2]):
        return 0.0 
    return wv.n_similarity(ws1, ws2) 

""" Looks like same as getSentenceSimilarityBuiltIn i.e cosine of sum of two vectors"""
def getSentenceSimilarityByCombiningVectors(sentence1, sentence2, wv):
    ws1 = nltk.word_tokenize(sentence1)
    ws2 = nltk.word_tokenize(sentence2)
    ws1 = filter(lambda x: checkWordInWV(x, wv), ws1)
    ws2 = filter(lambda x: checkWordInWV(x, wv), ws2)
    if not all([ws1, ws2]):
        return 0.0 

    vec1 = np.sum(map(lambda x: wv[x], ws1), axis=0)
    vec2 = np.sum(map(lambda x: wv[x], ws2), axis=0)
    #vec1 = map(lambda x: wv[x], ws1)
    return 1 - sc_dist.cdist([vec1], [vec2], 'cosine')[0][0]

if __name__ == "__main__":
    #fdist, total_words = getCorpusFreqDist()
    sentence1 = "support how india country of unknown are you ?"
    sentence2 = "help are word you good? rpn origin "
    #print getSentenceSimilarity(sentence1, sentence2, fdist=fdist, total_words=total_words)
    wv = getQuoraWord2Vec() 
    print getSentenceSimilarityByCombiningVectors(sentence1, sentence2, wv)
    print getSentenceSimilarityBuiltIn(sentence1, sentence2, wv)
    #print getSentenceSimilarityBuiltIn(sentence1, sentence2, wv)
    #print getSentenceSimilarity(sentence1, sentence2, fdist=fdist, total_words=total_words, wv=wv, default_type='word2vec_google_gensim')
    #model = getGloveModel()
    #print getSentenceSimilarity(sentence1, sentence2, fdist=fdist, total_words=total_words, model=model, default_type='glove')

