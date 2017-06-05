# Notes #
The question was about finding identical set of question pair that exsit in quora.
### Main link for competition ###
https://www.kaggle.com/c/quora-question-pairs/

Data set can be downloaded from 
https://www.kaggle.com/c/quora-question-pairs/data

## Datasets used for model training ##
1. Google Word2Vec [pretrained model](https://code.google.com/archive/p/word2vec/) 
2. Glove [pretrained Model](https://nlp.stanford.edu/projects/glove)
3. Gensim Word2Vec Implementation on quora dataset done through the script present in code.


Approaches
1. For generating semantic and wordorder similarity b/w sentences, following research paper is used [Sentence Similarity Based on Semantic Nets
and Corpus Statistics](http://ants.iis.sinica.edu.tw/3BkMJ9lTeWXTSrrvNoKNFDxRm3zFwRR/55/Sentence%20Similarity%20Based%20on%20Semantic%20Nets%20and%20corpus%20statistics.pdf)
Based on above research papers similarity is generated using  __wordnet, word2vec, glove__ methods.

## Timeline ##
- [x] Build semantic features using word2vec and PosTags.
- [x] Generate Model using XGB
- [ ] Apply Reverse LSTM on the features and sentences to get indepth semantic knowledge.
- [ ] [Latent Dirichlet Allocation](https://radimrehurek.com/gensim/wiki.html) to convert higher dimentional representation into lower dimentional.
