"""A baseline program for producing an impact score for . This program attempts at creating a fundamental program that
 establishes the minimum quality expected out of a training algorithm.

Given
 - It takes as given a set of words established as the domain. For example - {"NRA: ["gun", "nra", "2nd ammendment"]} is
  an example set of words for domain NRA.
 - It will also take in a set of words for each of the categories. Currently only two categories are
supported - Geography and Politics.
 - set of 15 links to articles about NRA domain.

Input
 - A choice of category

Output
 - A subset of articles and a corresponding impact score.
 """
import sys
import gensim
from collections import defaultdict
from typing import List

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words

from commons.datamodel import DataModel
from classifier.sgdClassifier import SgdClassifier
from classifier import Classifier
from commons.data.nyt import buildTestDataFromNYT
from src.commons.AspectSentimentExtraction import displayaspectandpolarity

CATEGORY_MAP = {
    'guns': 16
}

VOCABULARY = {
    'guns': ["guns", "shooting", "lobby", "rifles", "weapon", "nra", "handgun", "politics"]
}


class ImpactScoreUtil:
    # Loads the Glove vector files in memory so it can be used across the application

    # This function takes a word as input and then returns the closest 20 words similar to the passed words
    @staticmethod
    def find_similar_words_using_glove(word):
        # TODO : File path needs to be changed
        # gloveFile = r'../data/glove.6B/glove.6B.300d.txt'
        word2VecFile = r'../../data/glove.6B/glove.6B.300d_word2Vec.txt'
        # gensim.scripts.glove2word2vec.glove2word2vec(gloveFile, word2VecFile)
        model = gensim.models.KeyedVectors.load_word2vec_format(word2VecFile, binary=False)
        return model.most_similar(positive=[word], topn=20)


class ImpactScorer:
    """A class that contains generic methods for calculating impact scores"""

    @staticmethod
    def cleanupdata(doc):
        testdata1 = doc.lower()
        cleandata = ''

        for word in testdata1.split(' '):
            if word not in stop_words.ENGLISH_STOP_WORDS and len(word) > 1:
                cleandata = cleandata + ' ' + word

        # TODO can use this to do additional preprocessing
        # symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        # for i in symbols:
        #     cleandata = np.char.replace(cleandata, i, ' ')
        # cleandata = np.char.replace(cleandata, "'", '')
        return cleandata

    @staticmethod
    def calculatetfidf(classifier, vocabulary, document_indices):
        """For a given set of domain words and corresponding indices, the impact
           score for an article is the average of the frequencies of the words in the vocabulary. Returns a List sorted
           in descending order of these impact scores"""

        impact_score = {}
        for dIndex in document_indices:
            impact_score[dIndex] = 0
            freq = defaultdict(float)
            newdata = ImpactScorer.cleanupdata(classifier.testing_data.data[dIndex])

            for word in newdata.split(' '):
                freq[word] += 1.0

            count_vocab_words = 0
            for word in vocabulary:
                count_vocab_words += freq[word]
            tf = count_vocab_words / len(newdata.split(' '))
            # since we have already computed the importance of these documents using classification, we stop here and get the score
            impact_score[dIndex] = tf * 100
        s = sum(score for i, score in impact_score.items())
        for i, score in impact_score.items():
            impact_score[i] = score / s

        return sorted(impact_score.items(), key=lambda x: x[1], reverse=True)

#This function extracts the aspects and their polarity from the articles.
def extractaspectandpolarity(classifier):
    filepath = r"C:\\Nimisha\\Stanford\\Project\\data\\impactarticleindex\\"
    #filename="article.txt"
    impactarticleindex = [5893, 692, 6763, 7274, 326, 3613, 3831, 1665, 3781, 2135]
    for dIndex in impactarticleindex:
        datastring = classifier.testing_data.data[dIndex]
        filecommonname = str(dIndex)
        f = open(filepath + filecommonname + "article.txt", "w+")
        f.write(datastring)
        f.close()
        displayaspectandpolarity(filepath, filecommonname)


def main(category: str):
    """The Main function"""
    testData = fetch_20newsgroups(subset='train', shuffle=True)
    classifier_training_data = DataModel()
    classifier_training_data.setData(testData.data)
    classifier_training_data.setTarget(testData.target)

    classifier_testing_data = buildTestDataFromNYT(download=False, articlesDir="../data/nyt", writeToDir=True)

    classifier = SgdClassifier()
    classifier.trainModel(classifier_training_data)
    predictions_sgd = classifier.classify(classifier_testing_data)

    # identify indices for talk.politics.guns (index 16 in target_names)
    indices_sgd = [index for index, prediction in enumerate(predictions_sgd) if prediction == CATEGORY_MAP[category]]

    gloveVectors = ImpactScoreUtil.find_similar_words_using_glove(word="gun")
    enhanced_vocab = VOCABULARY[category]
    for word, score in gloveVectors:
        enhanced_vocab.append(word)

    #Uncomment the below for extracting Aspects and their polarity
    #extractaspectandpolarity(classifier)

    # identify the articles
    stack = ImpactScorer.calculatetfidf(classifier=classifier,
                                        vocabulary=enhanced_vocab,
                                        document_indices=indices_sgd)
    # print(classifier.testing_data.data[6620])
    print(stack[0:10])  # print top N


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect parameters. \n USAGE: $python3 main.py <CATEGORY:{guns}>")
    else:
        main(category=sys.argv[1])
