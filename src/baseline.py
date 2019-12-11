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
import os
import sys
import gensim
import requests
import pyjq
from newspaper import Article
from collections import defaultdict
from typing import List

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline

from commons.datamodel import DataModel

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


class Classifier:

    def trainModel(self, training_data: DataModel):
        pass

    def classify(self, testing_data: DataModel):
        pass


class SgdClassifier(Classifier):
    """The classifier uses SGD and two transformers - word count and TF-IDF to classify the 20 NewsGroup data into 20
categories."""

    # Reads the data and stores in class variables
    def __init__(self):
        self.model = None
        self.training_data = None
        self.testing_data = None

    # This function extracts features
    def extractfeatures(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(self.training_data.data)

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        return count_vect, tfidf_transformer

    # This function trains the model
    def trainModel(self, training_data: DataModel):
        self.training_data = training_data
        count_vect, tfidf_transformer = self.extractfeatures()
        self.model = Pipeline(steps=[('vect', count_vect),
                                     ('tfidf', tfidf_transformer),
                                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                               alpha=1e-3, max_iter=20, random_state=42,
                                                               early_stopping=True, tol=100))])
        self.model.fit(self.training_data.data, self.training_data.target)

    def classify(self, testing_data: DataModel):
        self.testing_data = testing_data
        predicted_svm = self.model.predict(testing_data.data)
        # score = metrics.accuracy_score(predicted_svm, testing_data.target)
        # print("Accuracy: {}".format(score))
        return predicted_svm

    def naiveBayesMB(self):
        count_vect = CountVectorizer(stop_words='english')
        X_train_counts = count_vect.fit_transform(self.training_data.data)
        X_test_data = count_vect.transform(self.testing_data.data)
        Y_test_target = count_vect.transform(self.testing_data.data)

        mnb = MultinomialNB()
        mnb.fit(X_train_counts, self.training_data.target)
        predict = mnb.predict(X_test_data)
        # score = metrics.accuracy_score(predict, self.testing_data.target)

        # print("Accuracy: {}".format(score))

        return predict


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

    @staticmethod
    def get_stack_rank(classifier: Classifier, vocabulary: List[str], document_indices: List[int]):
        """For a given set of domain words and corresponding indices, the impact
        score for an article is the average of the frequencies of the words in the vocabulary. Returns a List sorted
        in descending order of these impact scores"""

        impact_score = {}
        for dIndex in document_indices:
            impact_score[dIndex] = 0
            freq = defaultdict(float)
            for word in classifier.testing_data.data[dIndex].split(' '):
                freq[word] += 1.0
            for word in vocabulary:
                impact_score[dIndex] += freq[word]

        s = sum(score for i, score in impact_score.items())
        for i, score in impact_score.items():
            impact_score[i] = score / s

        return sorted(impact_score.items(), key=lambda x: x[1], reverse=True)


def cleanUpURLs(out):
    cleanUrls = []
    for url in out:
        if not url:  # some URLs are empty
            continue
        cleanUrls.append(url)
    return cleanUrls


def buildTestDataFromNYT(download=True, articlesDir=".", writeToDir=False):
    testData = DataModel()
    dataArr = []
    count = 0

    if not download:
        print("Building test dataset from NYT's archive of 12/2018.")
        print(f"Loading articles from f{articlesDir}")
        files = os.listdir(articlesDir)
        files = [os.path.join(articlesDir, f) for f in files]
        files.sort(key=lambda x: os.path.getmtime(x))
        for file in files:
            with open(articlesDir + '/' + file, "r") as f:
                dataArr.append(f.read())
                count += 1
    else:
        print("Building test dataset from NYT's archive of 12/2018. Downloading ...")
        key = "5AE0mAEtH2uXTpUjUnNr4kS9GVTVco8M"

        url = 'https://api.nytimes.com/svc/archive/v1/2018/12.json?&api-key=' + key
        r = requests.get(url)
        json_data = r.json()

        jq = f".response .docs [] | .web_url"
        out = pyjq.all(jq, json_data)

        # some URLs are empty, clean up
        cleanUrls = cleanUpURLs(out)

        for url in cleanUrls:
            # if count > 3000:  # if you don't want to download 6200 articles
            #     break
            a = Article(url=url)
            try:
                a.download()
                a.parse()
            except Exception as ex:
                print(f"caught {ex} continuing")
            if len(a.text):
                print(f"{len(dataArr)} - downloaded {len(a.text)} bytes")
                dataArr.append(a.text)
                if writeToDir:
                    with open(articlesDir + "/" + f"{count}.txt", "w") as f:
                        f.write(a.text)
                count += 1

    print(f"working with {len(dataArr)} articles")
    testData.setData(dataArr)
    return testData


def main(category: str):
    """The Main function"""
    testData = fetch_20newsgroups(subset='train', shuffle=True)
    classifier_training_data = DataModel()
    classifier_training_data.setData(testData.data)
    classifier_training_data.setTarget(testData.target)

    classifier_testing_data = buildTestDataFromNYT(download=False, articlesDir="../../data/nyt", writeToDir=True)

    classifier = SgdClassifier()
    classifier.trainModel(classifier_training_data)
    predictions_sgd = classifier.classify(classifier_testing_data)

    # predictions_nb = classifier.naiveBayesMB()

    # identify indices for talk.politics.guns (index 16 in target_names)
    indices_sgd = [index for index, prediction in enumerate(predictions_sgd) if prediction == CATEGORY_MAP[category]]
    # indices_nb = [index for index, prediction in enumerate(predictions_nb) if prediction == CATEGORY_MAP[category]]

    gloveVectors = ImpactScoreUtil.find_similar_words_using_glove(word="gun")
    enhanced_vocab = VOCABULARY[category]
    for word, score in gloveVectors:
        enhanced_vocab.append(word)

    print(f"current vocab: {VOCABULARY[category]}")
    print(f"enhanced vocab: {enhanced_vocab}")

    # identify the articles
    stack = ImpactScorer.calculatetfidf(classifier=classifier,
                                        vocabulary=enhanced_vocab,
                                        document_indices=indices_sgd)
    # print(classifier.testing_data.data[6620])
    print(stack[0:10])  # print top N


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect parameters. \n USAGE: $python3 baseline.py <CATEGORY:{guns}>")
    else:
        main(category=sys.argv[1])
