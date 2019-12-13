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
import sys, os, io, glob, re
from collections import defaultdict
from typing import List

from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


CATEGORY_MAP = {
    'guns': 16
}

VOCABULARY = {
    'guns': ["guns", "shooting", "victim", "mass", "kill", "murder", "weapon", "gun", "nra", "handgun", "assault"]
}


# This class trains a regularized linear models with stochastic gradient descent
class DataTransformer:
    """The classifier uses Logistic Reg and two transformers - word count and TF-IDF to classify the 20 NewsGroup data into 20
categories."""

    # Reads the data and stores in class variables. Prepares positive,negative or neutral sentiment features.
    def __init__(self):
        listoffiles = glob.glob('../data/nra-oracle/*')
        self.traindata = []
        self.cleandata = []
        self.target = []
        for file in listoffiles:
            if not os.path.isdir(file):
                if 'positive' in os.path.basename(file):
                    self.target.append(1)
                elif 'negative' in os.path.basename(file):
                    self.target.append(-1)
                else:
                    self.target.append(0)
                with io.open(file) as fileread:
                    self.traindata.append(fileread.read())
        listoftestfiles = os.listdir('../data/nra-oracle/test_data')
        self.testdata = []
        for file in listoftestfiles:
            with io.open(os.path.join('../data/nra-oracle/test_data', file)) as fileread:
                filedata = ''
                for line in fileread.readlines():
                    if not (line.startswith(':') or line.startswith('>')):
                        filedata += line + ' '

                self.testdata.append(filedata)

    def removeStopwords(self, input):
        output = []
        for text in input:
            output.append(' '.join([word for word in text.split() if word not in stop_words.ENGLISH_STOP_WORDS]))

        return output

    def preprocesstext(self, data):

        outdata = []
        for line in data:
            if not (line.startswith(':') or line.startswith('>')):
                outdata.append(line)

        return self.removeStopwords(outdata)

    # This function extracts features
    def extractfeatures(self, traindata, testdata):
        count_vect = CountVectorizer(binary=True, ngram_range=(1,3))
        count_vect.fit(traindata)
        X = count_vect.transform(traindata)
        X_test = count_vect.transform(testdata)

        print(count_vect.get_feature_names())

        X_train, X_val, y_train, y_val = train_test_split(X, self.target, train_size=0.75)

        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            lr = LogisticRegression(C = c)
            lr.fit(X_train, y_train)
            print('Accuracy score for C=%s: %s' %(c, accuracy_score(y_val, lr.predict(X_val))))

        final_model = LogisticRegression(C=1)
        final_model.fit(X, self.target)
        predict = final_model.predict(X_test)
        print(predict)
        print("Final Accuracy: %s"
              % accuracy_score(self.target, final_model.predict(X_test)))

    # This function extracts features
    def extractfeaturesNaiveBayes(self, traindata, testdata):
        count_vect = CountVectorizer(ngram_range=(1, 3))
        count_vect.fit(traindata)
        X = count_vect.transform(traindata)
        X_test = count_vect.transform(testdata)

        print(count_vect.get_feature_names())

        X_train, X_val, y_train, y_val = train_test_split(X, self.target, train_size=0.75)
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)
        predict = mnb.predict(X_test)

        print("Final Accuracy: %s"
              % accuracy_score(self.target, predict))




class SentimentClassifier:
    """A class that contains generic methods for calculating impact scores"""
    @staticmethod
    def cleanupdata(doc):
        testdata1 = doc.lower()
        cleandata = ''

        for word in testdata1.split(' '):
            if word not in stop_words.ENGLISH_STOP_WORDS and len(word) > 1:
                cleandata = cleandata + ' ' + word


        return cleandata



def main():
    """The Main function"""
    datat = DataTransformer()
    cleantrain = datat.preprocesstext(datat.traindata)
    cleantest = datat.preprocesstext(datat.testdata)
    # datat.removeStopwords(newtrain)
    datat.extractfeatures(cleantrain, cleantest)


if __name__ == "__main__":
    main()
