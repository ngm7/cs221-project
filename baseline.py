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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np

# This class trains a regularized linear models with stochastic gradient descent
class TrainClassifier:

    #Reads the data and stores in class variables
    def __init__(self):
        self.training_data = twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
        print(twenty_train.target_names) #prints all the categories
        print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file
        self.testing_data = fetch_20newsgroups(subset='test', shuffle=True)

    #This function extracts features
    def extractfeatures(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(self.training_data.data)
        print(X_train_counts.shape)

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        print(X_train_tfidf.shape)

        return count_vect, tfidf_transformer

    #This function trains the model
    def trainmodel(self):

        count_vect, tfidf_transformer = self.extractfeatures()

        text_clf_svm = Pipeline([('vect', count_vect),
                                 ('tfidf', tfidf_transformer),
                                 ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                           alpha=1e-3, max_iter=20, random_state=42, early_stopping=True, tol=100))])
        text_clf_svm.fit(self.training_data.data, self.training_data.target)
        predicted_svm = text_clf_svm.predict(self.testing_data.data)
        print(np.mean(predicted_svm == self.testing_data.target))


trainclassifier = TrainClassifier()
trainclassifier.trainmodel()
