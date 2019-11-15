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
from collections import defaultdict
from typing import List

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
import numpy as np

CATEGORY_MAP = {
    'guns': 16
}

VOCABULARY = {
    'guns': ["guns", "shooting", "victim", "mass", "kill", "murder", "weapon", "gun", "nra", "handgun", "assault"]
}


# This class trains a regularized linear models with stochastic gradient descent
class Classifier:
    """The classifier uses SGD and two transformers - word count and TF-IDF to classify the 20 NewsGroup data into 20
categories."""

    # Reads the data and stores in class variables
    def __init__(self):
        self.training_data = twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
        self.testing_data = fetch_20newsgroups(subset='test', shuffle=True)

    # This function extracts features
    def extractfeatures(self):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(self.training_data.data)

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        return count_vect, tfidf_transformer

    # This function trains the model
    def trainmodel(self):
        count_vect, tfidf_transformer = self.extractfeatures()

        text_clf_svm = Pipeline([('vect', count_vect),
                                 ('tfidf', tfidf_transformer),
                                 ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                           alpha=1e-3, max_iter=20, random_state=42,
                                                           early_stopping=True, tol=100))])
        text_clf_svm.fit(self.training_data.data, self.training_data.target)
        predicted_svm = text_clf_svm.predict(self.testing_data.data)
        # print(np.mean(predicted_svm == self.testing_data.target))
        # print(predicted_svm)
        # Keeping this here to see if we can use it later.(if needed)TODO Cleanup
        # naive_bayes = MultinomialNB()
        # naive_bayes.fit(self.training_data.data, modified_predicted_svm)
        # predict_nb = naive_bayes(self.testing_data.data)
        return predicted_svm

    # If we were to use NaiveBayes as the training model
    def trainmodelNaiveBayes(self):
        count_vect, tfidf_transformer = self.extractfeatures()

        text_clf_svm = Pipeline([('vect', count_vect),
                                 ('tfidf', tfidf_transformer),
                                 ('clf-svm', BernoulliNB(alpha=1))])
        text_clf_svm.fit(self.training_data.data, [['guns']])
        predicted_svm = text_clf_svm.predict(self.testing_data.data)
        print(np.mean(predicted_svm == self.testing_data.target))
        return predicted_svm


class ImpactScorer:
    """A class that contains generic methods for calculating impact scores"""

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


def main(category: str):
    """The Main function"""
    classifier = Classifier()
    predictions = classifier.trainmodel()

    # identify indices for talk.politics.guns (index 16 in target_names)
    indices = [index for index, prediction in enumerate(predictions) if prediction == CATEGORY_MAP[category]]

    # identify the articles
    stack = ImpactScorer.get_stack_rank(classifier=classifier,
                                        vocabulary=VOCABULARY[category],
                                        document_indices=indices)
    print(stack[0:10])  # print top N


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect parameters. \n USAGE: $python3 baseline.py <CATEGORY:{guns}>")
    else:
        main(category=sys.argv[1])
