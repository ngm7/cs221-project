"""The Stochastic Gradient Descent classifier. Uses a parameterized loss function with default parameters."""

from .Classifier import Classifier
from commons.datamodel import DataModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline


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

