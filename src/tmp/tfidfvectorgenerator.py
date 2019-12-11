import io

from sklearn.feature_extraction.text import TfidfVectorizer

countv = TfidfVectorizer()

# Uses the TFIDF method to build feature vectors.
# Generates training data set that can be used in the predictor
with io.open('53294') as myfile:
    lines = myfile.readlines()
    X = countv.fit_transform(lines)
    print(countv.get_feature_names())
    print(len(X.toarray()))
