"""The data model encodes any data set that we are using, either for training or learning."""


class DataModel:
    data = []  # A list of documents.
    target = []  # A corresponding list of classification if supervised learning requires.

    def __init__(self):
        self.data = None
        self.target = None

    def setData(self, data):
        self.data = data

    def setTarget(self, target):
        self.target = target
