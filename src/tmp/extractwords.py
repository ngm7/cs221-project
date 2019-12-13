import collections
import io
import re

import numpy


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words


def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab));

    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1

        print("{0} \n{1}\n".format(sentence, numpy.array(bag_vector)))



# Bag of Words algorithm to use as training data set.
with io.open('53294_negative') as myfile:
    lines = []
    word_dict = collections.defaultdict(int)
    for line in myfile:
        lines.append(line)
        words = word_extraction(line)

        for word in words:
            word_dict[word] += 1

    print(word_dict)
    # generates BOW sparsevector
    # generate_bow(lines)

