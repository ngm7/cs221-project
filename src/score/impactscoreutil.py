""""A helper class to encapsulate all the functions which help in calculating the impact score for the articles
"""
import gensim

class ImpactScoreUtil:
  #Loads the Glove vector files in memory so it can be used across the application
  #TODO : File path needs to be changed
  glovefile = r"../../data/wordvectors/glove.6B.300d.w2vformat.tx"
  model = gensim.models.KeyedVectors.load_word2vec_format(glovefile)

  #This function takes a word as input and then returns the closest 20 words similar to the passed words
  @staticmethod
  def find_similar_words_using_glove(self, word):
     self.model.most_similar(positive=[word], topn=20)

#Just added for testing purpose
print(ImpactScoreUtil.find_similar_words_using_glove(positive=["gun"], topn=20))
