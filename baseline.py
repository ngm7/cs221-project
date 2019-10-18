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
