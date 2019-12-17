The Impact Lens project attempts to identify potentially impactful information for a specific domain or brand. Imagine a fictitious client like NRA wants to identify information on the internet that could affect their product strategy or brand image. They would like to identify information on the internet that is relevant to guns. They have come to us to help them get ahead of the PR cycle and formulate a response / strategy. We will run our program to help them.

Our program presupposes that NRA gave us the following vocabulary to describe the domain they are looking for: _["guns", "shooting", "lobby", "rifles", "weapon", "nra", "handgun", "politics"]_. Ideally we would have taken this as input from the client. 

# HOW TO:
0. Before you start, you will need to download the gloVe dataset from their website: http://nlp.stanford.edu/data/glove.6B.zip
1. install dependencies - `$pip install requests, newspaper, pyjq, gensim, sklearn, flask, numpy`
1. Click on the zip link above, download and keep in repository's home directory
2. Unzip it. The gloVe files for various dimensions will be extracted into the home directory. Keep them there. The code uses this relative path
3. go to src folder
4. Run $python3 main.py guns

# Slighly Longer Explanation:
The program follows the following 4 stages:
1. Download/Load NYT Archive of December 2018 - we will use this dataset as a sample of the information on the internet and check for articles on guns. These articles have already been included in the git repo and can be found under data/nyt. 
2. Train Classifier - Train the categorizer to build categories. We use the Newsgroup 20 dataset for this and use the categories presented by that data set. Guns is one of the categories for us, conveniently. (More information: http://qwone.com/~jason/20Newsgroups/). Once trained, run the classifier on the NYT articles downloaded above.
3. GloVe learning - A set of glove embeddings that come from 'guns' is used to add to the vocabulary that was already presupposed. This step attempts to learn the word embeddings by training with the glove dataset downloaded before running the program.
4. Impact Score - Run the impact scorer to output a list of 10 indices in descending order of their scores. You can go to data/nyt folder and look at the indices of the articles to check their relevance.

