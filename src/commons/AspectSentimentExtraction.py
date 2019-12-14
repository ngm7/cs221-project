from commons import aspsentiment

def printResultChoice():
    userChoice = str(input('\nDo you want to print the result on output window? (Y/N) :'))
    if(userChoice=='Y' or userChoice=='y'):
        return True
    else:
        return False

def displayaspectandpolarity(folderpath, filename):
    #_FolderName='Data\\OppoF1\\'
    _FolderName='Data\\impact-analysis\\'
    _ReviewDataset=folderpath+filename + "article.txt"
    _PreProcessedData=folderpath+filename + '1.PreProcessedData.txt'
    _TokenizedReviews=folderpath+filename + '2.TokenizedReviews.txt'
    _PosTaggedReviews=folderpath+filename +'3.PosTaggedReviews.txt'
    _Aspects=folderpath+filename +'4.Aspects.txt'
    _Opinions=folderpath+filename + '5.Opinions.txt'
    aspsentiment.preProcessing(_ReviewDataset,_PreProcessedData,True)
    aspsentiment.tokenizeReviews(_ReviewDataset,_TokenizedReviews,True)
    aspsentiment.posTagging(_TokenizedReviews,_PosTaggedReviews,True)
    aspsentiment.aspectExtraction(_PosTaggedReviews,_Aspects,True)
    aspsentiment.identifyOpinionWords(_PosTaggedReviews,_Aspects,_Opinions,True)

