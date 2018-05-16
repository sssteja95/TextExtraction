import PyPDF2
import docx
import nltk
import string
import spacy
from nltk.corpus import stopwords
from nltk.collocations import *

# filePath = r'C:\Users\SSSTEJA\Desktop\Self Development\SUBRAHMANYA ASHTAKAM KARAVALAMBA STOTRAM.docx'
# filePath = r'C:\Users\SSSTEJA\Desktop\nowfloats\sample (1).pdf'
filePath = r'./sample.pdf'
filePath = r'./Machine Learning - An Introduction.docx'
similarityThreshold = 0.9



fileText = []
if filePath.endswith('pdf'):
    # print('in pdf block')
    fileObj = open(filePath, 'rb')
    pdfReader = PyPDF2.PdfFileReader(fileObj)
    pdfSize = pdfReader.numPages

    for i in range(pdfSize):
        fileText.append(pdfReader.getPage(i).extractText())

    print('********************PDF File Text**************************')
    print(fileText)

elif filePath.endswith('doc') or filePath.endswith('docx'):

    doc = docx.Document(filePath)
    for para in doc.paragraphs:
        fileText.append(para.text)
    print('********************Word File Text**************************')    
    print(fileText)

else:
    print('Could not determine the file type')

#------------------------------1-------------------------
def printWordFrequency():
    
    freqText = " ".join(fileText)
    data = freqText.split()
    punctuations = ['(',')',';',':','[',']',',', '/']
    data = [word for word in data if not word in stopwords.words('english') and  not word in string.punctuation]
    fdist1 = nltk.FreqDist(data)
    

    print('******************** Word Frequency of the file **************************')
    # print (fdist1.most_common(50))
    for sample in fdist1:
        print(sample, fdist1[sample])

#------------------------------2-------------------------
def printnGrams():
    freqText = " ".join(fileText)
    data = freqText.split()
    data = [word for word in data if not word in stopwords.words('english') and  not word in string.punctuation]

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(nltk.wordpunct_tokenize(" ".join(data)))
    # only bigrams that appear 3+ times
    finder.apply_freq_filter(3) 

    # return the 10 n-grams with the highest PMI
    print('******************** Common Bigrams of the file **************************')
    print(finder.nbest(bigram_measures.pmi, 10))  

    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(nltk.wordpunct_tokenize(" ".join(data)))
    # only trigrams that appear 3+ times
    finder.apply_freq_filter(3) 
    

    # return the 10 n-grams with the highest PMI
    print('******************** Common Trigrams of the file **************************')
    print(finder.nbest(trigram_measures.pmi, 10))  

#------------------------------3--------------------------

def printDuplicateSentences():

    fileTextSentences = []
    # print(fileText)
    for chunk in fileText:
        fileTextSentences += chunk.split('. ')
    fileTextSentences = [x.strip() for x in fileTextSentences]


    uniqueSet = set()
    repeatedSet = set()
    for sentence in fileTextSentences:
        if sentence in uniqueSet:
            repeatedSet.add(sentence)
        else:
            uniqueSet.add(sentence)
    

    print('******************** Duplicate sentences in the file **************************')        
    print(repeatedSet)

    return uniqueSet

def printSimilarSentences(uniqueSet):

    nlp = spacy.load('en')
    doc = nlp(" ".join(uniqueSet))
    uniqueList = list(uniqueSet)

    print('******************** Similar sentences in the file **************************')
    for i in range(len(uniqueSet)):
        for j in range(i+1, len(uniqueSet)):
            sen1 = nlp(uniqueList[i]) 
            sen2 = nlp(uniqueList[j])
            simi = sen1.similarity(sen2)    
            if simi > similarityThreshold:
                print("sen1:", sen1)
                print("sen2:", sen2)
                print("similarity:", simi)



if __name__ == "__main__": 
    printWordFrequency()
    printnGrams()
    sentenceSet = printDuplicateSentences()
    printSimilarSentences(sentenceSet)






