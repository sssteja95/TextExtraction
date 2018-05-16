import PyPDF2
import docx
import nltk
import string
from nltk.corpus import stopwords
from nltk.collocations import *

filePath = r'C:\Users\SSSTEJA\Desktop\Self Development\SUBRAHMANYA ASHTAKAM KARAVALAMBA STOTRAM.docx'
filePath = r'C:\Users\SSSTEJA\Desktop\nowfloats\sample (1).pdf'
filePath = r'./sample (1).pdf'
fileText = []
if filePath.endswith('pdf'):
    print('in pdf block')
    fileObj = open(filePath, 'rb')
    pdfReader = PyPDF2.PdfFileReader(fileObj)
    pdfSize = pdfReader.numPages

    for i in range(pdfSize):
        fileText.append(pdfReader.getPage(i).extractText())

    #print(fileText)

elif filePath.endswith('doc') or filePath.endswith('docx'):

    doc = docx.Document(filePath)
    for para in doc.paragraphs:
        fileText.append(para.text)
        
    #print(fileText)

else:
    print('Could not determine the file type')

#------------------------------1-------------------------
freqText = " ".join(fileText)
data = freqText.split()
punctuations = ['(',')',';',':','[',']',',', '/']
data = [word for word in data if not word in stopwords.words('english') and  not word in string.punctuation]

fdist1 = nltk.FreqDist(data)
#print (fdist1.most_common(50))

#------------------------------2-------------------------

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

finder = BigramCollocationFinder.from_words(nltk.wordpunct_tokenize(" ".join(data)))
# only bigrams that appear 3+ times
finder.apply_freq_filter(3) 
# return the 10 n-grams with the highest PMI
#print(finder.nbest(bigram_measures.pmi, 10))  

finder = TrigramCollocationFinder.from_words(nltk.wordpunct_tokenize(" ".join(data)))
# only trigrams that appear 3+ times
finder.apply_freq_filter(3) 
# return the 10 n-grams with the highest PMI
#print(finder.nbest(trigram_measures.pmi, 10))  

#------------------------------3--------------------------


fileTextSentences = []
# print(fileText)
for chunk in fileText:
    fileTextSentences += chunk.split('.')
fileTextSentences = [x.strip() for x in fileTextSentences]
# fileTextSentences = [x for x in fileTextSentences if x]

fileTextSentences = list(filter(bool, fileTextSentences))
for elem in fileTextSentences:
    print(elem)














