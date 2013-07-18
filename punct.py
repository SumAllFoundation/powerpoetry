def punct(poems):
	#N-gram Function 
	coca_word_count = 450000000
	#Import
	import csv
	import codecs
	import itertools
	from pandas import DataFrame, Series
	import datetime
	import numpy as np
	from numpy.random import randn, random_integers
	import re
	import nltk
	from nltk.stem import WordNetLemmatizer
	from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
	wnl = WordNetLemmatizer()
	#Setting up the params
	#Sentence List for Each Poem
	sentTokenizedPoems = [nltk.sent_tokenize(poem) for poem in poems]
	#Infrequentß Words - not in COCA top 20,000
	#Word Count
	punctFreq = np.zeros(len(poems))
	wordCount = np.zeros(len(poems))
	#calculate the frequency distribution for n-grams. 
	for i,sentTokenizedPoem in enumerate(sentTokenizedPoems):
		#print 'Unigram: Poem ' + str(i) + ': ' 
		#Tokenize each sentence in the poem
		#Tokenized Text with PoS Tag - Tuple
		tokenized = [word_tokenize(sent) for sent in sentTokenizedPoem]
		#Flatten the List of Tuple List
		tokenized = list(itertools.chain(*tokenized))
		#Remove Punctuation. Remove Numbers. Lower.
		nonPunct = re.compile('.*[A-Za-z].*')
		filtered = [w for w in tokenized if nonPunct.match(w)]
		#Number of Punctuations
		try:
			punctFreq[i] = 1- len(filtered)/len(tokenized)
			wordCount[i] = len(tokenized)
		except:
			print wordCount[i]
	return(punctFreq,wordCount)
