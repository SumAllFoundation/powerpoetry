#!user/bin/env python


def ngram(dbparams,poems):
	#N-gram Function 
	coca_word_count = 450000000
	#Import
	import csv
	import codecs
	import itertools
	from pandas import DataFrame, Series
	import matplotlib.pyplot as plt
	import datetime
	import numpy as np
	from numpy.random import randn, random_integers
	import re
	import MySQLdb
	import nltk
	from nltk.stem import WordNetLemmatizer
	from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
	from nltk.corpus import names
	from nltk.corpus import wordnet
	from nltk.tag.simplify import simplify_wsj_tag
	wnl = WordNetLemmatizer()
	#Setting up the params
	#Sentence List for Each Poem
	sentTokenizedPoems = [nltk.sent_tokenize(poem) for poem in poems]
	#Infrequent Words - not in COCA top 20,000
	infrequentUni=np.zeros(len(poems))
	infrequentBi=np.zeros(len(poems))
	infrequentTri=np.zeros(len(poems))
	#Misspelt Word Count - not in WordNet or COCA
	misspeltWord=np.zeros(len(poems))
	#Unigram Frequency Count
	unigramFreq = np.zeros(len(poems))
	#Bigram
	bigramFreq= np.zeros(len(poems))
	#Trigram
	trigramFreq = np.zeros(len(poems))
	#Sentence Count
	sentence_count = np.zeros(len(poems))
	#Word Count
	word_count = np.zeros(len(poems))
	#Punct Count
	punct_count = np.zeros(len(poems))
	#COCA Dictionaries
	w1 = {}
	w2 = {}
	w3 = {}
	##################
	#MySQL connection#
	##################
	db=MySQLdb.connect(cursorclass=MySQLdb.cursors.DictCursor,passwd=dbparams['passwd'],db=dbparams['db'],
		host=dbparams['host'],port=dbparams['port'],user=dbparams['user'])
	#Fetching the tables.
	c = db.cursor()
	for table in dbparams['table']:
		c.execute("SELECT * from %s"%(table))
		q = c.fetchall()
		#Unigram
		if table in 'w1':
			w1 = {(row['word1'],row['PoS']) : int(row['count']) for row in q}
		#Bigram
		if table in 'w2':
			w2 = {(row['word1'],row['word2']) : int(row['count']) for row in q}
		#Trigram
		if table in 'w3':
			w3 = {(row['word1'],row['word2'],row['word3']) : int(row['count']) for row in q}

	#calculate the frequency distribution for n-grams. 
	for i,sentTokenizedPoem in enumerate(sentTokenizedPoems):
		#print 'Unigram: Poem ' + str(i) + ': ' 
		#Tokenize each sentence in the poem
		#Tokenized Text with PoS Tag - Tuple
		tokenized = [nltk.pos_tag(word_tokenize(sent)) for sent in sentTokenizedPoem]
		#Sentence Count
		sentence_count[i] = len(tokenized)
		#Flatten the List of Tuple List
		tokenized = list(itertools.chain(*tokenized))
		#Remove Punctuation. Remove Numbers. Lower.
		nonPunct = re.compile('.*[A-Za-z].*')
		#Number of Punctuations
		punct_count[i] = len([(w[0].lower()) for w in tokenized if not nonPunct.match(w[0])])
		filtered = [(w[0].lower(),w[1]) for w in tokenized if nonPunct.match(w[0])]
		#Word Count
		word_count[i] = len(filtered)
		##Simplify PoS in order to be able to lemmatize. 
		#Wordnet Lemmatizer knows  Adj(a), Adverb(v), Noun(n),Verb(v)
		#Map Penn tree to wordnet for lemmatizing.
		#Map Penn tree to COCA for frequency analysis.
		#Lemmatized Word + COCA PoS
		lemmatized  = [(wnl.lemmatize(t[0],penn_to_wordnet(t[1])),penn_to_coca([1])) for t in filtered]
		#Calculate Frequency of the Words.
		#Number of words for the loop
		nw = 0
		for w,wordTuple in enumerate(lemmatized):
			#Reset count
			count = 0
			try:
				count = w1[wordTuple]
				nw+=1
				#print (wordTuple, ' found: ' + str(count))
			except:
			#Either the PoS convertsion from Penn is wrong. 
			#Find the word and corresponding PoS in Dict. 
				tup=[]
				if [tup for tup in w1.keys() if tup[0] == wordTuple[0]]:
					count=(w1[tup])
					nw+=1
					#print (wordTuple, ' appended : '+ str(count) )
				else:
					#No entries in the COCA dictionary. Check if it is misspelt.
					#Use WordNet 
					if wordnet.synsets(wordTuple[0]):
						count=100
						nw+=1
						infrequentUni[i] += 1 / word_count[i] 
						#print (wordTuple, ' infrequent')
					#See if the word is a name. Capitalize the first Letter!
					#If the letter is longer than three letters.
					elif not (wordTuple[0].capitalize() in names.words() or len(wordTuple[0])<=3):
					#Remove from body. Add to Misspelt.
						misspeltWord[i] += 1 / word_count[i]
						#print (wordTuple, ' misspelt?')
					#else:
						#print (wordTuple, ' not recognized.')
			#Calculate Frequency if count different than zero.
			if count > 0:
				unigramFreq[i]=(1/(nw))*log(1+(count/coca_word_count)) + ((nw-1)/(nw)) * unigramFreq[i]
		#####BIGRAM FUNCTION:
		#print 'Bigram: Poem ' + str(i) + ': ' 
		#Tokenize
		tokenized=[nltk.bigrams(word_tokenize(sentence)) for sentence in sentTokenizedPoem]
		#Flatten the List of Tuple List
		tokenized = list(itertools.chain(*tokenized))
		##Remove any tuple that has punctuation or number in it. Lower letter.
		nonPunct = re.compile('.*[A-Za-z].*')
		filtered = [(w[0].lower(),w[1].lower()) for w in tokenized if nonPunct.match(w[0]) and nonPunct.match(w[1])]
		#Freq Distribution
		#Number of words for the loop
		nw = 0
		for w,wordTuple in enumerate(filtered):
			#Reset count
			count = 0
			try:
				count = w2[wordTuple]
				nw+=1
				#print (wordTuple, ' found: ' + str(count))
			except:
				#Check if words exists in WordNet
				if (wordnet.synsets(wordTuple[0])) and (wordnet.synsets(wordTuple[1])):
					infrequentBi[i] += 1 / word_count[i]
					nw +=1 
					count = 10
					#print (wordTuple, ' is infrequent')
			#Append Frequency 
			if count > 0:
				bigramFreq[i] = (1/(nw))* log(count) + ((nw-1)/(nw)) * bigramFreq[i]
		#TRIGRAMS
		#print 'Trigram: Poem ' + str(i) + ': ' 
		#Tokenize
		tokenized=[nltk.trigrams(word_tokenize(sentence)) for sentence in sentTokenizedPoem]
		#Flatten the List of Tuple List
		tokenized = list(itertools.chain(*tokenized))
		##Remove any tuple that has punctuation or number in it. Lower letter.
		nonPunct = re.compile('.*[A-Za-z].*')
		filtered = [(w[0].lower(),w[1].lower(),w[2].lower()) for w in tokenized if nonPunct.match(w[0]) 
						and nonPunct.match(w[1]) and nonPunct.match(w[2])]
		#Freq Distribution
		#fdist = nltk.FreqDist(filtered)
		#Number of words for the loop
		nw = 0
		for w,wordTuple in enumerate(filtered):
			#Reset count
			count = 0
			try:
				count = w3[wordTuple]
				nw+=1
				print (wordTuple, ' found: ' + str(count))
			except:
				#Check if words exists in WordNet
				if (wordnet.synsets(wordTuple[0])) and (wordnet.synsets(wordTuple[1])) and (wordnet.synsets(wordTuple[2])):
					infrequentTri[i] += 1 / word_count[i]
					nw +=1 
					count = 10
					#print (wordTuple, ' is infrequent')
			#Append Frequency 
			if count > 0:
				trigramFreq[i] = (1/(nw))* log(count) + ((nw-1)/(nw)) * trigramFreq[i]
	return(unigramFreq,bigramFreq,trigramFreq,infrequentUni,infrequentBi,infrequentTri,misspeltWord,
		sentence_count,word_count,punct_count)

	#Utilities
	#Treebank PoS to WordNet in order to lemmatize
#Assign Noun if none of the conditions are satisfied. 
def penn_to_wordnet(treebank_tag):
	from nltk.corpus import wordnet
	#
	if treebank_tag.startswith('J'):
	    return wordnet.ADJ
	elif treebank_tag.startswith('V'):
	    return wordnet.VERB
	elif treebank_tag.startswith('N'):
	    return wordnet.NOUN
	elif treebank_tag.startswith('R'):
	    return wordnet.ADV
	else:
	    return wordnet.NOUN

def penn_to_coca(treebank_tag):
	#Mapping from Penn Tree PoS to COCA PoS
	mapCOCA =  {'-NONE-': 'n',
	'':'n',
	'CC': 'c',
	'CD': 'm',
	'DT': 'a',
	'EX': 'e',
	'FW': 'f',
	'IN': 'i',
	'JJ': 'j',
	'JJR': 'j',
	'JJS': 'j',
	'LS': 'g',
	'MD': 'v',
	'NN': 'n',
	'NNP': 'n',
	'NNPS': 'n',
	'NNS': 'n',
	'PDT': 'd',
	'POS': 'd',
	'PRP': 'p',
	'PRP$': 'p',
	'RB': 'r',
	'RBR': 'r',
	'RBS': 'r',
	'RP': 'r',
	'SYM': 'v',
	'TO': 't',
	'UH': 'u',
	'VB': 'v',
	'VBD': 'v',
	'VBG': 'v',
	'VBN': 'v',
	'VBP': 'v',
	'VBZ': 'v',
	'WDT': 'd',
	'WP': 'p',
	'WP$': 'p',
	'WRB': 'r'}
	try:
		mapped = mapCOCA[treebank_tag]
	except:
		#If Penn Tree Mapping not mapped explictly, return 'noun'
		mapped = 'n'
	return(mapped)
 