#!user/bin/env python

def sentiment(dbparams,poems):
	#import sys
	import re
	import itertools
	import nltk
	import collections
	import MySQLdb
	from nltk.stem import WordNetLemmatizer
	from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
	from nltk.corpus import names
	from nltk.corpus import wordnet
	from nltk.tag.simplify import simplify_wsj_tag
	wnl = WordNetLemmatizer()
	#Records instances of sound devices. 
	#Sentence List for Each Poem
	sentPoems = [nltk.sent_tokenize(poem) for poem in poems]
	#Perfect Rhyme,Slant Rhyme, Alliteration, Consonance, Assonance
	PosNeg=np.zeros(len(poems))
	ABS = np.zeros(len(poems))
	EnlTot=np.zeros(len(poems))
	Female=np.zeros(len(poems))
	Male = np.zeros(len(poems))
	Female = np.zeros(len(poems))
	Object = np.zeros(len(poems))
	Polit = np.zeros(len(poems))
	Race= np.zeros(len(poems))
	Relig=np.zeros(len(poems))
	St = np.zeros(len(poems))
	WlbPhycs=np.zeros(len(poems))
	WlbPsyc=np.zeros(len(poems))
	#Harvard Inquirer Modified Document
	#Fetch the document from the database.
	##################
	#MySQL connection#
	##################
	db=MySQLdb.connect(cursorclass=MySQLdb.cursors.DictCursor,passwd=dbparams['passwd'],
		db=dbparams['db'],host=dbparams['host'],port=dbparams['port'],user=dbparams['user'])
	#Fetching the entire table.
	c = db.cursor()
	c.execute(dbparams['query'])
	q = c.fetchall()
	#Here I take a simple approach and combine the signals if the word has multiple meanings.
	#A more robust way is to look up on the PoS. 
	inquirer = {}
	#Create the dictionary. 
	for row in q:
	 	inquirer[row['Entry'].lower()] =   dict([b for b in row.iteritems()])
	#Remove the duplicate value which is also the key. 
	for k,v in inquirer.iteritems():
	     v.pop('Entry',None)
	#Handle duplicate entries. 
	duplicates = [duplicate for duplicate in set([re.sub(r'\#[0-9]','',line) for line in 
		inquirer.keys() if re.search(r'\#[0-9]',line)])]
	#Counting entries for multiple entries
	for duplicate in duplicates:
		#Pull only exact string matches
		print duplicate
		st = '\\b'+duplicate + '\\b'
		dupvalues = [inquirer[item] for item in inquirer.keys() if re.findall(st,re.sub(r'\#[0-9]','',item))]
		dupkeys = [item for item in inquirer.keys() if re.findall(st,re.sub(r'\#[0-9]','',item))]
		#Create a np array
		count = np.zeros(len(dupvalues[0].items()))
		for dupitem in dupvalues:
			count = count + np.array(dupitem.values()).astype(int)
		count = sign(count)
		#Append the dictionary with a single item. Remove all other items. 
		print 'removing ', dupkeys
		[inquirer.pop(key) for key in dupkeys]
		inquirer[duplicate] = {k:int(v) for k,v in zip(dupitem.keys(),count.tolist())}
	##Take the words at the end of the line, ignoring punctuations
	nonPunct = re.compile('.*[A-Za-z].*')
	for i,sentPoem in enumerate(sentPoems):
		#print (i)
		#Create the lists for Alliteration and Rhymes. 	
		firstPhon=[]
		lastWords=[]
		#tokenize  the word list in each sentence. 
		tokenized = [nltk.pos_tag(word_tokenize(sent)) for sent in sentPoem]
		total_words = len(list(itertools.chain(*tokenized)))
		##########################
		####Sentiment Analysis###
		##########################
		#Negation String####
		negation = re.compile('n\'t|never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint')
		punct = re.compile('^[,.:;!?()]$|that')
		for sent in tokenized:
		#Sentence by Sentence
		#Create an array to measure the negation in a given sentence. 
			#negationArray=np.zeros(len(sent))
			negate = np.ones(1)
			for wordTuple in sent: 
				#Lemmatize to do the sentiment count. 
				word = wnl.lemmatize(wordTuple[0].lower(),penn_to_wordnet(wordTuple[1]))
				#print word
				#Flip the negation sign. 
				if negation.match(word):
					negate = (-1) * negate
				#Reset negation. If punctuation, do not match the word below. 
				if punct.match(word):
					negate = np.ones(1)
				elif word in inquirer.keys():
					#Word Match. Except Negative + Positive
					wm = inquirer[word]
					ABS[i]+=int(wm['ABS']) / total_words
					EnlTot[i]+= int(wm['EnlTot'])/ total_words
					Female[i]+=int(wm['Female'])/ total_words
					Male[i]+=int(wm['MALE'])/ total_words
					Object[i]+=int(wm['Object'])/ total_words
					Polit[i]+=int(wm['POLIT'])/ total_words
					Race[i]+=int(wm['Race'])/ total_words
					Relig[i]+=int(wm['Relig'])/ total_words
					St[i]+=int(wm['St'])/ total_words
					WlbPhycs[i]+=int(wm['WlbPhys'])/ total_words
					WlbPsyc[i]+= int(wm['WlbPsyc'])/ total_words
					#Positive
					PosNeg[i] += negate * int(wm['PosNeg']) / total_words
					#print negate, int(wm['PosNeg']), PosNeg[i]
					#PosNeg[i] =  PosNeg[i]/total_words
		print 'Sentiment Score is ' , PosNeg[i]
	return(ABS,EnlTot,Female,Male,Object,Polit,Race,Relig,St,WlbPhycs,WlbPsyc,PosNeg)

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

