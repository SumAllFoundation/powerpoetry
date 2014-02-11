
import csv
import pandas as pd
import numpy as np
import scipy as sp
import re
import MySQLdb
from MySQLdb import cursors
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.corpus import names, wordnet
import sys, getopt
import re
import itertools
import json

"The module creates featurespace for given iterable and ranks each feature vs the corpus characteristics"


#Extract Features
def ngram(poems,coca):
	#N-gram Function 
	coca_word_count = 450000000
	#Import
	wnl = WordNetLemmatizer()
	#Setting up the params
	#Sentence List for Each Poem
	sentTokenizedPoems=[]
	for i,poem in enumerate(poems):
		#Take out the new line and replace with a space. 	
		print i
		if isinstance(poem,float):
			sentTokenizedPoems.append('')
		else:
			poem = re.sub('\n',' ',poem)
			sentTokenizedPoems.append(nltk.sent_tokenize(poem))
	#sentTokenizedPoems = [nltk.sent_tokenize(poem) for poem in poems]
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
	wordCount = np.zeros(len(poems))
	logWordCount = np.zeros(len(poems))
	#Punct Count
	punctFreq = np.zeros(len(poems))

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
		filtered = [(w[0].lower(),w[1]) for w in tokenized if nonPunct.match(w[0])]
		if len(tokenized) == 0:
			punctFreq[i] = 0 
		else:
			punctFreq[i] = 1 - (len(filtered) / len(tokenized)) 
		#Word Count
		wordCount[i] = len(filtered)
		logWordCount[i] = np.log(wordCount[i])
		##Simplify PoS in order to be able to lemmatize. 
		#Wordnet Lemmatizer knows  Adj(a), Adverb(v), Noun(n),Verb(v)
		#Map Penn tree to wordnet for lemmatizing.
		#Map Penn tree to COCA for frequency analysis.
		#Lemmatized Word + COCA PoS
		lemmatized  = [(wnl.lemmatize(t[0],penn_to_wordnet(t[1])),penn_to_coca([1])) for t in filtered]
		#COCA Dictionary
		w1 = {item[0]:item[1] for item in coca['w1']}
		w2 = {item[0]:item[1] for item in coca['w2']}
		w3 = {item[0]:item[1] for item in coca['w3']}

		####UNIGRAM MEASUREMENT
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
						infrequentUni[i] += 1 / wordCount[i] 
						#print (wordTuple, ' infrequent')
					#See if the word is a name. Capitalize the first Letter!
					#If the letter is longer than three letters.
					elif not (wordTuple[0].capitalize() in names.words() or len(wordTuple[0])<=3):
					#Remove from body. Add to Misspelt.
						misspeltWord[i] += 1 / wordCount[i]
						#print (wordTuple, ' misspelt?')
					#else:
						#print (wordTuple, ' not recognized.')
			#Calculate Frequency if count different than zero.
			if count > 0:
				unigramFreq[i]=(1/(nw))*(count/coca_word_count) + ((nw-1)/(nw)) * unigramFreq[i]

		#####BIGRAM FUNCTION:
		#print 'Bigram: Poem ' + str(i) + ': ' 
		#Tokenize
		tokenized=[nltk.bigrams(word_tokenize(sentence)) for sentence in sentTokenizedPoem]
		#Flatten the List of Tuple List
		tokenized = list(itertools.chain(*tokenized))
		##Remove any tuple that has punctuation or number in it. Lower letter.
		nonPunct = re.compile('.*[A-Za-z].*')
		filtered = [(w[0].lower(),w[1].lower()) for w in tokenized if nonPunct.match(w[0]) and nonPunct.match(w[1])]

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
					infrequentBi[i] += 1 / wordCount[i]
					nw +=1 
					count = 10
					#print (wordTuple, ' is infrequent')
			#Append Frequency 
			if count > 0:
				bigramFreq[i] = (1/(nw))* (count) + ((nw-1)/(nw)) * bigramFreq[i]
		
		######TRIGRAMS
		#print 'Trigram: Poem ' + str(i) + ': ' 
		#Tokenize
		tokenized=[nltk.trigrams(word_tokenize(sentence)) for sentence in sentTokenizedPoem]
		#Flatten the List of Tuple List
		tokenized = list(itertools.chain(*tokenized))
		##Remove any tuple that has punctuation or number in it. Lower letter.
		nonPunct = re.compile('.*[A-Za-z].*')
		filtered = [(w[0].lower(),w[1].lower(),w[2].lower()) for w in tokenized if nonPunct.match(w[0]) 
						and nonPunct.match(w[1]) and nonPunct.match(w[2])]

		#Number of words for the loop
		nw = 0
		for w,wordTuple in enumerate(filtered):
			#Reset count
			count = 0
			try:
				#print wordTuple
				count = w3[wordTuple]
				nw+=1
				#print (wordTuple, ' found: ' + str(count))
			except:
				#Check if words exists in WordNet
				if (wordnet.synsets(wordTuple[0])) and (wordnet.synsets(wordTuple[1])) and (wordnet.synsets(wordTuple[2])):
					infrequentTri[i] += 1 / wordCount[i]
					nw +=1 
					count = 10
					#print (wordTuple, ' is infrequent')
			#Append Frequency 
			if count > 0:
				trigramFreq[i] = (1/(nw))* (count) + ((nw-1)/(nw)) * trigramFreq[i]
		#Log Frequnecy
		bigramFreq[i] = np.log(bigramFreq[i])
		trigramFreq[i] = np.log(trigramFreq[i])

	return(unigramFreq,bigramFreq,trigramFreq,misspeltWord,
		sentence_count,logWordCount,punctFreq)

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

def sound(poems):
	prondict = nltk.corpus.cmudict.dict()
	#Records instances of sound devices.
	nonPunct = re.compile('.*[A-Za-z].*') 
	#Split by new line 
	#Convert Float
	sentTokenizedPoems=[]
	for poem in poems:
		if isinstance(poem,float):
			sentTokenizedPoems.append('')
		else:
			sentTokenizedPoems.append(poem.split('\n'))
	#sentTokenizedPoems = [poem.split('\r\n') for poem in poems]
	#Perfect Rhyme,Slant Rhyme, Alliteration, Consonance, Assonance
	perfectRhymeFreq=np.zeros(len(poems))
	slantRhymeFreq=np.zeros(len(poems))
	alliterFreq=np.zeros(len(poems))
	#consonFreq=np.zeros(len(poems))
	#assonFreq=np.zeros(len(poems))
	##Take the words at the end of the line, ignoring punctuations
	for i,sentTokenizedPoem in enumerate(sentTokenizedPoems):
		print i
		#Create the lists for Alliteration and Rhymes. 	
		firstPhon=[]
		lastWords=[]
		#tokenize  the word list in each sentence. 
		tokenized = [word_tokenize(sent) for sent in sentTokenizedPoem]
		#total_words = len(list(itertools.chain(*tokenized)))
		#Excluding punctuations
		total_words = len([w for w in list(itertools.chain(*tokenized)) if nonPunct.match(w)])
		######################################################
		####Measure the Alliteration, Consonance, Assonance###
		######################################################
		##Alliteration:Find the first phoneme of first pronounciation of each word.
		##Consonance: Find the matching consonant phenomes.
		##Assonance: Find the matching vowel phenomes. 
		for sent in tokenized:
		#Sentence by Sentence	
			phenomes=[]
			for word in sent:
				#Retrieve the Phenomes
				try:
					phenomes.append(prondict[word.lower()][0])
				except:
					phenomes.append('')
					print ('item: ', word, ' not found in CMU Dictionary')
			#Alliteration: Find the first phenome.
			#Start from the second unit to match the consecutive ones.
			for n, phon in enumerate(phenomes[1:]):
				##Making sure (i) cons first ph match (ii) cons (iii) not NaN.
				nextfp = phenomes[n]
				lastfp = phenomes[n-1]
				#print nextfp, lastfp
				if nextfp and lastfp:
					#First Phenome for consecutive words. Hence [0] 
					if nextfp[0] in lastfp[0] and not re.search(r'\d+',nextfp[0]):
						print 'Alliteration match: ', nextfp,' ', lastfp
						alliterFreq[i] += (1/ total_words)
		###################################################
		###########Perfect and Slant Rhymes################
		##################################################
		#Find the last word in each line
		for line in tokenized:
			for z,word in enumerate(line):
				if nonPunct.match(line[-(1+z)]):
					#Lower
					lastWords.append(line[-(1+z)].lower())
					break
		#Retrieve the pronounciation from CMU Dictionary
		#[a for a in pdict if a[0] in 'world']
		plist=[]
		for word in lastWords:
			try:
				#The fist pronounciation.
				plist.append(prondict[word][0])
			except:
				print ('item: ', word, ' not found in CMU Dictionary.')
				plist.append([])
		##Divide the pronounciation of last words into rolling windows of 4 .
		window_length = 4
		windows = [plist[n:n+window_length] for n,p in enumerate(plist)  if n + window_length <= len(plist)]
		#Perfect and Slant Ryhmes within rolling windows
		for window in windows:
			#For each window go through the lines and match other lines.
			#End the loop at window[:-1] to avoid duplicates. 
			#eg. 1 vs 2,3,4; 2 vs 3,4; 3 vs 4. 
			for x, line in enumerate(window[:-1]):
				##Stressed Vowel Phoneme: Ends with 1. Tuple. Last one.
				##If there is no stressed vowel e.g. and. skip the line. 
				##because rhymes depend on the existince of stressed vowel.
				vpprim = [(n,p) for n,p in enumerate(line) if re.search('1',p)]
				if vpprim:
					#Primary Initial Phoneme
					ipprim = line[0]
					#phoneme sequences from the stressed vowel phoneme onward.
					#else stressed vowel phoneme is the last phoneme
					#vpprim[-1][1] is to access the last primary vowel. 
					if vpprim[-1][1] not in line[-1]:
						spprim = line[1+vpprim[-1][0]:]
					else:
						spprim = [vpprim[-1][1]]
					#Loop Remaning Lines to match the conditions.
					#Secondary. Match starts from x+1 on to avoid duplicates. 
					for y, match in enumerate(window[x+1:]):
						vpsec = [(n,p) for n,p in enumerate(match) if re.search('1',p)]
						if vpsec:
							#Secondary Initial Phoneme.
							ipsec = match[0]
							#Phoneme seq from the stressed vowel on
							#else stressed vowel phoneme is the last phoneme
							if vpsec[-1][1] not in match[-1]:
								spsec = match[1+vpsec[-1][0]:]
							else:
								spsec = [vpsec[-1][1]]
							#Perfect Rhyme and Slant Rhyme
							#Different initial consonants.
							#Matching stressed vowel phoneme:
							#Matching phoneme sequences after vowel.
							#Matching the last phoneme
							cona = (ipsec not in ipprim)
							conb = (vpsec[-1][1] in vpprim[-1][1])
							conc = (spprim == spsec)
							cond = (spprim[-1] == spsec[-1])
							#Perfect Rhyme
							if  cona and  conb and  conc:
								perfectRhymeFreq[i] += (1/len(windows[0]))
								#print 'Perfect:  ',vpprim, vpsec, spprim, spsec
							#Slant Rhyme
							if conb ^ cond:
								slantRhymeFreq[i] += (1/len(windows[0]))
								#print 'Slant: ',vpprim, vpsec, spprim, spsec

	return(perfectRhymeFreq,slantRhymeFreq,alliterFreq)


def sentiment(poems,inquirer):
	#Sentiment Analysis based on bag of words in Harvard Inquirer dictionary.
	#Poems are iterable object, inquirer is the dictionary. 

	wnl = WordNetLemmatizer()
	#Records instances of sound devices. 
	nonPunct = re.compile('.*[A-Za-z].*')
	#Sentence List for Each Poem
	sentPoems=[]
	for poem in poems:
		#Take out the new line and replace with a space. 	
		if isinstance(poem,float):
			sentPoems.append('')
		else:
			poem = re.sub('\n',' ',poem)
			sentPoems.append(nltk.sent_tokenize(poem))
	#sentPoems = [nltk.sent_tokenize(poem) for poem in poems]
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


	##Take the words at the end of the line, ignoring punctuations
	for i,sentPoem in enumerate(sentPoems):
		print 'sentence:'
		print sentPoem
		#Create the lists for Alliteration and Rhymes. 	
		firstPhon=[]
		lastWords=[]
		#tokenize  the word list in each sentence. 
		tokenized = [nltk.pos_tag(word_tokenize(sent)) for sent in sentPoem]
		#total_words = len(list(itertools.chain(*tokenized)))
		total_words = len([(w[0].lower(),w[1]) for w in list(itertools.chain(*tokenized)) if nonPunct.match(w[0])])
		##########################
		####Sentiment Analysis###
		##########################
		#Negation String####
		negation = re.compile('n\'t|never|no|nothing|signnowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint')
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

def execute(data,inquirer,coca,percentiles,save=True):

	#Sentiment
	sentiment_features = sentiment(data,inquirer)
	ABS,EnlTot,Female,Male,Object,Polit,Race,Relig,St,WlbPhycs,WlbPsyc,PosNeg = sentiment_features
	#NGrams
	ngram_features = ngram(data,coca)
	unigramFreq,bigramFreq,trigramFreq,misspeltWord, sentence_count,logWordCount,punctFreq = ngram_features
	#Sound
	sound_features = sound(data)
	perfectRhymeFreq,slantRhymeFreq,alliterFreq = sound_features
	#Output - Just few features
	features =pd.DataFrame([perfectRhymeFreq,slantRhymeFreq,Polit,Race,Relig,PosNeg,unigramFreq,bigramFreq,trigramFreq,misspeltWord, sentence_count,logWordCount,punctFreq]).transpose()
	features.columns= ['perfectRhymeFreq','slantRhymeFreq','Polit','Race','Relig','PosNeg','unigramFreq','bigramFreq','trigramFreq','misspeltWord', 'sentence_count','wordCount','punctFreq']		
	#Subset of Percentiles Data
	benchmarks = percentiles[features.columns]
	#Calculate Percentiles
	output =[]
	for i,feature in features.iterrows():
		output.append({k: int(sp.stats.percentileofscore(benchmarks[k],v)) for k,v in feature.iteritems()})
	if save:
		with open('output.json','w') as f: f.write(json.dumps(output))
		print 'file saved'
	return output


if __name__ == "__main__":


	print 'Deploy the powerpoetry script'
	#Input Parameters - Required Packages, Percentiles (Dataset the user input will be compared to), Inquirer (Inquirer bag of words), COCA dictionaries)
	try:
		opts,args = getopt.getopt(sys.argv[1:],'p:i:c:d', ['percentiles=','inquirer=','coca=','data='])
	except getopt.GetoptError:
		pass
	#Parse Arguments
	for opt, arg in opts:
		if opt in ('-p','--percentiles'):
			percentiles_filename = arg
		elif opt in ('-i','--inquirer'):
			inquirer_dict_filename = arg
		elif opt in ('-c','--coca'):
			coca_dict_filename = arg
		elif opt in ('-d','--data'):
			keypath = arg

		

	#Load the files
	percentiles = pd.read_csv(percentiles_filename)
	#Inquirer Dict
	with open(inquirer_dict_filename) as f:	
		inquirer = json.loads(f.read())
	#COCA Dict
	with open(coca_dict_filename) as f:	
		coca = json.loads(f.read())
	#Data - Input as Iterable. 
	with open('%s'% (keypath)) as f:
	    data  =  json.loads(f.read())

	#Run
	output = execute(data,inquirer,coca,percentiles)


	#python /Users/denizzorlu/Dropbox/code/powerpoetry/poetry_percentile_rank.py  --data /Users/denizzorlu/Dropbox/code/powerpoetry/data/data.json --coca /Users/denizzorlu/Dropbox/code/powerpoetry/data/coca.json --inquirer /Users/denizzorlu/Dropbox/code/powerpoetry/data/inquirer.json --percentiles /Users/denizzorlu/Dropbox/code/powerpoetry/data/percentiles.csv
	


