#!user/bin/env python
from __future__ import division

def sound(poems):
	import nltk
	prondict = nltk.corpus.cmudict.dict()
	from nltk import word_tokenize
	import sys
	import re
	import itertools
	import collections
	#Records instances of sound devices. 
	#Split by new line 
	sentTokenizedPoems = [poem.split('\r\n') for poem in poems]
	#Perfect Rhyme,Slant Rhyme, Alliteration, Consonance, Assonance
	perfectRhymeFreq=np.zeros(len(poems))
	slantRhymeFreq=np.zeros(len(poems))
	alliterFreq=np.zeros(len(poems))
	#consonFreq=np.zeros(len(poems))
	#assonFreq=np.zeros(len(poems))
	##Take the words at the end of the line, ignoring punctuations
	nonPunct = re.compile('.*[A-Za-z].*')
	for i,sentTokenizedPoem in enumerate(sentTokenizedPoems):
		#print (i)
		#Create the lists for Alliteration and Rhymes. 	
		firstPhon=[]
		lastWords=[]
		#tokenize  the word list in each sentence. 
		tokenized = [word_tokenize(sent) for sent in sentTokenizedPoem]
		total_words = len(list(itertools.chain(*tokenized)))
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
								perfectRhymeFreq[i] += (1/total_words)
								print 'Perfect:  ',vpprim, vpsec, spprim, spsec
							#Slant Rhyme
							if conb ^ cond:
								slantRhymeFreq[i] += (1/total_words)
								print 'Slant: ',vpprim, vpsec, spprim, spsec 
	return(perfectRhymeFreq,slantRhymeFreq,alliterFreq)


			#Collapse the phenomes list of lists. 
			#Measure consonance and assonance.
			#Note that flattening removes empty items in the list.
			#Count the number within 20 phonemes. 
			# window_length = 20
			# phenomes_flat = list(itertools.chain(*phenomes))
			# consonFreq[i] += [1]
			# assonFreq[i] += [1]
			# #Break into rolling windows. 
			# windows = [phenomes_flat[n:n+window_length] for n,p in enumerate(phenomes_flat) if n + window_length <= len(phenomes_flat)]
			# #Detect Duplicates
			# dup=[]
			# for window in windows:
			# 	dup.append(set([x for x, y in collections.Counter(window).items() if y > 1]))
			# #Collapse
			# dup = list(itertools.chain(*dup))
			# #Count
			# #Consonance
			# assonFreq[i] = len([p for p in dup if re.search(r'\d+',p)]) / total_words
			# consonFreq[i] = len([p for p in dup if not re.search(r'\d+',p)]) / total_words



