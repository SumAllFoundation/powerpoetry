from yhat import Yhat, YhatModel , preprocess
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

"The module creates dictionaries (COCA and Harvard Inquirer) to create the featurespace"

#Extraact Dictionaries
def coca_retrieve(dbparams,save=False):
	w1,w2,w3={},{},{}
	db=MySQLdb.connect(cursorclass=cursors.DictCursor,passwd=dbparams['mysql']['passwd'],db=dbparams['mysql']['db'],
		host=dbparams['mysql']['host'],port=dbparams['mysql']['port'],user=dbparams['mysql']['user'])
	#Fetching the tables.
	c = db.cursor()
	for table in dbparams['mysql']['table']:
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
	#Convert to List
	w1 = [(str(k),v) for k,v in w1.items()]
	w2 = [(str(k),v) for k,v in w2.items()]
	w3 = [(str(k),v) for k,v in w3.items()]
	coca['w1'],coca['w2'],coca['w3'] = w1,w2,w3
	#Save
	if save:
		with open('coca.json','w') as f: f.write(json.dumps(coca))
	return(coca)

def inqurier_retrieve(dbparams,save=False):
	# Harvard Inquirer Modified Document
	# Fetch the document from the database.
	#################
	#MySQL connection#
	#################
	db=MySQLdb.connect(cursorclass=cursors.DictCursor,passwd=dbparams['mysql']['passwd'],
		db=dbparams['mysql']['db'],host=dbparams['mysql']['host'],port=dbparams['mysql']['port'],user=dbparams['mysql']['user'])
	#Fetching the entire table.
	c = db.cursor()
	c.execute(dbparams['mysql']['query'])
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
		count = np.sign(count)
		#Append the dictionary with a single item. Remove all other items. 
		print 'removing ', dupkeys
		[inquirer.pop(key) for key in dupkeys]
		inquirer[duplicate] = {k:int(v) for k,v in zip(dupitem.keys(),count.tolist())}
	#Save
	if save:
		with open('inquirer.json','w') as f: f.write(json.dumps(inquirer))