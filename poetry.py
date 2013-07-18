
#DataFrame
import csv
import codecs
import itertools
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import datetime
import numpy as np
from numpy.random import randn, random_integers
import unicodecsv
from __future__ import division
import re
import nltk
from pandas import concat
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.corpus import names
from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from HTMLParser import HTMLParser
from numpy.random import randint
import MySQLdb

#RETRIEVEING POEMS FROM DATABASE.
dbparams = {'column': 'body_value',
 'db': 'poetry',
 'host': 'projects.cwkjvkbpph6y.us-east-1.rds.amazonaws.com',
 'passwd': 'ShYbjS70anhCnXpV',
 'port': 3306,
 'query': 'SELECT field_data_body.body_value, node.nid, node.title, node.created, node.uid, users.name FROM field_data_body, node, users WHERE field_data_body.entity_id =  node.nid and node.uid = users.uid;',
 'table': '',
 'index': 'nid',
 'user': 'admin',
 'column': 'title'}


#RETRIEVING LOCATIONS FROM DATABSE.
dbparams = {'db': 'poetry',
 'host': 'projects.cwkjvkbpph6y.us-east-1.rds.amazonaws.com',
 'index': 'nid',
 'passwd': 'ShYbjS70anhCnXpV',
 'port': 3306,
 'query': 'SELECT c.nid, b.latitude, b.longitude from field_data_field_location_one a, location b, node c where c.nid = a.entity_id and a.field_location_one_lid = b.lid;',
 'user': 'admin'}
 #loc = db_to_df(dbparams)
 #loc = loc.sort_index()
 #loc_with_features = features.join(loc,how='inner')
 #loc_with_features.index = [loc_with_features['latitude'],loc_with_features['longitude']]
#loc_with_features.drop(['latitude','longitude','created','uid'],axis=1)


def db_to_df(dbparams):
	from pandas import DataFrame, Series
	import MySQLdb
	import MySQLdb.cursors
	import pandas.io.sql as psql
	import re
	db=MySQLdb.connect(cursorclass=MySQLdb.cursors.DictCursor,passwd=dbparams['passwd'],db=dbparams['db'],host=dbparams['host'],port=dbparams['port'],user=dbparams['user'])
	#Fetching the entire table.
	mysqldf = psql.frame_query(dbparams['query'], con=db)
	#Index it to the index
	mysqldf.index = mysqldf[dbparams['index']]
	if 'columns' in dbparams.keys():
		#Processing the text
		processed_text = []
		#Iterate over DataFrame to decode and remove 'weird characters'
		processed_text = [temp[dbparams['column']].decode('ascii','ignore') for i, temp in mysqldf.T.iteritems()]
		#Recreated text Series
		df = Series(processed_text,index = mysqldf[dbparams['index']])
		#Drop the previous one, add the new one
		mysqldf=mysqldf.drop(dbparams['column'],axis=1)
		mysqldf[dbparams['column']] = df
	mysqldf = mysqldf.drop(dbparams['index'],axis=1)
	return(mysqldf)

def preprocess(mysqldf,dbparams):
	#Accepts a DataFrame. Corrects for start of the sentences and removes html notations. 
	import re
	from pandas import DataFrame, Series
	textDF = mysqldf[dbparams['column']]
	posttext = []
	for line in textDF:
		#No space between lines. 
		line = re.sub('[\r\n]+','',line)
		line = re.sub('[\t]+','',line)
		#(i)Preprocess punctional mistakes
		mistakes_dot = re.findall('[a-z]+[.!?][a-zA-Z]',line)
		#Find instances where there is no space between last word and first word.
		mistakes_nodot = re.findall('[a-z]+[A-Z]',line)
		#Clean the mistakes
		for mistake in mistakes_dot:
			punct = re.findall(r'\W',mistake)[0]
			search_term = mistake.split(punct)[0]+'\\' + punct + mistake.split(punct)[1]
			line = re.sub(search_term,mistake.split(punct)[0] + punct +' '+ mistake.split(punct)[1],line)
		for mistake in mistakes_nodot:	
			capital_letter = re.findall('[A-Z]',mistake)[0]
			line =re.sub(mistake,mistake.split(capital_letter)[0] + '. '+ capital_letter+ mistake.split(capital_letter)[1],line)
		#Append
		posttext.append(line)
	#Merge1
	#Recreated text Series
	s = Series(posttext,index = mysqldf[dbparams['index']])
	#Drop the previous one, add the new one
	df=mysqldf.drop(dbparams['column'],axis=1)
	df[dbparams['column']] = s
	df.index = df['nid']
	df = df.sort_index()
	return(df)

#####Creating the Feature Space
poems = mysqldf['body_value']
#Run randomized trials
poems = mysqldf['body_value'].iloc[randint(1,len(df),50)]
#Sound
[perfectRhymeFreq,slantRhymeFreq,alliterFreq] = sound(poems)
#Sentiment
dbparams = {'passwd': 'ShYbjS70anhCnXpV', 'db': 'opendata', 'host': 'projects.cwkjvkbpph6y.us-east-1.rds.amazonaws.com', 'port':3306, 'user':'admin','query':'select * from inquirer'}
[ABS,EnlTot,Female,Male,Object,Polit,Race,Relig,St,WlbPhycs,WlbPsyc,PosNeg] =sentiment(dbparams,poems)
#Ngram
dbparams = {'passwd': 'ShYbjS70anhCnXpV', 'db': 'COCA', 'host': 'projects.cwkjvkbpph6y.us-east-1.rds.amazonaws.com', 'port':3306, 'user':'admin','table':['w1','w2','w3']}
[unigramFreq,bigramFreq,trigramFreq,infrequentUni,infrequentBi,infrequentTri,misspeltWord, sentence_count,word_count,punct_count] = ngram(dbparams,poems)


#create the feature space indexed to #nid
# sentiment=read_csv('sentiment.csv')
# ngram= read_csv('ngram.csv')
# sound = read_csv('sound.csv')
# sentiment.index = sentiment['nid']
# sound.index = sound['nid']
# ngram.index= ngram['nid']
# sentiment = sentiment.drop('nid',axis=1)
# sound = sound.drop('nid',axis=1)
# ngram = ngram.drop(['nid','created','uid'],axis=1)
# features = sentiment.join(ngram,how='inner')
# features = features.join(sound,how='inner')

#Transform Data Index Structure from time posted to post count.
###Indexing-> Take out uid and created from all but one. use join 
def transformCount(df):
	df.index = [df['uid'],df['created']]
	df = df.sort_index()
	df = df.drop(['created','uid'],axis=1)
	#Reindex to count from data
	ix = DataFrame(df.index.tolist())[0].value_counts().sort_index()
	nix = [(user,i+1) for user,x in ix.iteritems() for i in range(x)]
	nix = DataFrame(nix)
	df.index = [nix[0],nix[1]]
	return(df)

#XML - Title Match
from xml.dom.minidom import parseString
import urllib2
f = urllib2.urlopen('http://www.powerpoetry.org/good-feed')
data=f.read()
f.close()
dom = parseString(data)
#Search by title
#Retrieve the quantified features for good poems.
good_titles = [title.firstChild.nodeValue.lower() for title in dom.getElementsByTagName('title')]
all_titles = [title.lower().decode('ascii','ignore') for  title in  df['title']]
good_poems_iloc= [] #nid is the index for data structure. 
for title in good_titles:
	if title in all_titles:
		#Extract the features
		good_poems_iloc.append(all_titles.index(title))
#DataFrame containing quality poems. 
good_poems = features.iloc[good_poems_iloc]
good_poems.index = df['nid'].iloc[good_poems_iloc]
#Drop good poems
ameteuer_poems = features
ameteuer_poems.iloc[good_poems_iloc] = np.nan
ameteuer_poems = ameteuer_poems.dropna()


#MODEL#########################################
#INPUTS# 
#random ameteuer poems for training/testing
rng = np.arange(len(ameteuer_poems))
np.random.shuffle(rng)
samp = rng[:len(good_poems)].tolist()
#Combine Datasets / Labels
data = concat([ameteuer_poems.iloc[samp],good_poems])
Y = np.concatenate((np.zeros(len(ameteuer_poems.iloc[samp])),np.ones(len(good_poems))))
#Concatenate Feature Space
X = np.column_stack((data.trigramFreq,data.misspeltWord, data.punctCount, data.wordCount, data.perfectRhymeFreq,data.slantRhymeFreq,data.alliterFreq,data.ABS,data.Object,data.PosNeg,data.Polit,data.Race))
#Divide into test/training
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)

#Regularization parameter for cross-validation
#A smaller number is a stronger regularized in Skit. 
C= np.arange(100,0.0,-0.5)[:-1]
score = np.zeros(len(C))
for i,c in enumerate(C):
	clf = LogisticRegression(fit_intercept=False,penalty='l2',C=c)
	#clf = clf.fit(X_train,y_train)
	#Leave One Out
	scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10)
	score[i]= scores.mean()
	print "Accuracy: %0.0f is %0.2f)" % (c, score[i])
# regResult = DataFrame(score,index=C)
# regResult.plot()
#C = 25 in this sample
clf = LogisticRegression(fit_intercept=False,penalty='l2',C=10).fit(X_train,y_train)
#sum(clf.predict(X_test)==y_test)/len(y_test)

#Feature Subset
subfeatures = features[['trigramFreq','misspeltWord','punctCount','wordCount',
	'perfectRhymeFreq','slantRhymeFreq','alliterFreq','ABS','Object','PosNeg','Polit','Race']]
test = np.column_stack((features.trigramFreq,features.misspeltWord,features.punctCount,features.wordCount,
	features.perfectRhymeFreq,features.slantRhymeFreq,features.alliterFreq,
	features.ABS,features.Object,features.PosNeg,features.Polit,features.Race))
for line in test:
	odds_ratio = np.exp(dot(clf.coef_.T, line).squeeze())
	scores.append(1/(1+odds_ratio))
score = Series(scores,index=features.index)
score.name = 'Score'
features = features.join(score,how='inner')








# #Time Series######################################
# #All the features

# #Lexical Diversity
# poemDF['lexical_diversity'] = [lexical_diversity(text) for text in poemDF.poem]
# lexDiv = Series([lexical_diversity(text) for text in poemDF.poem],index = [poemDF['user_id'],poemDF['poem_created']])
# lexDiv = lexDiv.sort_index()
# #Poets
# poets = [x[0] for x in lexDiv.index]
# poets = np.unique(np.array(poets))

# #Plotting the progress
# fig = plt.figure()
# ax1=fig.add_subplot(2,1,1)
# #Limit x and y axis
# plt.xlim([0,20])
# plt.ylim([1,5])
# #Recursively draw the progress plot. Calculate the mean lexical diversity.
# count = numpy.zeros(poemDF.user_id.value_counts().max(),np.integer)
# measure = numpy.zeros(poemDF.user_id.value_counts().max(),np.float64)
# for poet in poets:
#     plt.plot(lexDiv[poet])
#     for k,v in enumerate(lexDiv[poet]):
#     	count[k]+=1
#     	measure[k] = (count[k]- 1)/count[k] * measure[k] + 1/count[k]* v
# ax1.set_title('Progress for Lexical Diversity')
# ax1.set_ylabel('Lexical Diversity')
# ax2 = fig.add_subplot(2,1,2)
# ax2.plot(measure)
# ax2.set_xlabel('# of Poems Posted')
# ax2.set_ylabel('Lexical Diversity')
# plt.xlim([0,20])
# plt.ylim([1,2])











