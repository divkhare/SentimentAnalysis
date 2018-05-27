#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# for every text file say positive of negative
#
"""
Created on Sat Apr 24 23:33:23 2018

@author: divkhare
"""
## PLEASE CHANGE PATH AND PATH2 TO YOUR PATH 
#this code runs trhough given text files of movie reviews (2 given folders-- pos and neg) and labels it positive or 
#negative . It then takes this data as a train set and trains itself through multiple machine learning algorithms.
#Logistic regression is found to be the most accurate and is used to further take an input from the user and analyse 
# if that input is positive or negative. This code is user interactive. so please give an input at the end of the code.

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import glob
import errno

path = '/Users/divkhare/Downloads/train/pos/*.txt' #path to all postive reviews text files
path2 ='/Users/divkhare/Downloads/train/neg/*.txt'#path to all postive reviews text files
data = [] #list to save sentiment data
data_labels = [] #list to save review data 
print ("path of positive reviews folder: ",path)
print ("path of negative reviews folder: ",path)
glo = glob.glob(path) #glob used to read text file
glo2 = glob.glob(path2)
posdata=[]
negdata=[]
##############################################################
for text in glo: #read all text file
	try:
		with open(text) as f: #loop to run through all text files
			glob = f.read() #reads data within text file
	except IOError as exc:
		if exc.errno != errno.EISDIR: # to avoid fail if a directory is found, ignores it.
			raise # Propagates other kinds of IOError.
	posdata.append(glob) #appends every line to list called "m" (review)
for text in glo2: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
	try:
		with open(text) as f:
			glob2 = f.read()
	except IOError as exc:
		if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
			raise # Propagates other kinds of IOError.
	negdata.append(glob2)#appends every line to list called "n" (sentiment)
###############################################################
#Data in Data
#pos and neg respectively
for i in posdata: 
	data.append(i) #reads through the list m and appends data to the list "data"
	data_labels.append('1') #review is positive --> labeled '1'
for i in negdata: 
	data.append(i )#reads through the list m and appends data to the list "data"
	data_labels.append('0')	#review is negative --> labeled '0'
################################################################
	
data_tuples = list(zip(data,data_labels)) #merges both lists (data and data_labels)
data = pd.DataFrame(data_tuples, columns=['review','sentiment']) #use panda to create a table with columns "review,sentiment"

X = data.review
y = data.sentiment
vect = CountVectorizer(stop_words='english',analyzer='word',strip_accents='unicode', ngram_range = (1,1), max_df = .80, min_df = 4)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)
#Using training data to transform text into counts of features for each message
vect.fit(X_train)
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)

###############################################################
#Accuracy using Naive Bayes Model
NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)
print('\nNaive Bayes')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')

#Accuracy using Logistic Regression Model
LR = LogisticRegression()
LR.fit(X_train_dtm, y_train)
y_pred = LR.predict(X_test_dtm)
print('\nLogistic Regression')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')

#Accuracy using SVM Model
SVM = LinearSVC()
SVM.fit(X_train_dtm, y_train)
y_pred = SVM.predict(X_test_dtm)
print('\nSupport Vector Machine')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')

#Accuracy using KNN Model
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train_dtm, y_train)
y_pred = KNN.predict(X_test_dtm)
print('\nK Nearest Neighbors (NN = 3)')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')

###############################################################

tokens_words = vect.get_feature_names() #all features 
print('\nAnalysis')
print('No. of tokens: ',len(tokens_words)) #total number of words in data set without stopwords and characters
counts = NB.feature_count_ 
df_table = {'Token':tokens_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative']) #creates a table to store and count all positive/negative wods
positives = len(tokens[tokens['Positive']>tokens['Negative']])
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',len(tokens_words)-positives)

###############################################################

print("\nCheck positivity/negativity of specific tokens: ")
token_search = ['good'] # As an extra : code searches for a specific word: "good"
print('\nSearch Results for token :',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])

token_search = ['bad'] # As an extra : code searches for a specific word: "good"
print('\nSearch Results for token :',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])


###############################################################

#Custom Test: Tests a review on the best performing model (Logistic Regression)
trainingVector = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 5)
trainingVector.fit(X)
X_dtm = trainingVector.transform(X)
LR_complete = LogisticRegression()
LR_complete.fit(X_dtm, y)
#Input Review analysis
print('\nTest a custom review message')
print('Enter review to be analysed: ', end=" ")
test = [] 
test.append(input())
test_dtm = trainingVector.transform(test)
predLabel = LR_complete.predict(test_dtm)
predLabel = list(map(int, predLabel))
tags = ['Negative','Positive']
#Doutput
print('The review is predicted',tags[predLabel[0]])

###############################################################