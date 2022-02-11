import numpy as np
import pandas as pd
import random
from string import punctuation
from collections import OrderedDict

def load(path):
	df = None
	df = pd.read_csv("TRAIN_balanced_ham_spam.csv")
	return df

def prior(df):
	ham_prior = 0
	spam_prior =  0
	ham_prior = df["r"].value_counts()[0] / (df.shape[0])
	spam_prior = df["r"].value_counts[1] / (df.shape[0])
	return ham_prior, spam_prior

def likelihood(df):
	ham_likelihood = {}
	spam_likeihood= {}
	for i in range(df["r"].value_counts()[0]):
    		val = df.iloc[i,3].split()
			email_list = list(OrderedDict.fromkeys(val))
			for n in email_list:
    			if n.lower() not in ham_likelihood and n not in punctuation:
    				ham_likelihood[n] = 1
				elif n.lower() in ham_likelihood:
    					ham_likelihood[n] = ham_likelihood.get(n) + 1
	for i in ham_likelihood:
    		ham_likelihood[i] = ham_likelihood.get(i) / df["r"].value_counts()[0]
	for i in range(df["r"].value_counts()[0],df.shape[0]):
    		
	return ham_like_dict, spam_like_dict

def predict(ham_prior, spam_prior, ham_like_dict, spam_like_dict, text):
	'''
	prediction function that uses prior and likelihood structure to compute proportional posterior for a single line of text
	'''

	#ham_spam_decision = 1 if classified as spam, 0 if classified as normal/ham
	ham_spam_decision = None




	'''YOUR CODE HERE'''
	#ham_posterior = posterior probability that the email is normal/ham
	ham_posterior = None

	#spam_posterior = posterior probability that the email is spam
	spam_posterior = None

	'''END'''
	return ham_spam_decision


def metrics(ham_prior, spam_prior, ham_dict, spam_dict, df):
	'''
	Calls "predict"
	'''
    hh = 0 #true negatives, truth = ham, predicted = ham
    hs = 0 #false positives, truth = ham, pred = spam
    sh = 0 #false negatives, truth = spam, pred = ham
    ss = 0 #true positives, truth = spam, pred = spam
    num_rows = df.shape[0]
    for i in range(num_rows):
        roi = df.iloc[i,:]
        roi_text = roi.text
        roi_label = roi.label_num
        guess = predict(ham_prior, spam_prior, ham_dict, spam_dict, roi_text)
        if roi_label == 0 and guess == 0:
            hh += 1
        elif roi_label == 0 and guess == 1:
            hs += 1
        elif roi_label == 1 and guess == 0:
            sh += 1
        elif roi_label == 1 and guess == 1:
            ss += 1
    
    acc = (ss + hh)/(ss+hh+sh+hs)
    precision = (ss)/(ss + hs)
    recall = (ss)/(ss + sh)
    return acc, precision, recall
    
if __name__ == "__main__":
	df=load(1)
	ham_prior, spam_prior = prior(df)
	ham_
	#this cell is for your own testing of the functions above