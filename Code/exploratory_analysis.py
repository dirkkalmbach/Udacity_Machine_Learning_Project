# -*- coding: utf-8 -*-
"""
This file contains code snippets to explore the enron dataset.
"""


#--------------------------------->
# Import Statement
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import numpy as np
import matplotlib.pyplot as plt
import pprint
features_list = ['poi','salary',"strength_of_email_conn_to_POI"]
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#<---------------------------------


#--------------------------------->
### Task 2: Remove outliers
del data_dict['TOTAL']
#<---------------------------------


#--------------------------------->
### Task 3: Create new feature(s)

# 1. Create new feature for email-connection:
def create_email_conn_feature(data_dict):
	def email_conn(from_poi, to_poi, from_mess, to_mess):
		if from_poi=="NaN" or to_poi=="NaN" or from_mess=="NaN" or to_mess=="NaN":
			return 0
		else: 
			return (from_poi+to_poi)/float((from_mess+to_mess))
	
	for key in data_dict:
		data_dict[key]["strength_of_email_conn_to_POI"] = email_conn(	
				data_dict[key]["from_poi_to_this_person"],
				data_dict[key]["from_this_person_to_poi"],
				data_dict[key]["from_messages"],
				data_dict[key]["to_messages"] )
	return data_dict

# 2. Fraction-Feature from Lesson 11-4
def computeFraction( poi_messages, all_messages ):
	fraction = 0.
	if type(poi_messages)==int and type(all_messages)==int:
		fraction = float(poi_messages) / all_messages
	return fraction

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_dict[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_dict[name]["fraction_to_poi"] = fraction_to_poi
	
# Use new feature:
data_dict = create_email_conn_feature(data_dict)

print "Number of persons: ", len(data_dict)
print "Number of features: ", len(data_dict.items()[0][1])
#<---------------------------------


#--------------------------------->
### convert data_dict to np.array
#Create list of colnames:
colnames=[]
for i in data_dict.values()[0]:
	colnames.append(i)
# get poi first entry:
i=colnames.index("poi")
colnames[0], colnames[i] = colnames[i], colnames[0]
j=colnames.index("email_address")
colnames.pop(j) #del email-adress (problems with @)

colnames.remove("from_messages")
colnames.remove("to_messages")

d = featureFormat(data_dict, colnames, sort_keys = True)
l, f = targetFeatureSplit(d)
#<---------------------------------


#--------------------------------->
#Feature Selection
print "------- SelectKBest --------"
#Use sklearn for feature selection
from sklearn.feature_selection import SelectKBest, f_classif
#selection = SelectKBest(k=3)
#f = SelectKBest(k=5).fit_transform(f,l)
t=SelectKBest(f_classif, k=7)
t.fit(f,l)

print t.get_support() #shows selected features

# to identify features print out features list (without poi)
colnames.remove('poi')
print "Feature list to compare: "
print colnames
#<---------------------------------

# Feature Importance
print "------- Extra Tree --------"
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
dataset = datasets.load_iris()
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(f, l)
# display the relative importance of each attribute
print(model.feature_importances_)
#<---------------------------------


# Recursive Feature Elimination
print "------- Logistic Regression --------"
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(f,l)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)





#data = featureFormat(data_dict, features_list, sort_keys = True)


#--------------------------------->
## FEATURE EXAMINATION
def countNaN(a):
	names_poi=[]
	names_nonpoi=[]	
	d={}
	n_poi=0
	n=0
	for key in data_dict:
		if  data_dict[key]['poi']==False:
			if data_dict[key][a] == "NaN":
				n+=1
				names_nonpoi.append(key)
			else:
				if data_dict[key][a] == "NaN":
					n_poi+=1
					names_poi.append(key)
	d = {'POI': {'count': n_poi, 'names': names_poi}, 'NoPOI': {'count': n, 'names': names_nonpoi}}
	return d	#, names_nonpoi, names_poi

print countNaN('email_address')
#<---------------------------------



#--------------------------------->
# Plotting Outliers
def PlotOutlier(f1,f2):
	x=[]; y=[]; x_poi=[]; y_poi=[]
	for key in data_dict:
		if data_dict[key][f1]!="NaN" and data_dict[key][f2]!="NaN":
			if data_dict[key]["poi"]==False:
				x.append(data_dict[key][f1])
				y.append(data_dict[key][f2])
	for key in data_dict:
		if data_dict[key][f1]!="NaN" and data_dict[key][f2]!="NaN":
			if data_dict[key]["poi"]==True:
				x_poi.append(data_dict[key][f1])
				y_poi.append(data_dict[key][f2])	
	p=plt.scatter(x_poi,y_poi,color='red')
	n=plt.scatter(x,y,color='blue', alpha=0.5)
	#plt.legend((p,n),("POI","No-POI"), loc="lower right")
	plt.xlabel(f1)
	plt.ylabel(f2)
	plt.title("Plot to detect Outliers")	
	sub_text = "No-POIs (n): " + str(len(x)) + "     POIs (n): " + str(len(x_poi))
	plt.suptitle(sub_text)
	#plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5) )
            
            
	plt.show()
	print len(x_poi), len(y_poi)
	print len(x), len(y)
#PlotOutlier("strength_of_email_conn_to_POI",'shared_receipt_with_poi')
PlotOutlier('salary', "bonus")
#<---------------------------------