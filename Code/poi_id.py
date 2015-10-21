#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary','bonus']
features_list = ['poi', 'deferred_income', 'long_term_incentive', 'bonus', 'total_stock_value', 'salary', 'fraction_to_poi', 'exercised_stock_options']
#features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'fraction_to_poi']
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


### Task 2: Remove outliers
del data_dict['TOTAL']



### Task 3: Create new feature(s)
##########################################################
# Feature Scaling-Function (if necessar):
def feature_scaling(feature):
	from sklearn import preprocessing
	min_max_scaler = preprocessing.MinMaxScaler()
	#min_max_scaler.fit_transform(value)
	return min_max_scaler.fit_transform(feature)
	
# 1. Fraction-Feature from Lesson 11-4
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


# 2. Create new feature for email-connection:
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
	
# Use new feature:
data_dict = create_email_conn_feature(data_dict)

##########################################################



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Split the dataset in 70:30 (training : testing)
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=40)




#I added this code snippet only for showing how validation can be done with 
#KFold. For the project report I use the scores from tester.py.

# Splitting using KFold

#kfold
from sklearn.cross_validation import KFold
kf=KFold(len(labels), 2)
for train_indices, test_indices in kf:
	#make training and testing datasets
	features_train = [features[ii] for ii in train_indices]
	features_test = [features[ii] for ii in test_indices]
	labels_train = [labels[ii] for ii in train_indices]
	labels_test = [labels[ii] for ii in test_indices]

# TRAIN DATA:
t0 = time()

# Pick one:
clf = GaussianNB()
#clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=2)
#clf = RandomForestClassifier(min_samples_split=11,n_estimators=8)

#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME", n_estimators=200)

clf.fit(features_train, labels_train) # Train it on the training set
print "training time:", round(time()-t0, 3), "s"
print

# PREDICTION:
pred = clf.predict(features_test)


# VALIDATE MODEL:
print "MODEL VALIDATION:"
print "================="
print "accuracy_score: ", metrics.accuracy_score(labels_test, pred)
print "precision_score: ", metrics.precision_score(labels_test, pred) 
print "recall_score: ", metrics.recall_score(labels_test, pred)
print "f_score: ", metrics.f1_score(labels_test, pred)
print

# Naive BAYES:
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

# DECISION TREE:
#clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=4)


# SVM:
#clf = svm.SVC(kernel='rbf', C=10000)


#Random Forest
#clf = RandomForestClassifier(min_samples_split=11,n_estimators=6)


# Adaboost
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME", n_estimators=200)
																						



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.



dump_classifier_and_data(clf, my_dataset, features_list)

print "Done!"