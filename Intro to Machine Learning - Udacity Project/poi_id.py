#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
df = pd.DataFrame.from_dict(data_dict, orient = "index")




### Task 2: Remove outliers
df = df.drop(["TOTAL", "THE TRAVEL AGENCY IN THE PARK"])
df = df[["salary", "deferral_payments", "total_payments", "exercised_stock_options", "bonus", "restricted_stock", "restricted_stock_deferred", "total_stock_value", "expenses", "loan_advances", "other", "director_fees", "deferred_income", "long_term_incentive", "to_messages", "shared_receipt_with_poi", "from_messages", "from_this_person_to_poi","from_poi_to_this_person", "poi" ]]
##df = df.fillna(0)
for i in range(19):
    df.iloc[:,i] = df.iloc[:,i].astype(float)
for i in range(14):
    df.iloc[:,i] = df.iloc[:,i].fillna(0)
df['to_messages'] = df['to_messages'].fillna(df['to_messages'].median())
df['shared_receipt_with_poi'] = df['shared_receipt_with_poi'].fillna(df['shared_receipt_with_poi'].median())
df['from_messages'] = df['from_messages'].fillna(df['from_messages'].median())
df['from_this_person_to_poi'] = df['from_this_person_to_poi'].fillna(df['from_this_person_to_poi'].median())
df['from_poi_to_this_person'] = df['from_poi_to_this_person'].fillna(df['from_poi_to_this_person'].median())



### Task 3: Create new feature(s)
df['percent_emails_to_poi'] = (df.from_poi_to_this_person / df.to_messages) * 100
df['percent_emails_from_poi'] = (df.from_this_person_to_poi / df.from_messages) * 100
df.percent_emails_to_poi = df.percent_emails_to_poi.round(2)
df.percent_emails_from_poi = df.percent_emails_from_poi.round(2)
df = df.fillna(0)
##df = df[["poi", "salary", "deferral_payments", "total_payments", "exercised_stock_options", "bonus", "restricted_stock", "restricted_stock_deferred", "total_stock_value", "expenses", "loan_advances", "other", "director_fees", "deferred_income", "long_term_incentive", "shared_receipt_with_poi", "percent_emails_to_poi", "percent_emails_from_poi"]]
df = df[["poi","exercised_stock_options", "total_stock_value", "bonus", "shared_receipt_with_poi", "percent_emails_to_poi", "percent_emails_from_poi" ]]
##df = df[["poi","exercised_stock_options", "total_stock_value", "bonus", "shared_receipt_with_poi", "percent_emails_to_poi"]]
features_list = list(df)

### Store to my_dataset for easy export below.
df_dict = df.to_dict('index')
my_dataset = df_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)
ssplit = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=42)
##dt = DecisionTreeClassifier()
##skb = SelectKBest()
##parameters = {'skb__k':range(1,10), 'dtc__min_samples_split' : range(2,10,1),'dtc__max_depth': range(1,10), 'dtc__random_state': [42]}
##pipe = Pipeline([('skb', skb),('dtc', dt)])
##gs = GridSearchCV(pipe, parameters, scoring = 'f1', cv = ssplit)
##gs.fit(features,labels)
##K_best = gs.best_estimator_.named_steps['skb']
##feature_scores = ['%.2f' % elem for elem in K_best.scores_ ]
##feature_scores_pvalues = ['%.3f' % elem for elem in  K_best.pvalues_ ]
##features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in K_best.get_support(indices=True)]
##features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
##print features_selected_tuple
##dt = DecisionTreeClassifier()
##param = {'min_samples_split' : range(2,10,1), 'max_depth': range(1,10), 'random_state': [42]}
##gs = GridSearchCV(dt, param, scoring = 'f1', cv = ssplit)
##gs.fit(features,labels)
##importances = gs.best_estimator_.feature_importances_
##indices = np.argsort(importances)[::-1]
##for i in range(len(importances)):
    ##print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.




##clf = GaussianNB()
clf = DecisionTreeClassifier()
##clf = LogisticRegression()
##clf = RandomForestClassifier()
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
print(accuracy_score(predict, labels_test))
print(recall_score(predict, labels_test))
print(precision_score(predict, labels_test))





### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
param = {'min_samples_split' : range(2,10,1), 'max_depth': range(1,10), 'random_state': [42], 'criterion': ('gini','entropy'), 'splitter': ('best','random')}
grid = GridSearchCV(clf, param_grid = param, scoring = 'f1', cv = ssplit)
grid.fit(features, labels)
print grid.best_score_
print grid.best_params_
clf = grid.best_estimator_
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
