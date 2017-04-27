#!/usr/bin/python
import warnings
warnings.filterwarnings("ignore")
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "salary", "bonus_salary_ratio", "fraction_from_poi", "fraction_to_poi", "deferral_payments", "total_payments", \
                     "loan_advances", "bonus", "restricted_stock_deferred", "deferred_income", "total_stock_value", \
                     "expenses", "exercised_stock_options", "other", "long_term_incentive", "restricted_stock", \
                     "director_fees", "to_messages", "from_poi_to_this_person", "from_messages", \
                     "from_this_person_to_poi", "shared_receipt_with_poi"] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
### Each key in dataset is name of poi.
### Remove outliers which are not name of poi
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BHATNAGAR SANJAY']['director_fees'] = 0
data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864

data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BELFER ROBERT']['total_stock_value'] = 0
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

### Task 3: Create new feature(s)
# Bonus-salary ratio
for employee, features in data_dict.iteritems():
    if features['bonus'] == "NaN" or features['salary'] == "NaN":
        features['bonus_salary_ratio'] = "NaN"
    else:
        features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])

# from_this_person_to_poi as a percentage of from_messages
for employee, features in data_dict.iteritems():
    if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN":
        features['fraction_to_poi'] = "NaN"
    else:
        features['fraction_to_poi'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])

# from_poi_to_this_person as a percentage of to_messages
for employee, features in data_dict.iteritems():
    if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN":
        features['fraction_from_poi'] = "NaN"
    else:
        features['fraction_from_poi'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])

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

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

skb = SelectKBest(f_classif)

nb = GaussianNB()

dt = DecisionTreeClassifier()

rf = RandomForestClassifier()

pca = PCA()

pipeline1 = Pipeline([("kbest", skb), ("nb", nb )])

pipeline2 = Pipeline([("kbest", skb), ("dt", dt )])

pipeline3 = Pipeline([("kbest", skb), ("rf", rf )])

pipeline4 = Pipeline([("kbest", skb), ("PCA", pca ),("dt", dt )])

params1 = {"kbest__k": range(5, 10)}

params2 = {"kbest__k": range(5, 10),
           "dt__min_samples_split": [2, 4, 6],
           "dt__min_samples_leaf": [2, 4, 6],
           "dt__criterion": ["gini", "entropy"]}

params3 = {"kbest__k": range(5, 10),
           "rf__max_depth": [None, 5, 10],
           "rf__n_estimators": [10, 15, 20]}

params4 = {"kbest__k": range(5, 10),
           "PCA__whiten": [True, False],
           "dt__min_samples_split": [2, 4, 6],
           "dt__min_samples_leaf": [2, 4, 6],
           "dt__criterion": ["gini", "entropy"]}

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=60)


###################################
####### K Best + Naive Bayes
###################################
pipeline1 = Pipeline([("kbest", skb), ("nb", nb )])
params1 = {"kbest__k": range(5, 10)}
gs = GridSearchCV(pipeline1, params1, n_jobs=-1, cv=sss)
gs.fit(features, labels)

clf = gs.best_estimator_
print "Tester Classification report"
test_classifier(clf.named_steps["nb"], data_dict, features_list)

features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
print "A total of %d features were used" % len(features_used)
#Note 1: You use 'features_list[i+1]', instead of 'features_list[i]',
#because the first feature in that list is 'poi'
#which you didn't include in the variable 'features'
feature_names = [features_list[i + 1] for i in features_used]
print "The features used are:", feature_names

###################################
####### K-Best + Decision Tree:
###################################
pipeline2 = Pipeline([("kbest", skb), ("dt", dt )])
params2 = {"kbest__k": range(5, 10),
           "dt__min_samples_split": range(2,10),
           "dt__min_samples_leaf": range(2,10),
           "dt__criterion": ["gini", "entropy"]}

gs = GridSearchCV(pipeline2, params2, n_jobs=-1, cv=sss, scoring="f1")
gs.fit(features, labels)
clf= gs.best_estimator_
print "Tester Classification report"
test_classifier(clf.named_steps["dt"], data_dict, features_list)

features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
print "A total of %d features were used" % len(features_used)
#Note 1: You use 'features_list[i+1]', instead of 'features_list[i]',
#because the first feature in that list is 'poi'
#which you didn't include in the variable 'features'
feature_names = [features_list[i + 1] for i in features_used]
#print "The features used are:", feature_names
scores = [clf.named_steps['kbest'].scores_[i + 1] for i in features_used]
#print 'Scores: ', scores
importances = [clf.named_steps['dt'].feature_importances_[i+1] for i in features_used]
#print 'Importance: ', importances
import numpy as np
indices = np.argsort(importances)[::-1]
#print 'Indices: ', indices
for i in range(len(feature_names)):
    print "feature no. {}: {} ({}) ({})".format(i+1, feature_names[indices[i]], importances[indices[i]], scores[indices[i]])

###############
## Add Scaling
##############

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
pipeline2 = Pipeline([("scale", scale), ("kbest", skb), ("dt", dt )])
params2 = {"kbest__k": range(5, 10),
           "dt__min_samples_split": range(2,10),
           "dt__min_samples_leaf": range(2,10),
           "dt__criterion": ["gini", "entropy"]}

gs = GridSearchCV(pipeline2, params2, n_jobs=-1, cv=sss, scoring="f1")
gs.fit(features, labels)
clf= gs.best_estimator_
print "Tester Classification report"
test_classifier(clf.named_steps["dt"], data_dict, features_list)

features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
print "A total of %d features were used" % len(features_used)
#Note 1: You use 'features_list[i+1]', instead of 'features_list[i]',
#because the first feature in that list is 'poi'
#which you didn't include in the variable 'features'
feature_names = [features_list[i + 1] for i in features_used]
#print "The features used are:", feature_names
scores = [clf.named_steps['kbest'].scores_[i + 1] for i in features_used]
#print 'Scores: ', scores
importances = [clf.named_steps['dt'].feature_importances_[i+1] for i in features_used]
#print 'Importance: ', importances
import numpy as np
indices = np.argsort(importances)[::-1]
#print 'Indices: ', indices
for i in range(len(feature_names)):
    print "feature no. {}: {} ({}) ({})".format(i+1, feature_names[indices[i]], importances[indices[i]], scores[indices[i]])


###################################
####### K-Best + PCA + Decision Tree:
###################################

pipeline4 = Pipeline([("kbest", skb), ("PCA", pca ),("dt", dt )])
params4 = {"kbest__k": range(5, 10),
           "PCA__whiten": [True, False],
           "dt__min_samples_split": [2, 4, 6],
           "dt__min_samples_leaf": [2, 4, 6],
           "dt__criterion": ["gini", "entropy"]}
gs = GridSearchCV(pipeline4, params4, n_jobs=-1, cv=sss, scoring="f1")
gs.fit(features, labels)
clf= gs.best_estimator_
print "Tester Classification report"
test_classifier(clf.named_steps["dt"], data_dict, features_list)

features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
print "A total of %d features were used" % len(features_used)
#Note 1: You use 'features_list[i+1]', instead of 'features_list[i]',
#because the first feature in that list is 'poi'
#which you didn't include in the variable 'features'
feature_names = [features_list[i + 1] for i in features_used]
#print "The features used are:", feature_names
scores = [clf.named_steps['kbest'].scores_[i + 1] for i in features_used]
#print 'Scores: ', scores
importances = [clf.named_steps['dt'].feature_importances_[i+1] for i in features_used]
#print 'Importance: ', importances
import numpy as np
indices = np.argsort(importances)[::-1]
#print 'Indices: ', indices
for i in range(len(feature_names)):
    print "feature no. {}: {} ({}) ({})".format(i+1, feature_names[indices[i]], importances[indices[i]], scores[indices[i]])


###################################
####### K-Best + Random Forest:
###################################
pipeline3 = Pipeline([("kbest", skb), ("rf", rf )])
params3 = {"kbest__k": range(5, 10),
           "rf__max_depth": [None, 5, 10],
           "rf__n_estimators": [10, 15, 20]}
gs = GridSearchCV(pipeline3, params3, n_jobs=-1, cv=sss, scoring="f1")
gs.fit(features, labels)
clf= gs.best_estimator_
print "Tester Classification report"
test_classifier(clf.named_steps["rf"], data_dict, features_list)

features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
print "A total of %d features were used" % len(features_used)
#Note 1: You use 'features_list[i+1]', instead of 'features_list[i]',
#because the first feature in that list is 'poi'
#which you didn't include in the variable 'features'
feature_names = [features_list[i + 1] for i in features_used]
#print "The features used are:", feature_names
scores = [clf.named_steps['kbest'].scores_[i + 1] for i in features_used]
#print 'Scores: ', scores
importances = [clf.named_steps['rf'].feature_importances_[i+1] for i in features_used]
#print 'Importance: ', importances
indices = np.argsort(importances)[::-1]
#print 'Indices: ', indices
for i in range(len(feature_names)):
    print "feature no. {}: {} ({}) ({})".format(i+1, feature_names[indices[i]], importances[indices[i]], scores[indices[i]])



##########################################
####### Final Algorithm For This Project:
#######    Decision Tree
#########################################
pipeline2 = Pipeline([("kbest", skb), ("dt", dt )])
params2 = {"kbest__k": [9],
           "dt__min_samples_split": [6],
           "dt__min_samples_leaf": [9],
           "dt__criterion": ["entropy"]}

gs = GridSearchCV(pipeline2, params2, n_jobs=-1, cv=sss, scoring="f1")
gs.fit(features, labels)
clf= gs.best_estimator_
print "Tester Classification report"
test_classifier(clf.named_steps["dt"], data_dict, features_list)

features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
print "A total of %d features were used" % len(features_used)
#Note 1: You use 'features_list[i+1]', instead of 'features_list[i]',
#because the first feature in that list is 'poi'
#which you didn't include in the variable 'features'
feature_names = [features_list[i + 1] for i in features_used]
#print "The features used are:", feature_names
scores = [clf.named_steps['kbest'].scores_[i + 1] for i in features_used]
#print 'Scores: ', scores
importances = [clf.named_steps['dt'].feature_importances_[i+1] for i in features_used]
#print 'Importance: ', importances
import numpy as np
indices = np.argsort(importances)[::-1]
#print 'Indices: ', indices
for i in range(len(feature_names)):
    print "feature no. {}: {} ({}) ({})".format(i+1, feature_names[indices[i]], importances[indices[i]], scores[indices[i]])



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf.named_steps["dt"], data_dict, features_list)
