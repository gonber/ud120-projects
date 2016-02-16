#!/usr/bin/python

import sys
import math
from sets import Set
import matplotlib.pyplot as plt
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def Draw(features, labels, f1_name="feature 1", f2_name="feature 2"):
    colors = ["b", "r"]
    for ii, pp in enumerate(features):
        plt.scatter(features[ii][0], features[ii][1], color=colors[int(labels[ii])])

    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.show()

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.

# Find features with a high proportion of NaN values and exclude those
all_features = list(data_dict[list(data_dict.viewkeys())[0]].viewkeys())
nan_counter = [0] * len(all_features)
for features in data_dict.viewvalues():
    for feature in features.viewkeys():
        try:
            if math.isnan(float(features[feature])):
                nan_counter[all_features.index(feature)] += 1
        except:
            continue

ignore_features = []
for i in range(0, len(all_features)):
    nan_rate = float(nan_counter[i])/float(len(data_dict))
    if nan_rate > 0.5:
        ignore_features.append(all_features[i])

# print Set(all_features) - Set(ignore_features)
# ['salary', 'to_messages', 'total_payments', 'bonus', 'restricted_stock',
#'total_stock_value', 'shared_receipt_with_poi', 'exercised_stock_options',
# 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'expenses',
# 'email_address', 'from_poi_to_this_person']

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', \
                 'from_this_person_to_poi_rate', \
                 'from_poi_to_this_person_rate', \
                #  'salary', \
                #  'to_messages', \
                #  'total_payments', \
                #  'bonus', \
                #  'restricted_stock', \
                #  'total_stock_value', \
                #  'shared_receipt_with_poi', \
                #  'exercised_stock_options', \
                #  'from_messages', \
                #  'other', \
                #  'from_this_person_to_poi', \
                #  'expenses', \
                #  'from_poi_to_this_person'
                 ]

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for name in data_dict.viewkeys():
    data_dict[name]['from_this_person_to_poi_rate'] = str(\
        float(data_dict[name]['from_this_person_to_poi'])/ \
        float(data_dict[name]['from_messages']))

    data_dict[name]['from_poi_to_this_person_rate'] = str(\
        float(data_dict[name]['from_poi_to_this_person'])/ \
        float(data_dict[name]['to_messages']))

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, remove_NaN=True, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.decomposition import RandomizedPCA

# pca = RandomizedPCA(n_components=5)
# pca.fit(features)
# features = pca.transform(features)
# print pca.explained_variance_ratio_

# Visualiazation
Draw(features, labels, f1_name="feature 1", f2_name="feature 2")


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=2)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, recall_score

precision = []
recall = []
kf = StratifiedKFold(labels, n_folds=10, shuffle=True)
for train_index, test_index in kf:
    features_train = [features[i] for i in train_index]
    labels_train = [labels[i] for i in train_index]
    features_test = [features[i] for i in test_index]
    labels_test = [labels[i] for i in test_index]
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))

print precision
print recall

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
