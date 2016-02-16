#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below

for key in data_dict:
    if data_dict[key]['bonus'] > 5e6 and data_dict[key]['bonus'] < 1.8e8:
        print key
        print data_dict[key]

for i,point in enumerate(data):
    salary = point[0]
    bonus = point[1]
    if bonus > 0.8e8:
        print '{0}, {1}'.format(salary, bonus)
        print i
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
