#!/usr/bin/python

import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    residual_errors = (predictions - net_worths)**2
    index_sorted = residual_errors.argsort(0)

    cleaned_data = []
    for i in range(0, 81):
        index = index_sorted[i][0]
        cleaned_data.append([ages[index],net_worths[index], residual_errors[index]])

    return cleaned_data
