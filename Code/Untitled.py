# -*- coding: utf-8 -*-
"""

"""
import pickle



### The first feature must be "poi".
features_list = ['poi','salary',"strength_of_email_conn_to_POI"] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
