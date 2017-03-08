import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import json
import folium
import re
import operator
from haversine import *
from pandas.io.json import json_normalize

age_group_order = ["0 to 4 years",
                  "5 to 9 years",
                  "10 to 14 years",
                  "15 to 19 years",
                  "15 years",
                  "16 years",
                  "17 years",
                  "18 years",
                  "19 years",
                  "20 to 24 years",
                  "25 to 29 years",
                  "30 to 34 years",
                  "35 to 39 years",
                  "40 to 44 years",
                  "45 to 49 years",
                  "50 to 54 years",
                  "55 to 59 years",
                  "60 to 64 years",
                  "65 to 69 years",
                  "70 to 74 years",
                  "75 to 79 years",
                  "80 to 84 years",
                  "85 years and over"]
                  
# a = np.load('postings_to_ward.npy').item()

# total = 0
# for ward in range(1, 44):
#    # print(a[ward])
#    cat = max(a[ward].iteritems(), key=operator.itemgetter(1))[0]
#    print("Ward " + str(ward) + ": " + cat + " " + str(a[ward][cat]))
#    # raw_input("waiting")
# #    for category in a[ward]:
# #       total = total + a[ward][topic]
# #    # wardkey = "ward_" + str(ward)
# #    # for ag in age_group_order: 
# #    #    print("Ward " + str(ward) + ": " + ag + " " + str(a[ward][ag]))
# #    #    total = total + a[ward][ag]
# # print(total)

a = np.load('sim_cat_list.npy')
b = np.load('sim_geoloc_list.npy')
c = np.load('sim_person_list.npy')
d = np.load('sim_ward_list.npy')

e = np.load('np_target_list_age.npy')
print(len(e))
if(78 in a):
   print("yes")



