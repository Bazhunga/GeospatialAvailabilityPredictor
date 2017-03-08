from __future__ import print_function
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

# Used to graph the distributions of people in each group for different
# ward features (Agegroup, females, males, etc)

def get_ward_info(basePath, feature_type):
   path = (basePath + "/2011_ward_info_csv/2011_Ward_POP_by_" + feature_type + ".csv")
   # print path
   df_items = pd.read_csv(path) # names=['GEO_ID','CREATE_ID','NAME', 'SCODE_NAME', 'LCODE_NAME', 'TYPE_DESC', 'TYPE_CODE', 'OBJECTID', 'xcoord', 'ycoord'])
   # df_items = df_items.drop(['GEO_ID', 'CREATE_ID', 'NAME', 'LCODE_NAME', 'TYPE_DESC', 'TYPE_CODE', 'OBJECTID'], 1)
   # df_items = df_items.sort_values(by=['SCODE_NAME'])
   return df_items

def get_ward_info_as_dict(df_wpop): 
   wi_dict = {}
   # for row in df_wpop.iterrows():
   #    # print(row[1]['Category'])
   #    temp_wpop_feature_str = row[1]['Category'].replace(" ", "_")
   #    # print(temp_wpop_feature_str)
   #    wi_dict[temp_wpop_feature_str]={}
   #    for ward in range(1,45):
   #       ward_string = "ward_" + str(ward)
   #       # print(row[1][ward_string])
   #       ward_df_accessor = "Ward " + str(ward)
   #       wi_dict[temp_wpop_feature_str][ward_string] = row[1][ward_df_accessor]

   for ward in range(1,45):
      ward_string = "ward_" + str(ward)
      wi_dict[ward_string]={}
      # print(row[1][ward_string])
      ward_df_accessor = "Ward " + str(ward) # Column in the CSV
      for row in df_wpop.iterrows():
         wi_dict[ward_string][row[1]['Category']] = row[1][ward_df_accessor]

   return wi_dict

# Graph the distributions 
def graph_ward_info(wi_dict, basePath, poptype): 
   directory = basePath + "/2011_ward_info_csv/2011_Ward_POP_by_" + poptype + "/"

   #TODO: NEED TO NORMALIZE 
   for ward in wi_dict:
      total = 0
      # This is inefficient. You should precalculate these. 
      for category in wi_dict[ward]:
         total += wi_dict[ward][category]

      # print (total)
      # print (ward)
      plt.rcParams.update({'figure.autolayout': True})
      plt.clf()
      plt.bar(range(len(wi_dict[ward])), [float(x) / float(total) for x in wi_dict[ward].values()], align='center')
      plt.xticks(range(len(wi_dict[ward])), wi_dict[ward].keys(), rotation=90, ha='center')
      plt.title("Population by " + poptype)
      fig = plt.gcf()
      fig.set_size_inches(18.5, 10.5)
      # plt.show()
      # fig = plt.figure()
      print(directory + str(ward)+".png")
      plt.savefig(directory + str(ward)+".png")


if __name__ == "__main__":
   basePath = os.path.dirname(os.path.abspath(__file__))
   # Usage
   # Put a csv file with the fn format being 2011_Ward_POP_by_<POPTYPE>.csv
   # Create a directory with fn format being 2011_Ward_POP_by_<POPTYPE>
   # poptype = "agegroup"
   # poptype = "females"
   # poptype = "males"
   # poptype = "householdtype"
   # poptype = "familytype"
   poptype = "malestofemales"

   # Load the ward csv into dataframe
   df_wpop = get_ward_info(basePath, poptype)
   # print(df_wpop)
   wi_dict = get_ward_info_as_dict(df_wpop)
   # print(wi_dict)
   # graph_ward_info(wi_dict, basePath, poptype)

   np.save("ward_"+poptype+"_training.npy", wi_dict)

