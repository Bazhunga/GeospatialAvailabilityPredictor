import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import json
import re
from pandas.io.json import json_normalize

def debugItemDataPrint(item_data):
   print item_data

def debugItemDataColumns(item_data):
   print(list(item_data.columns.values))

def print_full(x):
   pd.set_option('display.max_rows', len(x))
   pd.set_option('display.max_columns', len(list(item_data.columns.values)))
   print(x)
   pd.reset_option('display.max_rows')

def partitionDataToronto(item_data):
   # item_data = item_data[item_data['where'].str.contains('toronto', case=False, na='nan')]
   # TODO
   print_full(item_data)

def getUniqueCategoryCode(item):
   m = re.search('org/tor/(.+?)/', item)
   return m

def graph_occurences(oc_dict):
   plt.bar(range(len(oc_dict)), oc_dict.values(), align='center')
   plt.xticks(range(len(oc_dict)), oc_dict.keys(), rotation='vertical')
   plt.show()



def plot_occurence_of_items(item_data):
   item_data = item_data.drop('geotag', 1)
   item_data = item_data.drop('where', 1)
   item_data = item_data.drop('datetime', 1)

   item_data = item_data[item_data['name'].str.contains('shoe', case=False, na='nan')]

   print_full(item_data)


if __name__ == "__main__":
   basePath = os.path.dirname(os.path.abspath(__file__))
   path = (basePath + "/bigger.json")
   print path
   with open(path) as toronto_json: 
      master_data = json.load(toronto_json)

   item_data = json_normalize(master_data["tor"])

   # Debug
   # debugItemDataPrint(item_data)
   # debugItemDataColumns(item_data)

   # Get rid of useless columns 
   item_data = item_data.drop('has_image', 1)
   item_data = item_data.drop('has_map', 1)
   item_data = item_data.drop('id', 1)

   # Debug
   # debugItemDataColumns(item_data)

   # Order by where
   # item_data = item_data.sort_values(by='where')
   # debugItemDataPrint(item_data)
   # print_full(item_data)
   debugItemDataColumns(item_data)
   # print_full(item_data['price'])

   # partitionDataToronto(item_data)

   # plot_occurence_of_items(item_data)
   # partitionDataToronto(item_data)

   # print_full(item_data['url'])
   occurence_dict = {}
   for item_url in item_data['url']:
      code = str(getUniqueCategoryCode(item_url).group(1))
      if code in occurence_dict:
         occurence_dict[code] = occurence_dict[code] + 1
      else:
         occurence_dict[code] = 1

   print occurence_dict.keys()

   # graph_occurences(occurence_dict)



