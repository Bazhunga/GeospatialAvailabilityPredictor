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

def debugItemDataPrint(item_data):
   print item_data

def debugItemDataColumns(item_data):
   print(list(item_data.columns.values))

def get_population_map(basePath):
   path = (basePath + "/2011_Ward_Population.csv")
   print path
   df_items = pd.DataFrame.from_csv(path)
   df_items = df_items.drop(['Etobicoke York', 'North York', 'Toronto & East York', 'Scarborough'] , 1)
   return df_items

def get_centroid_map(basePath):
   path = (basePath + "/centroids/centroid_data_toronto.csv")
   # print path
   df_items = pd.read_csv(path) # names=['GEO_ID','CREATE_ID','NAME', 'SCODE_NAME', 'LCODE_NAME', 'TYPE_DESC', 'TYPE_CODE', 'OBJECTID', 'xcoord', 'ycoord'])
   df_items = df_items.drop(['GEO_ID', 'CREATE_ID', 'NAME', 'LCODE_NAME', 'TYPE_DESC', 'TYPE_CODE', 'OBJECTID'], 1)
   df_items = df_items.sort_values(by=['SCODE_NAME'])
   return df_items

def get_centroids_aslist(df_centroids):
   ward_centroid_dict = {}
   for row in df_centroids.iterrows():
      # print row[1]['SCODE_NAME']
      ward_centroid_dict[int(row[1]['SCODE_NAME'])] = list([row[1]['ycoord'], row[1]['xcoord']])
   return ward_centroid_dict

def get_items_asdataframe(filename):
   path = (basePath + "/" + filename)
   # print("Getting json from " + path)
   with open(path) as toronto_json: 
      master_data = json.load(toronto_json)
   item_data = json_normalize(master_data["tor"])
   item_data = item_data.drop('has_image', 1)
   item_data = item_data.drop('has_map', 1)
   item_data = item_data.drop('id', 1)
   return item_data

# Gets the unique category code from the url that's passed in
def get_unique_category_code(item):
   m = re.search('org/tor/(.+?)/', item)
   return m.group(1) # Returns the category string

def get_empty_ward_dict(ward_centroid_dict):
   ward_category_dict = {}
   for key in ward_centroid_dict:
      ward_category_dict[key] = {}
   # print ward_category_dict
   return ward_category_dict

def get_most_common_categories_per_ward(ward_centroid_dict, df_items):
   # Iterate through the items in df_items and find the closest ward
   # We'd have a ward dictionary 
   # Key --> Ward number
   # Val --> Dictionary of 
   #           Key --> Category string (3 letters)
   #           Val --> Number of occurences 

   ward_category_dict = get_empty_ward_dict(ward_centroid_dict)

   # Iterate through the df_items
   pass_count = 0;
   for item in df_items.iterrows():
      # Get the coordinates of the item
      try: 
         item_coords = (item[1]['geotag'][0], item[1]['geotag'][1])
         # print item_coords

         # Get the type of item
         item_type = str(get_unique_category_code(item[1]['url']))
         # print item_type

         # Get the nearest neighbour
         distances = []
         for key in ward_centroid_dict:
            # Create a vector of distances, get the index of the smallest distance
            # print(str(item_coords[0]) + str(item_coords[1]) + str(ward_centroid_dict[key][0]))
            # print(item_coords[0] + " " + item_coords[1]+ " " + ward_centroid_dict[key][0] + " " + ward_centroid_dict[key][1])
            distances.append(haversine(item_coords[0], item_coords[1], ward_centroid_dict[key][0], ward_centroid_dict[key][1]))
         # print distances

         # Find the index of smallest number
         # print ("Smallest: " + str(distances.index(min(distances))))
         ward_to_increment = distances.index(min(distances))
         if item_type in ward_category_dict.get(ward_to_increment):
            ward_category_dict.get(ward_to_increment, {})[item_type] += 1
         else:
            ward_category_dict.get(ward_to_increment, {})[item_type] = 1


      except TypeError:
         # No geotag. Skip this one
         pass_count += 1
         pass
   # print("Passcount: " + str(pass_count))
   return ward_category_dict

def print_full(x):
   pd.set_option('display.max_rows', len(x))
   pd.set_option('display.max_columns', len(list(x.columns.values)))
   print(x)
   pd.reset_option('display.max_rows')

def graph_occurences(oc_dict, ward_key, directory_to_save_to):
   plt.clf()
   plt.bar(range(len(oc_dict)), oc_dict.values(), align='center')
   plt.xticks(range(len(oc_dict)), oc_dict.keys(), rotation='vertical')
   plt.title("Most common category occurrences for Ward" + str(ward_key))
   # plt.show()
   # fig = plt.figure()
   plt.savefig('Ward_' + str(ward_key)+".png")

def get_most_common_category_per_ward_as_dict(ward_category_dict):
   print("WARD CATEGORY DICTIONARY")
   for key in ward_category_dict:
      print(str(key) + ": " + str(ward_category_dict.get(key)))

   ward_top = {}
   print("Ward category most common occurrences")
   for ward_key in ward_category_dict:
      total = sum(ward_category_dict[ward_key].values())
      print total
      temp_freq_dict = ward_category_dict.get(ward_key)
      try:
         temp_max = max(temp_freq_dict.iteritems(), key=operator.itemgetter(1)) # Gets the category with max # occurrences
      except ValueError:
         temp_max = ['none', 0]
      ward_top[ward_key] = str(temp_max[0])
      print(str(ward_key) + ": " + ward_top[ward_key] + " with " + str(temp_max[1]) + " occurrences")

   np.save("ward_most_common_targets.npy", ward_top)


def verify_wardcatdict(ward_category_dict):
   total = 0
   for key in ward_category_dict:
      for key2 in ward_category_dict.get(key):
         total += ward_category_dict.get(key).get(key2)

   print total


if __name__ == "__main__":
   basePath = os.path.dirname(os.path.abspath(__file__))
   #path = get_population_map(basePath)

   # GET THE CENTROIDS INTO DATAFRAME
   df_centroids = get_centroid_map(basePath)
   
   # debugItemDataColumns(df_centroids)
   # print_full(df_centroids)

   # DATAFRAME TO DICTIONARY
   ward_centroid_dict = get_centroids_aslist(df_centroids)
   # print ward_centroid_dict

   # GET THE ITEMS INTO DATAFRAME
   # json_file_name = "bigger.json"
   # df_items = get_items_asdataframe(json_file_name)
   first_item = True
   # df_items = get_ite
   dataframelist = []
   for filename in os.listdir(os.getcwd() + "/data"):
      print filename

      # if (first_item == True):
      #    print("TRUE")
      #    df_items = get_items_asdataframe("data/" + filename)
      #    print df_items.shape
      #    first_item = False
      # else:
      #    print("FALSE")
      #    df_items_temp = get_items_asdataframe("data/" + filename)
      #    df_items.append(df_items_temp, ignore_index=True)
      #    print df_items.shape
      dataframelist.append(get_items_asdataframe("data/" + filename))

   df_items = pd.concat(dataframelist)
   # print df_items.shape

   # print_full(df_items)

   # FIND NEAREST NEIGHBOUR OF THE ITEM USING HAVERSINE
   print "Working on the haversine boss."
   ward_category_dict = get_most_common_categories_per_ward(ward_centroid_dict, df_items)


   # SAVE THE MOST COMMON CATEGORY
   get_most_common_category_per_ward_as_dict(ward_category_dict)

   # verify_wardcatdict(ward_category_dict)

   # Graphing the occurences
   # png_folder = os.getcwd() + "/ward_cat_occurence_graphs"
   # for ward_key in ward_category_dict:
   #    temp_freq_dict = ward_category_dict.get(ward_key)
   #    graph_occurences(temp_freq_dict, ward_key, png_folder)
