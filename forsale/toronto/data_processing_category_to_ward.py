import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import json
import folium
import re
import operator
from geo_utils import *
from haversine import *
from pandas.io.json import json_normalize

#============================================================
# Kevin Zhu
# Used to find the most common categories per ward 
#============================================================

#============================================================
# Adjustable items
#============================================================

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
      ward_category_dict[key] = {
      "app":0,
      "ard":0,
      "art":0,
      "atd":0,
      "atq":0,
      "bab":0,
      "bad":0,
      "bar":0,
      "bdp":0,
      "bfd":0,
      "bfs":0,
      "bid":0,
      "bik":0,
      "bkd":0,
      "bks":0,
      "boa":0,
      "bod":0,
      "bop":0,
      "bpo":0,
      "cbd":0,
      "cld":0,
      "clo":0,
      "clt":0,
      "ctd":0,
      "cto":0,
      "eld":0,
      "ele":0,
      "emd":0,
      "emq":0,
      "fod":0,
      "for":0,
      "fuo":0,
      "grd":0,
      "grq":0,
      "hab":0,
      "had":0,
      "hsd":0,
      "hsh":0,
      "hvd":0,
      "hvo":0,
      "jwd":0,
      "jwl":0,
      "mad":0,
      "mat":0,
      "mcd":0,
      "mcy":0,
      "mob":0,
      "mod":0,
      "mpo":0,
      "msd":0,
      "msg":0,
      "phd":0,
      "pho":0,
      "ppd":0,
      "ptd":0,
      "pts":0,
      "rvs":0,
      "sdp":0,
      "sgd":0,
      "snd":0,
      "snw":0,
      "sop":0,
      "spo":0,
      "syd":0,
      "sys":0,
      "tad":0,
      "tag":0,
      "tix":0,
      "tld":0,
      "tls":0,
      "tro":0,
      "vgd":0,
      "vgm":0,
      "wad":0,
      "wan":0,
      "wtd":0,
      "wto":0,
      "zip":0
   }
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
         for key in range(1,45):
            # Create a vector of distances, get the index of the smallest distance
            # print(str(item_coords[0]) + str(item_coords[1]) + str(ward_centroid_dict[key][0]))
            # print(item_coords[0] + " " + item_coords[1]+ " " + ward_centroid_dict[key][0] + " " + ward_centroid_dict[key][1])
            distances.append(haversine(item_coords[0], item_coords[1], ward_centroid_dict[key][0], ward_centroid_dict[key][1]))

         # Find the index of smallest number
         # print ("Smallest: " + str(distances.index(min(distances))))
         ward_to_increment = distances.index(min(distances)) + 1 # Need the plus one because wards are not 0-indexed 


         if item_type in ward_category_dict.get(ward_to_increment):
            ward_category_dict.get(ward_to_increment, {})[item_type] += 1
         else:
            ward_category_dict.get(ward_to_increment, {})[item_type] = 1


      except TypeError:
         # No geotag. Skip this one
         pass_count += 1
         pass
   print("Passcount: " + str(pass_count))
   return ward_category_dict

def print_full(x):
   pd.set_option('display.max_rows', len(x))
   pd.set_option('display.max_columns', len(list(x.columns.values)))
   print(x)
   pd.reset_option('display.max_rows')

def graph_occurences(oc_dict, ward_key, directory_to_save_to):
   # print(ward_key)
   # print(oc_dict)
   norm_oc_dict = normalize(oc_dict)
   # print norm_oc_dict
   # print(norm_oc_dict)
   #==================
   y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   plt.clf()
   plt.bar(range(len(norm_oc_dict)), norm_oc_dict.values(), align='center')
   plt.xticks(range(len(norm_oc_dict)), norm_oc_dict.keys(), rotation='vertical')
   plt.yticks(y_ticks)
   plt.title("Most common category occurrences for Ward" + str(ward_key))
   fig = plt.gcf()
   fig.set_size_inches(13, 4)
   fig.tight_layout()
   #==================
   #plt.show()
   # raw_input()
   # fig = plt.figure()
   plt.savefig(directory_to_save_to + 'Ward_' + str(ward_key)+".png")

def normalize(oc_dict):
   total = 0.0;
   for item in oc_dict:
      total += float(oc_dict[item])

   for item in oc_dict:
      try: 
         oc_dict[item] = float(oc_dict[item])/total
      except ZeroDivisionError:
         pass
   return oc_dict


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

   try:
      ward_centroid_dict = np.load("ward_centroid_dict.npy").item()
      ward_category_dict = np.load("postings_to_ward.npy").item()

      print "Loaded your stuff!"

      png_folder = os.getcwd() + "/ward_cat_occurence_graphs/"
      for ward_key in ward_category_dict:            
         temp_freq_dict = ward_category_dict.get(ward_key)
         graph_occurences(temp_freq_dict, ward_key, png_folder)

   except IOError: 
      # GET THE CENTROIDS INTO DATAFRAME
      df_centroids = get_centroid_map(basePath)
      
      # debugItemDataColumns(df_centroids)
      # print_full(df_centroids)

      # DATAFRAME TO DICTIONARY
      ward_centroid_dict = get_centroids_aslist(df_centroids)
      np.save("ward_centroid_dict.npy", ward_centroid_dict)
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
      print "Calculating the nearest neighbours."
      ward_category_dict = get_most_common_categories_per_ward(ward_centroid_dict, df_items)

      # >>>> Save the raw association of postings to ward
      print("Saving raw association of postings to ward")
      np.save("postings_to_ward.npy", ward_category_dict)
      # >>>> Save the most common category
      # get_most_common_category_per_ward_as_dict(ward_category_dict)

      # >>>> Graphing the occurences
      # png_folder = os.getcwd() + "/ward_cat_occurence_graphs/"
      # for ward_key in ward_category_dict:
      #    temp_freq_dict = ward_category_dict.get(ward_key)
      #    graph_occurences(temp_freq_dict, ward_key, png_folder)
