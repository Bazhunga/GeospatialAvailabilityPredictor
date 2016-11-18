import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import json
import folium
from pandas.io.json import json_normalize

# This file plots craigslist items on a map


def debugItemDataPrint(item_data):
   print item_data

def debugItemDataColumns(item_data):
   print(list(item_data.columns.values))

def map_data(df_items):
   coordinates = df_items['geotag'].tolist()
   # print coordinates
   coordinates = filter(None, coordinates)
   # print coordinates
   TORONTO_COORDINATES = (43.6532, -79.3832)
   tormap = folium.Map(location=TORONTO_COORDINATES, zoom_start=12)
   for gt in coordinates:
      folium.Marker([gt[0], gt[1]]).add_to(tormap)
   tormap.save('tormap2.html')


def print_full(x):
   pd.set_option('display.max_rows', len(x))
   pd.set_option('display.max_columns', len(list(df_items.columns.values)))
   print(x)
   pd.reset_option('display.max_rows')

if __name__ == "__main__":
   basePath = os.path.dirname(os.path.abspath(__file__))
   path = (basePath + "/bigger.json")
   print path
   with open(path) as toronto_json: 
      master_data = json.load(toronto_json)

   df_items = json_normalize(master_data["tor"])
   # Get rid of useless columns 
   df_items = df_items.drop(['has_image', 'has_map', 'id', 'datetime'] , 1)
   debugItemDataColumns(df_items)

   map_data(df_items)

   

   # Debug
   # debugItemDataPrint(item_data)
   # debugItemDataColumns(item_data)

   # Debug
   # debugItemDataColumns(item_data)

   # Order by where
   # item_data = item_data.sort_values(by='where')
   # debugItemDataPrint(item_data)
   # print_full(item_data)
   # debugItemDataColumns(item_data)
   # print_full(item_data['price'])

   # partitionDataToronto(item_data)

   # plot_occurence_of_items(item_data)
   # partitionDataToronto(item_data)