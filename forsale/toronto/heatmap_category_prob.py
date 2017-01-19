from __future__ import print_function
import random as rand
import folium
from folium import plugins
from functions import *
from geo_utils import *
from data_processing_category_to_ward import *

#======================================================================
# Kevin Zhu
# This program heatmaps interesting category appearances
# over Toronto. This is used to visualize the spread and clustering
# of different categories. 
#
# Each posting has a geotag associated with it. 
# We use this to heatmap the frequency of productivity of label
#======================================================================

# Raw heatmap of postings
heatmap_category = ["fuo"] 
heatmap_name = "heatmap_" + str(heatmap_category)
posting_lats = "posting_lats.npy"
posting_lons = "posting_lons.npy"
posting_mags = "posting_mags.npy"

def get_unique_category_code(item_url):
   m = re.search('org/tor/(.+?)/', item_url)
   return m.group(1) # Returns the category string

def get_items_asdataframe(filename):
   with open(filename) as toronto_json: 
      master_data = json.load(toronto_json)

   item_data = json_normalize(master_data["tor"])
   item_data = item_data.drop('has_image', 1)
   item_data = item_data.drop('has_map', 1)
   item_data = item_data.drop('id', 1)
   return item_data

def getGeotagsForCategory(df_items):
   lats = []
   lons = []
   mags = []
   
   for item in df_items.iterrows():
      if(get_unique_category_code(item[1]['url']) in heatmap_category):
         try:
            lats.append(item[1]['geotag'][0])
            lons.append(item[1]['geotag'][1])
            mags.append(15.0)
         except TypeError:
            pass

   print(len(lats))
   print(len(lons))

   return lats, lons, mags

if __name__ == "__main__":
   # 1. Load all the data into a data frame. Same as usual
   #    Make sure this is done in the proper day order. 
   # 2. Iterate through and throw out all the ones without geotags
   # 3. Create a slider thing. Have tick on the first day. Have a thing on the last day.
   #    This can control how much data we want to put in 
   # 4. Plot the stuff on the map using add mface
  
   basePath = os.path.dirname(os.path.abspath(__file__))
   dataPath = basePath + "/data/"

   json_list = get_json_list(basePath)  #get all data files

   dataframelist = []
   for jsonfile in json_list:
      print(jsonfile)
      dataframelist.append(get_items_asdataframe(dataPath + jsonfile))

   # Get items in data frame format
   # datetime,geotag, name, price, url, where 
   df_items = pd.concat(dataframelist)

   print("Gathering Latitutes and Longitudes")
   lats, lons, mags = getGeotagsForCategory(df_items)


   url_base = 'http://server.arcgisonline.com/ArcGIS/rest/services/'
   service = 'NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
   tileset = url_base + service
   m = folium.Map(location=TORONTO_COORDINATES, zoom_start=10,\
                   control_scale = True, tiles=tileset, attr='USGS style')

   m.add_children(plugins.HeatMap(zip(lats, lons, mags), radius = 10))

   m.save("heatmap_" + str(heatmap_name) + ".html")