from __future__ import print_function
import random as rand
import folium
from folium import plugins
from functions import *
from geo_utils import *
from data_processing_category_to_ward import *

#======================================================================
# Kevin Zhu
# Plot the probabilities of a given category on the toronto map
# according to the centroids of the wards
#======================================================================

# Raw heatmap of postings
heatmap_category = "bfd" #bab  clo ele fuo hab hsh jwl tag vgm
heatmap_name = "probs_rad2k_" + heatmap_category

centroid_dict = "ward_centroid_dict.npy"

# posting_lats = "posting_lats.npy"
# posting_lons = "posting_lons.npy"
# posting_mags = "posting_mags.npy"

def processCentroids(dict_centroids, list_probs):
   lats = []
   lons = []
   mags = []
   
   for ward in range(1,45):
      ward_cent = dict_centroids[ward]
      lats.append(ward_cent[0])
      lons.append(ward_cent[1])
      ward_probs = list_probs[ward - 1] # Get probabilities of ward
      mags.append(ward_probs[target_to_class[heatmap_category] - 1] * 2000)

   return lats, lons, mags

if __name__ == "__main__":
  
   basePath = os.path.dirname(os.path.abspath(__file__))
   dataPath = basePath + "/data/"

   # Get the centroids
   dict_centroids = np.load(centroid_dict).item()
   # Get the probabilities
   list_probs = np.load("category_probabilities_age.npy")

   lats, lons, mags = processCentroids(dict_centroids, list_probs)

   # print(len(lats))
   # print(len(lons))
   # print(lats)
   # print(lons)
   # print(mags)

   url_base = 'http://server.arcgisonline.com/ArcGIS/rest/services/'
   service = 'NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
   tileset = url_base + service
   m = folium.Map(location=TORONTO_COORDINATES, zoom_start=10,\
                   control_scale = True, tiles=tileset, attr='USGS style')

   m.add_children(plugins.HeatMap(zip(lats, lons, mags), radius = 15))

   m.save("heatmap_" + str(heatmap_name) + ".html")