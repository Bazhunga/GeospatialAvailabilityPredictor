from __future__ import print_function
from geo_utils import *
import random as rand
import re
# from sklearn import datasets
# from sklearn import linear_model, datasets
# from sklearn.svm import LinearSVC
# from sklearn.externals import joblib

'''
Kevin Zhu
Jan 8, 2017

Graphs the productivities of the wards. Wards are not equal in the postings 
that they put out. 

The output is a graph over the course over the course of October. There
'''

omit_missing_days = True
basePath = os.path.dirname(os.path.abspath(__file__))
dataPath = basePath + "/data/"
ward_prod_dir = basePath + "/ward_productivity/"


# Sort the file names to get the correct order 
def rotate(l, n):
    return l[n:] + l[:n]

# Get the number of posts for each ward in a day
# Returns a dictionary of wards and number of postings
def get_post_counts_dict(json_days, ward_centroid_dict):
   wards = {}
   print(json_days)
   count = 0
   passcount = 0
   for json_day in json_days: 
      with open(dataPath + json_day) as jfile:
         day_data = json.load(jfile)

      day_data = json_normalize(day_data["tor"])
      day_data = day_data.drop('has_image', 1)
      day_data = day_data.drop('has_map', 1)
      day_data = day_data.drop('id', 1)

      # print(day_data.shape)
      for item in day_data.iterrows():
         try:
            item_coords = (item[1]['geotag'][0], item[1]['geotag'][1])
            responsible_ward = get_items_responsible_ward(item_coords, ward_centroid_dict) + 1
            # print(responsible_ward)
            try:
               wards[responsible_ward] += 1 # Need to catch an error
               count += 1
            except KeyError:
               wards[responsible_ward] = 1
               count += 1

         except TypeError:
            passcount += 1
            pass

   # for thing in wards:
   #    print("Ward " + str(thing) + ": " + str(wards[thing]))
   # print("useable: " + str(count))
   # print("passcount: " + str(passcount))
   return wards, count

# Graph the stuff
def graph_wrt_wards(days_list, master_post_count_list, count_list):
   # Iterate through all the days
   # Ignore the ones where count == 0
   # Plot the ones 
   if (len(days_list) != len(count_list)):
      print("number of days != totals, recheck your lists")

   if (omit_missing_days == True): 
      nz_days_list = [] #Non zero days_list
      nz_count_list = [] #Non zero counts_list
      for i, count in enumerate(count_list): 
         if (count != 0):
            nz_days_list.append(days_list[i])
            nz_count_list.append(count_list[i])

      days_list = nz_days_list
      count_list = nz_count_list
   else:
      pass

   print(days_list)
   print(count_list)

   # initialize a list of 44 lists (these will hold values for our 44 lines)
   list_ward_points = []
   for i in range(1, 45): 
      list_ward_points.append([])

   print("Creating points")
   for d, day in enumerate(days_list):
      for index in range(0, 44):
         print("Day: " + str(day) + " index: " + str(index))
         ward = index + 1
         try: 
            list_ward_points[index].append(float(master_post_count_list[day][ward])/float(count_list[d]))
         except KeyError:
            list_ward_points[index].append(0.0)

   # Do the graphing!
   # x_labels are the days_list
   # y_values are the list_ward_points


   fig, ax = plt.subplots()
   fig.canvas.draw()

   plt.xticks(np.arange(0, len(days_list), 1.0))
   
   labels = [item.get_text() for item in ax.get_xticklabels()]
   for i, label in enumerate(labels):
      labels[i] = days_list[i]

   ax.set_xticklabels(labels)

   legend_handles = []
   for wp, ward_points in enumerate(list_ward_points): 
      plt.plot(ward_points, label="Ward " + str(wp + 1))

   ax.legend(prop={'size':6})
   plt.xticks(rotation=90)
   plt.tight_layout()
   plt.title("Proportion of postings of wards over time")
   plt.xlabel("date")
   plt.ylabel("proportion of occurrences")
   plt.show()

   # Plot individuals
   # legend_handles = []
   # for wp, ward_points in enumerate(list_ward_points):
   #    plt.cla()
   #    fig, ax = plt.subplots()
   #    fig.canvas.draw()
   #    plt.xticks(np.arange(0, len(days_list), 1.0))
   
   #    labels = [item.get_text() for item in ax.get_xticklabels()]
   #    for i, label in enumerate(labels):
   #       labels[i] = days_list[i]
   #    ax.set_xticklabels(labels)
   #    plt.xticks(rotation=90)
   #    plt.tight_layout()
   #    plt.title("Proportion of postings of ward " + str(wp + 1))
   #    plt.xlabel("date")
   #    plt.ylabel("proportion of occurrences")
   #    plt.plot(ward_points, label="Ward " + str(wp + 1))
   #    plt.savefig(ward_prod_dir + 'ward_' + str(wp + 1))

   return

if __name__ == "__main__":
   # Get the raw list of file names we can open
   json_list = get_json_list(basePath)

   # Get the list of days (since there are multiple files for each day)
   days_list = get_days_list(json_list)

   # get centroids
   df_centroids = get_centroid_map(basePath)
   ward_centroid_dict = get_centroids_aslist(df_centroids)

   # Iterate through the days, find the file for that day, create an array of 
   # dictionary entries
   master_post_count_list = {} #day{ward{}}
   count_list = [] # Accompanies master_post_count_list, contains totals for each day

   for day in days_list:
      day_files = [filename for filename in json_list if day in filename]
      master_post_count_list[day], tempcount = get_post_counts_dict(day_files, ward_centroid_dict)
      count_list.append(tempcount)

   for i, day in enumerate(days_list):
      if (count_list[i] != 0):
         print("Day: " + str(day) + " " + str(master_post_count_list[day]))
         print(count_list[i])
   # Graph with respect to other wards
   graph_wrt_wards(days_list, master_post_count_list, count_list)

   # Graph each ward's trend



