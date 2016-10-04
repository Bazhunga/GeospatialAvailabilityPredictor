from __future__ import print_function
from craigslist import CraigslistForSale
from datetime import datetime
import json
import re
import time

regionArray = ['sfc', 'sby', 'eby', 'pen', 'nby', 'scz']
# ['sfc', 'sby', 'eby', 'pen', 'nby', 'scz']
# Apa is apartments/housing for rent
# Hou is apartments wanted
# Rooms and shares is roo

filename = datetime.now()
filename = filename.strftime('%m_%d_%Y_%H_%M_%S.txt')
f = open(filename, 'a+')

def main(): 
  f.write("{")
  first = True;
  for area in regionArray:
    # f.write("REGION: " + area + "\n")
    if(first == False): 
      f.write(",")
    else:
      first = False
    latest_item_details = {} # getLatestItemDetails(area)
    f.write("\"" + area + "\":[")
    cl = CraigslistForSale(site='sfbay', area=area, category='sss',
                    filters={'max_price': 4000, 'min_price': 0})
    new_most_recent_item = printResults(latest_item_details, cl.get_results(sort_by='newest', geotagged=True, limit=2500));
    # most_recent_file.write("}")
    # writeToMostRecentFile(area, new_most_recent_item)
    f.write("]")

  f.write("}")
  f.close()

def printResults(latest_item_details, results):
  firstItem = True
  for result in results:
    # if (str(result["name"].encode('utf-8')) == latest_item_details["name"] and str(result["datetime"].encode('utf-8')) == latest_item_details["datetime"]):
    #   if(firstItem == True):
    #     return {}
    #   else:
    #     return new_most_recent_item
    # else:
    if (firstItem == False): 
      f.write(",\n")
    else: 
      new_most_recent_item = result
      firstItem = False
    result = sanitizeOutput(result)
    f.write(json.dumps(result))
    # print("Comparing " + " \"" + str(result["name"].encode('utf-8')) + "\" " + " with " + " \"" +latest_item_details["name"] + "\" ")
    # print("Comparing " + str(result["datetime"].encode('utf-8')) + " with " + latest_item_details["datetime"])
  # return new_most_recent_item # not needed at the moment
  return {}

def writeToMostRecentFile(area, new_most_recent_item):
  if(new_most_recent_item != {}):
    mrf = open('most_recent_item.json', 'r+')
    try: 
      master_recent_list = json.load(mrf)
      master_recent_list[area] = new_most_recent_item
      mrf.seek(0)
      # most_recent_file.write("{")
      mrf.write(json.dumps(master_recent_list))
      mrf.truncate()
      mrf.close()
    except:
      master_recent_list = {}
      master_recent_list[area] = new_most_recent_item
      mrf.write(json.dumps(master_recent_list))
      mrf.close()


def getLatestItemDetails(area):
  latest_item_details = {}
  with open('most_recent_item.json') as json_data:
    try:
      d = json.load(json_data)
    except ValueError:
      d = {}
  try: 
    latest_item_details["name"] = str(d[area]["name"])
  except KeyError:
    latest_item_details["name"] = ""

  try:
    latest_item_details["datetime"] = str(d[area]["datetime"])
  except KeyError:
    latest_item_details["datetime"] = ""

  return latest_item_details
      

def sanitizeOutput(result):
  # Sanitize the string to make it free of double quotes and backslashes, which screw with json
  result["name"] = re.sub('["\\\\]', '', result["name"]) #Regex to get rid of a single backslash, found through \\, is \\\\
  return result

if __name__ == "__main__":
   main()
 