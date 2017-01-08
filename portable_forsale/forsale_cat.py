from __future__ import print_function
from craigslist import CraigslistForSale
from datetime import datetime
import json
import re
import time
from support import *

# Variant of the forsale scraper that gathers 35 postings 

regionArray = ['tor']
# ['tor', 'drh', 'yrk', 'bra', 'mss', 'oak']
# ['sfc', 'sby', 'eby', 'pen', 'nby', 'scz']
# Apa is apartments/housing for rent
# Hou is apartments wanted
# Rooms and shares is roo

filename = datetime.now()
filename = filename.strftime('%m_%d_%Y_%H_%M_%S.json')
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
    for i in range(1,79):
      # print(target_to_class[str(i)])
      f.write("{\"" + target_to_class[str(i)] + "\":[")
      cl = CraigslistForSale(site='toronto', area=area, category=target_to_class[str(i)],
                      filters={'max_price': 4000, 'min_price': 0})
      printResults(latest_item_details, cl.get_results(sort_by='newest', geotagged=True, limit=1));
      if(i == 78): 
        f.write("]}")
      else:
        f.write("]},")
    f.write("]")
  f.write("}")
  f.close()

def printResults(latest_item_details, results):
  firstItem = True
  for result in results:
    if (firstItem == False): 
      f.write(",\n")
    else: 
      new_most_recent_item = result
      firstItem = False
    result = sanitizeOutput(result)
    f.write(json.dumps(result))
    time.sleep(2)
  return

def sanitizeOutput(result):
  # Sanitize the string to make it free of double quotes and backslashes, which screw with json
  result["name"] = re.sub('["\\\\]', '', result["name"]) #Regex to get rid of a single backslash, found through \\, is \\\\
  return result

if __name__ == "__main__":
   main()
 
