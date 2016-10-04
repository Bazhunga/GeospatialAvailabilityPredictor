import json 
from datetime import datetime

def main():
   with open('most_recent_item.json') as json_data:
      d = json.load(json_data)
      print(d)

   latest_name = d["eby"]["name"]
   latest_datetime = d["eby"]["datetime"]
   print latest_name
   print latest_datetime

   filename = datetime.now().time()
   filename = filename.strftime('%m_%d_%Y:%H_%M_%S.txt')

   with open(filename, 'a+') as f:
      f.write(latest_datetime + "\n")
      f.write(latest_name)
      f.close()



if __name__ == "__main__":
   main()