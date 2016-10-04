from craigslist import CraigslistHousing
import time

regionArray = ['eby', 'pen', 'nby', 'scz']
#['sfc', 'sby', 'eby', 'pen', 'nby', 'scz']
# Apa is apartments/housing for rent
# Hou is apartments wanted
# Rooms and shares is roo

def main(): 
   for area in regionArray:
      #for cat in categoryArray: 
      print("REGION: " + area)
      cl = CraigslistHousing(site='sfbay', area=area, category='hhh',
                         filters={'max_price': 3000, 'min_price': 0})
      printResults(cl.get_results(sort_by='newest', geotagged=True, limit=1000));
      time.sleep(30)


def printResults(results):
   for result in results:
    print result

if __name__ == "__main__":
   main()
