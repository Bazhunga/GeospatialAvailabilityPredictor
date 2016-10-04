from craigslist import CraigslistJobs
import time

regionArray = ['sfc', 'sby', 'eby', 'pen', 'nby', 'scz']

def main(): 
   for area in regionArray:
      print("REGION: " + area)
      cl = CraigslistJobs(site='sfbay', area=area,
                            filters={'is_internship': False, 'employment_type': ['full-time']})
      printResults(cl.get_results(sort_by='newest', geotagged=True, limit=1000));
      time.sleep(30)


def printResults(results):
   for result in results:
    print result

if __name__ == "__main__":
   main()
