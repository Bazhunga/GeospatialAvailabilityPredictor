from craigslist import CraigslistHousing
cl_h = CraigslistHousing(site='sfbay', area='sfc', category='sss',
                         filters={'max_price': 1200})

for result in cl_h.get_results(sort_by='newest', geotagged=True):
    print result
