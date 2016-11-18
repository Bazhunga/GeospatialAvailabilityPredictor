from __future__ import print_function
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import json
import folium
import re
import operator
from haversine import *
from pandas.io.json import json_normalize

a = np.load('ward_agegroup_training.npy').item()
print(a['ward_33']['65 to 69 years'])