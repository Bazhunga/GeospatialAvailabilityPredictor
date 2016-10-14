import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os

pd.set_option('display.mpl_style', 'default')

basePath = os.path.dirname(os.path.abspath(__file__))
data = pd.read_json(basePath + "/10_10_2016_17_33_21.json")

data[:3]