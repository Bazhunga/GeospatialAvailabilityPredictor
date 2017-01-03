from __future__ import print_function
from math import radians, cos, sin, asin, sqrt
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

target_to_class = {
"app":1,"1":"app: appliances by owner",
"ard":2,"2":"ard: arts and crafts by dealer",
"art":3,"3":"art: arts & crafts - by owner",
"atd":4,"4":"atd: antiques - by dealer",
"atq":5,"5":"atq: antiques - by owner",
"bab":6,"6":"bab: baby & kid stuff - by owner",
"bad":7,"7":"bad: baby & kid stuff - by dealer",
"bar":8,"8":"bar: barter",
"bdp":9,"9":"bdp: bicycle parts - by dealer",
"bfd":10,"10":"bfd: business/commercial - by dealer",
"bfs":11,"11":"bfs: business/commercial - by owner",
"bid":12,"12":"bid: bicycles - by dealer",
"bik":13,"13":"bik: bicycles - by owner",
"bkd":14,"14":"bkd: books & magazines - by dealer",
"bks":15,"15":"bks: books & magazines - by owner",
"boa":16,"16":"boa: boats - by owner",
"bod":17,"17":"bod: boats - by dealer",
"bop":18,"18":"bop: bicycle parts - by owner",
"bpo":19,"19":"bpo: boat parts - by owner",
"cbd":20,"20":"cbd: collectibles - by dealer",
"cld":21,"21":"cld: clothing and accessories by dealer",
"clo":22,"22":"clo: clothing and accessories by owner",
"clt":23,"23":"clt: collectibles - by owner",
"ctd":24,"24":"ctd: cars & trucks - by dealer",
"cto":25,"25":"cto: cars & trucks - by owner",
"eld":26,"26":"eld: electronics - by dealer",
"ele":27,"27":"ele: electronics - by owwner",
"emd":28,"28":"emd: cds / dvds / vhs - by owner",
"emq":29,"29":"emq: cds / dvds / vhs - by dealer",
"fod":30,"30":"fod: general for sale - by dealer",
"for":31,"31":"for: general for sale by owner",
"fuo":32,"32":"fuo: furniture - by owner",
"grd":33,"33":"grd: farm & garden - by owner",
"grq":34,"34":"grq: farm and garden by dealer",
"hab":35,"35":"hab: health and beauty - by owner",
"had":36,"36":"had: health and beauty - by dealer",
"hsd":37,"37":"hsd: household items - by dealer",
"hsh":38,"38":"hsh: household items - by owner",
"hvd":39,"39":"hvd: heavy equipment - by dealer",
"hvo":40,"40":"hvo: heavy equipment - by owner",
"jwd":41,"41":"jwd: jewelry - by dealer",
"jwl":42,"42":"jwl: jewelry - by owner",
"mad":43,"43":"mad: materials - by dealer",
"mat":44,"44":"mat: materials by owner",
"mcd":45,"45":"mcd: motorcycles/scooters - by dealer",
"mcy":46,"46":"mcy: motorcycles/scooters - by owner",
"mob":47,"47":"mob: cell phones - by owner",
"mod":48,"48":"mod: cell phones - by dealer",
"mpo":49,"49":"mpo: motorcycle parts - by owner",
"msd":50,"50":"msd: musical instruments - by dealer",
"msg":51,"51":"msg: musical instruments - by owner",
"phd":52,"52":"phd: photo/video - by dealer",
"pho":53,"53":"pho: photo/video - by owner",
"ppd":54,"54":"ppd: appliances by dealer",
"ptd":55,"55":"ptd: auto parts by dealer",
"pts":56,"56":"pts: auto parts by owner",
"rvs":57,"57":"rvs: rvs by owner",
"sdp":58,"58":"sdp: computer parts by dealer",
"sgd":59,"59":"sgd: sporting goods - by dealer",
"snd":60,"60":"snd: atvs, utvs, snowmobiles - by dealer",
"snw":61,"61":"snw: atvs, utvs, snowmobiles - by owner",
"sop":62,"62":"sop: computer parts by owner",
"spo":63,"63":"spo: sporting good by owner",
"syd":64,"64":"syd: computers - by dealer",
"sys":65,"65":"sys: computers - by owner",
"tad":66,"66":"tad: toys & games - by dealer",
"tag":67,"67":"tag: toys and games by owner",
"tix":68,"68":"tix: tix by owner",
"tld":69,"69":"tld: tools - by dealer",
"tls":70,"70":"tls: tools - by owner",
"tro":71,"71":"tro: trailers - by owner",
"vgd":72,"72":"vgd: video gaming - by dealer",
"vgm":73,"73":"vgm: video gaming by owner",
"wad":74,"74":"wad: wanted - by dealer",
"wan":75,"75":"wan: wanted by owner",
"wtd":76,"76":"wtd: auto wheels & tires - by dealer",
"wto":77,"77":"wto: auto wheels & tires - by owner",
"zip":78,"78":"zip: free stuff"}

popfeat_age = ["0 to 4 years",
               "5 to 9 years",
               "10 to 14 years",
               "15 to 19 years",
               "15 years",
               "16 years",
               "17 years",
               "18 years",
               "19 years",
               "20 to 24 years",
               "25 to 29 years",
               "30 to 34 years",
               "35 to 39 years",
               "40 to 44 years",
               "45 to 49 years",
               "50 to 54 years",
               "55 to 59 years",
               "60 to 64 years",
               "65 to 69 years",
               "70 to 74 years",
               "75 to 79 years",
               "80 to 84 years",
               "85 years and over"]

popfeat_hhtype = ["Census family households",
                  "One-family only households",
                  "Couple family households",
                  "Without children",
                  "With children",
                  "Lone-parent family households",
                  "Other family households",
                  "One-family households with persons not in a census family",
                  "Couple family households",
                  "Without children",
                  "With children",
                  "Lone-parent family households",
                  "Two-or-more-family households",
                  "Non-census family households",
                  "One-person households",
                  "Two-or-more-person households"]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km