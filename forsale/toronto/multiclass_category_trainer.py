from __future__ import print_function
from thesis_imports import *
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

target_one_hot_list = ["app", "ard", "art", "atq", "bab", "bdp", "bfs", "bid", "bik", "bkd", "bks", "boa", "bop", "bpo", "cld", "clo", "clt", "ctd", "cto", "eld", "ele", "emd", "fod", "for", "fuo", "grd", "grq", "hab", "had", "hsd", "hsh", "hvo", "jwl", "mat", "mcy", "mob", "mod", "mpo", "msg", "phd", "pho", "ppd", "ptd", "pts", "rvs", "sdp", "sgd", "snw", "sop", "spo", "syd", "sys", "tag", "tix", "tld", "tls", "vgm", "wan", "wto"]

def features_dict_to_matrix(features_dict):
   print (features_dict)


def target_dict_to_matrix(target_dict):
   # Target 1 hot vector is coded in the following order [1 0 0 0 ...] would be "app"
   '''
   app: appliances by owner
   ard: arts and crafts by dealer
   art: arts & crafts - by owner
   atq: antiques - by owner
   bab: baby & kid stuff - by owner
   bdp: bicycle parts - by dealer
   bfs: business/commercial - by owner
   bid: bicycles - by dealer
   bik: bicycles - by owner
   bkd: books & magazines - by dealer
   bks: books & magazines - by owner
   boa: boats - by owner
   bop: bicycle parts - by owner
   bpo: boat parts - by owner
   cld: clothing and accessories by dealer
   clo: clothing and accessories by owner
   clt: collectibles - by owner
   ctd: cars & trucks - by dealer
   cto: cars & trucks - by owner
   eld: electronics - by dealer
   ele: electronics - by owwner
   emd: cds / dvds / vhs - by owner
   fod: general for sale - by dealer
   for: general for sale by owner
   fuo: furniture - by owner
   grd: farm & garden - by owner
   grq: farm and garden by dealer
   hab: health and beauty - by owner
   had: health and beauty - by dealer
   hsd: household items - by dealer
   hsh: household items - by owner
   hvo: heavy equipment - by owner
   jwl: jewelry - by owner
   mat: materials by owner
   mcy: motorcycles/scooters - by owner
   mob: cell phones - by owner
   mod: cell phones - by dealer
   mpo: motorcycle parts - by owner
   msg: musical instruments - by owner
   phd: photo/video - by dealer
   pho: photo/video - by owner
   ppd: appliances by dealer
   ptd: auto parts by dealer
   pts: auto parts by owner
   rvs: rvs by owner
   sdp: computer parts by dealer
   sgd: sporting goods - by dealer
   snw: atvs, utvs, snowmobiles - by owner
   sop: computer parts by owner
   spo: sporting good by owner
   syd: computers - by dealer
   sys: computers - by owner
   tag: toys and games by owner
   tix: tix by owner
   tld: tools - by dealer
   tls: tools - by owner
   vgm: video gaming by owner
   wan: wanted by owner
   wto: auto wheels & tires - by owner
   '''
   print (target_dict)
   target_list = []
   for ward in range(1, 45):
      ward_one_hot = []
      target_list.append(target_dict[ward])      


if __name__ == "__main__":
   ag_features_dict = np.load('ward_agegroup_training.npy').item()
   ag_targets_dict = np.load('ward_most_common_targets.npy').item()

   # ag_feature_matrix = features_dict_to_matrix(ag_features_dict)
   ag_targets_matrix = target_dict_to_matrix(ag_targets_dict)


