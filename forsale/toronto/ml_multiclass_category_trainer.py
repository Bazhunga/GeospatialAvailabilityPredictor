from __future__ import print_function
from thesis_imports import *
from sklearn import datasets
from sklearn import linear_model, datasets
from sklearn.svm import LinearSVC
import random as rand
from sklearn.externals import joblib

# The purpose of this file is to guess the most common posting given its ward distribution
# The model will be trained with the data set (x,t) where x is the ward feature distribution 
# corresponding to t, the posting that occurred in that ward
# We're essentially trying to guess which 

target_to_class = {"app":1,
                  "ard":2,
                  "art":3,
                  "atd":4,
                  "atq":5,
                  "bab":6,
                  "bad":7,
                  "bar":8,
                  "bdp":9,
                  "bfd":10,
                  "bfs":11,
                  "bid":12,
                  "bik":13,
                  "bkd":14,
                  "bks":15,
                  "boa":16,
                  "bod":17,
                  "bop":18,
                  "bpo":19,
                  "cbd":20,
                  "cld":21,
                  "clo":22,
                  "clt":23,
                  "ctd":24,
                  "cto":25,
                  "eld":26,
                  "ele":27,
                  "emd":28,
                  "emq":29,
                  "fod":30,
                  "for":31,
                  "fuo":32,
                  "grd":33,
                  "grq":34,
                  "hab":35,
                  "had":36,
                  "hsd":37,
                  "hsh":38,
                  "hvd":39,
                  "hvo":40,
                  "jwd":41,
                  "jwl":42,
                  "mad":43,
                  "mat":44,
                  "mcd":45,
                  "mcy":46,
                  "mob":47,
                  "mod":48,
                  "mpo":49,
                  "msd":50,
                  "msg":51,
                  "phd":52,
                  "pho":53,
                  "ppd":54,
                  "ptd":55,
                  "pts":56,
                  "rvs":57,
                  "sdp":58,
                  "sgd":59,
                  "snd":60,
                  "snw":61,
                  "sop":62,
                  "spo":63,
                  "syd":64,
                  "sys":65,
                  "tad":66,
                  "tag":67,
                  "tix":68,
                  "tld":69,
                  "tls":70,
                  "tro":71,
                  "vgd":72,
                  "vgm":73,
                  "wad":74,
                  "wan":75,
                  "wtd":76,
                  "wto":77,
                  "zip":78 }
cm_cats = []
for i in range(0,78):
   cm_cats.append({}); # initialize the confusion matrix counter


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


# DEPRECATED
def features_dict_to_matrix(features_dict):
   entire_feature_list = []
   for ward in features_dict:
      print(features_dict[ward])
      ward_feature_list = []
      for agegroup in popfeat_age: 
         # print (agegroup)
         ward_feature_list.append(features_dict[ward][agegroup])
      entire_feature_list.append(ward_feature_list)

   # for lst in entire_feature_list:
   #    print(lst)
   return np.asarray(entire_feature_list)

# DEPRECATED
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

   There are more (like wad, which is wanted by dealer)
   '''
   print (target_dict)
   target_list = []
   for ward in range(1, 45):
      # print(str(ward) + " " + target_dict[ward])
      try:
         target_list.append(target_to_class[target_dict[ward]])     
      except KeyError :
         target_list.append(0)
   
   return np.asarray(target_list) 

# Takes a population distribution dictionary and transforms it into an ordered list 
# according to the order specified by popfeat_age
def dist_to_list(ward_distribution_dict):
   # print (ward_distribution_dict)
   wd_list = [];
   for ag in popfeat_age: 
      wd_list.append(ward_distribution_dict[ag])
   return wd_list
   # for ag in popfeat_hhtype: 
   #    wd_list.append(ward_distribution_dict[ag])
   # return wd_list
   

# Params
# ag_features_dict contains the distributions of each ward
# ag_ward_to_posting_dict contains ward:{category: number of occurrences}
def organize_training_data(ag_features_dict, ag_ward_to_posting_dict):
   # Each ward as a distribution of population, these are the input features
   # The output would be a single posting
   # for ward in range(1,45):
   #    wardkey = "ward_" + str(ward)
   #    print(dist_to_list(ag_features_dict[wardkey]))
   entire_feature_list = []  
   target_list = []
   for ward in ag_ward_to_posting_dict:
      # Iterate through the posting dictionary. Add feature row (the ward distribution) 
      # and then add an entry to the target_list (which is the feature # code)
      wardkey = "ward_" + str(ward)
      for category in ag_ward_to_posting_dict[ward]:
         # print(str(ward) + " " + str(category) + " " + str(ag_ward_to_posting_dict[ward][category]))
         for i in range(0, ag_ward_to_posting_dict[ward][category]):
            # TODO: Stop calling dist_to_list every time. You can pre-compute this
            entire_feature_list.append(dist_to_list(ag_features_dict[wardkey]))
            target_list.append(target_to_class[category])
            # print (entire_feature_list)
            # print (target_list)
   np_entire_feature_list = np.asarray(entire_feature_list)
   np_target_list = np.asarray(target_list)

   np.save("np_entire_feature_list_hhtype.npy", np_entire_feature_list)
   np.save("np_target_list.npy", np_target_list)

   return np_entire_feature_list, np_target_list

def find_unused_categories(ag_targets_matrix):
   list_cats = range(1,79)   
   for targ in ag_targets_matrix:
      try:
         list_cats.remove(targ)
      except ValueError:
         pass
   print("REMAINING LIST")
   print (list_cats)
   return

def train_mc_lr(X_train, Y_train):
   logreg = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
   logreg.fit(X_train, Y_train)
   # print(logreg)
   # print(logreg.predict(X_train))
   # print(Y_train)
   # >>>> Save the model
   joblib.dump(logreg, 'mclr_age.pkl') 

   return logreg

def run_mc_lr(logreg, X_test, Y_test):
   # Given a certain distribution, what is the most probable category that will be posted
   # here? 
   # Of course this is going to run into problems in terms of accuracy, since logically, any
   # sort of posting can be put in any ward, so we can never get 100%
   # During training, we have the same distribution corresponding to multiple targets, so
   # naturally we'd get stuff wrong.
   results = logreg.predict(X_test)
   results_probs = logreg.predict_proba(X_test)
   total_right = 0
   total = len(Y_test)
   for i in range(0, len(results)):
      try: 
         cm_cats[int(Y_test[i] - 1)][results[i]] += 1
      except KeyError: 
         cm_cats[int(Y_test[i] - 1)][results[i]] = 1

      print("Result: " + str(results[i]) + " Desired: " + str(Y_test[i]), end=" ")
      if(results[i] == Y_test[i]):
         print(" CORRECT!")
         total_right += 1
      else:
         print("\n")

   print("Percent correct")
   print(str(float(total_right)/float(total)))

   print(results)
   print(results_probs.shape)
   for i, thing in enumerate(cm_cats):
      print(str(i + 1) + str(cm_cats[i]))




if __name__ == "__main__":
   num_test_set = 300

   try: 
      logreg = joblib.load('mclr_age.pkl')
      ag_feature_matrix = np.load("np_entire_feature_list_age.npy")
      ag_targets_matrix = np.load("np_target_list.npy") 
      
      print("Aww yuss loaded")

      print("Crafting your test set")
      num_of_targs = ag_targets_matrix.shape[0]
      random_indices = rand.sample(xrange(0, num_of_targs), num_of_targs)
      ag_X_test = np.empty([0, ag_feature_matrix.shape[1]])
      ag_Y_test = []
      for index in range(len(random_indices) - num_test_set, len(random_indices)):
         # print("index2: " + str(random_indices[index]))
         ag_X_test = np.vstack((ag_X_test, ag_feature_matrix[random_indices[index]]))
         ag_Y_test.append(ag_targets_matrix[random_indices[index]])

      run_mc_lr(logreg, ag_X_test, ag_Y_test)
   except IOError:

      ag_features_dict = np.load('ward_agegroup_training.npy').item()
      ag_ward_to_posting_dict = np.load('ward_dict_of_postings.npy').item()

      # ag_feature_matrix = features_dict_to_matrix(ag_features_dict)
      # ag_targets_matrix = target_dict_to_matrix(ag_targets_dict)
      
      ag_feature_matrix, ag_targets_matrix = organize_training_data(ag_features_dict, ag_ward_to_posting_dict)

      # >>>> Debug by printing the feature and target matrix
      # np.set_printoptions(threshold=np.nan)
      # print(ag_feature_matrix)
      # print(ag_targets_matrix)

      # >>>> See if all categories are used
      # find_unused_categories(ag_targets_matrix)

      # >>>> Slice into training examples!
      num_of_targs = ag_targets_matrix.shape[0]
      # We want to randomly sample the training and the validation   

      # Mix up the indices from 0 to the number of targets
      random_indices = rand.sample(xrange(0, num_of_targs), num_of_targs)
      print("Slicin' and Dicin'")

      print(len(random_indices))

      print("Crafting your training set")
      ag_X_train = np.empty([0, ag_feature_matrix.shape[1]])
      ag_Y_train = []
      for index in range(0, len(random_indices) - num_test_set):
         # print(ag_feature_matrix[random_indices[index]])
         # print(ag_targets_matrix[random_indices[index]])
         # print("index: " + str(random_indices[index]))
         ag_X_train = np.vstack((ag_X_train, ag_feature_matrix[random_indices[index]]))
         ag_Y_train.append(ag_targets_matrix[random_indices[index]])

      print("Crafting your test set")
      ag_X_test = np.empty([0, ag_feature_matrix.shape[1]])
      ag_Y_test = []
      for index in range(len(random_indices) - num_test_set, len(random_indices)):
         # print("index2: " + str(random_indices[index]))
         ag_X_test = np.vstack((ag_X_test, ag_feature_matrix[random_indices[index]]))
         ag_Y_test.append(ag_targets_matrix[random_indices[index]])

      print("Shape information")
      print("Total")
      print(ag_feature_matrix.shape)
      print(ag_targets_matrix.shape)
      print("Train")
      print(ag_X_train.shape)
      print(len(ag_Y_train))
      print("Test")
      print(ag_X_test.shape)
      print(len(ag_Y_test))

      print(ag_feature_matrix)
      print(ag_targets_matrix)

      # >>>> Verify proper transfer of data
      print("Random index: " + str(random_indices[45806]))
      print("Feature matrix: " + str(ag_feature_matrix[random_indices[45806]]))
      print("Targ: " + str(ag_targets_matrix[random_indices[45806]]))
      print("ag_X_train: " + str(ag_X_test[0]))
      print("ag_Y_test: " + str(ag_Y_test[0]))
      # >>>> Run logistic regression
   
      logreg = train_mc_lr(ag_X_train, ag_Y_train)
      run_mc_lr(logreg, ag_X_test, ag_Y_test)



