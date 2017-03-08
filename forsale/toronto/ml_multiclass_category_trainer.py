from __future__ import print_function
from geo_utils import *
from sklearn import datasets
from sklearn import linear_model, datasets
from sklearn.svm import LinearSVC
import random as rand
from sklearn.externals import joblib
from math import log

# Crafted by this file

# Models crafted by these files
model_age = 'mclr_age.pkl'
model_females = 'mclr_females.pkl'
model_males = 'mclr_males.pkl'
model_hhtype = 'mclr_hhtype.pkl'
model_familytype = 'mclr_familytype.pkl'
model_malestofemales = 'mclr_malestofemales_1000.pkl'

c_val = 1000

# Age
feature_set_npy = "np_entire_feature_list_age.npy"
target_set_npy = "np_target_list_age.npy"
wardlist_set_npy = "np_feature_list_wards_age.npy"
unique_ward_feature_set_npy = "np_unique_ward_feature_set_age.npy"
model_used = model_age
raw_training_npy = "ward_agegroup_training.npy"
popfeat = popfeat_age

# MalestoFemales
# feature_set_npy = "np_entire_feature_list_malestofemales.npy"
# target_set_npy = "np_target_list_malestofemales.npy"
# wardlist_set_npy = "np_feature_list_wards_malestofemales.npy"
# unique_ward_feature_set_npy = "np_unique_ward_feature_set_malestofemales.npy"
# model_used = model_malestofemales
# raw_training_npy = "ward_malestofemales_training.npy"
# popfeat = popfeat_age

# Females
# feature_set_npy = "np_entire_feature_list_females.npy"
# target_set_npy = "np_target_list_females.npy"
# wardlist_set_npy = "np_feature_list_wards_females.npy"
# unique_ward_feature_set_npy = "np_unique_ward_feature_set_females.npy"
# model_used = model_females
# raw_training_npy = "ward_females_training.npy"
# popfeat = popfeat_age


# Males
# feature_set_npy = "np_entire_feature_list_males.npy"
# target_set_npy = "np_target_list_males.npy"
# wardlist_set_npy = "np_feature_list_wards_males.npy"
# unique_ward_feature_set_npy = "np_unique_ward_feature_set_males.npy"
# model_used = model_males
# raw_training_npy = "ward_makmles_training.npy"
# popfeat = popfeat_agekys 

# HHtype
# feature_set_npy = "np_entire_feature_list_hhtype.npy"
# target_set_npy = "np_target_list_hhtype.npy"
# wardlist_set_npy = "np_feature_list_wards_hhtype.npy"
# unique_ward_feature_set_npy = "np_unique_ward_feature_set_hhtype.npy"
# model_used = model_hhtype
# raw_training_npy = "ward_householdtype_training.npy"
# popfeat = popfeat_hhtyp


# FamilyType
# feature_set_npy = "np_entire_feature_list_familytype.npy"
# target_set_npy = "np_target_list_familytype.npy"
# wardlist_set_npy = "np_feature_list_wards_familytype.npy"
# unique_ward_feature_set_npy = "np_unique_ward_feature_set_familytype.npy"
# model_used = model_familytype
# raw_training_npy = "ward_familytype_training.npy"
# popfeat = popfeat_familytype


#============================================================
# Kevin Zhu
# The purpose of this file is to guess the most common posting given its ward distribution
# The model will be trained with the data set (x,t) where x is the ward feature distribution 
# corresponding to t, the posting that occurred in that ward
# We're essentially trying to guess which 
#============================================================

# Takes a population distribution dictionary and transforms it into an ordered list 
# according to the order specified by popfeat_age

def dist_to_list(ward_distribution_dict):
   # print (ward_distribution_dict)
   wd_list = [];
   # for ag in popfeat_age: 
   #    wd_list.append(ward_distribution_dict[ag])
   # return wd_list
   # for ag in popfeat_hhtype: 
   #    wd_list.append(ward_distribution_dict[ag])
   # return wd_list

   for ag in popfeat: 
      wd_list.append(ward_distribution_dict[ag])
   return wd_list
   
# Params
# ag_features_dict contains the distributions of each ward
# ag_ward_to_posting_dict contains ward:{category: number of occurrences}
def organize_training_data(ag_features_dict, ag_ward_to_posting_dict):
   # Each ward as a distribution of population, these are the input features
   # The output would be a single posting
   # for ward in range(1,45):
   #    wardkey = "ward_" + str(ward)
   #    print(dist_to_list(ag_features_dict[wardkey]))
   unique_ward_features = [] # Contains the feature set for each ward. Must be size 44
   entire_feature_list = [] # Bias term
   target_list = []
   ward_list = [] # The ward that the feature set belongs to (entire_feature_list[3] describes ward ward_list[3])
   for ward in ag_ward_to_posting_dict:
      # Iterate through the posting dictionary. Add feature row (the ward distribution) 
      # and then add an entry to the target_list (which is the feature # code)
      wardkey = "ward_" + str(ward)
      unique_ward_features.append(dist_to_list(ag_features_dict[wardkey]))
      for category in ag_ward_to_posting_dict[ward]:
         # print(str(ward) + " " + str(category) + " " + str(ag_ward_to_posting_dict[ward][category]))
         for i in range(0, ag_ward_to_posting_dict[ward][category]):
            # TODO: Stop calling dist_to_list every time. You can pre-compute this
            ward_list.append(wardkey)
            # ag_features_dict[wardkey] gets the ward distribution
            # dist_to_list converts this into a list
            entire_feature_list.append(dist_to_list(ag_features_dict[wardkey]))
            target_list.append(target_to_class[category])
            # print (entire_feature_list)
            # print (target_list)
   np_entire_feature_list = np.asarray(entire_feature_list)
   np_target_list = np.asarray(target_list)
   np_ward_list = np.asarray(ward_list)
   np_unique_ward_features = np.asarray(unique_ward_features)

   np.save(feature_set_npy, np_entire_feature_list)
   np.save(target_set_npy, np_target_list)
   np.save(wardlist_set_npy, np_ward_list)
   np.save(unique_ward_feature_set_npy, np_unique_ward_features)

   print("Length of unique: " + str(len(unique_ward_features)))

   return np_entire_feature_list, np_target_list, np_ward_list

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

# We train multiclass logistic regression on a subset of the examples
# Each posting (category) is a data point. The features are the distributions
# X: The distributions Y: The class
def train_mc_lr(X_train, Y_train):
   logreg = linear_model.LogisticRegression(C=c_val, solver='lbfgs', multi_class='multinomial', max_iter=1000)
   logreg.fit(X_train, Y_train)

   # Save the model
   joblib.dump(logreg, model_used) 

   return logreg

# With the unused examples, we try to guess what posting would come out of a given distribution
def run_mc_lr(logreg, X_test, Y_test, X_labels):
   # Given a certain distribution, what is the most probable category that will be posted
   # here? 
   # Of course this is going to run into problems in terms of accuracy, since logically, any
   # sort of posting can be put in any ward, so we can never get 100%
   # During training, we have the same distribution corresponding to multiple targets, so
   # naturally we'd get stuff wrong.
   results = logreg.predict(X_test)
   results_probs = logreg.predict_proba(X_test)
   total_right = 0
   total_close = 0
   total = len(Y_test)

   num_top = 5

   log_lik = 0


   for i in range(0, len(results)):
      # try: 
      #    cm_cats[int(Y_test[i] - 1)][results[i]] += 1
      # except KeyError: 
      #    cm_cats[int(Y_test[i] - 1)][results[i]] = 1
      ordered_indices = np.argsort(np.array(results_probs[i]))[::-1]

      print(X_labels[i], end = " ")
      if(results[i] == Y_test[i]):
         print(" CORRECT!")
         total_right += 1
      else:
         top_five = [x+1 for x in ordered_indices[:num_top]] 
         if(Y_test[i] in top_five):
            print(" Close!")
            total_close += 1
         print("")
      #print("Result: " + str(results[i]) + ": " + str(target_to_class[str(results[i])]) + " Desired: " + str(Y_test[i]) + ": " + str(target_to_class[str(Y_test[i])]), end=" ")
      print("Desired: " + str(Y_test[i]) + " --> " + str(target_to_class[str(Y_test[i])]))
      print("Top prediction: " + str(results[i]) + " --> " + str(target_to_class[str(results[i])]))
      print("Top " + str(num_top) + ":")
      for k in range(0, num_top): # We take the top 5, which are indexed in ordered_indices
         category = ordered_indices[k] + 1
         print(str(category) + " --> " + str(target_to_class[str(category)]) + " " + str(results_probs[i][ordered_indices[k]]))
      print("\n")

      log_lik += log(results_probs[i][ordered_indices[0]]) 

   print("Log likelihood")
   print(str(-1 * log_lik))

   print("Percent correct")
   print(str(float(total_right)/float(total)))

   print("Percent close (close count + correct count)")
   print(str(float(total_right + total_close)/float(total)))

   # print(results)
   # print(results_probs.shape)
   # for i, thing in enumerate(cm_cats):
   #    print(str(i + 1) + str(cm_cats[i]))

# def get_prob_from_all_wards():
#    # Now that we have the trained model, we want to know the probabilities of 
#    # predicting each of the 78 classes. 
#    # We can then heat map these probabilities to see where the model is guessing


if __name__ == "__main__":
   num_test_set = 3000

   try: 
      logreg = joblib.load(model_used)
      ag_feature_matrix = np.load(feature_set_npy)
      ag_targets_matrix = np.load(target_set_npy) 
      ag_feature_labels = np.load(wardlist_set_npy) # Labels for the feature set. 
      ag_unique_ward_features = np.load(unique_ward_feature_set_npy)
      
      print("Load successful")

      print("Crafting your test set")
      num_of_targs = ag_targets_matrix.shape[0]
      random_indices = rand.sample(xrange(0, num_of_targs), num_of_targs)
      ag_X_test = np.empty([0, ag_feature_matrix.shape[1]])
      ag_Y_test = []
      ag_X_labels_test = [] 
      for index in range(len(random_indices) - num_test_set, len(random_indices)):
         # print("index2: " + str(random_indices[index]))
         ag_X_test = np.vstack((ag_X_test, ag_feature_matrix[random_indices[index]]))
         ag_Y_test.append(ag_targets_matrix[random_indices[index]])
         ag_X_labels_test.append(ag_feature_labels[random_indices[index]])

      run_mc_lr(logreg, ag_X_test, ag_Y_test, ag_X_labels_test)
   except IOError:

      ag_features_dict = np.load(raw_training_npy).item()
      ag_ward_to_posting_dict = np.load('postings_to_ward.npy').item()

      # ag_feature_matrix = features_dict_to_matrix(ag_features_dict)
      # ag_targets_matrix = target_dict_to_matrix(ag_targets_dict)
      
      ag_feature_matrix, ag_targets_matrix, ag_feature_labels = organize_training_data(ag_features_dict, ag_ward_to_posting_dict)

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

      print("Number of samples: " + str(len(random_indices)))

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
      ag_X_labels_test = [] 
      for index in range(len(random_indices) - num_test_set, len(random_indices)):
         # print("index2: " + str(random_indices[index]))
         ag_X_test = np.vstack((ag_X_test, ag_feature_matrix[random_indices[index]]))
         ag_Y_test.append(ag_targets_matrix[random_indices[index]])
         ag_X_labels_test.append(ag_feature_labels[random_indices[index]])


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
      run_mc_lr(logreg, ag_X_test, ag_Y_test, ag_X_labels_test)



