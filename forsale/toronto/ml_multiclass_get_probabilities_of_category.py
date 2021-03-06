from __future__ import print_function
from ml_multiclass_category_trainer import *
from geo_utils import *

feature_set_toggle = "age"
unique_ward_feature_set_npy = ""
popfeat = []

if __name__ == "__main__":
   # Load the model
   if(feature_set_toggle == "age"):
      print("age")
      logreg_model = joblib.load(model_age)
      unique_ward_feature_set_npy = "np_unique_ward_feature_set_age.npy"
      popfeat = popfeat_age

   elif(feature_set_toggle == "females"):
      print("females")
      logreg_model = joblib.load(model_females)
      unique_ward_feature_set_npy = "np_unique_ward_feature_set_females.npy"
      popfeat = popfeat_age


   elif(feature_set_toggle == "males"):
      print("males")
      logreg_model = joblib.load(model_males)
      unique_ward_feature_set_npy = "np_unique_ward_feature_set_males.npy"
      popfeat = popfeat_age


   elif(feature_set_toggle == "hhtype"):
      print("hhtype")
      logreg_model = joblib.load(model_hhtype)
      unique_ward_feature_set_npy = "np_unique_ward_feature_set_hhtype.npy"
      popfeat = popfeat_hhtype

   elif(feature_set_toggle == "familytype"):
      print("familytype")
      logreg_model = joblib.load(model_familytype)
      unique_ward_feature_set_npy = "np_unique_ward_feature_set_familytype.npy" 
      popfeat = popfeat_familytype

   else:
      print("Hasn't been implemented yet")

   print(popfeat)

   # Load the feature set
   
   # Unique feature sets are the distributions for each ward
   # For age, that would be the # of people in each age bucket
   ag_unique_ward_features = np.load(unique_ward_feature_set_npy)
   print(ag_unique_ward_features)
   # ag_unique_ward_features = [1]
   # ag_unique_ward_features.append(ag_unique_ward_features_no_bias)
   print("WFS: " + str(len(ag_unique_ward_features)))

   # Find the probabilities
   probs = logreg_model.predict_proba(ag_unique_ward_features)
   # This would spit out the probabilities over the 78 classes
   print(ag_unique_ward_features.shape)
   print(probs.shape) # 44 wards x 78 categories
   # np.save("category_probabilities_" + feature_set_toggle, probs)

   # Find the top 5 features influencing output
   # Remember that there are a different set of weights for each class
   parms = logreg_model.coef_
   # print(len(parms)) # 78 classes
   # print(len(parms[0])) # 18 weights
   # print(parms)
   for i, parm in enumerate(parms): # Iterate through all the parameters b
      print("i: " + str(i) + " parm: " + str(parm))
      tclass = i + 1 # Get the class (1 - 78)
      highest_features = np.argsort(np.array(parm))[::-1][:5] # Find top 5 features
      print("highest_features\n" + str(highest_features))
      print("Category: " + target_to_class[str(tclass)])
      for feature in highest_features :
         print("feature: " + str(feature))
         print("     " + str(popfeat[feature]) + ": " + str(parm[feature])) # The feature: value

