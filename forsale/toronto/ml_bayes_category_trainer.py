from __future__ import print_function
from geo_utils import *
from sklearn import datasets
from sklearn import linear_model, datasets
from sklearn.svm import LinearSVC
import random as rand
from sklearn.externals import joblib
from math import log

from population_simulator import *
# Gets functions to get P(Item | Ward) --> getNormalizedWardPostProbability()
#                       P(Age | Ward)  --> getPeopleMatrix()

'''
Kevin Zhu
Feb 7, 2016

Bayes Predictor 
We know P(item|Ward) and P(age|ward)
What is P(age|item)?

P(age|item) = P(age, item)/P(item)
P(item) = sum_over(ward, P(item|ward))
P(age|ward) and P(item|ward) are conditionally independent
We can get P(age,ward) by summing over ward for age and item and then multiplying them together
'''

sim_cat_npy = "sim_cat_list.npy"
sim_geoloc_npy = "sim_geoloc_list.npy"
sim_person_npy = "sim_person_list.npy"
sim_ward_npy = "sim_ward_list.npy"


# Get P(I|A) or P(I|W)
test_type = "p_i_a" # p_i_w



# Get P(age|category)
# Output -> A vector representing prob distribution over age groups
def predict_item_by_ageward(ward):
    #          Ward
    #           /\
    #       item  age
    # Therefor p(W,I,A) = p(W)p(I|W)p(A|W)
    # So p(I,A) = sumout_ward p(W,I,A)

    # Direct Comparison with MCLR
    # ===========================================
    # MCLR gives p(Item|Ward)
    # Get the probability distribution of each ward
    prob_list = getPostCategoryProbability(ward) 
    # This is just predicting the most frequent class for
    # each ward!

    return prob_list

def predict_item_by_age(age_bucket):
    

    # Investigating age group only
    #============================================
    # Need to find P(I | A)
    # P(I|A) = P(I^A)/P(A)
    #        = sumout_W P(I,W,A)/sumout_I sumout_W P(I,W,A)

    # Calculate the numerator
    p_item_agegroup = np.zeros(78)
    post_prob_ward = getNormalizedWardPostProbability()
    for ward in range (1,45): # Here we are summing out the ward
        index = ward - 1
        tmp_p_w = post_prob_ward[index] # one num
        tmp_p_igw = np.array(getPostCategoryProbability(ward)) # distribution --> shape 78 x 1
        tmp_p_agw = getPeopleMatrix(ward)[age_bucket] # Single number

        tmp_tot = tmp_p_w*tmp_p_igw*tmp_p_agw

        p_item_agegroup += tmp_tot


    # Sum out the numerator over the item to get the denominator
    p_agegroup = 0
    for category in range (1, 79):
        index = category - 1
        p_agegroup += p_item_agegroup[index]    

    p_item_given_agegroup = p_item_agegroup/p_agegroup

    # CORRECT YOUR MISTAKE
    # Compare one hot LR with bayesian
    # Math is correct, need to marginalize correctly. Use the numerator to generate the denominator. 
    # Motivate with data fusion --> use this as a use case
    # Empirical comparison
    # 

    return p_item_given_agegroup


    # # P(I|W)
    # prob_list == post_prob_ward(ward)
    # p_item_gward == prob_list[age_bucket]

    # # P(A|W)
    # temp_ward_profile = np.array(getPeopleMatrix(ward))
    # p_age_gward = temp_ward_profile[age_bucket]

    # # P(W)
    # # Assume uniform distribution
    # # p_ward = 1.0/44.0
    # # Take productivity into account
    # p_ward = getNormalizedWardPostProbability()[ward - 1]

    # # P(A,W)


    # Investigating Age Groups + Ward
    #=============================================
    # #p(item) = sumout_ward p(item|ward)
    # p_item = 0
    # # Sum out ward for a particular item_category
    # for ward in range(1,45):
    #     prob_list == post_prob_ward(ward)
    #     p_item += prob_list[target_to_class[item_category]]
    # p_item = float(p_item) / 44.0 # Normalize over 44 wards

    # #Age probability distribution summed over ward
    # p_age = np.zeros(18)
    # for ward in range (1,45):
    #     temp_ward_profile = np.array(getPeopleMatrix(ward))
    #     p_age = np.sum(p_age, temp_ward_profile)
    # p_age = p_age / 44.0 # normalize

   

    # p_item_age = np.zeros(18)
    # p_item_age = 

if __name__ == "__main__":
    # Take in the bunch of generated test cases
    # Get hit rate of 5
    # Get accuracy
    # Log likelihood

    # Currently the model is "trained" on past data. Being tested on new synthetic data
    # Iterate through al 55000 test cases. 
    ag_cat_list = np.load(sim_cat_npy)
    ag_geoloc_list = np.load(sim_geoloc_npy)
    ag_person_list = np.load(sim_person_npy)
    ag_ward_list = np.load(sim_ward_npy)

    if(test_type == "p_i_w"):

        limit = 3000


        count = 1
        total_right = 0
        total_close = 0
        prob_list = []
        log_lik = 0
        for i, ward in enumerate(ag_ward_list):
            if(count > limit):
                break

            # Predictions 
            # Hit rate of 5, accuracy and log likelihood
            #=================================================
            prob_list = predict_item_by_ageward(ward)
            ordered_indices = np.argsort(np.array(prob_list))[::-1]

            top_cat = ordered_indices[0]
            top_five = [x+1 for x in ordered_indices[:5]]

            if(top_cat == ag_cat_list[i]):
                print(" Correct!")
                total_right += 1
            if ((ag_cat_list[i] + 1) in top_five):
                print(" CLOSE!")
                total_close +=1 
                print("")

            print("Desired: "  + str(target_to_class[str(ag_cat_list[i] + 1)]))
            print("Top prediction: " + str(target_to_class[str(top_cat)]))
            print("Top 5: ")
            for k in range(0,5):
                print(target_to_class[str(top_five[k])])

            print("Probability of top: " + str(prob_list[ordered_indices[0]]))
            print("Probability of correct: " + str(prob_list[ag_cat_list[i]]))
            log_lik += log(prob_list[ag_cat_list[i]])

            count += 1


        print("Total correct: " + str(float(total_right)/float(limit)))
        print("Total close: " + str(float(total_close) / float(limit)))
        print("Log likelihood: " + str(log_lik))
    
    # 
    elif(test_type == "p_i_a"):
        # Run through the age groups 
        age_group_to_category = []
        for i, age_group in enumerate(popfeat_age): # Run through the age groups
            cat_list = predict_item_by_age(i) # Get the category list for this age bucket
            ordered_indices = np.argsort(np.array(cat_list))[::-1]

            top_cat = ordered_indices[0] + 1
            #print(cat_list[top_cat - 1])
            raw_input()
            top_five = [x+1 for x in ordered_indices[:10]] # These are categories!
            print(popfeat_age[i])
            for one in top_five:
                print(target_to_class[str(one)] + ": ", end='')
                print(cat_list[one - 1])

            age_group_to_category.append(top_cat)

        for i, cat in enumerate(age_group_to_category):
            print(popfeat_age[i] + ": " +  target_to_class[str(cat)])




# Thing I've discovered
# Trained on the same model, using category | ward is actually more effective 


# Not 
# What the differences between wards are isn't immensely defined.
# Age buckets less granular vs more granular
# Cross product for more granular (age bucket and stuff)
# Tabular?
# T, D , Frequency. Then assign probabilitye
# /Side note: Try decision trees?
# Less biased model, don't have data to train large models
# Logistic regression could be better because more biased (fewer parameters and learn things with higher confidence)
# Compute the P(I|A) based on the frequency. Type 1, Demograph1, frequency 5. Type 2, Demograph1, frequency 5. Sum it up should be 1
# 3 models to compare P(I|A) is certainly the ground truth. Types don't differ accross demographics it says.
# 

# See if the rank order is preserved in log reg (the one hot encoding hack)
# 2 things to do: P(I|A) from the data, then one hot the thing.
# At least probabioiticly, bayes net approach is better probaiblities assuming structure is correct. 
# Saving grace is is ranking is still preserved despite crazy probaiblities. 