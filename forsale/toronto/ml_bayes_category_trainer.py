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

# Get P(age|category)
# Output -> A vector representing prob distribution over age groups
def predict_item_by_ageward(age_bucket, ward):
    #          Ward
    #           /\
    #       item  age
    # Therefor p(W,I,A) = p(W)p(I|W)p(A|W)
    # So p(I,A) = sumout_ward p(W,I,A)

    # Direct Comparison with MCLR
    # ===========================================
    # MCLR gives p(Item|Ward)
    # Get the probability distribution of each ward
    prob_list == getPostCategoryProbability(ward) 
    # This is just predicting the most frequent class for
    # each ward!
    

    # Investigating age group only
    #============================================
    # Need to find P(I | A)
    p_item_agegroup = np.zeros(78)
    p_agegroup = 0
    post_prob_ward = getNormalizedWardPostProbability()
    for ward in range (1,45)
        tmp_p_w = post_prob_ward[ward] # one num
        tmp_p_igw = np.array(getPostCategoryProbability(ward)) # distribution
        tmp_p_agw = getPeopleMatrix(ward)[age_bucket]

        tmp_tot = tmp_p_w*tmp_p_igw*tmp_p_agw
        p_item_agegroup = p_item_agegroup + tmp_tot

        tmp_p_a = getPeopleMatrix(ward)[age_bucket]
        p_agegroup += tmp_p_a



    p_item_agegroup = p_item_agegroup/np.sum()

    p_item_given_agegroup = p_item_agegroup/p_agegroup



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
    pass
