from __future__ import print_function
from geo_utils import *
import random as rand
from math import log
import sys
from numpy.random import choice
# Important variables
# ward_post_norm_list --> contains the probabilities of a post originating from a certain ward
# ward_people_age_dist --> 44 x 18 matrix. 44 wards, each having a distribution of people
ward_list = []
for i in range(1,45):
    ward_list.append(i)

post2ward_dict = np.load('postings_to_ward.npy').item()
ward_feature_set_age = np.load("ward_agegroup_training.npy").item()

def getNormalizedWardPostProbability():

    ward_post_count_list = []
    total = 0
    for ward in range (1, 45):
        i = ward - 1
        ward_post_count_list.append(0)
        for key in post2ward_dict[ward]:
            ward_post_count_list[i] += post2ward_dict[ward][key]
            total += post2ward_dict[ward][key]

    # Normalize counts over total
    ward_post_norm_list = []
    for item in ward_post_count_list:
        ward_post_norm_list.append(float(item)/float(total))

    # print(ward_post_norm_list)

    return ward_post_norm_list

def getPeopleMatrix(ward):
    # Normalized wrt entirety of Toronto to get likelihood of pulling a person 
    wardkey = "ward_" + str(ward)
    people_total = 0
    ward_people_dist = ward_feature_set_age[wardkey]

    ward_people_count_list = []
    for age_feature in popfeat_age:
        people_total += ward_feature_set_age[wardkey][age_feature]
        ward_people_count_list.append(ward_feature_set_age[wardkey][age_feature])

    ward_people_norm_list = []

    i = 0
    for bucket in ward_people_count_list:
        ward_people_norm_list.append(float(bucket) / float(people_total))

    assert(len(ward_people_norm_list) == 18)
    return ward_people_norm_list

def getPostCategoryProbability(ward):
    ward_posts = post2ward_dict[ward]
    # print("Ward posts")
    # print(ward_posts)
    posting_list = []
    post_ward_total = 0
    for i, category in enumerate(ward_posts):
        wpc = ward_posts[reverse_targ_to_class[str(i + 1)]]
        post_ward_total += wpc
        posting_list.append(wpc)
 
    # print(posting_list)
    # print(post_ward_total)

    posting_norm_list = []
    for item in posting_list:
        posting_norm_list.append(float(item)/float(post_ward_total))

    assert(len(posting_norm_list) == 78)
    return posting_norm_list


if __name__ == "__main__":

    # Currently we're using ward feature sets to learn categories
    # The prediction is not extreemly strong, as it is 20% accurate 
    # during cross validation and only 10% accurate on a test set

    # Another thing we can look at is its hit rate of 5 accuracy. 
    # This is markedly better, as it remains over 40%. 
    # As a result, it's not really worth looking at accuracy.
    # However, it is worth looking ito the weights on features 
    # By looking at the weights, we can see which ones are emphasized
    # more due to their higher weight
    # For example, for heavy machinery, perhaps the most prominent group that 
    # was inferred by the program is peopl from 30-35. 
    # While this is nice, and can confirm with our biases towards society, 
    # there is no way to verify whether this is correct
    
    # What we can do is create our own "Toronto" with its own population 
    # distribution. Except we can create "people" with an age, ethnicity, and
    # anything else. We can generate posts from those people, then run the 
    # predictor on them. 

    '''
    Variables describing a person

    ward_choice
    person_bucket

    '''
    sim_cat_list = []
    sim_ward_list = []
    sim_person_list = []
    sim_geoloc_list = []
    for iters in range(1,55000):
        print(iters)
    
        # >>> Get the probability of ward from the month of data
        ward_post_norm_list = getNormalizedWardPostProbability()
        # print(choice(np.array(ward_list), np.array(ward_post_norm_list)))
        ward_choice = choice(ward_list, 1, p=ward_post_norm_list)[0]
        # print("Ward " + str(ward_choice))
        sim_ward_list.append(ward_choice)


        
        # >>> Generate a person based on distribution of people 
        distribution_choice = getPeopleMatrix(ward_choice)

        person_bucket = choice(popfeat_age, 1, p=distribution_choice)[0]
        # Sanity check for proper choicing
        # first = 0
        # total = 0
        # for c in person_bucket:
        #     total += 1
        #     if(c == "0 to 4 years"):
        #         first+=1
        # print(str(first) + "/" + str(total))

        # print(person_bucket)
        sim_person_list.append(person_bucket)

        
        # >>> Generate a post category based on distribution captured by the data
        # At this point, we would have a the ward to originate from
        # We'd have generated a person profile (aged 0-5, etc)
        # Though we're still missing the probabilties of a person posting a certain thing. Since it's very unlikely for 0-5 year olds to post on craigslist
        # Assume the buckets are all equally likely to generate posts for now 

        ward_category_prob_list = getPostCategoryProbability(ward_choice)
        # print(ward_category_prob_list)
        # for i, cat in enumerate(ward_category_prob_list):
        #     print(target_to_class[str(i + 1)] + ": " + str(cat))
        category_choice = choice(78, 1, p=ward_category_prob_list)[0]
        # category_choice += 1 
        # fuo = 0
        # total = 0
        # for c in category_choice:
        #     total += 1
        #     if(c == 31):
        #         fuo+=1
        # print(str(fuo) + "/" + str(total))
        # raw_input("sigh")
        # print("Category: " + target_to_class[str(category_choice)])
        sim_cat_list.append(category_choice)

        # Assume a 2D gaussian distribution of postings for each ward
        # Generate the place
        ward_centroid_dict = np.load("ward_centroid_dict.npy").item()
        ward_cent = ward_centroid_dict[ward_choice]
        # print(ward_cent)
        # print(ward_centroid_dict)

        x_gauss_mult = 0.007 # 1.2 # 1.2 km radius on average
        y_gauss_mult = 0.009 # 1.7 km radius on average 

        mu = 0
        sigma = 1
        x_gauss = rand.gauss(mu, sigma)
        y_gauss = rand.gauss(mu, sigma)

        # print(x_gauss)
        # print(y_gauss)

        ward_cent_mod = (ward_cent[0] + x_gauss*x_gauss_mult, ward_cent[1] + y_gauss*y_gauss_mult)

        # print(ward_cent_mod)
        sim_geoloc_list.append(ward_cent_mod)

    np.save("sim_cat_list", sim_cat_list)
    np.save("sim_ward_list", sim_ward_list)
    np.save("sim_person_list", sim_person_list)
    np.save("sim_geoloc_list", sim_geoloc_list)


    # By doing this, we're able to verify if our predictions are correct,

    # We can take a look into the features that best define the categories 
    # by looking at weights, but there is no way of finding out 
    # whether the weights are correct
    # 
    # In order to bridge the gap, we should create a simulated population
    # based on the distributions of people we know 
    # 
    # Get distributions for each of the wards
    # Age 
    # Female/Male
    # Household type
    # 
    # P(coordloc | ward loc)  we have to build this, this would be gaussian over ward centre.

