# Handoff Document by Kevin Zhu

There were many experiments conducted over the course of a year, so unfortunately the code is a bit scattered. 

I'll explain all the files in this document


==================
= OTHER FILES
==================


>> 2011_Ward_Profiles.xlsx
Contains all distributions for a variety of different ward features of Toronto in 2011

>> Occurences_Legend
Contains 78 categories and their corresponding names that were found when scraping craigslist 

>> pkl files
These are multiclass logisitc regression models that can be fed into the ml_multiclass_category_trainer.py

>> mutual information
some meeting notes

>> predictions_and_top_five.txt
Example file of predictions that are made, hit rate of 5, etc

==================
= PYTHON SCRIPTS
==================
>> data_processing_category_to_ward.py
Used to visualize the category distributions per ward
    - Requires the postings_to_ward.npy to get the posting associates
    - Uses the ward_centroid_dict.npy to get centroids of wards
    - Outputs the postings_to_ward.npy which associates a posting to a ward based on the 

>> geo_utils.py
Helper script (should not be run on its own, it doesn't do anything) that contains definitions to everything you need. 
Additional declarations should be placed here for ease of access
Also has helper functions 
    - Haversine for distances between 2 geographic points
    - other functions you might find usefule

>> heatmap_category_prob.py
Visualizes the spread and clustering of different categories using folium

>> heatmap_model_probs.py
Plots the probabilities of given categories on the torotno map. 
This uses category_probabilities_xxx.npy which contains the probabiltiies for a particular post, and then creates a heatmap

>> ml_bayes_category_trainer.py
The bayes net used to make predictions
Nodes are represented by lists of categories, geolocations, people (that generated the posting) and ward. 
The element of each represent one post (posting is created by person[0], has category[0], ward[0], etc)

There are 3 modes: 
    "p_i_w" gets probability of item given the ward. 
    "p_i_a" gives probability given the age group 
    "classify", which makes predictions

>> ml_multiclass_category_trainer.py
The file used to train the MCLR model and use it to make predictions
The try,except portion allows for speed ups if a model has already been created
Ensure you have a postings_to_ward.npy to work with

>> ml_multiclass_category_trainer_simulated.py
same thing as before, but with simulated data

>> ml_multiclass_category_trainer_with_test.py
Same thing as before but strictly for testing

>> ml_multiclass_get_probabilities_of_category
Used to get probabiltiies of category predictions


>> population_simulator.py
Generator for posts

>> visualizer_mapper.py 
takes in a CSV, and then does visualizer of data on a folium map

>> ward_feature_grapher.py
Takes in census data and outputs bar graphs for each ward. Good for visualizing distributions

