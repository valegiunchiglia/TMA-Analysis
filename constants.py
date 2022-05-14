"""
Constants required for the analysis of Formalin-Fixed-Paraffin-Embedded Tumor Microarray (FFPE-TMA)
samples for Deep-Learning based tumor prediction
"""

from itertools import product

# Only the cores that were marked as containing tumor or being normal tissue by
# a pathologist were used for the analysis. If all cores must be included, the 
# list as to include the range of number from 1 to n, with n equal to the total 
# number of cores available in each TMA image


areas_keep = {
    "TMA52": [0,1,2,3,4,5,6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
    27,28,29,30,31, 33,34,35,36,37,38,39,42,43,44,45,46,47,48,49,50,51,52,53,54,
    55,56,57,58,59,60,61,62,63,64,65,66,68,69,70,71,72, 73,74,75,76,77,78,79,80,
    81, 82, 83,84,85,86,87],
    
    "TMA53": [0,1,2,3,4,5,6,8,9,11,12,13,14,15, 16,17,18,19,20,21,22,23,24,25,26,
    28, 29,30,31,32,34,36,37,39,40,41,42,43,44,45,46,47,49,50,51,52,53,54,55,56, 
    57,58, 59,60,61,62,63,64, 65,66, 67, 69,70,71,72,73,74,75,76,77,78, 79], 
    
    "TMA51": [1,2,3,4,5,8,9,10, 11,13,14,15,16,17,18,19,20,21,22, 23,24,25,26,27,
    28,29,30,31,32,33,34,35,36, 38,39,40,41,42,44, 45,46,48,49,52,53,54,55,56,57,
    58,59,60,61,62,63, 66, 67, 68, 69, 70, 71,74,75,76,77,78, 79,81, 82, 83, 84,
    85,86],
    
    "TMA58":[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15,17,19, 20, 21, 22, 23,24, 
    25, 27, 28,32, 34, 35, 36, 37, 38, 39,40, 41, 42, 44, 45, 47,48, 49, 50, 51, 
    52, 53, 54,56, 57,58,59,60,61,63,64,65,67,68,69,70,72,73,76,77,78,79,80,82,83,
    84,85,86,87, 88],
    
    "TMA59": [0,1,2,3,4,5,6,8,9,11,12,13,14,15,16,18,19,20,21,22,23,25,26,28,29,
    30,31,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,50,51,52,53,54,55,56,57,58,
    59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79], 
    
    "TMA54": [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,
    27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
    54,55,56,58,60,61,62,53, 64,65,66,67,68,69,70,71,72, 75, 76],
    
    "TMA63": [0,1,2,3,4,5,6,9,10,11,12,13,14,15,16,18,19,20,22,23,24,25,26,27,31,
    32,33,34,35,36,38,41,42,43,44,49,50,51,52,54,55],
    
    "TMA55":[0, 1,2,3, 5,6,8,9,10,11,12,13,14,16,17,18,19,20,21,24,25,26,28,29,
    32,33,34,35,36,37,38,40,41,45,48,49,50,53, 55,56,57,58,60, 61,62],
    
    "TMA60":[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
    26,27,28,30,32,34,35,36,37,38,39,41,43,44,45,46,47,48,49,51,52,53,54,56,57,
    59,60,61, 62,63,65,66,67,68,69,70], 
    
    "TMA44":[0,2,4,5,9,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,
    48,49,50,52,56,58,59,61,63,64],
    
    "TMA45":[0,1,2,3,4,9,11,12,13,14,15,16,18,19,20,21,23,26,27,30,31,32,33,35,
    37,41,46,47,48,49,50,51,53,54,55,56,57,62,63]
}

# List of the TMA file names containing tumor cores
img_names_tumor = ["TMA52", "TMA53", "TMA51", "TMA58", "TMA54", "TMA63", "TMA55", 
"TMA59", "TMA60"]

# List of the TMA file names containing healthy cores
img_names_healthy = ["TMA44", "TMA45"]

# Hyperparameter to fine-tune in order to obtain the best separation grid in the 
# automated approach to split a TMA in the calculation of performance metrics 
# step into separate patches, each one including only one core. 
windows = {"TMA52": 100, "TMA53": 100,"TMA51": 100,"TMA58":100 ,
           "TMA59": 60,"TMA54": 40,"TMA63": 100,"TMA55":100,
           "TMA60":100, "TMA44":100, "TMA45":75}

# List of thresholds on the number of pixels that must be detected as being 
# tumour in order for the core to be classified as tumour
thresh_area = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 
                1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200,
                3400, 3600, 3800, 4000, 4200, 4400, 4600]

# List of thresholds on the probability output from the neural network to classify
# a tile as tumour
thresh_probs = [0.4, 0.5, 0.6]

thresh_combis = list(product(thresh_area, thresh_probs))

# Hyperparameter to fine-tune in order to obtain the best separation grid in the 
# automated approach to split a TMA in the pre-processing step into separate patches 
# each one including only one core.  
new_windows_hsv = {"OLOF_TMA52_26_xylene_2018-08-23.ndpi": 300, 
                   "OLOF_TMA53_6_xylene_2018-08-23.ndpi":300,
                   "OLOF_TMA51_18_xylene_ 2018-08-23.ndpi": 300,
                   "OLOF_TMA55_17_xylene_ 2018-09-04.ndpi":300,
                   "OLOF_TMA59_28_xylene_2018-09-04.ndpi": 300,
                   "OLOF_TMA54_7_xylene_2018-08-24.ndpi": 300,
                   "OLOF_TMA63_7_xylene_2018-09-04.ndpi": 500,
                   "OLOF_TMA58_5_xylene_2018-09-04.ndpi":300, 
                   "OLOF_TMA60_5_xylene_2018-09-04.ndpi":400, 
                   "OLOF_TMA44_3_xylene_2018-08-23.ndpi":300, 
                   "OLOF_TMA45_3_xylene_2018-08-24.ndpi":300}

# Dictionary of the label of each TMA (1 for tumour and 0 for healthy)
targets_dict = {"OLOF_TMA52_26_xylene_2018-08-23": 1,
                "OLOF_TMA53_6_xylene_2018-08-23":1,
                "OLOF_TMA51_18_xylene_ 2018-08-23": 1,
                "OLOF_TMA55_17_xylene_ 2018-09-04":1,
                "OLOF_TMA59_28_xylene_2018-09-04": 1,
                "OLOF_TMA54_7_xylene_2018-08-24": 1,
                "OLOF_TMA63_7_xylene_2018-09-04": 1,
                "OLOF_TMA58_5_xylene_2018-09-04":1, 
                "OLOF_TMA60_5_xylene_2018-09-04":1, 
                "OLOF_TMA44_3_xylene_2018-08-23":0, 
                "OLOF_TMA45_3_xylene_2018-08-24":0}


windows = {"TMA52": 100, "TMA53": 100,"TMA51": 100,"TMA58":100 ,
           "TMA59": 60,"TMA54": 40,"TMA63": 100,"TMA55":100,
           "TMA60":100, "TMA44":100, "TMA45":75}

thresh_area = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 
                1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200,
                3400, 3600, 3800, 4000, 4200, 4400, 4600]

thresh_probs = [0.4, 0.5, 0.6]

thresh_combis = list(product(thresh_area, thresh_probs))
