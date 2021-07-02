import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math

#open the file where the Enums are stored.
with open('counter.json') as json_file1:
    data1 = json.load(json_file1)

#print(max(data1.values())) #outputs "the" : 12552
#total_count = len(data1) #outputs number of pairs : 12458

#divide the counts into incremental ranges of 5 or whatever is in interval
#add a list of dictionaries containing all the words&counts separated by their frequncy range
ranges_dict = {}
interval = 5

for key, val in data1.items():
    interval_place = math.ceil(val / interval)
    if interval_place in ranges_dict:
        ranges_dict[interval_place].append({key : val})
    else:
        ranges_dict[interval_place] = [{key : val}]

#make a list of all the range container sizes
size_of_ranges = []
for key, val in ranges_dict.items():
    size_of_ranges.append(len(ranges_dict[key]))

#Setting up the plotting
#x: ranges of counts , y: how many words fall into the ranges
width = 1
barlist = plt.bar(ranges_dict.keys(), size_of_ranges, width, color='g', align="center")
plt.title('Pre Truncate')
plt.ylabel('Frequnecy ')
plt.xlabel('Ranges of ' + str(interval))

#plt.show()

"""
#set every other bar to red for visual assessment
for i in range(len(barlist)):
    if (i % 2):
        barlist[i].set_color('r')
#outputs all of the original input data counts on one graph
#plt.plot(data1.keys(),data1.values(), width, color='r')
"""


#remove the word ranges
acceptable_range = 10

vocabulary_dict_ranges = {}
for key, val in ranges_dict.copy().items():
    if key >= acceptable_range:
        vocabulary_dict_ranges[key] = val


#make a list of all the range container sizes
size_of_ranges = []
for key, val in vocabulary_dict_ranges.items():
    size_of_ranges.append(len(vocabulary_dict_ranges[key]))

#Setting up the plotting
#x: ranges of counts , y: how many words fall into the ranges
width = 1
barlist = plt.bar(vocabulary_dict_ranges.keys(), size_of_ranges, width, color='r', align="center")
plt.title('Post Truncate'+ str(acceptable_range))
plt.ylabel('Frequnecy ')
plt.xlabel('Ranges of ' + str(interval))

#plt.show()

#save vocab sorted by ranges
with open("vocabulary_dict_ranges.json", "w") as f:
        json.dump(vocabulary_dict_ranges, f)


"""
# save just a list of the words in the vocabulary alone
vocabulary_list = []
for key, val in vocabulary_dict_ranges.items():
    for i in range(len(vocabulary_dict_ranges[key])):
        for sub_key, sub_val in vocabulary_dict_ranges[key][i].items():
            vocabulary_list.append(sub_key)

#print(len(vocabulary_list))

with open("vocabulary_list.json", "w") as f:
        json.dump(vocabulary_list, f)
"""

# save just a list of the words in the vocabulary alone
vocabulary_dict = {}
for key, val in vocabulary_dict_ranges.items():
    for i in range(len(vocabulary_dict_ranges[key])):
        for sub_key, sub_val in vocabulary_dict_ranges[key][i].items():
            vocabulary_dict[sub_key] = sub_val

#print(len(vocabulary_list))

with open("vocabulary_dic.json", "w") as f:
        json.dump(vocabulary_dict, f)
