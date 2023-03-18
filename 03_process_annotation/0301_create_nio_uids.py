from itertools import groupby
import os
from collections import defaultdict


nio_raw_path = "/Users/yidesdo21/Projects/outputs/12_time_slicing/nio_results/"
ptc_raw_path = "/Users/yidesdo21/Projects/outputs/12_time_slicing/ptc_results/"

## copied from create_time_sliced_dataset.ipynb
## subject to change, this .py file is created to help to do the link prediction work
# extract the uids from each period of time


def create_nio_uids(year,path=nio_raw_path):
    """extract the uids in each period of time for the NIO annotator outputs,
    return a dictionary {citation_id:(set_of_uids)}"""
    
    with open(path+year+".txt") as f:
        output = f.read().split(']], [NLM')
        
    cit_dict = dict()
    
#     indices, titles, uids = list(), list(), list()
#     rubbish = 0
    
    for o in output:
        cit = o.split("], [")   # each citation has 4 parts
        
        c_title = "NLM"+"_"+cit[0].split("_")[1]
        c_uids = cit[1].strip("[]").split(", ")
    #     c_start, c_end = ast.literal_eval(cit[2]), ast.literal_eval(cit[3]+"]")  # starting indices and ending indices, need fix

        cit_dict[c_title] = sorted(list(set(c_uids)))     
    
    return cit_dict

