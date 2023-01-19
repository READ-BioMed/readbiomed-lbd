from itertools import groupby
import os
from collections import defaultdict


nio_raw_path = "/Users/yidesdo21/Projects/outputs/12_time_slicing/nio_results/"
ptc_raw_path = "/Users/yidesdo21/Projects/outputs/12_time_slicing/ptc_results/"

## copied from create_time_sliced_dataset.ipynb
## subject to change, this .py file is created to help to do the link prediction work
# extract the uids from each period of time


def create_ptc_uids(year,path=ptc_raw_path):
    """extract the uids in each period of time for the PTC annotator outputs
    another output is a dictionary with the uid as the key, and category of the uid as the value """
    file_name = year   # this line need to change when dealing with different dataset
    
    # sanity checks for indexing PTC UIDs and categories
    categories = ['CellLine','Chemical','Chromosome','DNAAcidChange','DNAMutation','Disease','Gene','ProteinAcidChange','ProteinMutation','RefSeq','SNP','Species']
    
    ptc_contents = list()
    pubtator_files = os.listdir(path)

    for pubtator_file in pubtator_files:
        if pubtator_file.startswith(file_name) and pubtator_file.endswith(".PubTator"):
            with open(ptc_raw_path+pubtator_file) as f:
                pmid_results = f.read().replace("\t", " ").split("\n")

                # group each article and corresponding annotations by using the split ''
                pmid_groups = (list(g) for _, g in groupby(pmid_results, key=''.__ne__))
                pmid_content = [a + b for a, b in zip(pmid_groups, pmid_groups)]

                ptc_contents.extend(pmid_content)


    ptc_id_type = defaultdict(set)
       
    for p in ptc_contents:
        # annotation output format: "article id" "starting index" "ending index" "annotated text" "category" "uid"
        if len(p) > 3:   # exclude the articles when the length of the list is three, because they have no annotations
            annos = p[2:-1]
            for anno in annos:
                anno_split = anno.split(" ")

            # sometimes annotation is blank [""], it is not valid annotations,
            #      the blank annotation is ignored.
                if len(anno_split) > 1:
                # if the second to last index is not in the category, 
                #      then it means the UID might not be in the normal format.                
                    if anno_split[-2] not in categories:
                        # change the category and uid index
                        category, uid = anno_split[-3], anno_split[-2]+" "+anno_split[-1]
                    else: 
                        category, uid = anno_split[-2], anno_split[-1]

                    if uid not in ["-", ""]:
                        if not (category in ['Gene'] and len(uid.split(";")) > 1):
                            ptc_id_type[uid].add(category)

    filtered_dict = dict()
    
    # some UIDs happend to be the same from different categories
    # they have to be written in a more granular UIDs
    for k,v in ptc_id_type.items():
        if len(v) > 1:  
            for t in list(v):
                filtered_dict[t+"_"+k] = t
                
        else:
            filtered_dict[k] = list(v)[0]
    
    uids = list(filtered_dict.keys())
    

    return uids,filtered_dict

