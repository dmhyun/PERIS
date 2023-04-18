import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import Counter

fn = sys.argv[1]
# fn = 'reviews_yelp21_academic_dataset.json'

data = data = [json.loads(l) for l in open(fn)]

users = []
items = []
for row in tqdm(data):
    users.append(row['user_id'])
    items.append(row['business_id'])


# It will be used reduce item interaction density
idict = {}
for d in tqdm(data):
    iid = d['business_id']
    if iid not in idict: idict[iid] = []
    idict[iid].append(d)      

keep_ratio = 0.20 # To make Yelp data similar to other raw data

drop_data = {}
for iid in tqdm(idict):
    data4item = idict[iid]
    num_keep = int(len(data4item) * keep_ratio)
    
    random.shuffle(data4item)
    
    keepdata = data4item[:num_keep]
    
    drop_data[iid] = keepdata       

flatten_data = []
for iid in tqdm(drop_data):
    flatten_data += drop_data[iid]
print(len(flatten_data))

# To make the data into Amazon data format
uirt = []
for row in tqdm(flatten_data):
    uid = row['user_id']
    iid = row['business_id']
    time = row['date']
    time = int(datetime.strptime(time.split()[0], "%Y-%m-%d").strftime('%s'))
    star = 5 # dummy (not used later)
    
    uirt.append([uid, iid, star, time])
    
uirt = np.array(uirt)

np.save('uirt_yelp.npy', uirt)