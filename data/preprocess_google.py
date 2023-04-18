import re
import sys
import csv
import json
import numpy as np 
from tqdm import tqdm

def extract_numbers(dirty_text): 
    try:
        return int(re.findall('[0-9]+', dirty_text)[0])
    except:
        return None

fn = sys.argv[1]
# fn = 'reviews.clean.json'

data = open(fn).readlines()

uirt = []
rating = 5 # dummy variable to make it same as the format of Amazon data
for d in tqdm(data):
    uid, iid, time = None, None, None
    for di in d.split(','):
        if 'gPlusUserId' in di: uid = extract_numbers(di)
        if 'gPlusPlaceId' in di: iid = extract_numbers(di)
        if 'unixReviewTime' in di: time = extract_numbers(di)
            
    if (uid != None) and (iid != None) and (time != None):            
        uirt.append([uid, iid, rating, time])

np.save('uirt_google.npy', uirt)