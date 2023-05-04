import os
import pdb
import pickle
import numpy as np

import os
import pdb
import time
import torch
import random
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm

from torch.utils import data
from torch.utils.data.dataloader import default_collate

random.seed(2023)

class SEQ_Dataset(data.Dataset):
    
    # def get_seq_negatives(self, udict, maxlen, allitems):      
    #     seqs, poss, negs, intvs = [], [], [], []
        
    #     for u in udict:
    #         items = udict[u]['seq']
                        
    #         seq = [0] * (maxlen - len(items[:-1])) + items[:-1]
    #         pos = [0] * (maxlen - len(items)) + items            
            
    #         negitems = np.random.choice(list(allitems - set(items)), len(items))
            
    #         neg = [0] * (maxlen - len(negitems)) + list(negitems)
            
    #         intv = [0] * (maxlen - len(udict[u]['time'])) + udict[u]['time']
            
    #         seqs.append(seq)
    #         poss.append(pos)
    #         negs.append(neg)
    #         intvs.append(intv)
            
    #     return seqs, poss, negs, intvs
        
    def toseqdata(self, data, slotlen, timescale): # these data are positive ones
        data = data.astype(float).astype(int) # The data are already sorted by the time
        
        udict = {}
        for u, i, _, t in data: 
            if u not in udict: udict[u] = [] 
            udict[u].append(i)

        return udict
        
    def __init__(self, path, slotlen, timescale, numneg, numnext, trninfo=None):
        dpath = '/'.join(path.split('/')[:-1]) # example) path : ...../[trn|vld|tst]
        if dpath[-1] != '/': dpath += '/'
        dtype = path.split('/')[-1]
        
        self.dtype = dtype
        self.numneg = numneg
        
        st = time.time()
        
        # Load data designed for rating prediction
        if dtype == 'trn': # [u, i, r, t]
            self.uir = np.load(dpath+'trn')

            # Make the interaction data as sequential data            
            self.udict = self.toseqdata(self.uir, slotlen, timescale) # udict includes sqeuence

        elif dtype == 'vld' or dtype == 'tst': 
            self.uir = np.load(dpath+dtype) # [u, pi] + [ni] * N
            self.udict = trninfo
            
        # Refining the data for ranking
        if dtype == 'trn': 
            # input: user, seq (considering L+T), pos (considering T), neg (obtained in collate)
            user = self.uir[:,0] # User IDs
            self.allitems = set(self.uir[:,1].astype(int)) # Item IDs

            # self.numitem = len(self.allitems)
            
            L = slotlen
            T = numnext
            
            users = []
            input_seqs = []
            next_items = []
            # self.neg_cands = {}
            for u in self.udict: # [udict] key: user ID, value: the user's sequence
                
                seq = [0] * (L+T-1) + self.udict[u] # Left padding
                
                # Data instances to predict next bucket (extension of next item prediction)
                for i in range(len(seq) - (L+T-1)):
                    subseq = seq[i: i+(L+T)]
                    
                    input_seq = subseq[:L]
                    next_item = subseq[L:]
                    
                    users.append(u)
                    input_seqs.append(input_seq)
                    next_items.append(next_item)
                    
                # self.neg_cands[u] = np.array(list(self.allitems - set(self.udict[u]))) # NOTE: too slow
            
            self.first = np.array(users)
            self.second = np.array(input_seqs)
            self.third = np.array(next_items)
            self.fourth = np.zeros(len(self.first)) # It will be the negative samples
            
            self.numuser = len(set(self.uir[:,0].astype(int)))
            self.numitem = len(set(self.uir[:,1].astype(int)))

        elif dtype == 'vld' or dtype == 'tst': 
            self.evalinfo = np.load(dpath+dtype+'_info')[:, [0,1]].astype(int) # only u-i pairs
            
            L = slotlen
            T = numnext
            
            if dtype == 'tst': # Augment the user's history in validation data when evaluating on test data
                self.vldinfo = np.load(dpath+'vld_info')
                self.vldudict = self.toseqdata(self.vldinfo, slotlen, timescale)
            
            all_seqfeature = []
            for i, row in enumerate(self.uir): # Each row: [user ID, item ID, binary label]
                u = row[0]
                
                if u in self.udict:            
                    seq = self.udict[u]
                    
                    if dtype == 'tst' and u in self.vldudict: seq += self.vldudict[u]
                                                
                    seqfeature = seq[-L:]
                    seqfeature = [0] * (L - len(seqfeature)) + seq[-L:]
                else: # New user case
                    seqfeature = np.zeros(L)
                all_seqfeature.append(seqfeature)
                     
                    
            self.first = self.uir[:,0] # user ID
            self.second = np.array(all_seqfeature)
            self.third = self.uir[:,1:] # a positive item and other negative items
            self.fourth = np.zeros(len(self.first)) # dummy (not used in the model)

        print('Data building time : %.1fs' % (time.time()-st))

    def __getitem__(self, index):           
        return self.first[index], self.second[index], self.third[index], self.fourth[index]
    
    def __len__(self):
        """Returns the total number of user-item pairs."""
        if self.dtype == 'trn':
            return len(self.first)
        else: # vld and tst
            return len(self.uir)
    
    def train_collate(self, batch):
        batch = [i for i in filter(lambda x:x is not None, batch)]
        # Generate negative samples and replace the last one in the batch
        
        newbatch = []
        for i in range(len(batch)):
            u, seq, pos, _= batch[i]
            
            neg_items = np.random.randint(self.numitem, size=self.numneg)            

            newbatch.append([u, seq, pos, neg_items])
    
        return default_collate(newbatch)

def test_collate(batch):    
    batch = [i for i in filter(lambda x:x is not None, batch)]
    return default_collate(batch)    
    
def my_collate(batch):
    try:
        return default_collate(np.array(batch).swapaxes(0,1)) 
    except:
        seqs, times, items, _ = [torch.LongTensor(i) for i in zip(*batch)]
        
        return seqs, times, items, _

def get_each_loader(data_path, batch_size, slotlen, timescale, numneg=None, numnext=None, trninfo=None, shuffle=True, num_workers=0):
    """Builds and returns Dataloader."""

    dataset = SEQ_Dataset(data_path, slotlen, timescale, numneg, numnext, trninfo)
    
    if data_path.endswith('trn') == True:
        collate = dataset.train_collate
    else:
        collate = test_collate

    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate)

    return data_loader

class DataLoader:
    def __init__(self, opt):
        # Data loader specific arguments
        self.dpath = opt.dataset_path + '/'
        self.batch_size = opt.batch_size
        self.slotlen = opt.maxhist
        self.timescale = 1 # opt.timescale
        self.trn_numneg = opt.numneg
        self.numnext = opt.num_next
        
        self.trn_loader, self.vld_loader, self.tst_loader = self.get_loaders_for_factorization(self.trn_numneg)
            
        print(("train/val/test/ divided by batch size {:d}/{:d}/{:d}".format(len(self.trn_loader), len(self.vld_loader),len(self.tst_loader))))
        print("==================================================================================")
        
    def get_loaders_for_factorization(self, numneg):
        print("Loading data...")
        trn_loader = get_each_loader(self.dpath+'trn', self.batch_size, self.slotlen, self.timescale, numneg=numneg, numnext=self.numnext)
        print('\tTraining data loaded')
        
        trninfo = trn_loader.dataset.udict
        
        vld_loader = get_each_loader(self.dpath+'vld', self.batch_size, self.slotlen, self.timescale, trninfo=trninfo, shuffle=False)
        print('\tValidation data loaded')
        
        tst_loader = get_each_loader(self.dpath+'tst', self.batch_size, self.slotlen, self.timescale, trninfo=trninfo, shuffle=False)
        print('\tTest data loaded')
        
        return trn_loader, vld_loader, tst_loader
    
    def get_loaders(self):
        return self.trn_loader, self.vld_loader, self.tst_loader
    
    def get_embedding(self):
        return self.input_embedding
        
            