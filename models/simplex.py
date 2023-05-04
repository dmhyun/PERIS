import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SIMPLEX(nn.Module):
    def __init__(self, opt):
        super(SIMPLEX, self).__init__()

        self.lamb = opt.lamb
        self.margin = opt.margin
        
        self.ebd_size = opt.K
        self.maxlen = opt.maxhist        
                
        self.ebd_item = nn.Embedding(opt.numitem+1, self.ebd_size, padding_idx=0)
        nn.init.xavier_normal_(self.ebd_item.weight)
    
    def compute_loss(self, batch_data):    
        _, seq, pos, neg = batch_data
        
        pos_score = self.forward([seq, pos])
        neg_score = self.forward([seq, neg])           

        pos_term = torch.relu(1- pos_score.mean(dim=-1))
        neg_term = torch.relu(neg_score - self.margin).mean(dim=-1)                

        loss = pos_term + self.lamb * neg_term       

        return loss.mean()    
    
    def predict_for_multiple_item(self, batch_data):    
        _, seq, items, _ = batch_data
        
        dist = self.forward([seq, items]) * -1
                
        return dist

    def forward(self, batch_data):
        item_seq, items_to_predict = [i.cuda() for i in batch_data]   

        item_embs = self.ebd_item(item_seq)
        items_to_predict_emb = self.ebd_item(items_to_predict)

        # Averaging item history 
        mask = (item_seq != 0)
        denom = mask.sum(-1)
        denom[denom==0] = 1 # to prevent zero-division
        avg_history = (item_embs * mask[:,:,None]).sum(1) / denom[:,None]

        # Cosine similarity
        product = (avg_history[:,None,:] * items_to_predict_emb).sum(-1)
        norm_a = avg_history.norm(dim=-1)[:,None] + 1e-9
        norm_b = items_to_predict_emb.norm(dim=-1)
        cos = product / (norm_a * norm_b)

        return cos