import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda()


class LSAN(nn.Module):
    def __init__(self, opt):
        super(LSAN, self).__init__()

        L = opt.maxhist
        dims = opt.K
        self.ebd_size = opt.K
        self.kernel_size = opt.kernel_size
        self.maxlen = opt.maxhist
        self.numhead = 1
        
        self.lamb = opt.lamb

        # user and item embeddings
        self.ebd_user = nn.Embedding(opt.numuser+1, dims)
        self.ebd_item = nn.Embedding(opt.numitem+1, dims, padding_idx=0)

        self.feature_gate_item = nn.Linear(dims, dims)
        self.feature_gate_user = nn.Linear(dims, dims)

        self.instance_gate_item = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).cuda()
        self.instance_gate_user = Variable(torch.zeros(dims, L).type(torch.FloatTensor), requires_grad=True).cuda()
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

        self.W2 = nn.Embedding(opt.numitem+1, dims, padding_idx=0)
        self.b2 = nn.Embedding(opt.numitem+1, 1, padding_idx=0)
        
        nn.init.xavier_normal_(self.ebd_user.weight)
        nn.init.xavier_normal_(self.ebd_item.weight)
        nn.init.xavier_normal_(self.W2.weight)
        
        self.b2.weight.data.zero_()
        
        self.mask = _generate_square_subsequent_mask(self.maxlen)  
        
        self.dcnn = nn.Conv1d(self.ebd_size, self.ebd_size, kernel_size=self.kernel_size,  padding='same', groups=self.ebd_size)
                
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.ebd_size, nhead=1,
                                                   dropout=0.01, 
                                                   dim_feedforward=self.ebd_size)        
        self.sa = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.prediction_layer = nn.Sequential(
                                    nn.Linear(self.ebd_size*self.numhead*2, self.ebd_size),            
                                )            
        
    def compute_loss(self, batch_data):      
        user, seq, pos, neg = batch_data

        pos_score = self.forward([user, seq, pos])
        neg_score = self.forward([user, seq, neg])        

        posterm = - (pos_score.sigmoid()+1e-24).log()
        # Masking padding 
        posterm = (pos != 0) * posterm
        posterm = posterm.mean()

        negterm = - (1-neg_score.sigmoid()+1e-24).log().mean()

        loss = posterm + negterm
        
        return loss
    
    def predict_for_multiple_item(self, batch_data):    
        user, seq, item, _ = batch_data
        dist = self.forward([user, seq, item]) * -1
                
        return dist

    def forward(self, batch_data):
        user_ids, item_seq, items_to_predict = [i.cuda() for i in batch_data]
        
        item_embs = self.ebd_item(item_seq)
        # user_emb = self.ebd_user(user_ids)            
        
        interm_output = self.dcnn(item_embs.permute([0,2,1]))
        cnn_features = interm_output.permute([0,2,1])
        
        interm_sa_output = self.sa(item_embs.permute([1,0,2]), mask=self.mask)       
        sa_features = interm_sa_output.permute([1,0,2])     
                
        valid_mask = (item_seq!=0)
        
        sum_features = (1-self.lamb) * cnn_features + self.lamb * sa_features
        denorm = valid_mask.sum(-1)[:, None]
        denorm[denorm==0] = 1
        union_out = sum_features.sum(1) / denorm
        
        w2 = self.W2(items_to_predict) # item embedding vector
        b2 = self.b2(items_to_predict) # item bias        
        
        # union-level
        res = torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).reshape(w2.shape[:2])      
        
        return res 
        

