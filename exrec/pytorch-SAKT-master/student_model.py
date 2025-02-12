import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import DefaultConfig
from utils import subsequent_mask
from torch.autograd import Variable
from multihead_attn import MultiHeadedAttention
from torch.nn import LayerNorm

opt = DefaultConfig()

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, state_size, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, state_size)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, state_size, 2) *
                             -(math.log(10000.0) / state_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class student_model(nn.Module):

    def __init__(self, num_skills, state_size, num_heads=2, dropout=0.2, infer=False):
        super(student_model, self).__init__()
        self.infer = infer
        self.num_skills = num_skills
        self.state_size = state_size
        # we use the (num_skills * 2 + 1) as key padding_index
        '''
        Embedding- drugi argument je maksimalna duljina tensora, prvi argument je broj tensora
            ako je num_embeddings velicina dictionariya onda bi mozda trebao biti velicine num_skills?
            
            (10,3) 10- broj razlicitih elemenata, 3-u koliko se dimenzija embeddaju elementi
            
            posto je 10 broj elemenata, najveci element moze biti 9, znaci num moze biti len(ex_id_converter ako indeksi pocinju od 0)
            state size je po defaultu 200, s tim bi se moglo igrati
        '''
        self.embedding = nn.Embedding(num_embeddings=num_skills*2, #promijenjen s a *2+1 na +2, ovaj embedding sadrzi pitanja i appendane odgovore duljina je 2n
                                      embedding_dim=state_size)       #je li ispravno stavljati indekse pitanja i odgovore tocno netocno u isti embedding?
                                      # padding_idx=num_skills*2
        # self.position_embedding = PositionalEncoding(state_size)
        self.position_embedding = nn.Embedding(num_embeddings=opt.max_len, #max len je najveci broj exercisea s kojim moze raditi, maknut je -1
                                               embedding_dim=state_size)
        # we use the (num_skills + 1) as query padding_index
        self.problem_embedding = nn.Embedding(num_embeddings=num_skills, #maknut +1
                                      embedding_dim=state_size)
                                      # padding_idx=num_skills)
        self.multi_attn = MultiHeadedAttention(h=num_heads, d_model=state_size, dropout=dropout, infer=self.infer)
        self.feedforward1 = nn.Linear(in_features=state_size, out_features=state_size)
        self.feedforward2 = nn.Linear(in_features=state_size, out_features=state_size)
        self.pred_layer = nn.Linear(in_features=state_size, out_features=num_skills)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(state_size)

    def forward(self, x, problems, target_index):
        # self.key_masks = torch.unsqueeze( (x!=self.num_skills*2).int(), -1)
        # self.problem_masks = torch.unsqueeze( (problems!=self.num_skills).int(), -1)
        x = self.embedding(x)
        pe = self.position_embedding(torch.arange(x.size(1)).unsqueeze(0).cuda()) #Sto tocno radi position embedding
        x += pe  #interactional embedding + positional embedding
        # x = self.position_embedding(x)
        '''
        A simple lookup table that stores embeddings of a fixed dictionary and size.
        This module is often used to store word embeddings and retrieve them using indices. 
        The input to the module is a list of indices, and the output is the corresponding word embeddings.
        '''
        problems = self.problem_embedding(problems)
        # self.key_masks = self.key_masks.type_as(x)
        # self.problem_masks = self.problem_masks.type_as(problems)
        # x *= self.key_masks
        # problems *= self.problem_masks
        x = self.dropout(x)
        #Ovo bi trebala biti attention matrica povezanosti zadataka, poziva se funkcija attention
        "Compute 'Scaled Dot Product Attention'"
        #Zasto su key i value ista varijabla, query je normiran problems
        res = self.multi_attn(query=self.layernorm(problems), key=x, value=x,
                              key_masks=None, query_masks=None, future_masks=None)
        outputs = F.relu(self.feedforward1(res))
        outputs = self.dropout(outputs)
        outputs = self.dropout(self.feedforward2(outputs))
        # Residual connection
        outputs += self.layernorm(res)
        outputs = self.layernorm(outputs)
        logits = self.pred_layer(outputs)
        logits = logits.contiguous().view(logits.size(0) * opt.max_len, -1)
        logits = logits.contiguous().view(-1)
        selected_logits = torch.gather(logits, 0, torch.LongTensor(target_index).cuda())

        print ("Selected logits je")
        print(selected_logits.type())
        print(selected_logits)
        return selected_logits
        #treba skuziti kakvu tocno informaciju output daje