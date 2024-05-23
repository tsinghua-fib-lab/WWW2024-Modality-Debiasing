# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F
from torch.nn.functional import normalize

class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        item_embeddings = self.item_linear(torch.cat((self.t_feat, self.v_feat), -1))
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)

        v_embedding = self.item_linear(torch.cat((torch.zeros(self.t_feat.size()).cuda(), self.v_feat), -1))
        t_embedding = self.item_linear(torch.cat((self.t_feat, torch.zeros(self.v_feat.size()).cuda()), -1))

        self.v_embedding = v_embedding
        self.t_embedding = t_embedding

        return user_e, item_e, v_embedding, t_embedding
    
    # def bpr_loss(self, input_users, positive_items, negative_items):
    #     user_embeddings, item_embeddings, v_embedding, t_embedding = self.forward()
    #     users_emb = user_embeddings[input_users]
    #     pos_emb = item_embeddings[positive_items]
    #     neg_emb = item_embeddings[negative_items]
    #     pos_score = torch.mul(users_emb, pos_emb)
    #     neg_score = torch.mul(users_emb, neg_emb)
    #     loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    #     return torch.mean(loss)
    
    def min_max_scale(self,x):
        min_vals, _ = torch.min(x, dim=1, keepdim=True)
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        scaled_x = (x - min_vals) / (max_vals - min_vals)
        return scaled_x

    def calculate_loss(self, protect_type, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings, v_embedding, t_embedding = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        #neg_e = self.get_item_embedding(neg_item)
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss

        # if protect_type == 'v' and self.v_feat is not None and self.t_feat is not None and self.fusion_mode=='concat':
        if protect_type == 'vt' and self.v_feat is not None and self.t_feat is not None:
            user_v = user_embeddings[user, self.u_embedding_size:]
            pos_v = v_embedding[pos_item, :]
            neg_v = v_embedding[neg_item, :]
            pos_item_score_v, neg_item_score_v = torch.mul(user_v, pos_v).sum(dim=1), torch.mul(user_v, neg_v).sum(dim=1)
            v_loss = self.loss(pos_item_score_v, neg_item_score_v)

            user_t = user_embeddings[user, self.u_embedding_size:]
            pos_t = t_embedding[pos_item, :]
            neg_t = t_embedding[neg_item, :]
            pos_item_score_t, neg_item_score_t = torch.mul(user_t, pos_t).sum(dim=1), torch.mul(user_t, neg_t).sum(dim=1)
            t_loss = self.loss(pos_item_score_t, neg_item_score_t)

        return loss, v_loss, t_loss


    def full_sort_predict(self, interaction,item_v_ranking,item_t_ranking,flag):
        user = interaction[0]
        item_embeddings = self.item_linear(torch.cat((self.t_feat, self.v_feat), -1))
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_embeddings = F.dropout(self.u_embedding, 0.0)
        item_embeddings = F.dropout(item_embeddings, 0.0)
        
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))

        score_v = self.full_sort_predict_v(interaction)
        score_t = self.full_sort_predict_t(interaction)

        score = score * torch.sigmoid(score_v) * torch.sigmoid(score_t)

        #counterfactual-visual
        v_star_feat = torch.mean(self.v_feat,dim=0,keepdim=True).repeat(self.v_feat.size(0),1)
        item_embeddings_star_v = self.item_linear(torch.cat((self.t_feat, v_star_feat), -1))
        item_embeddings_star_v = torch.cat((self.i_embedding, item_embeddings_star_v), -1)
        all_item_e_star_v = item_embeddings_star_v
        score_star_v = torch.matmul(user_e, all_item_e_star_v.transpose(0, 1))
        score_star_v = score_star_v * torch.sigmoid(score_v) * torch.sigmoid(score_t)

        #counterfactual-textual
        t_star_feat = torch.mean(self.t_feat,dim=0,keepdim=True).repeat(self.t_feat.size(0),1)
        item_embeddings_star_t = self.item_linear(torch.cat((t_star_feat, self.v_feat), -1))
        item_embeddings_star_t = torch.cat((self.i_embedding, item_embeddings_star_t), -1)
        all_item_e_star_t = item_embeddings_star_t
        score_star_t = torch.matmul(user_e, all_item_e_star_t.transpose(0, 1))
        score_star_t = score_star_t * torch.sigmoid(score_v) * torch.sigmoid(score_t)

        if flag >-2:
            _, idx_v = score_v.sort(dim=1,descending=True)
            _,rank_v = idx_v.sort(dim=1)
            rank_v = torch.clamp(rank_v,0,99)

            _, idx_t = score_t.sort(dim=1,descending=True)
            _,rank_t = idx_t.sort(dim=1)
            rank_t = torch.clamp(rank_t,0,99)

            v_fre_ranking_u = torch.tensor(item_v_ranking,dtype=float)
            v_fre_ranking_u = v_fre_ranking_u.unsqueeze(0)
            v_fre_ranking_u = v_fre_ranking_u.repeat(rank_v.size(0),1)

            t_fre_ranking_u = torch.tensor(item_t_ranking,dtype=float)
            t_fre_ranking_u = t_fre_ranking_u.unsqueeze(0)
            t_fre_ranking_u = t_fre_ranking_u.repeat(rank_t.size(0),1)
            
            # s_ui_v = self.min_max_scale(torch.cosine_similarity(rank_v,v_fre_ranking_u.cuda(),dim=-1))
            # s_ui_t = self.min_max_scale(torch.cosine_similarity(rank_t,t_fre_ranking_u.cuda(),dim=-1))

            #best
            # s_ui_v = torch.cosine_similarity(rank_v,v_fre_ranking_u.cuda(),dim=-1)
            # s_ui_t = torch.cosine_similarity(rank_t,t_fre_ranking_u.cuda(),dim=-1)

            # s_ui_v = (1-torch.sigmoid(rank_v[:,:,0]).cuda())+(1-torch.sigmoid(v_fre_ranking_u[:,:,0]).cuda())
            # s_ui_t = (1-torch.sigmoid(rank_t[:,:,0]).cuda())+(1-torch.sigmoid(t_fre_ranking_u[:,:,0]).cuda())

            k=0.0001
            # s_ui_v = 0.5*(torch.exp(-k*rank_v.cuda())+torch.exp(-k*v_fre_ranking_u.cuda()))
            # s_ui_t = 0.5*(torch.exp(-k*rank_t.cuda())+torch.exp(-k*t_fre_ranking_u.cuda()))
            # s_ui_v = torch.exp(-k*v_fre_ranking_u[:,:,0].cuda())
            # s_ui_t = torch.exp(-k*t_fre_ranking_u[:,:,0].cuda())
            s_ui_v = torch.exp(-k*(rank_v.cuda()-v_fre_ranking_u.cuda()))
            s_ui_t = torch.exp(-k*(rank_t.cuda()-t_fre_ranking_u.cuda()))


            # s_ui_v = torch.exp(-k*rank_v[:,:,0]).cuda()+torch.exp(-k*v_fre_ranking_u[:,:,0]).cuda()
            # s_ui_t = torch.exp(-k*rank_t[:,:,0]).cuda()+torch.exp(-k*t_fre_ranking_u[:,:,0]).cuda()

            # s_ui_v = torch.exp(1/(rank_v[:,:,0].cuda()+1))*torch.exp(1/(v_fre_ranking_u[:,:,0].cuda()+1))
            # s_ui_t = torch.exp(1/(rank_t[:,:,0].cuda()+1))*torch.exp(1/(t_fre_ranking_u[:,:,0].cuda()+1))

            print('mean v_score:',torch.mean(s_ui_v.view(-1)))
            print('mean t_score:',torch.mean(s_ui_t.view(-1)))

            #debias
            score = score - s_ui_v*score_star_v + score - s_ui_t*score_star_t
            # score = score - 0.9*score_star_v + score - 0.0*score_star_t
            return score
        else:
            return score
            # return (score - 1*score_star_v) + (score - 0*score_star_t)  #内部乘系数好一些，视觉模态去偏重要一些

    def full_sort_predict_v(self, interaction):
        user = interaction[0]
        item_embeddings = self.item_linear(torch.cat((torch.zeros(self.t_feat.size()).cuda(), self.v_feat), -1))
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)
        user_embeddings = F.dropout(self.u_embedding, 0.0)
        item_embeddings = F.dropout(item_embeddings, 0.0)
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def full_sort_predict_t(self, interaction):
        user = interaction[0]
        item_embeddings = self.item_linear(torch.cat((self.t_feat, torch.zeros(self.v_feat.size()).cuda()), -1))
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)
        user_embeddings = F.dropout(self.u_embedding, 0.0)
        item_embeddings = F.dropout(item_embeddings, 0.0)
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score
