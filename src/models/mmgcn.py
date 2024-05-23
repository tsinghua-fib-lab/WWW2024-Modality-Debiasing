# coding: utf-8
"""
MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. 
In ACM MM`19,
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class MMGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGCN, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']         # not used
        self.aggr_mode = 'mean'
        self.concate = 'False'
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0

        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode,
                             self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256, device=self.device)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256, device=self.device)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).to(self.device)
        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self):
        representation = None
        v_rep = None
        t_rep = None
        if self.v_feat is not None:
            v_rep = self.v_gcn(self.v_feat, self.id_embedding)
            representation = v_rep
        if self.t_feat is not None:
            if representation is None:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = t_rep
            else:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = representation + t_rep
                representation /= self.num_modal

        self.result = representation
        # print(representation.size())
        return representation, v_rep, t_rep
    
    def bpr_loss(self, input_users, positive_items, negative_items):
        out, _, _ = self.forward()
        users_emb = out[input_users]
        pos_emb = out[positive_items]
        neg_emb = out[negative_items]
        pos_score = torch.mul(users_emb, pos_emb)
        neg_score = torch.mul(users_emb, neg_emb)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)
    
    def min_max_scale(self,x):
        min_vals, _ = torch.min(x, dim=1, keepdim=True)
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        scaled_x = (x - min_vals) / (max_vals - min_vals)
        return scaled_x
    
    def calculate_loss(self, protect_type, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] 
        neg_items = interaction[2] 
        pos_items += self.n_users
        neg_items += self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out, ori_v_rep, ori_t_rep = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]

        if protect_type == 'vt':
            v_s =  torch.sum(ori_v_rep[user_tensor] * ori_v_rep[item_tensor], dim=1).view(-1, 2)
            v_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(v_s, self.weight))))

            t_s =  torch.sum(ori_t_rep[user_tensor] * ori_t_rep[item_tensor], dim=1).view(-1, 2)
            t_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(t_s, self.weight))))

        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))

        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        if self.t_feat is not None:
            reg_embedding_loss += (self.t_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        return loss + reg_loss, v_loss, t_loss

    def full_sort_predict(self, interaction,item_v_ranking,item_t_ranking,flag):
        representation = None
        if self.v_feat is not None:
            v_rep = self.v_gcn(self.v_feat, self.id_embedding)
            representation = v_rep
        if self.t_feat is not None:
            if representation is None:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = t_rep
            else:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = representation + t_rep
                representation /= self.num_modal

        self.result = representation

        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score = torch.matmul(temp_user_tensor, item_tensor.t())

        score_v = self.full_sort_predict_v(interaction)
        score_t = self.full_sort_predict_t(interaction)

        score = score * torch.sigmoid(score_v) * torch.sigmoid(score_t)

        #counterfactual-visual
        v_star_feat = torch.mean(self.v_feat,dim=0,keepdim=True).repeat(self.v_feat.size(0),1)

        v_rep = self.v_gcn(v_star_feat, self.id_embedding)
        representation = v_rep
        t_rep = self.t_gcn(self.t_feat, self.id_embedding)
        representation = representation + t_rep
        representation /= self.num_modal
        user_tensor = representation[:self.n_users]
        item_tensor = representation[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        score_star_v = torch.matmul(temp_user_tensor, item_tensor.t())

        score_star_v = score_star_v * torch.sigmoid(score_v) * torch.sigmoid(score_t)

        #counterfactual-textual
        t_star_feat = torch.mean(self.t_feat,dim=0,keepdim=True).repeat(self.t_feat.size(0),1)

        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        representation = v_rep
        t_rep = self.t_gcn(t_star_feat, self.id_embedding)
        representation = representation + t_rep
        representation /= self.num_modal
        user_tensor = representation[:self.n_users]
        item_tensor = representation[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        score_star_t = torch.matmul(temp_user_tensor, item_tensor.t())

        score_star_t = score_star_t * torch.sigmoid(score_v) * torch.sigmoid(score_t)

        if flag >-2:
            # _, idx_v = score_v.sort(dim=1,descending=True)
            # _,rank_v = idx_v.sort(dim=1)
            # rank_v = torch.clamp(rank_v,0,99)

            # _, idx_t = score_t.sort(dim=1,descending=True)
            # _,rank_t = idx_t.sort(dim=1)
            # rank_t = torch.clamp(rank_t,0,99)

            # v_fre_ranking_u = torch.tensor(item_v_ranking,dtype=float)
            # v_fre_ranking_u = v_fre_ranking_u.unsqueeze(0)
            # v_fre_ranking_u = v_fre_ranking_u.repeat(rank_v.size(0),1)

            # t_fre_ranking_u = torch.tensor(item_t_ranking,dtype=float)
            # t_fre_ranking_u = t_fre_ranking_u.unsqueeze(0)
            # t_fre_ranking_u = t_fre_ranking_u.repeat(rank_t.size(0),1)

            
            # s_ui_v = self.min_max_scale(torch.cosine_similarity(rank_v,v_fre_ranking_u.cuda(),dim=-1))
            # s_ui_t = self.min_max_scale(torch.cosine_similarity(rank_t,t_fre_ranking_u.cuda(),dim=-1))

            #best
            # s_ui_v = torch.cosine_similarity(rank_v,v_fre_ranking_u.cuda(),dim=-1)
            # s_ui_t = torch.cosine_similarity(rank_t,t_fre_ranking_u.cuda(),dim=-1)
            k=0.5
            # s_ui_v = torch.exp(-k*abs(rank_v.cuda()-v_fre_ranking_u.cuda()))
            # s_ui_t = torch.exp(-k*abs(rank_t.cuda()-t_fre_ranking_u.cuda()))

            # print('mean v_score:',torch.mean(s_ui_v.view(-1)))
            # print('mean t_score:',torch.mean(s_ui_t.view(-1)))

            # s_ui_v = torch.sigmoid(-abs(rank_v-v_fre_ranking_u.cuda())[:,:,0])
            # s_ui_t = torch.sigmoid(-abs(rank_t-t_fre_ranking_u.cuda())[:,:,0])

            #debias
            # print(s_ui_v)
            # score = score - s_ui_v*score_star_v + score - s_ui_t*score_star_t
            score = score - score_star_v + score - score_star_t

            return score
        else:
            return score


    def full_sort_predict_v(self, interaction):
        representation = None
        if self.v_feat is not None:
            v_rep = self.v_gcn(self.v_feat, self.id_embedding)
            representation = v_rep
        if self.t_feat is not None:
            if representation is None:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = t_rep
            else:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = representation + t_rep
                representation /= self.num_modal

        self.result = representation

        v_rep_u = v_rep[:self.n_users]
        v_rep_i = v_rep[self.n_users:]
        temp_user_tensor = v_rep_u[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, v_rep_i.t())
        return score_matrix

    def full_sort_predict_t(self, interaction):
        representation = None
        if self.v_feat is not None:
            v_rep = self.v_gcn(self.v_feat, self.id_embedding)
            representation = v_rep
        if self.t_feat is not None:
            if representation is None:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = t_rep
            else:
                t_rep = self.t_gcn(self.t_feat, self.id_embedding)
                representation = representation + t_rep
                representation /= self.num_modal

        self.result = representation

        t_rep_u = t_rep[:self.n_users]
        t_rep_i = t_rep[self.n_users:]
        temp_user_tensor = t_rep_u[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, t_rep_i.t())
        return score_matrix


class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, dim_latent=None, device='cpu'):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device
        

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent))))

            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            # self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat))))

            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

    def forward(self, features, id_embedding):
        # print(features)
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))  # equation 5
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))  # equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))  # equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + x_hat)

        return x


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)