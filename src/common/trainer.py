# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""
import gc
import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from time import time
from logging import getLogger
from copy import deepcopy
from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator
import numpy as np
from common.distillation import Linf_distillation, get_adv_scores, Adv_distillation 
import seaborn as sns
from collections import Counter
class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
        
    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """

        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            #####normal loss#####
            self.optimizer.zero_grad()
            losses, v_loss, t_loss = loss_func('vt',interaction)
            # print(strong_pos.size())

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                # print('aa')
                loss = losses + v_loss + t_loss
                # print(losses)
                # print(v_loss)
                # print(t_loss)
                total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_batches.append(loss.detach())
 

        return total_loss, loss_batches

    def _valid_epoch(self, valid_data,is_test=False,flag=0):

        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """

        valid_result = self.evaluate(valid_data,is_test,flag) 
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                if epoch_idx == self.epochs-1:
                    f = 1
                else:
                    f = 0
                _, test_result = self._valid_epoch(test_data,is_test=True,flag=f)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []

        # vis_class_info = np.loadtxt('/nfs/shangy/mmrec_fair/src/visual_feat_label_baby.txt')
        # text_class_info = np.loadtxt('/nfs/shangy/mmrec_fair/src/texual_feat_label_baby.txt')
        # inter = pd.read_csv('/nfs/shangy/mmrec_fair/data/baby/baby.inter',header=0,sep='\t')

        vis_class_info = np.loadtxt('./visual_feat_label_clothing.txt')
        text_class_info = np.loadtxt('./texual_feat_label_clothing.txt')
        inter = pd.read_csv('../data/clothing/clothing.inter',header=0,sep='\t')

        inter_item = list(inter['itemID'])

        v_inter = [vis_class_info[i] for i in inter_item]
        v_inter = Counter(v_inter)
        label_v = sorted(v_inter.items(), key=lambda x: x[1], reverse=True)
        inter_rank_v = [x[0] for x in label_v]
        item_v_inter_ranking = [inter_rank_v.index(i) for i in vis_class_info]

        t_inter = [text_class_info[i] for i in inter_item]
        t_inter = Counter(t_inter)
        label_t = sorted(t_inter.items(), key=lambda x: x[1], reverse=True)
        inter_rank_t = [x[0] for x in label_t]
        item_t_inter_ranking = [inter_rank_t.index(i) for i in text_class_info]
        
        with torch.no_grad():
            for batch_idx, batched_data in enumerate(eval_data):
                # predict: interaction without item ids
                scores = self.model.full_sort_predict(batched_data,item_v_inter_ranking,item_t_inter_ranking,idx)
                masked_items = batched_data[1]
                # mask out pos items
                scores[masked_items[0], masked_items[1]] = -1e10
                # rank and get top-k
                _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
                # print(topk_index.size())
                batch_matrix_list.append(topk_index)

        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
    
    def clean(self,test_data):
        # print(self.model.t_feat)
        _, clean_result = self._valid_epoch(test_data)
        # #GRCN
        # # clean_u = self.model.result[:self.model.num_user]
        # # clean_i = self.model.result[self.model.num_user:]
        # #vbpr
        # clean_u = self.model.u_embedding
        # item_emb = self.model.item_linear(torch.cat((self.model.t_feat, self.model.v_feat), -1))
        # clean_i = torch.cat((self.model.i_embedding, item_emb), -1)
        # #slmrec
        # clean_u = 
        # print(clean_u.size())
        # print(clean_i.size())
        return clean_result
    
    def attack(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        #random noise
        # eps = 100000
        # noise = torch.randn_like(self.model.t_feat)
        # noise = F.normalize(noise,dim=1)
        # self.model.t_feat += eps*noise/torch.norm(noise)
        # noise = torch.randn_like(self.model.t_feat)
        # noise = F.normalize(noise,dim=1)
        # self.model.t_feat += eps*noise/torch.norm(noise)
        # self.model.t_feat = 0*torch.randn_like(self.model.t_feat)
        # self.model.t_feat = 0*torch.randn_like(self.model.t_feat)
        
        #FGSM-VBPR
        # self.model.item_raw_features = nn.Parameter(self.model.item_raw_features)
        # ori_feature = deepcopy(self.model.item_raw_features)
        # self.model.pre_epoch_processing()
        # eps = 14
        # c = 0
        # print(self.model.item_raw_features)
        # print(torch.norm(self.model.item_raw_features))
        # for batch_idx, interaction in enumerate(test_data):
        #     scores = self.model.full_sort_predict(interaction)
        #     # print(scores.size())
        #     pos_pair = []
        #     user_list = []
        #     pos_item_list = []
        #     for u in interaction[0]:
        #         for v in test_data.get_eval_items()[u]:
        #             user_list.append(int(u.cpu().numpy())-4096*c)
        #             pos_item_list.append(v)
        #     pos_pair.append(user_list)
        #     pos_pair.append(pos_item_list)
        #     att_loss = -torch.sum(scores[pos_pair])  #attack loss
        #     item_grad = torch.autograd.grad(att_loss, self.model.item_raw_features)[0]
        #     # self.model.item_raw_features.data.add_(eps*item_grad)
        #     c = c+1
        # #random noise
        # noise = torch.randn_like(self.model.item_raw_features)
        # self.model.item_raw_features.data.add_(0.05*noise*torch.norm(self.model.item_raw_features)/torch.norm(noise))

        #     # print(att_loss)
        #     # total_grad = item_grad.detach() if batch_idx==0 else total_grad+item_grad.detach()
        # # total_grad /= F.normalize(total_grad+1e-10,p=2,dim=1)
        # # self.model.item_raw_features.data.add_(eps*total_grad)
        # print(self.model.item_raw_features)
        # print('norm change:{}%'.format(100*torch.norm(self.model.item_raw_features-ori_feature)/torch.norm(ori_feature)))

        #FGSM-others
        ori_feature_t = deepcopy(self.model.t_feat).cpu()
        ori_feature_v = deepcopy(self.model.v_feat).cpu()
        self.model.t_feat = nn.Parameter(self.model.t_feat.cuda())
        self.model.v_feat = nn.Parameter(self.model.v_feat.cuda())

        eps_t = 0.008
        eps_v = 50
        c = 0
        # print(self.model.t_feat)
        # print(torch.norm(self.model.t_feat))
        score_margin = 0
        score_margin_v = 0
        score_margin_t = 0
        inter_test_num = 0
        for batch_idx, interaction in enumerate(test_data):
            # print(batch_idx)
            scores = self.model.full_sort_predict(interaction)
            scores_v = self.model.full_sort_predict_v(interaction)
            scores_t = self.model.full_sort_predict_t(interaction)

            pos_pair = []
            user_list = []
            pos_item_list = []
            for u in interaction[0]:
                for v in test_data.get_eval_items()[u]:
                    user_list.append(int(u.cpu().numpy())-4096*c)
                    pos_item_list.append(v)
            pos_pair.append(user_list)
            pos_pair.append(pos_item_list)

            masked_items = interaction[1]
            avg_u_score = torch.mean(scores,dim=1)
            avg_u_score_v = torch.mean(scores_v,dim=1)
            avg_u_score_t = torch.mean(scores_t,dim=1)
            att_loss = -torch.sum(scores[pos_pair]-avg_u_score[pos_pair[0]])  #attack bpr loss
            # att_loss = -torch.sum(scores[pos_pair])  #attack positive score loss
            # score_margin = score_margin + torch.sum(scores[pos_pair]-avg_u_score[pos_pair[0]]).item()
            # score_margin_v = score_margin_v + torch.sum(scores_v[pos_pair]-avg_u_score_v[pos_pair[0]]).item()
            # score_margin_t = score_margin_t + torch.sum(scores_t[pos_pair]-avg_u_score_t[pos_pair[0]]).item()
            # inter_test_num = inter_test_num + len(pos_item_list)
            v_grad = torch.autograd.grad(att_loss, self.model.v_feat,retain_graph=True)[0].cpu()
            del self.model.v_feat
            del scores
            del scores_t
            del scores_v
            del avg_u_score
            del avg_u_score_t
            del avg_u_score_v
            t_grad = torch.autograd.grad(att_loss, self.model.t_feat)[0].cpu()
            # self.model.t_feat.data.add_(eps_t*t_grad)
            # self.model.v_feat.data.add_(eps_v*v_grad)

            total_grad_v = v_grad.detach() if batch_idx==0 else total_grad_v+v_grad.detach()
            total_grad_t = t_grad.detach() if batch_idx==0 else total_grad_t+t_grad.detach()
            # print(total_grad_t)
            self.model.v_feat = nn.Parameter(ori_feature_v).cuda()
            # if batch_idx == 0:
            #     np.save('slm_score.npy',scores.cpu().detach().numpy())
            #     np.save('slm_score_v.npy',scores_v.cpu().detach().numpy())
            #     np.save('slm_score_t.npy',scores_t.cpu().detach().numpy())
                # np.save('pos_pair.npy',pos_pair)
            del att_loss
            del v_grad
            del t_grad
            c = c+1
        # print('score:',score_margin/inter_test_num)
        # print('score visual:',score_margin_v/inter_test_num)
        # print('score text:',score_margin_t/inter_test_num)
        total_grad_v = total_grad_v.cuda()
        self.model.v_feat.data.add_(0.05*torch.norm(ori_feature_v)*total_grad_v/torch.norm(total_grad_v))
        del total_grad_v
        total_grad_t = total_grad_t.cuda()
        # self.model.t_feat.data.add_(0.05*torch.norm(ori_feature_t)*total_grad_t/torch.norm(total_grad_t))
        del total_grad_t
        # del ori_feature_t
        # del ori_feature_v
        gc.collect()
        torch.cuda.empty_cache()
        print('text feature norm change:{}%'.format(100*torch.norm(self.model.t_feat.cpu()-ori_feature_t)/torch.norm(ori_feature_t)))
        print('visual feature norm change:{}%'.format(100*torch.norm(self.model.v_feat.cpu()-ori_feature_v)/torch.norm(ori_feature_v)))
        # print('3',torch.cuda.memory_summary())
        del ori_feature_t
        del ori_feature_v

        #test score change
        self.model.v_feat.grad = None
        self.model.t_feat.grad = None
        with torch.no_grad():
            score_margin = 0
            score_margin_v = 0
            score_margin_t = 0
            inter_test_num = 0
            c = 0
            for batch_idx, interaction in enumerate(test_data):
                scores = self.model.full_sort_predict(interaction)
                scores_v = self.model.full_sort_predict_v(interaction)
                scores_t = self.model.full_sort_predict_t(interaction)

                pos_pair = []
                user_list = []
                pos_item_list = []
                for u in interaction[0]:
                    for v in test_data.get_eval_items()[u]:
                        user_list.append(int(u.cpu().numpy())-4096*c)
                        pos_item_list.append(v)
                pos_pair.append(user_list)
                pos_pair.append(pos_item_list)

                avg_u_score = torch.mean(scores,dim=1)
                avg_u_score_v = torch.mean(scores_v,dim=1)
                avg_u_score_t = torch.mean(scores_t,dim=1)
                score_margin = score_margin + torch.sum(scores[pos_pair]-avg_u_score[pos_pair[0]]).item()
                score_margin_v = score_margin_v + torch.sum(scores_v[pos_pair]-avg_u_score_v[pos_pair[0]]).item()
                score_margin_t = score_margin_t + torch.sum(scores_t[pos_pair]-avg_u_score_t[pos_pair[0]]).item()
                inter_test_num = inter_test_num + len(pos_item_list)
                # if batch_idx == 0:
                #     np.save('slm_adv_score.npy',scores.cpu().detach().numpy())
                #     np.save('slm_adv_score_v.npy',scores_v.cpu().detach().numpy())
                #     np.save('slm_adv_score_t.npy',scores_t.cpu().detach().numpy())
                    # np.save('pos_pair.npy',pos_pair)

                
                c = c+1
        print('score after attack:',score_margin/inter_test_num)
        print('score visual after attack:',score_margin_v/inter_test_num)
        print('score text after attack:',score_margin_t/inter_test_num)
    
        #test
        self.model.pre_epoch_processing()
        self.model.train()
        loss_func = self.model.calculate_loss
        train_loss = None
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses, weak_pos, weak_neg, strong_pos, strong_neg = loss_func('t',interaction)
            # if batch_idx == 0:
            #     np.save('slmrec_attack_weak_pos.npy',np.array(weak_pos.detach().cpu()))
            #     np.save('slmrec_attack_weak_neg.npy',np.array(weak_neg.detach().cpu())) 
            #     np.save('slmrec_attack_strong_pos.npy',np.array(strong_pos.detach().cpu())) 
            #     np.save('slmrec_attack_strong_neg.npy',np.array(strong_neg.detach().cpu()))
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                train_loss = loss_tuple if train_loss is None else tuple(map(sum, zip(train_loss, loss_tuple)))
            else:
                loss = losses
                train_loss = losses.item() if train_loss is None else train_loss + losses.item()
        c=0

        #DualGNN
        # batch_matrix_list = []
        # for batch_idx, interaction in enumerate(test_data):
        #     # predict: interaction without item ids
        #     representation = None
        #     if self.model.v_feat is not None:
        #         self.model.v_rep, self.model.v_preference = self.model.v_gcn(self.model.edge_index_dropv, self.model.edge_index, self.model.v_feat)
        #         representation = self.model.v_rep
        #     if self.model.t_feat is not None:
        #         self.model.t_rep, self.model.t_preference = self.model.t_gcn(self.model.edge_index_dropt, self.model.edge_index, self.model.t_feat)
        #         if representation is None:
        #             representation = self.model.t_rep
        #         else:
        #             representation += self.model.t_rep
        #     # representation = self.v_rep+self.a_rep+self.t_rep

        #     # pdb.set_trace()
        #     if self.model.construction == 'weighted_sum':
        #         if self.model.v_rep is not None:
        #             self.model.v_rep = torch.unsqueeze(self.model.v_rep, 2)
        #             user_rep = self.model.v_rep[:self.model.num_user]
        #         if self.model.t_rep is not None:
        #             self.model.t_rep = torch.unsqueeze(self.model.t_rep, 2)
        #             user_rep = self.model.t_rep[:self.model.num_user]
        #         if self.model.v_rep is not None and self.model.t_rep is not None:
        #             user_rep = torch.matmul(torch.cat((self.model.v_rep[:self.model.num_user], self.model.t_rep[:self.model.num_user]), dim=2),
        #                                     self.model.weight_u)
        #         user_rep = torch.squeeze(user_rep)

        #     item_rep = representation[self.model.num_user:]
        #     ############################################ multi-modal information aggregation
        #     h_u1 = self.model.user_graph(user_rep, self.model.epoch_user_graph, self.model.user_weight_matrix)
        #     user_rep = user_rep + h_u1
        #     self.model.result_embed = torch.cat((user_rep, item_rep), dim=0)
        #     # user_tensor = self.model.result_embed[:self.model.n_users]
        #     item_tensor = self.model.result_embed[self.model.n_users:]
        #     # item_tensor = clean_i
        #     user_tensor = clean_u
        #     # user_tensor = result[:self.model.num_user]
        #     # item_tensor = result[self.model.num_user:]
        #     temp_user_tensor = user_tensor[interaction[0], :]
        #     scores = torch.matmul(temp_user_tensor, item_tensor.t())

        batch_matrix_list = []
        with torch.no_grad():
            for batch_idx, interaction in enumerate(test_data):
                # predict: interaction without item ids
                scores = self.model.full_sort_predict(interaction)
                # if batch_idx == 0:
                #     np.save('vbpr_weak_pos.npy',np.array(weak_pos.detach().cpu()))
                #     np.save('vbpr_weak_neg.npy',np.array(weak_neg.detach().cpu())) 
                #     np.save('vbpr_strong_pos.npy',np.array(strong_pos.detach().cpu())) 
                #     np.save('vbpr_strong_neg.npy',np.array(strong_neg.detach().cpu()))
                masked_items = interaction[1]
                # print(masked_items)
                # print('scores:',torch.mean(scores.view(-1)))
                # mask out pos items
                scores[masked_items[0], masked_items[1]] = -1e10
                
                # rank and get top-k
                _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
                batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, test_data, is_test=False, idx=0)
        # return test_result



