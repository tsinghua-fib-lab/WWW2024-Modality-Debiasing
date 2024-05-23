import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood

def gradient_wrt_feature(model, x_adv, m_type, source_id, target_id, criterion=nn.MSELoss()):
    if m_type == 't':
        x_adv.requires_grad = True
        #dualgnn
        # t_rep, t_preference = model.t_gcn(model.edge_index_dropt, model.edge_index, x_adv)
        #grcn
        # t_rep, t_preference = model.t_gcn(model.edge_index, x_adv)
        #mmgcn
        # t_rep  = model.t_gcn(x_adv,model.id_embedding)
        #vbpr
        # t_rep = model.item_linear(torch.cat((x_adv, torch.zeros(model.v_feat.size()).cuda()), -1))
        # t_rep = torch.cat((model.i_embedding, t_rep), -1)
        #slmrec
        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            embs = [all_emb]
            g_droped = model.norm_adj
            for _ in range(model.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            return light_out
        users_emb = model.embedding_user.weight
        items_emb = model.embedding_item.weight
        t_emb = compute_graph(users_emb, model.t_dense(x_adv))
        t_emb_u, t_rep = torch.split(t_emb, [model.num_users, model.num_items])

        out = t_rep[source_id,:]
        target = t_rep[target_id,:].data.clone().detach()  
        loss = criterion(out, target)
        model.zero_grad()
        loss.backward()
        data_grad = x_adv.grad.data
    elif m_type == 'v':
        x_adv.requires_grad = True
        #dualgnn
        # v_rep, v_preference = model.v_gcn(model.edge_index_dropv, model.edge_index, x_adv)
        #grcn
        v_rep, v_preference = model.v_gcn(model.edge_index, x_adv)
        #mmgcn
        # v_rep = model.v_gcn(x_adv,model.id_embedding)
        #vbpr
        # v_rep = model.item_linear(torch.cat((torch.zeros(model.t_feat.size()).cuda(),x_adv), -1))
        # v_rep = torch.cat((model.i_embedding, v_rep), -1)
        #slmrec
        # def compute_graph(u_emb, i_emb):
        #     all_emb = torch.cat([u_emb, i_emb])
        #     embs = [all_emb]
        #     g_droped = model.norm_adj
        #     for _ in range(model.n_layers):
        #         all_emb = torch.sparse.mm(g_droped, all_emb)
        #         embs.append(all_emb)
        #     embs = torch.stack(embs, dim=1)
        #     light_out = torch.mean(embs, dim=1)
        #     return light_out
        # users_emb = model.embedding_user.weight
        # items_emb = model.embedding_item.weight
        # v_emb = compute_graph(users_emb, model.v_dense(x_adv))
        # v_emb_u, v_rep = torch.split(v_emb, [model.num_users, model.num_items])

        out = v_rep[source_id,:]
        target = v_rep[target_id,:].data.clone().detach()
        loss = criterion(out, target)
        # print(loss)
        model.zero_grad()
        loss.backward()
        data_grad = x_adv.grad.data
    return data_grad.clone().detach()

def get_adv_scores(model, m_type, x_adv, user_id, item_id):
    if m_type == 't':
        #dualgnn
        # t_rep, t_preference = model.t_gcn(model.edge_index_dropt, model.edge_index, x_adv)
        # v_rep, v_preference = model.v_gcn(model.edge_index_dropv, model.edge_index, model.v_feat)
        #mmgcn
        t_rep = model.t_gcn(x_adv,model.id_embedding)
        v_rep = model.v_gcn(model.v_feat,model.id_embedding)
        #vbpr
        # t_rep = model.item_linear(torch.cat((x_adv, torch.zeros(model.t_feat.size()[0],4096).cuda()), -1))
        # t_rep = torch.cat((model.i_embedding, t_rep), -1)
        # item_rep = t_rep
        #slmrec
        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            embs = [all_emb]
            g_droped = model.norm_adj
            for _ in range(model.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            return light_out
        users_emb = model.embedding_user.weight
        items_emb = model.embedding_item.weight
        t_emb = compute_graph(users_emb, model.t_dense(x_adv))
        t_emb_u, t_rep = torch.split(t_emb, [model.num_users, model.num_items])
        item_rep = t_rep

    elif m_type == 'v':
        #dualgnn
        # t_rep, t_preference = model.t_gcn(model.edge_index_dropt, model.edge_index, model.t_feat)
        # v_rep, v_preference = model.v_gcn(model.edge_index_dropv, model.edge_index, x_adv)
        #grcn
        t_rep, t_preference = model.t_gcn(model.edge_index, model.t_feat)
        v_rep, v_preference = model.v_gcn(model.edge_index, x_adv)
        #mmgcn
        # t_rep = model.t_gcn(model.t_feat,model.id_embedding)
        # v_rep = model.v_gcn(x_adv,model.id_embedding)
        #vbpr
        # v_rep = model.item_linear(torch.cat((torch.zeros(model.v_feat.size()[0],384).cuda(),x_adv), -1))
        # v_rep = torch.cat((model.i_embedding, v_rep), -1)
        #slmrec
        # def compute_graph(u_emb, i_emb):
        #     all_emb = torch.cat([u_emb, i_emb])
        #     embs = [all_emb]
        #     g_droped = model.norm_adj
        #     for _ in range(model.n_layers):
        #         all_emb = torch.sparse.mm(g_droped, all_emb)
        #         embs.append(all_emb)
        #     embs = torch.stack(embs, dim=1)
        #     light_out = torch.mean(embs, dim=1)
        #     return light_out
        # users_emb = model.embedding_user.weight
        # items_emb = model.embedding_item.weight
        # v_emb = compute_graph(users_emb, model.v_dense(x_adv))
        # v_emb_u, v_rep = torch.split(v_emb, [model.num_users, model.num_items])
        item_rep = v_rep
    #dualgnn 
    # user_tensor = model.result_embed[user_id]
    # score = torch.sum(user_tensor[:model.dim] * item_rep[item_id])
    #grcn
    user_tensor = model.result[user_id]
    score = torch.sum(user_tensor[model.dim:2*model.dim] * item_rep[item_id])
    #mmgcn
    # user_tensor = model.result[user_id]
    #vbpr
    # user_tensor = model.u_embedding[user_id]
    #slmrec
    # user_tensor = v_emb_u[user_id] if m_type == 'v' else t_emb_u[user_id]
    # score = torch.sum(user_tensor * item_rep[item_id])

    return score

def Linf_distillation(model, m_type, source_id, target_id, eps, alpha, steps, mu=1, momentum=False, rand_start=False):
    if m_type == 't':
        x_nat = model.t_feat.clone().detach()
        x_adv = None

        if rand_start:
            x_adv = model.t_feat.clone().detach() + torch.FloatTensor(model.t_feat.shape).uniform_(-eps, eps).cuda()
        else:
            x_adv = model.t_feat.clone().detach()
        # g = torch.zeros_like(x_adv)

        # Iteratively Perturb data
        for i in range(steps):
            # Calculate gradient w.r.t. data
            grad = gradient_wrt_feature(model, x_adv, m_type, source_id, target_id)
            
            with torch.no_grad():
                if momentum:
                    # Accumulate the gradient
                    new_grad = mu * g + grad # calc new grad with momentum term
                    g = new_grad
                else:
                    new_grad = grad
                x_adv = x_adv - alpha * new_grad.sign()  # perturb the data to MINIMIZE loss 
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
    
    if m_type == 'v':
        # print(source_id)
        # print(target_id)
        x_nat = model.v_feat.clone().detach()
        x_adv = None

        if rand_start:
            x_adv = model.v_feat.clone().detach() + torch.FloatTensor(model.v_feat.shape).uniform_(-eps, eps).cuda()
        else:
            x_adv = model.v_feat.clone().detach()
        # g = torch.zeros_like(x_adv)

        # Iteratively Perturb data
        # print('------------')
        for i in range(steps):
            # Calculate gradient w.r.t. data
            grad = gradient_wrt_feature(model, x_adv, m_type, source_id, target_id)

            with torch.no_grad():
                if momentum:
                    # Accumulate the gradient
                    new_grad = mu * g + grad # calc new grad with momentum term
                    g = new_grad
                else:
                    new_grad = grad
                x_adv = x_adv - alpha * new_grad.sign() 
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
            # print(torch.norm(x_adv-model.v_feat)/torch.norm(model.v_feat))
        # del g
    return x_adv.clone().detach()

def gradient_wrt_feature_adv_train(model, x_adv, m_type, user_id, pos_id, neg_id):
    if m_type == 't':

        loss_adv = -model.bpr_loss(torch.tensor([user_id]), torch.tensor([pos_id]), torch.tensor([neg_id]))     
        model.zero_grad()
        loss_adv.backward(retain_graph=True)
        data_grad = model.t_feat.grad.data
    elif m_type == 'v':
        
        loss_adv = -model.bpr_loss(torch.tensor([user_id]), torch.tensor([pos_id]), torch.tensor([neg_id]))    
        loss_adv.backward(retain_graph=True)
        # print(loss_adv)
        data_grad = model.v_feat.grad.data
        # print(data_grad)
    return data_grad.clone().detach()

def Adv_distillation(model, m_type, user, source_id, target_id, eps, alpha, steps, mu=0, momentum=False, rand_start=False):
    # print(m_type)
    if m_type == 't':
        x_nat = model.t_feat.clone().detach()
        x_adv = None

        if rand_start:
            x_adv = model.t_feat.clone().detach() + torch.FloatTensor(model.t_feat.shape).uniform_(-eps, eps).cuda()
        else:
            x_adv = model.t_feat.clone().detach()
        for i in range(steps):
            grad = gradient_wrt_feature_adv_train(model, x_adv, m_type, user, source_id, target_id)
            with torch.no_grad():
                if momentum:
                    # Accumulate the gradient
                    new_grad = mu * g + grad # calc new grad with momentum term
                    g = new_grad
                else:
                    new_grad = grad
                x_adv = x_adv - alpha * new_grad.sign()  # perturb the data to MINIMIZE loss 
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
    
    if m_type == 'v':
        x_nat = model.v_feat.clone().detach()
        x_adv = None

        if rand_start:
            x_adv = model.v_feat.clone().detach() + torch.FloatTensor(model.v_feat.shape).uniform_(-eps, eps).cuda()
        else:
            x_adv = model.v_feat.clone().detach()
        for i in range(steps):
            grad = gradient_wrt_feature_adv_train(model, x_adv, m_type, user, source_id, target_id)
            with torch.no_grad():
                if momentum:
                    # Accumulate the gradient
                    new_grad = mu * g + grad # calc new grad with momentum term
                    g = new_grad
                else:
                    new_grad = grad
                x_adv = x_adv - 0.05*torch.norm(x_adv)/torch.norm(new_grad) * new_grad # perturb the data to MINIMIZE loss 
                # print(torch.norm(alpha * new_grad)/torch.norm(x_adv))
            x_adv = torch.max(torch.min(x_adv, x_nat+eps), x_nat-eps)
    return x_adv.clone().detach()