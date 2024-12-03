import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class MATPN_TRI(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, N=5, Q=1):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        self.fc1 = nn.Linear(768, 768 * 2)
        
        
        self.relation_encoder = relation_encoder
        self.hidden_size = 768
    
    
    def __dist__(self, x, y, dim):
        self.dot = False
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        # return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
        return self.__dist__(S, Q.unsqueeze(2), 3)

    # Rectification Loss
    def Triplet_Loss(self, anchor, positive_class, negative_class, margin=0.1, p=2, reduction='none'):
        '''
            positive_class(B, NQ, N, D)
            negative_class(B, NQ, N, D)
            query(B, NQ)[0, 2, 1, 3, 4, 2, 1, ...]
        '''
        triplet_criterion = torch.nn.TripletMarginLoss(margin=margin, p=p, reduction=reduction)
        return triplet_criterion(anchor, positive_class, negative_class)

    def N_Triplet_Loss(self, query, positive, negative, query_label, N, K, Q, B, D):
        '''
            anchor等价于query_ [NQ, D]
            positive等价于support_proto [NQ, N, D]
            negative等价于support_proto(出去q的所属类原型) [NQ, N, D]
            query_label表示q的标签 [NQ]
            '''
        triplet_loss = 0.0
        for j in range(len(query)):
            q = query[j].unsqueeze(0).expand(N - 1, -1)
            label_q = query_label[j]
            pos = positive[j][label_q].unsqueeze(0).expand(N - 1, -1)
            neg = self.delete_tensor(positive[j], label_q)
            triplet_loss += self.Triplet_Loss(q, pos, neg).sum(-1)
        return triplet_loss / (N * Q)

    def delete_tensor(self, tensor, index):
        '''
           tensor: 输入的张量tensor
           index: 需要删除的行数，或者说是索引
            '''
        t1 = tensor[0:index]
        t2 = tensor[index + 1:]
        return torch.cat((t1, t2), dim=0)

    def forward(self, support, query, rel_txt, N, K, total_Q, label):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        label: label of the query [BN]
        '''

        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False) # # rel_gol [B*N, D]

        
        #support,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        #query,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)
        
        
        support_h, support_t,  s_loc, s_gol = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_h, query_t,  q_loc, q_gol = self.sentence_encoder(query) # (B * total_Q, D)
        #support = self.global_atten_entity(support_h, support_t, s_loc, rel_loc, None)
        #query = self.global_atten_entity(query_h, query_t, q_loc, None, None)

        # 拼接实体特征
        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)



        # 拼接实体需要使用双倍隐藏层
        support = support.view(-1, N, K, self.hidden_size*2) # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size*2) # (B, total_Q, D)
        #
        # support = self.fc1(s_gol.view(-1, N, K,  self.hidden_size)) + support  # [B*N*K, D]
        # query = self.fc1(q_gol.view(-1, total_Q,  self.hidden_size)) + query
        # 使用全局s_gol需要添加
        # support = s_gol
        # query = q_gol
        # support = support.view(-1, N, K,  self.hidden_size)
        # query = query.view(-1, total_Q,  self.hidden_size)

        Q = int(query.size(1) / N)
        NQ = total_Q
        B = support.size(0)
        D = support.size(-1)


        # Instance Attention Module
        support_for_ins = support.unsqueeze(1).expand(B, NQ, -1, -1, -1)  # (B, NQ, N, K, D)
        query_for_att = query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1)
        ins_att_score = F.softmax(torch.tanh(support_for_ins * query_for_att).sum(-1), dim=-1)  # (B, NQ, N, K)
        support_proto_ins = (support_for_ins * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, D)).sum(
            3)  # (B, NQ, N, D)

        # No Instance Attention Module
        # support_mean = torch.mean(support, 2)
        # support_proto_ins = support_mean
        # support_proto_ins = support_mean.unsqueeze(1).expand(B, NQ, -1, -1)  # (B, NQ, N, D)

        # 拼接loc
        # rel_loc = torch.mean(rel_loc, 1)  # [B*N, D]
        # rel_rep = torch.cat((rel_gol, rel_loc), -1)
        # rel_words_emb = rel_rep

        # Label Awareness Module
        rel_words_emb = rel_gol

        # 只使用最后一层隐藏层的输出
        # rel_loc_cls = rel_loc[torch.arange(B*N), 0, :]

        # 使用最后一层隐藏层的输出的平均拼接整体特征
        # rel_loc_cls_ave = torch.mean(rel_loc, 1)
        # rel_words_emb = rel_loc_cls_ave + rel_gol

        # print(rel_words_emb.size())
        if D == 768:
            # 使用s_gol需要统一隐藏层
            support_label_word = rel_words_emb.view(B, N, -1)
        else:
            support_label_word = self.fc1(rel_words_emb).view(B, N, -1)
            # support_label_word = rel_words_emb.view(B, N, -1)

        # Label Awareness Module
        support_proto_ada_la = support_label_word.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, D)
        score_la = F.softmax(torch.tanh(support * support_proto_ada_la).sum(-1), dim=-1)  # (B, N, K)

        # 不进行归一化
        # score_la = torch.tanh(support * support_proto_ada_la).sum(-1)  # (B, N, K)

        support_proto_la = (support * score_la.unsqueeze(3).expand(-1, -1, -1, D)).sum(2)  # (B, N, D)
        support_proto_la = support_proto_la.unsqueeze(1).expand(-1, NQ, -1, -1)

        # No Label Awareness Module
        # support_proto_la = support_label_word.unsqueeze(1).expand(-1, NQ, -1, -1)
        # print(support_proto_la.size())


        # Adaptive Fusion Module
        score = F.softmax(torch.tanh(support_proto_ins * support_proto_la).sum(-1), dim=-1)  # (B, NQ, N)
        score = score.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, NQ, N, D)
        support_proto = support_proto_ins * (1 - score) + support_proto_la * score  # (B, NQ, N, D)

        # No Adaptive Fusion Module
        # support_proto = support_proto_ins + support_proto_la

        # 不注入外部标签信息
        # support_proto = support_proto_ins



        # Rectification Loss Loss
        query_label = label.view(B, NQ)
        query = query.view(-1, NQ, D)
        triplet_loss = 0.0
        for b in range(B):
            triplet_loss += self.N_Triplet_Loss(query[b], support_proto[b], support_proto[b], query_label[b], N, K, Q, B, D)
        triplet_loss = triplet_loss / B
        
        
        logits = self.__batch_dist__(support_proto, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred, triplet_loss

    
    
    
