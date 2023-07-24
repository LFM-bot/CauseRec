# @Time   : 2023/2/14
# @Author : Chenglong Shi
# @Email  : sclzufe@163.com

r"""
CauseRec
################################################
Reference:
    Zhang Yao et al. "CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation." in SIGIR 2021.
Reference code:
    https://github.com/gzy-rgb/CauseRec
"""

import random

import torch.nn.functional as F
from src.model.abstract_recommeder import AbstractRecommender
import argparse
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_
from src.model.sequential_encoder import Transformer
from src.utils.utils import HyperParamDict


class CauseRec(AbstractRecommender):
    def __init__(self, config, additional_data_dict):
        super(CauseRec, self).__init__(config)
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.batch_size = config.train_batch
        self.lamda_1 = config.lamda_1
        self.lamda_2 = config.lamda_2
        self.sample_size = config.sample_size
        self.neg_size = config.neg_size
        self.pos_size = config.pos_size
        self.bank_size = config.bank_size
        self.interest_num = config.interest_num
        self.replace_ratio = config.replace_ratio
        self.initializer_range = config.initializer_range

        # module
        self.item_embedding = nn.Embedding(self.num_items, self.embed_size, padding_idx=0)
        self.input_dropout = nn.Dropout(config.hidden_dropout)
        self.W1 = torch.nn.Parameter(data=torch.randn(self.hidden_size, self.embed_size), requires_grad=True)
        self.W2 = torch.nn.Parameter(data=torch.randn(self.interest_num, self.hidden_size), requires_grad=True)
        self.mlp = nn.Sequential(nn.Linear(self.embed_size, self.hidden_size), nn.Tanh(),
                                 nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
                                 nn.Linear(self.hidden_size, self.embed_size))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cosine_contrastive_loss = nn.CosineEmbeddingLoss(0.5)
        self.contrastive_loss = nn.TripletMarginLoss(margin=1, p=2)
        self.embedding_memory_bank = None  # [bank_size, D]
        self.item_memory_bank = None  # [bank_size]

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def train_forward(self, data_dict: dict):
        item_seq, item_seq_len, next_item = self.load_basic_SR_data(data_dict)

        # item importance score
        ranked_item_idx, restore_idx = self.get_interest_and_ranking(item_seq, next_item)

        # obtain interest embedding, [B, K, D]
        interest_embedding = self.get_interest_embedding(item_seq)

        # fill memory bank
        if data_dict['epoch'] == 0 and data_dict['step'] == 0:
            with torch.no_grad():
                batch_embedding_set = interest_embedding.reshape(-1, self.embed_size)  # [B * K, D]
                batch_item_set = item_seq[item_seq > 0]  # only enqueue valid items
                self.embedding_memory_bank = batch_embedding_set[:self.bank_size]
                self.item_memory_bank = batch_item_set[:self.bank_size]
                return None  # skip training

        # memory bank is filled, start training.
        seq_embedding = self.feature_aggregation(interest_embedding)

        if self.sample_size > 0:
            target = torch.tensor([0] * item_seq.size(0), dtype=torch.int64).to(self.dev)  # [B]
            candidate_set = list(set(range(self.num_items)) ^ set(next_item.tolist()))  # ^: A + B - (A and B)
            candidate_set = torch.tensor(random.sample(candidate_set, self.sample_size)).to(self.dev)  # [sample_size]
            logits = self.sampled_softmax(seq_embedding, next_item, candidate_set)  # [B, 1 + S]
        else:  # use all items
            logits = seq_embedding @ self.item_embedding.weight.t()
            target = next_item

        rec_loss = self.cross_entropy_loss(logits, target)  # SR task loss

        negative_seq_embedding = self.counterfactual_neg_sample_for_loop(item_seq, item_seq_len, ranked_item_idx,
                                                                         restore_idx)  # [neg_size, B, D]
        positive_seq_embedding = self.counterfactual_pos_sample_for_loop(item_seq, item_seq_len, ranked_item_idx,
                                                                         restore_idx)  # [pos_size, B, D]

        # contrast between interest and item
        next_item_embedding = self.item_embedding(next_item)
        pos_target = torch.ones_like(item_seq_len, device=self.dev)
        neg_target = -torch.ones_like(item_seq_len, device=self.dev)
        loss_pos = self.cosine_contrastive_loss(next_item_embedding, positive_seq_embedding.mean(dim=0), pos_target)
        loss_neg = self.cosine_contrastive_loss(next_item_embedding, negative_seq_embedding.mean(dim=0), neg_target)
        cl_loss1 = loss_pos + loss_neg

        # contrast between counterfactual and observation (triplet margin loss)
        cl_loss2 = self.contrastive_loss(seq_embedding, positive_seq_embedding.mean(dim=0),
                                         negative_seq_embedding.mean(dim=0))

        loss = rec_loss + self.lamda_1 * cl_loss1 + self.lamda_2 * cl_loss2

        with torch.no_grad():  # update memory bank
            batch_embedding_set = interest_embedding.reshape(-1, self.embed_size)  # [B * K, D]
            self.embedding_memory_bank = batch_embedding_set[:self.bank_size]

            batch_item_set = item_seq[item_seq > 0]  # only enqueue valid items
            new_item_bank = batch_item_set[:self.bank_size]
            new_item_bank = torch.cat([self.item_memory_bank, new_item_bank], 0)
            self.item_memory_bank = new_item_bank[-self.bank_size:]

        return loss

    def forward(self, data_dict):
        item_seq, seq_len, target = self.load_basic_SR_data(data_dict)

        interest_embedding = self.get_interest_embedding(item_seq)  # [B, K, D]
        seq_embedding = self.feature_aggregation(interest_embedding)  # [B, D]
        logits = torch.matmul(seq_embedding, self.item_embedding.weight.t())

        return logits

    def get_interest_and_ranking(self, item_seq, next_item):
        """
        Parameters
        ----------
        item_seq: [B, L]
        next_item:  [B]

        Returns
        -------
        ranked_item_idx: [B, L]
        recover_idx: [B, L]
        """
        valid_mask = (item_seq > 0).bool()
        item_seq_embedding = self.input_dropout(self.item_embedding(item_seq))
        next_item_embedding = self.item_embedding(next_item).unsqueeze(1)  # [B, 1, D]

        # calculate item-level score, dot-product
        item_score = item_seq_embedding @ next_item_embedding.transpose(-2, -1)
        item_score = item_score.squeeze()
        item_score = torch.masked_fill(item_score, ~valid_mask, -1.)
        _, ranked_item_idx = torch.sort(item_score)  # [B, L]
        _, restore_idx = torch.sort(ranked_item_idx)

        return ranked_item_idx, restore_idx

    def get_interest_embedding(self, item_seq, return_score=False):
        """
        Parameters
        ----------
        item_seq: [*, B, L]
        return_score: bool, if return interest score

        Returns
        -------
        interest_embedding: [B, D]
        """
        item_seq_embedding = self.item_embedding(item_seq)
        item_seq_embedding = self.input_dropout(item_seq_embedding)
        interest_logits = torch.matmul(self.W2, torch.tanh(torch.matmul(self.W1, item_seq_embedding.transpose(-2, -1))))
        interest_score = F.softmax(interest_logits, dim=2)
        interest_embedding = torch.matmul(interest_score, item_seq_embedding)
        if not return_score:
            return interest_embedding
        return interest_embedding, interest_score

    def feature_aggregation(self, feature_embeddings):
        """
        Parameters
        ----------
        feature_embeddings: [*, B, *, D]

        Returns
        -------
        sequence_embedding: [B, D]

        """
        seq_embedding = torch.mean(feature_embeddings, dim=-2)
        seq_embedding = self.mlp(seq_embedding)
        return seq_embedding

    def sampled_softmax(self, user_embedding, next_item, candidate_set):
        """
        Parameters
        ----------
        user_embedding: [B, D]
        next_item: [B, D]
        candidate_set: [S], S: sample_size

        Returns
        -------
        logits: [B, 1 + S]
        """
        target_embedding = torch.sum(self.item_embedding(next_item) * user_embedding, dim=1).view(
            user_embedding.size(0), 1)  # [B, 1]
        sampled_logits = torch.matmul(user_embedding,
                                      torch.transpose(self.item_embedding(candidate_set), 0, 1))  # [B, S]
        sampled_logits = torch.cat([target_embedding, sampled_logits], dim=1)
        return sampled_logits

    def counterfactual_neg_sample(self, item_seq, seq_len, ranked_item_idx, restore_item_idx):
        """

        Parameters
        ----------
        item_seq: [B, L]
        seq_len: [B]
        ranked_item_idx: [B, L]
        restore_item_idx: [B, L]

        Returns
        -------
        counterfactual_neg_interest_embedding: [neg_size, B, D]
        """
        # obtain item mask
        ranked_item_seq = item_seq.gather(dim=-1, index=ranked_item_idx)  # ascending order, [B, L]
        replace_size = (seq_len * self.replace_ratio).ceil().long()  # ceil for negative, [B]
        position_tensor = torch.arange(self.max_len, device=self.dev).expand(item_seq.size(0), -1)
        nontrivial_item_mask = (position_tensor >= self.max_len - replace_size.unsqueeze(-1)).bool()

        # sample replacing items
        other_item = self.item_memory_bank  # [bank_size]
        other_item_embedding = self.item_embedding(other_item)
        l2_other_item_embedding = F.normalize(other_item_embedding, p=2, dim=-1)
        nontrivial_item = ranked_item_seq[nontrivial_item_mask]  # [nontrivial_size]
        nontrivial_item_embedding = self.item_embedding(nontrivial_item)
        l2_nontrivial_item_embedding = F.normalize(nontrivial_item_embedding, p=2, dim=-1)
        replace_sample_prob = l2_nontrivial_item_embedding @ l2_other_item_embedding.transpose(-2, -1)
        replace_sample_prob = F.softmax(replace_sample_prob, dim=-1)
        replace_sample_prob = 1 - replace_sample_prob  # choose dissimilar items
        multi_replace_sample_prob = replace_sample_prob.repeat(self.neg_size, 1, 1).float()
        multi_replace_sample_prob = multi_replace_sample_prob.view(-1, self.bank_size)
        nontrivial_item_sampled_idx = torch.multinomial(multi_replace_sample_prob,
                                                        1).view(self.neg_size, -1)  # [neg_size, nontrivial_size]
        other_item = other_item.expand(self.neg_size, -1)
        nontrivial_replacing_item_set = other_item.gather(dim=-1, index=nontrivial_item_sampled_idx).flatten()

        # replace operation
        nontrivial_item_mask = nontrivial_item_mask.expand(self.neg_size, -1, -1)
        counterfactual_neg_item_seq = ranked_item_seq.repeat(self.neg_size, 1, 1)  # [neg, B, L]
        counterfactual_neg_item_seq[nontrivial_item_mask] = nontrivial_replacing_item_set

        # re-sort to original position
        restore_item_idx = restore_item_idx.expand(self.neg_size, -1, -1)  # [neg_size, B, L]
        counterfactual_neg_item_seq = counterfactual_neg_item_seq.gather(dim=-1,
                                                                         index=restore_item_idx)  # [neg_size, B, L]

        counterfactual_interest_embedding = self.get_interest_embedding(
            counterfactual_neg_item_seq)  # [neg_size, B, K, D]

        counterfactual_neg_interest_embedding = self.feature_aggregation(counterfactual_interest_embedding)

        return counterfactual_neg_interest_embedding

    def counterfactual_neg_sample_for_loop(self, item_seq, seq_len, ranked_item_idx, restore_item_idx):
        """
        Args:
            item_seq: [B, L]
            seq_len: [B]
            ranked_item_idx: [B, L]
            restore_item_idx: [B, L]

        Returns:
            counterfactual_neg_interest_embedding: [neg_size, B, D]
        """
        counterfactual_neg_interest_embedding = []

        ranked_item_seq = item_seq.gather(dim=-1, index=ranked_item_idx)  # ascending order, [B, L]
        replace_size = (seq_len * self.replace_ratio).long()  # [B, 1]
        position_tensor = torch.arange(self.max_len, device=self.dev).expand(item_seq.size(0), -1)
        nontrivial_item_mask = (position_tensor >= self.max_len - replace_size.unsqueeze(-1)).bool()

        # sample items for replacement
        other_item = self.item_memory_bank  # [bank_size]
        other_item_embedding = self.item_embedding(other_item)
        l2_other_item_embedding = F.normalize(other_item_embedding, p=2, dim=-1)
        nontrivial_item = ranked_item_seq[nontrivial_item_mask]  # [nontrivial_size]
        nontrivial_item_embedding = self.item_embedding(nontrivial_item)
        l2_nontrivial_item_embedding = F.normalize(nontrivial_item_embedding, p=2, dim=-1)
        replace_sample_prob = l2_nontrivial_item_embedding @ l2_other_item_embedding.transpose(-2, -1)
        replace_sample_prob = F.softmax(replace_sample_prob, dim=-1)
        replace_sample_prob = 1 - replace_sample_prob  # choose dissimilar items

        for j in range(self.neg_size):
            nontrivial_item_sampled_idx = torch.multinomial(replace_sample_prob, 1).squeeze()  # [nontrivial_size]
            nontrivial_replacing_item_set = other_item[nontrivial_item_sampled_idx]  # [nontrivial_size]
            counterfactual_neg_item_seq = ranked_item_seq.clone()
            counterfactual_neg_item_seq[nontrivial_item_mask] = nontrivial_replacing_item_set

            # re-sort to origin position
            counterfactual_neg_item_seq = counterfactual_neg_item_seq.gather(dim=-1, index=restore_item_idx)  # [B, L]

            # obtain negative sequence embedding
            counterfactual_interest_embedding = self.get_interest_embedding(counterfactual_neg_item_seq)
            counterfactual_neg_interest_embedding.append(self.feature_aggregation(counterfactual_interest_embedding))

        counterfactual_neg_interest_embedding = torch.stack(counterfactual_neg_interest_embedding, dim=0)

        return counterfactual_neg_interest_embedding

    def counterfactual_pos_sample(self, item_seq, seq_len, ranked_item_idx, restore_item_idx):
        """
        Parameters
        ----------
        item_seq: [B, L]
        seq_len: [B]
        ranked_item_idx: [B, L]
        restore_item_idx: [B, L]

        Returns
        -------
        counterfactual_pos_interest_embedding: [pos_size, B, D]
        """
        # obtain item mask
        ranked_item_seq = item_seq.gather(dim=-1, index=ranked_item_idx)  # ascending order, [B, L]
        non_replace_size = seq_len - (seq_len * self.replace_ratio).floor().long()  # floor for positive, [B]
        position_tensor = torch.arange(self.max_len, device=self.dev).expand(item_seq.size(0), -1)
        valid_item_mask = (position_tensor >= self.max_len - seq_len.unsqueeze(-1)).bool()
        trivial_item_mask = (position_tensor < self.max_len - non_replace_size.unsqueeze(-1)).bool()
        trivial_item_mask = trivial_item_mask & valid_item_mask

        # sample replacing items
        other_item = self.item_memory_bank  # [bank_size]
        other_item_embedding = self.item_embedding(other_item)
        l2_other_item_embedding = F.normalize(other_item_embedding, p=2, dim=-1)
        trivial_item = ranked_item_seq[trivial_item_mask]
        trivial_item_embedding = self.item_embedding(trivial_item)
        l2_trivial_item_embedding = F.normalize(trivial_item_embedding, p=2, dim=-1)
        replace_sample_prob = l2_trivial_item_embedding @ l2_other_item_embedding.transpose(-2, -1)
        replace_sample_prob = F.softmax(replace_sample_prob, dim=-1)
        multi_replace_sample_prob = replace_sample_prob.repeat(self.pos_size, 1, 1).float()
        multi_replace_sample_prob = multi_replace_sample_prob.view(-1, self.bank_size)
        trivial_item_sampled_idx = torch.multinomial(multi_replace_sample_prob,
                                                     1).view(self.neg_size, -1)  # [neg_size, nontrivial_size]
        other_item = other_item.expand(self.pos_size, -1)
        trivial_replacing_item_set = other_item.gather(dim=-1, index=trivial_item_sampled_idx).flatten()

        # replace operation
        trivial_item_mask = trivial_item_mask.expand(self.neg_size, -1, -1)
        counterfactual_pos_item_seq = ranked_item_seq.repeat(self.neg_size, 1, 1)  # [neg, B, L]
        counterfactual_pos_item_seq[trivial_item_mask] = trivial_replacing_item_set

        # re-sort to original position
        restore_idx = restore_item_idx.expand(self.neg_size, -1, -1)  # [neg_size, B, L]
        counterfactual_pos_item_seq = counterfactual_pos_item_seq.gather(dim=-1, index=restore_idx)  # [neg_size, B, L]

        counterfactual_interest_embedding = self.get_interest_embedding(
            counterfactual_pos_item_seq)  # [neg_size, B, K, D]

        counterfactual_pos_interest_embedding = self.feature_aggregation(counterfactual_interest_embedding)

        return counterfactual_pos_interest_embedding

    def counterfactual_pos_sample_for_loop(self, item_seq, seq_len, ranked_item_idx, restore_item_idx):
        """
        Args:
            item_seq: [B, L]
            seq_len: [B]
            ranked_item_idx: [B, L]
            restore_item_idx: [B, L]

        Returns:
            counterfactual_pos_interest_embedding: [pos_size, B, D]
        """

        counterfactual_pos_interest_embedding = []

        ranked_item_seq = item_seq.gather(dim=-1, index=ranked_item_idx)  # ascending order, [B, L]
        non_replace_size = seq_len - (seq_len * self.replace_ratio).long()
        position_tensor = torch.arange(self.max_len, device=self.dev).expand(item_seq.size(0), -1)
        valid_item_mask = (position_tensor >= self.max_len - seq_len.unsqueeze(-1)).bool()
        trivial_item_mask = (position_tensor < self.max_len - non_replace_size.unsqueeze(-1)).bool()
        trivial_item_mask = trivial_item_mask & valid_item_mask

        # sample items for replacement
        other_item = self.item_memory_bank  # [bank_size]
        other_item_embedding = self.item_embedding(other_item)
        l2_other_item_embedding = F.normalize(other_item_embedding, p=2, dim=-1)
        trivial_item = ranked_item_seq[trivial_item_mask]
        trivial_item_embedding = self.item_embedding(trivial_item)
        l2_trivial_item_embedding = F.normalize(trivial_item_embedding, p=2, dim=-1)
        replace_sample_prob = l2_trivial_item_embedding @ l2_other_item_embedding.transpose(-2, -1)
        replace_sample_prob = F.softmax(replace_sample_prob, dim=-1)

        for j in range(self.pos_size):
            trivial_item_sampled_idx = torch.multinomial(replace_sample_prob, 1).squeeze()  # [trivial_size]
            trivial_replacing_item_set = other_item[trivial_item_sampled_idx]  # [trivial_size]
            counterfactual_pos_item_seq = ranked_item_seq.clone()
            counterfactual_pos_item_seq[trivial_item_mask] = trivial_replacing_item_set

            # re-sort to origin position
            counterfactual_pos_item_seq = counterfactual_pos_item_seq.gather(dim=-1, index=restore_item_idx)

            # obtain positive sequence embedding
            counterfactual_interest_embedding = self.get_interest_embedding(counterfactual_pos_item_seq)
            counterfactual_pos_interest_embedding.append(self.feature_aggregation(counterfactual_interest_embedding))
        counterfactual_pos_interest_embedding = torch.stack(counterfactual_pos_interest_embedding, dim=0)

        return counterfactual_pos_interest_embedding


def CauseRec_config():
    parser = HyperParamDict('CauseRec default hyper-parameters')
    parser.add_argument('--model', default='CauseRec', type=str)
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--interest_num', default=20, type=int)
    parser.add_argument('--replace_ratio', default=0.5, type=float)
    parser.add_argument('--neg_size', default=8, type=int, help='counterfactual neg sample size')
    parser.add_argument('--pos_size', default=1, type=int, help='counterfactual pos sample size')
    parser.add_argument('--bank_size', default=1024, type=int, help='volume of memory bank')
    parser.add_argument('--lamda_1', default=1., type=float,
                        help='weight of FO (Counterfactual and Observation) contrastive loss')
    parser.add_argument('--lamda_2', default=1., type=float, help='weight of II (Interest and Items) contrastive loss')
    parser.add_argument('--sample_size', default=-1, type=int, help='sampled neg item for cross-entropy loss')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')
    parser.add_argument('--loss_type', default='CUSTOM', type=str, choices=['CE', 'BPR', 'BCE', 'CUSTOM'])
    return parser
