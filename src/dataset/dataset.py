import copy
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from src.utils.utils import neg_sample
from src.model.data_augmentation import Crop, Mask, Reorder
from src.model.data_augmentation import AUGMENTATIONS


def load_specified_dataset(model_name, config):
    return SequentialDataset


class BaseSequentialDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(BaseSequentialDataset, self).__init__()
        self.batch_batch_dict = {}
        self.num_items = config.num_items
        self.config = config
        self.train = train
        self.dataset = config.dataset
        self.max_len = config.max_len
        self.item_seq = data_pair[0]
        self.label = data_pair[1]

    def get_SRtask_input(self, idx):
        item_seq = self.item_seq[idx]
        target = self.label[idx]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))

    def __getitem__(self, idx):
        return self.get_SRtask_input(idx)

    def __len__(self):
        return len(self.item_seq)

    def collate_fn(self, x):
        return self.basic_SR_collate_fn(x)

    # def basic_SR_collate_fn(self, x):
    #     """
    #     x: [(seq_1, len_1, tar_1), ..., (seq_n, len_n, tar_n)]
    #     """
    #     tensor_dict = {}
    #     tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
    #
    #     item_seq, seq_len, target = tensor_list
    #     tensor_dict['item_seq'] = item_seq
    #     tensor_dict['seq_len'] = seq_len
    #     tensor_dict['target'] = target
    #     return tensor_dict

    def basic_SR_collate_fn(self, x):
        """
        x: [(seq_1, len_1, tar_1), ..., (seq_n, len_n, tar_n)]
        """
        item_seq, seq_len, target = default_collate(x)
        self.batch_batch_dict['item_seq'] = item_seq
        self.batch_batch_dict['seq_len'] = seq_len
        self.batch_batch_dict['target'] = target
        return self.batch_batch_dict


class SequentialDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SequentialDataset, self).__init__(config, data_pair, additional_data_dict, train)


class SequentialNoAugDataset(BaseSequentialDataset):
    """
    No sequential data augmentation.
    """

    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SequentialNoAugDataset, self).__init__(config, data_pair, additional_data_dict, train)

    def __getitem__(self, idx):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(idx)

        item_seq = self.item_seq[idx]
        target_seq = item_seq[1:]
        if not isinstance(target_seq, list):
            target_seq = [target_seq]
        last_target = self.label[idx]
        target_seq.append(last_target)

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        target_seq = target_seq[-self.max_len:]
        target_seq = target_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len
        assert len(target_seq) == self.max_len

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long))


class KERLDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(KERLDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.tar_length = config.episode_len

    def __getitem__(self, idx):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(idx)

        # for training
        item_seq = self.item_seq[idx]
        target_seq = self.label[idx]
        if not isinstance(target_seq, list):
            target_seq = [target_seq]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        target_len = len(target_seq)

        # crop
        item_seq = item_seq[-self.max_len:]
        # padding
        item_seq = item_seq + (self.max_len - seq_len) * [0]
        target_seq = target_seq + (self.tar_length - target_len) * [0]

        assert len(item_seq) == self.max_len
        assert len(target_seq) == self.tar_length

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long),
                torch.tensor(target_len, dtype=torch.long))

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)
        tensor_list = default_collate(x)
        item_seq, seq_len, target_seq, target_len = tensor_list

        self.batch_batch_dict['item_seq'] = item_seq
        self.batch_batch_dict['seq_len'] = seq_len
        self.batch_batch_dict['target_seq'] = target_seq
        self.batch_batch_dict['target_len'] = target_len

        return self.batch_batch_dict


class BERTDataset(BaseSequentialDataset):
    """
    For bert training.
    """

    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(BERTDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.mask_id = self.num_items
        self.num_items = self.num_items + 1
        self.max_len = config.max_len + 1  # add mask at last pos

        self.mask_ratio = config.mask_ratio
        self.last_mask_ratio = config.last_mask_ratio  # only mask last

    def __getitem__(self, index):
        sequence = self.item_seq[index]  # pos_items

        # eval and test phase
        if not self.train:
            item_sequence = sequence
            item_sequence.append(self.mask_id)
            seq_len = len(item_sequence) if len(item_sequence) < self.max_len else self.max_len
            target = self.label[index]

            item_sequence = item_sequence[-self.max_len:]
            item_sequence = item_sequence + (self.max_len - seq_len) * [0]

            return (torch.tensor(item_sequence, dtype=torch.long),
                    torch.tensor(seq_len, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long))

        # for training: Masked Item Prediction
        masked_item_sequence = []
        pos_items = copy.deepcopy(sequence)

        # add mask with raito p
        for item in sequence:
            prob = random.random()
            if prob < self.mask_ratio:
                masked_item_sequence.append(self.mask_id)
            else:
                masked_item_sequence.append(item)

        # add mask at last position
        if random.random() < self.last_mask_ratio:
            masked_item_sequence.append(self.mask_id)
            pos_items.append(self.label[index])
        else:  # padding
            masked_item_sequence.append(0)
            pos_items.append(0)

        assert len(masked_item_sequence) == len(sequence) + 1
        assert len(pos_items) == len(sequence) + 1

        # crop sequence
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]

        # padding sequence
        pad_len = self.max_len - len(masked_item_sequence)
        masked_item_sequence = masked_item_sequence + [0] * pad_len
        pos_items = pos_items + [0] * pad_len

        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len

        cur_tensors = (torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long))
        return cur_tensors

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        masked_item_seq, target = tensor_list

        tensor_dict['masked_item_seq'] = masked_item_seq
        tensor_dict['target'] = target

        return tensor_dict


class CL4SRecDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(CL4SRecDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.mask_id = self.num_items
        self.aug_types = config.aug_types
        self.n_views = 2
        self.augmentations = []

        self.load_augmentor()

    def load_augmentor(self):
        for aug in self.aug_types:
            if aug == 'mask':
                self.augmentations.append(Mask(gamma=self.config.mask_ratio, mask_id=self.mask_id))
            else:
                self.augmentations.append(AUGMENTATIONS[aug](getattr(self.config, f'{aug}_ratio')))

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        # contrast learning augmented views
        item_seq = self.item_seq[index]
        target = self.label[index]
        aug_type = np.random.choice([i for i in range(len(self.augmentations))],
                                    size=self.n_views, replace=True)
        aug_seq_1 = self.augmentations[aug_type[0]](item_seq)
        aug_seq_2 = self.augmentations[aug_type[1]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(aug_seq_1, dtype=torch.long),
                       torch.tensor(aug_seq_2, dtype=torch.long),
                       torch.tensor(aug_len_1, dtype=torch.long),
                       torch.tensor(aug_len_2, dtype=torch.long))

        return cur_tensors

    # def collate_fn(self, x):
    #     if not self.train:
    #         return self.basic_SR_collate_fn(x)
    #
    #     tensor_dict = {}
    #     tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
    #
    #     item_seq, seq_len, target, aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = tensor_list
    #
    #     tensor_dict['item_seq'] = item_seq
    #     tensor_dict['seq_len'] = seq_len
    #     tensor_dict['target'] = target
    #     tensor_dict['aug_seq_1'] = aug_seq_1
    #     tensor_dict['aug_seq_2'] = aug_seq_2
    #     tensor_dict['aug_len_1'] = aug_len_1
    #     tensor_dict['aug_len_2'] = aug_len_2
    #
    #     return tensor_dict

    # def collate_fn(self, x):
    #     if not self.train:
    #         return self.basic_SR_collate_fn(x)
    #
    #     item_seq, seq_len, target, aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = zip(*x)
    #
    #     tensor_dict = {}
    #     tensor_dict['item_seq'] = torch.stack(item_seq, 0)
    #     tensor_dict['seq_len'] = torch.stack(seq_len, 0)
    #     tensor_dict['target'] = torch.stack(target, 0)
    #     tensor_dict['aug_seq_1'] = torch.stack(aug_seq_1, 0)
    #     tensor_dict['aug_seq_2'] = torch.stack(aug_seq_2, 0)
    #     tensor_dict['aug_len_1'] = torch.stack(aug_len_1, 0)
    #     tensor_dict['aug_len_2'] = torch.stack(aug_len_2, 0)
    #
    #     return tensor_dict

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)

        item_seq, seq_len, target, aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = default_collate(x)

        self.batch_batch_dict['item_seq'] = item_seq
        self.batch_batch_dict['seq_len'] = seq_len
        self.batch_batch_dict['target'] = target
        self.batch_batch_dict['aug_seq_1'] = aug_seq_1
        self.batch_batch_dict['aug_seq_2'] = aug_seq_2
        self.batch_batch_dict['aug_len_1'] = aug_len_1
        self.batch_batch_dict['aug_len_2'] = aug_len_2

        return self.batch_batch_dict


class CL4SRec_Separated_Dataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(CL4SRec_Separated_Dataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.mask_id = self.num_items
        self.aug_types = config.aug_types
        self.n_views = 2

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        # contrast learning augmented views
        item_seq = self.item_seq[index]
        target = self.label[index]
        aug_type = np.random.choice(self.aug_types,
                                    size=1, replace=True)

        # aug_seq_len = len(item_seq) // 2
        aug_seq_len = int(len(item_seq) / 3)
        gap = 3
        if aug_type[0] == 'crop':
            start = int(random.random() * aug_seq_len)
            aug_seq_1 = item_seq[start:aug_seq_len + start]
            aug_seq_2 = item_seq[start + gap:aug_seq_len + start + gap]
            if len(aug_seq_2) == 0:
                aug_seq_2 = item_seq[-aug_seq_len:]
        else:  # mask
            aug_seq_1 = [self.mask_id] * (len(item_seq) - aug_seq_len) + item_seq[-aug_seq_len:]
            aug_seq_2 = item_seq[:aug_seq_len] + [self.mask_id] * (len(item_seq) - aug_seq_len)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(aug_seq_1, dtype=torch.long),
                       torch.tensor(aug_seq_2, dtype=torch.long),
                       torch.tensor(aug_len_1, dtype=torch.long),
                       torch.tensor(aug_len_2, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        item_seq, seq_len, target, aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = tensor_list

        tensor_dict['item_seq'] = item_seq
        tensor_dict['seq_len'] = seq_len
        tensor_dict['target'] = target
        tensor_dict['aug_seq_1'] = aug_seq_1
        tensor_dict['aug_seq_2'] = aug_seq_2
        tensor_dict['aug_len_1'] = aug_len_1
        tensor_dict['aug_len_2'] = aug_len_2

        return tensor_dict


class CL4SRec_SeCoDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(CL4SRec_SeCoDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.mask_id = self.num_items
        self.aug_types = config.aug_types
        self.stable_ratio = config.stable_ratio
        self.n_views = 2
        self.broken_augmentations = []
        self.safe_augmentations = []
        self.threshold = config.threshold
        self.repeat_aug = None
        self.load_augmentor()

    def load_augmentor(self):
        for aug in self.aug_types:
            if aug == 'mask':
                self.broken_augmentations.append(Mask(gamma=self.config.mask_ratio, mask_id=self.mask_id))
            else:
                self.broken_augmentations.append(AUGMENTATIONS[aug](getattr(self.config, f'{aug}_ratio')))

        # self.safe_augmentations = copy.deepcopy(self.broken_augmentations)
        self.repeat_aug = AUGMENTATIONS['repeat'](getattr(self.config, 'repeat_ratio'))
        # self.safe_augmentations.append(self.repeat_aug)

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        # contrast learning augmented views
        item_seq = self.item_seq[index]
        target = self.label[index]
        total_seq_len = len(item_seq)

        # augmentation
        # semantic_aug_type = np.random.choice([i for i in range(len(self.safe_augmentations))],
        #                                      size=1, replace=True)
        structure_aug_type = np.random.choice([i for i in range(len(self.broken_augmentations))],
                                              size=1, replace=True)

        # if total_seq_len <= self.threshold:  # use repeat augmentation
        #     aug_seq_1 = self.repeat_aug(item_seq)
        # else:
        #     # split
        #     leave_seq_len = math.ceil(self.stable_ratio * total_seq_len)
        #     right_item_seq = item_seq[-leave_seq_len:]
        #     left_item_seq = item_seq[:-leave_seq_len]
        #
        #     aug_seq_1 = self.broken_augmentations[structure_aug_type[0]](left_item_seq)
        #
        #     # concatenation
        #     aug_seq_1 = aug_seq_1 + right_item_seq

        aug_seq_1 = self.repeat_aug(item_seq)
        aug_seq_2 = self.broken_augmentations[structure_aug_type[0]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))

        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(aug_seq_1, dtype=torch.long),
                       torch.tensor(aug_seq_2, dtype=torch.long),
                       torch.tensor(aug_len_1, dtype=torch.long),
                       torch.tensor(aug_len_2, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        item_seq, seq_len, target, aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = tensor_list

        tensor_dict['item_seq'] = item_seq
        tensor_dict['seq_len'] = seq_len
        tensor_dict['target'] = target
        tensor_dict['aug_seq_1'] = aug_seq_1
        tensor_dict['aug_seq_2'] = aug_seq_2
        tensor_dict['aug_len_1'] = aug_len_1
        tensor_dict['aug_len_2'] = aug_len_2

        return tensor_dict


class SeqWithReorderAugDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SeqWithReorderAugDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.aug_size = 1
        self.reorder_augmentor = AUGMENTATIONS['reorder'](getattr(self.config, f'reorder_ratio'))
        self.reorder_augmentor = Reorder(beta=self.config.reorder_ratio)

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        # reordered sequence
        item_seq = self.item_seq[index]
        target = self.label[index]
        reorder_seq = self.reorder_augmentor(item_seq)

        reorder_seq = reorder_seq[-self.max_len:]
        reorder_seq = reorder_seq + [0] * (self.max_len - len(reorder_seq))

        assert len(reorder_seq) == self.max_len

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(reorder_seq, dtype=torch.long))

        return cur_tensors


class SeqWithSessionGraphDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SeqWithSessionGraphDataset, self).__init__(config, data_pair, additional_data_dict, train)

    def __getitem__(self, index):
        item_seq = self.item_seq[index]
        target = self.label[index]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]
        assert len(item_seq) == self.max_len

        # generate adjacency matrix
        max_n_node = self.max_len  # we simply use max length to replace max node number
        unique_node = np.unique(item_seq)
        node = unique_node.tolist() + (max_n_node - len(unique_node)) * [0]  # padded unique nodes
        assert len(node) == self.max_len

        u_A = np.zeros((max_n_node, max_n_node))
        for i in range(len(item_seq) - 1):
            if item_seq[i + 1] == 0:
                break
            u = np.where(unique_node == item_seq[i])[0][
                0]  # np.where return a tuple,so need use [0][0] to show the value
            v = np.where(unique_node == item_seq[i + 1])[0][0]
            u_A[u][v] += 1

        # normalize in-degree
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[u_sum_in == 0] = 1
        u_A_in = (u_A / u_sum_in).T

        # normalize out-degree
        u_sum_out = np.sum(u_A, -1)
        u_sum_out[u_sum_out == 0] = 1
        u_A_out = u_A / u_sum_out.reshape(-1, 1)

        u_A = np.concatenate([u_A_in, u_A_out], -1)
        alias_item_seq = [np.where(unique_node == item)[0][0] for item in item_seq]

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(alias_item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(node, dtype=torch.long),
                       torch.tensor(u_A, dtype=torch.float))

        return cur_tensors

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        item_seq, alias_item_seq, seq_len, target, node, u_A = tensor_list

        tensor_dict['item_seq'] = item_seq
        tensor_dict['alias_item_seq'] = alias_item_seq
        tensor_dict['seq_len'] = seq_len
        tensor_dict['target'] = target
        tensor_dict['node'] = node
        tensor_dict['u_A'] = u_A

        return tensor_dict


class MKMSRDataset(SeqWithSessionGraphDataset):
    """For MKMSR"""

    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(MKMSRDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.entity_num = config.entity_num
        self.train_triplet = additional_data_dict['triple_list']
        self.relation_tph = additional_data_dict['relation_tph']
        self.relation_hpt = additional_data_dict['relation_hpt']
        self.triplet_size = len(self.train_triplet)

    def __getitem__(self, index):
        cur_tensors = list(super().__getitem__(index))
        # kg triplets
        n = index // self.triplet_size
        index -= self.triplet_size * n
        pos_triplet = self.train_triplet[index]

        corrupted_triplet = copy.deepcopy(pos_triplet)

        pr = np.random.random(1)[0]
        relation = int(corrupted_triplet[1])
        p = self.relation_tph[relation] / (
                self.relation_tph[relation] + self.relation_hpt[relation])
        if pr > p:
            # change the head entity
            corrupted_triplet[0] = random.randint(0, self.entity_num - 1)
            while corrupted_triplet[0] == pos_triplet[0]:
                corrupted_triplet[0] = random.randint(0, self.entity_num - 1)
        else:
            # change the tail entity
            corrupted_triplet[2] = random.randint(0, self.entity_num - 1)
            while corrupted_triplet[2] == pos_triplet[2]:
                corrupted_triplet[2] = random.randint(0, self.entity_num - 1)

        cur_tensors.extend([torch.tensor(pos_triplet, dtype=torch.long),
                            torch.tensor(corrupted_triplet, dtype=torch.long)])

        return cur_tensors

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        item_seq, alias_item_seq, seq_len, target, node, u_A, pos_triplet, corrupted_triplet = tensor_list

        tensor_dict['item_seq'] = item_seq
        tensor_dict['alias_item_seq'] = alias_item_seq
        tensor_dict['seq_len'] = seq_len
        tensor_dict['target'] = target
        tensor_dict['node'] = node
        tensor_dict['u_A'] = u_A
        tensor_dict['pos_triplet'] = pos_triplet
        tensor_dict['corrupted_triplet'] = corrupted_triplet

        return tensor_dict


class DuoRecDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(DuoRecDataset, self).__init__(config, data_pair, additional_data_dict, train)
        # self.aug_size = 1
        self.same_target_index_w_self = additional_data_dict['same_target_index']

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        item_seq = self.item_seq[index]
        target = self.label[index]
        same_target_seq = self.get_same_target_seq(target)

        same_target_seq_len = len(same_target_seq) if len(same_target_seq) < self.max_len else self.max_len
        same_target_seq = same_target_seq[-self.max_len:]
        same_target_seq = same_target_seq + [0] * (self.max_len - len(same_target_seq))

        assert len(same_target_seq) == self.max_len

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(same_target_seq, dtype=torch.long),
                       torch.tensor(same_target_seq_len, dtype=torch.long))

        return cur_tensors

    def get_same_target_seq(self, target):
        targets = self.same_target_index_w_self[target]
        sampled_index = np.random.choice(targets)
        sup_pos_seq = self.item_seq[sampled_index]
        sup_pos_target = self.label[sampled_index]

        assert sup_pos_target == target, 'The target of sample seq do not equal to origin seq!'

        return sup_pos_seq

    def collate_fn(self, x):
        """
        Args:
            x: List of tensor, [(x_1, y_1, ..., z_1), ..., (x_B, y_B, ..., z_B)]
        Returns:
            data_dict: stores data for specified model
        """
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        item_seq, seq_len, target, same_target_seq, same_target_seq_len = tensor_list

        tensor_dict['item_seq'] = item_seq
        tensor_dict['seq_len'] = seq_len
        tensor_dict['target'] = target
        tensor_dict['same_target_seq'] = same_target_seq
        tensor_dict['same_target_seq_len'] = same_target_seq_len
        return tensor_dict


class KGSCLDataset(BaseSequentialDataset):
    """
    Use KG relations to guide the data augmentation for contrastive learning.
    """

    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(KGSCLDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.insert_ratio = config.insert_ratio
        self.substitute_ratio = config.substitute_ratio
        self.kg_relation_dict = additional_data_dict['kg_relation_dict']
        self.co_occurrence_dict = additional_data_dict['co_occurrence_dict']

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # raw data
        origin_item_seq = self.item_seq[index]
        target = self.label[index]

        # standardized data
        seq_len = len(origin_item_seq) if len(origin_item_seq) < self.max_len else self.max_len
        item_seq = origin_item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]
        assert len(item_seq) == self.max_len

        # aug seq 1
        aug_seq_1 = self.KG_guided_augmentation(origin_item_seq)
        aug_seq_len_1 = len(aug_seq_1) if len(aug_seq_1) < self.max_len else self.max_len
        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        assert len(aug_seq_1) == self.max_len

        # aug seq 2
        aug_seq_2 = self.KG_guided_augmentation(origin_item_seq)
        aug_seq_len_2 = len(aug_seq_2) if len(aug_seq_2) < self.max_len else self.max_len
        aug_seq_2 = aug_seq_2[-self.max_len:]
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_2) == self.max_len

        # augment target item
        aug_target, pos_item_set = self.target_substitution(target)
        pos_item_set = pos_item_set + [0] * (60 - len(pos_item_set))

        batch_tensors = (torch.tensor(item_seq, dtype=torch.long),
                         torch.tensor(seq_len, dtype=torch.long),
                         torch.tensor(target, dtype=torch.long),
                         torch.tensor(aug_seq_1, dtype=torch.long),
                         torch.tensor(aug_seq_len_1, dtype=torch.long),
                         torch.tensor(aug_seq_2, dtype=torch.long),
                         torch.tensor(aug_seq_len_2, dtype=torch.long),
                         torch.tensor(aug_target, dtype=torch.long),
                         torch.tensor(pos_item_set, dtype=torch.long))

        return batch_tensors

    def KG_guided_augmentation(self, item_seq):
        if random.random() < 0.5:
            return self.KG_insert(item_seq)
        return self.KG_substitute(item_seq)

    def KG_insert(self, item_seq):
        copied_item_seq = copy.deepcopy(item_seq)
        insert_num = int(self.insert_ratio * len(copied_item_seq))
        insert_index = random.sample([i for i in range(len(copied_item_seq))], k=insert_num)
        new_item_seq = []
        for index, item in enumerate(copied_item_seq):
            new_item_seq.append(item)
            if index in insert_index:
                shifted_item = item - 1  # origin item id
                insert_candidates = self.kg_relation_dict[shifted_item]['c']  # c: complement
                if len(insert_candidates) > 0:  # if complement items exist
                    insert_frequency = self.co_occurrence_dict[shifted_item]['c']
                    insert_item = np.random.choice(insert_candidates, size=1, p=insert_frequency)[0]
                    shifted_insert_item = insert_item + 1
                    new_item_seq.append(shifted_insert_item)
                else:
                    new_item_seq.append(item)  # Item-repeat
        return new_item_seq

    def KG_substitute(self, item_seq):
        copied_item_seq = copy.deepcopy(item_seq)
        substitute_num = int(self.substitute_ratio * len(copied_item_seq))
        substitute_index = random.sample([i for i in range(len(copied_item_seq))], k=substitute_num)
        new_item_seq = []
        for index, item in enumerate(copied_item_seq):
            if index in substitute_index:
                shifted_item = item - 1
                substitute_candidates = self.kg_relation_dict[shifted_item]['s']  # s: substitute
                if len(substitute_candidates) > 0:  # if substitute items exist
                    substitute_frequency = self.co_occurrence_dict[shifted_item]['s']
                    substitute_item = np.random.choice(substitute_candidates, size=1, p=substitute_frequency)[0]
                    shifted_substitute_item = substitute_item + 1
                    new_item_seq.append(shifted_substitute_item)
                else:
                    new_item_seq.append(item)
                    new_item_seq.append(item)  # Item-repeat
            else:
                new_item_seq.append(item)
        return new_item_seq

    def target_substitution(self, target_item):
        shifted_target_item = target_item - 1
        substitute_candidates = self.kg_relation_dict[shifted_target_item]['s']  # s: substitute
        if len(substitute_candidates) == 0:
            return target_item, []  # if no substitute items, don't change
        substitute_frequency = self.co_occurrence_dict[shifted_target_item]['s']
        substitute_item = np.random.choice(substitute_candidates, size=1, p=substitute_frequency)[0]
        shifted_substitute_item = substitute_item + 1
        substitute_candidates = [item + 1 for item in substitute_candidates]
        substitute_candidates.remove(shifted_substitute_item)
        return shifted_substitute_item, substitute_candidates

    def collate_fn(self, x):
        """
        Args:
            x: List of tensor, [(x_1, y_1, ..., z_1), ..., (x_B, y_B, ..., z_B)]
        Returns:
            data_dict: stores data for specified model
        """
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_list = default_collate(x)

        self.batch_batch_dict['item_seq'] = tensor_list[0]
        self.batch_batch_dict['seq_len'] = tensor_list[1]
        self.batch_batch_dict['target'] = tensor_list[2]
        self.batch_batch_dict['v2v_aug_seq'] = tensor_list[3]
        self.batch_batch_dict['v2v_aug_len'] = tensor_list[4]
        self.batch_batch_dict['v2t_aug_seq'] = tensor_list[5]
        self.batch_batch_dict['v2t_aug_len'] = tensor_list[6]
        self.batch_batch_dict['aug_target'] = tensor_list[7]
        self.batch_batch_dict['pos_item_set'] = tensor_list[8]

        return self.batch_batch_dict


class KGDataAugDataset(KGSCLDataset):
    """
    Use KG relation to do data and target augmentation, and just for rec task.
    """

    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(KGDataAugDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.target_aug = config.target_aug

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for rec training
        origin_item_seq = self.item_seq[index]
        target = self.label[index]

        seq_len = len(origin_item_seq) if len(origin_item_seq) < self.max_len else self.max_len
        item_seq = origin_item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        aug_seq = self.KG_guided_augmentation(origin_item_seq)

        aug_seq_len = len(aug_seq) if len(aug_seq) < self.max_len else self.max_len
        aug_seq = aug_seq[-self.max_len:]
        aug_seq = aug_seq + [0] * (self.max_len - len(aug_seq))

        assert len(aug_seq) == self.max_len
        if not self.target_aug:
            cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                           torch.tensor(seq_len, dtype=torch.long),
                           torch.tensor(target, dtype=torch.long),
                           torch.tensor(aug_seq, dtype=torch.long),
                           torch.tensor(aug_seq_len, dtype=torch.long))

            return cur_tensors
        else:
            target_aug = self.target_substitution(target)
            cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                           torch.tensor(seq_len, dtype=torch.long),
                           torch.tensor(target, dtype=torch.long),
                           torch.tensor(aug_seq, dtype=torch.long),
                           torch.tensor(aug_seq_len, dtype=torch.long),
                           torch.tensor(target_aug, dtype=torch.long))
            return cur_tensors

    def collate_fn(self, x):
        """
        Args:
            x: List of tensor, [(x_1, y_1, ..., z_1), ..., (x_B, y_B, ..., z_B)]
        Returns:
            data_dict: stores data for specified model
        """
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        if self.target_aug:
            item_seq, seq_len, target, aug_item_seq, aug_seq_len, target_aug = tensor_list
            tensor_dict['target_aug'] = target_aug
        else:
            item_seq, seq_len, target, aug_item_seq, aug_seq_len = tensor_list

        total_item_seq = torch.cat([item_seq, aug_item_seq], 0)
        total_seq_len = torch.cat([seq_len, aug_seq_len], 0)
        total_target = torch.cat([target, target_aug], 0) if self.target_aug else target.repeat(2)
        tensor_dict['item_seq'] = total_item_seq
        tensor_dict['seq_len'] = total_seq_len
        tensor_dict['target'] = total_target
        return tensor_dict


class SeqWithReorderRepeatAugDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SeqWithReorderRepeatAugDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.aug_size = 1
        self.reorder_augmentor = AUGMENTATIONS['reorder'](getattr(self.config, 'reorder_ratio'))
        self.repeat_augmentor = AUGMENTATIONS['repeat'](getattr(self.config, 'repeat_ratio'))

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        item_seq = self.item_seq[index]
        target = self.label[index]
        reorder_seq = self.reorder_augmentor(item_seq)
        repeat_seq = self.repeat_augmentor(item_seq)
        reorder_repeat_seq = self.repeat_augmentor(reorder_seq)

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]
        # pure reordered seq as training sample too
        reorder_seq = reorder_seq[-self.max_len:]
        reorder_seq = reorder_seq + [0] * (self.max_len - len(reorder_seq))

        assert len(reorder_seq) == self.max_len
        assert len(item_seq) == self.max_len

        # cl aug seq
        repeat_seq_len = len(repeat_seq) if len(repeat_seq) < self.max_len else self.max_len
        repeat_seq = repeat_seq[-self.max_len:]
        repeat_seq = repeat_seq + [0] * (self.max_len - len(repeat_seq))

        reorder_repeat_len = len(reorder_repeat_seq) if len(reorder_repeat_seq) < self.max_len else self.max_len
        reorder_repeat_seq = reorder_repeat_seq[-self.max_len:]
        reorder_repeat_seq = reorder_repeat_seq + [0] * (self.max_len - len(reorder_repeat_seq))

        assert len(repeat_seq) == self.max_len
        assert len(reorder_repeat_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(reorder_seq, dtype=torch.long),
                       torch.tensor(repeat_seq, dtype=torch.long),
                       torch.tensor(repeat_seq_len, dtype=torch.long),
                       torch.tensor(reorder_repeat_seq, dtype=torch.long),
                       torch.tensor(reorder_repeat_len, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        """
        Args:
            x: List of tensor, [(x_1, y_1, ..., z_1), ..., (x_B, y_B, ..., z_B)]
        Returns:
            data_dict: stores data for specified model
        """
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        item_seq, seq_len, target, reorder_seq, repeat_seq, \
        repeat_seq_len, reorder_repeat_seq, reorder_repeat_seq_len = tensor_list

        total_item_seq = torch.cat([item_seq, reorder_seq], 0)
        total_seq_len = seq_len.repeat(2)  # [B * 2]
        total_target = target.repeat(2)  # [B * 2]
        total_aug_seq = torch.cat([repeat_seq, reorder_repeat_seq], 0)
        total_aug_len = torch.cat([repeat_seq_len, reorder_repeat_seq_len], 0)

        tensor_dict['item_seq'] = total_item_seq
        tensor_dict['seq_len'] = total_seq_len
        tensor_dict['target'] = total_target
        tensor_dict['aug_item_seq'] = total_aug_seq
        tensor_dict['aug_seq_len'] = total_aug_len

        return tensor_dict


class SeqWithRepeatAugDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SeqWithRepeatAugDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.aug_size = 1
        self.repeat_augmentor = AUGMENTATIONS['repeat'](getattr(self.config, 'repeat_ratio'))

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        item_seq = self.item_seq[index]
        target = self.label[index]
        repeat_seq = self.repeat_augmentor(item_seq)

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        # cl aug seq
        repeat_seq_len = len(repeat_seq) if len(repeat_seq) < self.max_len else self.max_len
        repeat_seq = repeat_seq[-self.max_len:]
        repeat_seq = repeat_seq + [0] * (self.max_len - len(repeat_seq))

        assert len(repeat_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(repeat_seq, dtype=torch.long),
                       torch.tensor(repeat_seq_len, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        """
        Args:
            x: List of tensor, [(x_1, y_1, ..., z_1), ..., (x_B, y_B, ..., z_B)]
        Returns:
            data_dict: stores data for specified model
        """
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        item_seq, seq_len, target, repeat_seq, repeat_seq_len = tensor_list

        tensor_dict['item_seq'] = item_seq
        tensor_dict['seq_len'] = seq_len
        tensor_dict['target'] = target
        tensor_dict['aug_item_seq'] = repeat_seq
        tensor_dict['aug_seq_len'] = repeat_seq_len

        return tensor_dict


class UniContraDataset(DuoRecDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(UniContraDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.reorder_augmentor = AUGMENTATIONS['reorder'](getattr(self.config, 'reorder_ratio'))
        self.repeat_augmentor = AUGMENTATIONS['repeat'](getattr(self.config, 'repeat_ratio'))

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # origin train data
        item_seq = self.item_seq[index]
        reorder_seq = self.reorder_augmentor(item_seq)  # reorder augmentation
        target = self.label[index]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        reorder_seq = reorder_seq[-self.max_len:]
        reorder_seq = reorder_seq + [0] * (self.max_len - len(reorder_seq))

        assert len(reorder_seq) == self.max_len
        assert len(item_seq) == self.max_len

        # repeat view
        repeat_seq = self.repeat_augmentor(item_seq)
        reorder_repeat_seq = self.repeat_augmentor(reorder_seq)

        repeat_seq_len = len(repeat_seq) if len(repeat_seq) < self.max_len else self.max_len
        repeat_seq = repeat_seq[-self.max_len:]
        repeat_seq = repeat_seq + [0] * (self.max_len - len(repeat_seq))

        reorder_repeat_len = len(reorder_repeat_seq) if len(reorder_repeat_seq) < self.max_len else self.max_len
        reorder_repeat_seq = reorder_repeat_seq[-self.max_len:]
        reorder_repeat_seq = reorder_repeat_seq + [0] * (self.max_len - len(reorder_repeat_seq))

        assert len(repeat_seq) == self.max_len
        assert len(reorder_repeat_seq) == self.max_len

        # supervised pos view
        super_pos_seq = self.get_same_target_seq(target)
        reorder_super_pos_seq = self.get_same_target_seq(target)

        super_seq_len = len(super_pos_seq) if len(super_pos_seq) < self.max_len else self.max_len
        super_pos_seq = super_pos_seq[-self.max_len:]
        super_pos_seq = super_pos_seq + [0] * (self.max_len - len(super_pos_seq))

        reorder_super_seq_len = len(reorder_super_pos_seq) if len(
            reorder_super_pos_seq) < self.max_len else self.max_len
        reorder_super_pos_seq = reorder_super_pos_seq[-self.max_len:]
        reorder_super_pos_seq = reorder_super_pos_seq + [0] * (self.max_len - len(reorder_super_pos_seq))

        assert len(super_pos_seq) == self.max_len
        assert len(reorder_super_pos_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(reorder_seq, dtype=torch.long),
                       torch.tensor(repeat_seq, dtype=torch.long),
                       torch.tensor(repeat_seq_len, dtype=torch.long),
                       torch.tensor(reorder_repeat_seq, dtype=torch.long),
                       torch.tensor(reorder_repeat_len, dtype=torch.long),
                       torch.tensor(super_pos_seq, dtype=torch.long),
                       torch.tensor(super_seq_len, dtype=torch.long),
                       torch.tensor(reorder_super_pos_seq, dtype=torch.long),
                       torch.tensor(reorder_super_seq_len, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        """
        Args:
            x: List of tensor, [(x_1, y_1, ..., z_1), ..., (x_B, y_B, ..., z_B)]
        Returns:
            data_dict: stores data for specified model
        """
        if not self.train:
            return self.basic_SR_collate_fn(x)

        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]

        item_seq, seq_len, target, reorder_seq, repeat_seq, repeat_seq_len, \
        reorder_repeat_seq, reorder_repeat_len, super_pos_seq, super_seq_len, \
        reorder_super_pos_seq, reorder_super_seq_len = tensor_list

        # recommendation seq
        total_item_seq = torch.cat([item_seq, reorder_seq], 0)
        total_seq_len = seq_len.repeat(2)  # [B * 2]
        total_target = target.repeat(2)  # [B * 2]

        # augmented seq
        total_aug_seq = torch.cat([repeat_seq, reorder_repeat_seq], 0)
        total_aug_len = torch.cat([repeat_seq_len, reorder_repeat_len], 0)

        # supervised positive seq
        total_pos_seq = torch.cat([super_pos_seq, reorder_super_pos_seq], 0)
        total_pos_len = torch.cat([super_seq_len, reorder_super_seq_len], 0)

        tensor_dict['item_seq'] = total_item_seq
        tensor_dict['seq_len'] = total_seq_len
        tensor_dict['target'] = total_target
        tensor_dict['aug_item_seq'] = total_aug_seq
        tensor_dict['aug_seq_len'] = total_aug_len
        tensor_dict['pos_item_seq'] = total_pos_seq
        tensor_dict['pos_seq_len'] = total_pos_len

        return tensor_dict


class ReversedSeqDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(ReversedSeqDataset, self).__init__(config, data_pair, additional_data_dict, train)

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        item_seq = self.item_seq[index]
        reversed_seq = list(reversed(item_seq))
        target = self.label[index]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        reversed_seq = reversed_seq[-self.max_len:]
        reversed_seq = reversed_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        return (torch.tensor(reversed_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))


class ContraRecDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(ContraRecDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.aug_type = config.aug_type
        self.mask_token = self.num_items
        self.beta_a = config.beta_a
        self.beta_b = config.beta_a

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        # contrast learning augmented views
        item_seq = self.item_seq[index]
        target = self.label[index]
        aug_seq_1 = self.augment(item_seq)
        aug_seq_2 = self.augment(item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]
        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))

        # recommendation item sequence
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(aug_seq_1, dtype=torch.long),
                       torch.tensor(aug_seq_2, dtype=torch.long))

        return cur_tensors

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        item_seq, seq_len, target, aug_seq_1, aug_seq_2 = tensor_list

        tensor_dict['item_seq'] = item_seq
        tensor_dict['seq_len'] = seq_len
        tensor_dict['target'] = target
        tensor_dict['aug_seq_1'] = aug_seq_1
        tensor_dict['aug_seq_2'] = aug_seq_2

        return tensor_dict

    def reorder_op(self, seq):
        ratio = np.random.beta(a=self.beta_a, b=self.beta_b)
        select_len = int(len(seq) * ratio)
        start = np.random.randint(0, len(seq) - select_len + 1)
        idx_range = np.arange(len(seq))
        np.random.shuffle(idx_range[start: start + select_len])
        return seq[idx_range].tolist()

    def mask_op(self, seq):
        ratio = np.random.beta(a=self.beta_a, b=self.beta_b)
        selected_len = int(len(seq) * ratio)
        mask = np.full(len(seq), False)
        mask[:selected_len] = True
        np.random.shuffle(mask)
        seq[mask] = self.mask_token
        return seq.tolist()

    def augment(self, seq):
        aug_seq = np.array(seq).copy()
        if self.aug_type == 'random':
            if np.random.rand() > 0.5:
                return self.mask_op(aug_seq)
            else:
                return self.reorder_op(aug_seq)
        elif self.aug_type == 'reorder':
            return self.reorder_op(aug_seq)
        elif self.aug_type == 'mask':
            return self.mask_op(aug_seq)
        else:
            raise KeyError(f"Invalid aug_type:{self.aug_type}. Choose from ['random', 'reorder', 'mask'].")


class MISPPretrainDataset(Dataset):
    """
    Masked Item & Segment Prediction (MISP)
    """

    def __init__(self, config, data_pair, additional_data_dict=None):
        self.mask_id = config.num_items
        self.mask_ratio = config.mask_ratio
        self.num_items = config.num_items + 1
        self.config = config
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.max_len = config.max_len
        self.long_sequence = []

        for seq in self.item_seq:
            self.long_sequence.extend(seq)

    def __len__(self):
        return len(self.item_seq)

    def __getitem__(self, index):
        sequence = self.item_seq[index]  # pos_items

        # Masked Item Prediction
        masked_item_sequence = []
        neg_items = []
        pos_items = sequence

        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.mask_ratio:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.num_items))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)
        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.num_items))

        assert len(masked_item_sequence) == len(sequence)
        assert len(pos_items) == len(sequence)
        assert len(neg_items) == len(sequence)

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.mask_id] * sample_length + sequence[
                                                                                             start_id + sample_length:]
            pos_segment = [self.mask_id] * start_id + pos_segment + [self.mask_id] * (
                    len(sequence) - (start_id + sample_length))
            neg_segment = [self.mask_id] * start_id + neg_segment + [self.mask_id] * (
                    len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # crop sequence
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]
        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = masked_item_sequence + [0] * pad_len
        pos_items = pos_items + [0] * pad_len
        neg_items = neg_items + [0] * pad_len
        masked_segment_sequence = masked_segment_sequence + [0] * pad_len
        pos_segment = pos_segment + [0] * pad_len
        neg_segment = neg_segment + [0] * pad_len

        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long))
        return cur_tensors

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        masked_item_sequence, pos_items, neg_items, \
        masked_segment_sequence, pos_segment, neg_segment = tensor_list

        tensor_dict['masked_item_sequence'] = masked_item_sequence
        tensor_dict['pos_items'] = pos_items
        tensor_dict['neg_items'] = neg_items
        tensor_dict['masked_segment_sequence'] = masked_segment_sequence
        tensor_dict['pos_segment'] = pos_segment
        tensor_dict['neg_segment'] = neg_segment

        return tensor_dict


class MIMPretrainDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None):
        self.config = config
        self.aug_types = config.aug_types
        self.mask_id = config.num_items

        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.max_len = config.max_len
        self.n_views = 2
        self.augmentations = []
        self.load_augmentor()

    def load_augmentor(self):
        for aug in self.aug_types:
            if aug == 'mask':
                self.augmentations.append(Mask(gamma=self.config.mask_ratio, mask_id=self.mask_id))
            else:
                self.augmentations.append(AUGMENTATIONS[aug](getattr(self.config, f'{aug}_ratio')))

    def __getitem__(self, index):
        aug_type = np.random.choice([i for i in range(len(self.augmentations))],
                                    size=self.n_views, replace=False)
        item_seq = self.item_seq[index]
        aug_seq_1 = self.augmentations[aug_type[0]](item_seq)
        aug_seq_2 = self.augmentations[aug_type[1]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        aug_seq_tensors = (torch.tensor(aug_seq_1, dtype=torch.long),
                           torch.tensor(aug_seq_2, dtype=torch.long),
                           torch.tensor(aug_len_1, dtype=torch.long),
                           torch.tensor(aug_len_2, dtype=torch.long))

        return aug_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        aug_seq_1, aug_seq_2, aug_len_1, aug_len_2 = tensor_list

        tensor_dict['aug_seq_1'] = aug_seq_1
        tensor_dict['aug_seq_2'] = aug_seq_2
        tensor_dict['aug_len_1'] = aug_len_1
        tensor_dict['aug_len_2'] = aug_len_2

        return tensor_dict


class PIDPretrainDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None):
        self.num_items = config.num_items
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.config = config
        self.max_len = config.max_len
        self.pseudo_ratio = config.pseudo_ratio

    def __getitem__(self, index):
        item_seq = self.item_seq[index]
        pseudo_seq = []
        target = []

        for item in item_seq:
            if random.random() < self.pseudo_ratio:
                pseudo_item = neg_sample(item_seq, self.num_items)
                pseudo_seq.append(pseudo_item)
                target.append(0)
            else:
                pseudo_seq.append(item)
                target.append(1)

        pseudo_seq = pseudo_seq[-self.max_len:]
        target = target[-self.max_len:]

        pseudo_seq = pseudo_seq + [0] * (self.max_len - len(pseudo_seq))
        target = target + [0] * (self.max_len - len(target))
        assert len(pseudo_seq) == self.max_len
        assert len(target) == self.max_len
        pseudo_seq_tensors = (torch.tensor(pseudo_seq, dtype=torch.long),
                              torch.tensor(target, dtype=torch.float))

        return pseudo_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)

    def collate_fn(self, x):
        tensor_dict = {}
        tensor_list = [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]
        pseudo_seq, target = tensor_list

        tensor_dict['pseudo_seq'] = pseudo_seq
        tensor_dict['target'] = target

        return tensor_dict


if __name__ == '__main__':
    index = np.arange(10)
    res = np.random.choice(index, size=1)
    print(index)
    print(res)
