"""
# -*- coding: utf-8 -*-
for the importance of each client, we use MCI-MI of all groups
"""

import time
import math
import sys
import numpy as np
import torch
import torch.distributed as dist

from utils.distance import square_euclidean_np
from utils.comm_op import gather, sum_sqrt_all_reduce, sum_all_reduce
from utils.fagin_utils import suggest_size,  master_count_label


def get_utility_key(client_attendance):
    key = 0
    for i in reversed(client_attendance):
        key = 2 * key + i
    return key


def utility_key_to_groups(key, world_size):
    client_attendance = [0] * world_size
    for i in range(world_size):
        flag = key % 2
        client_attendance[i] = flag
        key = key // 2
    return client_attendance


class LSHAdaptiveFaginBatchTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.n_data = len(data)
        self.targets = targets
        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))
        self.n_candidates = []
        self.n_sizes = []

    # def get_size(self):
    #     sys.getsizeof()

    @staticmethod
    def digamma(x):
        return math.log(x, math.e) - 0.5 / x

    def find_top_k(self, test_data, test_target, k, lsh_candidates ,group_keys):
        start_time = time.time()
        if self.args.rank == 0:
            print(">>> start find top-{} <<<".format(k))
        n_lsh_candidates = len(lsh_candidates)
        local_dist_start = time.time()
        lsh_candidate_data = self.data[lsh_candidates]
        local_dist = square_euclidean_np(lsh_candidate_data, test_data)
        local_dist_time = time.time() - local_dist_start

        sort_start = time.time()
        local_dist_ind = np.argsort(local_dist)
        true_lsh_local_dist_ind = np.array(lsh_candidates)

        # print("local dist index = {}".format(local_dist_ind[:10]))
        # print("local dist = {}".format(local_dist[local_dist_ind[:10]]))
        sort_time = time.time() - sort_start

        send_size = suggest_size(n_lsh_candidates, self.args.k, self.args.world_size)
        if self.args.rank == 0:
            print("suggest batch size = {}".format(send_size))
        send_ind = 0

        fagin_start = time.time()
        gather_time = 0
        bc_time = 0
        count_time = 0
        top_k_ids = []
        counts = [0 for _ in range(n_lsh_candidates)]
        cur_n_top = 0
        n_iter = 0
        rank = dist.get_rank()

        while cur_n_top < self.args.k and send_ind <= n_lsh_candidates:
            gather_start = time.time()
            new_lists = gather(local_dist_ind[send_ind:min(n_lsh_candidates, send_ind + send_size)])
            gather_time += time.time() - gather_start
            send_ind += send_size
            if rank == 0:
                count_start = time.time()
                master_count_label(new_lists, counts, top_k_ids, self.args.k, self.targets, test_target)
                count_time += time.time() - count_start
                bc_start = time.time()
                cur_n_top = len(top_k_ids)
                dist.broadcast(torch.tensor(cur_n_top), 0)
                bc_time += time.time() - bc_start
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
            else:
                bc_start = time.time()
                tmp_tensor = torch.tensor(0)
                dist.broadcast(tmp_tensor, 0)
                bc_time += time.time() - bc_start
                cur_n_top = tmp_tensor.item()
                # print("iter {}, scan {} rows, current top k = {}".format(n_iter, send_size, cur_n_top))
                n_iter += 1
                # dist.barrier()
        fagin_time = time.time() - fagin_start

        # get candidates for top-k, i.e, the instances seen so far in fagin
        candidate_start = time.time()
        n_candidate = 0
        candidate_ind = []
        if rank == 0:
            candidate_ind = [i for i, e in enumerate(counts) if e > 0]
            n_candidate = len(candidate_ind)
            # print("number of candidates = {}".format(n_candidate))
            dist.broadcast(torch.tensor(n_candidate), 0)
            dist.broadcast(torch.tensor(candidate_ind, dtype=torch.int32), 0)
        else:
            tmp_tensor = torch.tensor(0)
            dist.broadcast(tmp_tensor, 0)
            n_candidate = tmp_tensor.item()
            # print("number of candidates = {}".format(n_candidate))
            tmp_tensor = torch.zeros([n_candidate], dtype=torch.int32)
            dist.broadcast(tmp_tensor, 0)
            candidate_ind = tmp_tensor.tolist()
            # print("top-k candidates = {}".format(candidate_ind))
            # print("number of candidates = {}".format(n_candidate))
        candidate_time = time.time() - candidate_start

        self.n_candidates.append(len(candidate_ind))
        # sync candidates for top-k, i.e, the instances seen so far in fagin
        candidate_dist_start = time.time()
        candidate_local_dist = local_dist[candidate_ind]

        # true_candidate_ind = local_dist_ind[candidate_ind]
        # true_fagin_candidate_ind = true_lsh_local_dist_ind[candidate_ind]

        # for each group cal its group global distance
        group_candidate_dist_list = []
        for key in group_keys:
            group_flags = utility_key_to_groups(key, self.args.world_size)
            group_local_dist = group_flags[rank] * candidate_local_dist
            group_dist = sum_sqrt_all_reduce(group_local_dist)
            group_candidate_dist_list.append(group_dist)
        group_candidate_dist = np.array(group_candidate_dist_list)

        # sort group distance
        select_top_start = time.time()
        groups_sorted_ids = np.argsort(group_candidate_dist, axis=1)
        fagin_candidate_ind = np.array(candidate_ind)[groups_sorted_ids]

        true_fagin_candidate_ind = true_lsh_local_dist_ind[fagin_candidate_ind]

        # all_groups_sorted_ids = np.array(true_fagin_candidate_ind)[all_groups_sorted_ids]

        sort_time = time.time() - sort_start

        # calculate label
        count_label_start = time.time()


        groups_mi_values = []
        groups_mi_values_dict = {key: 0. for key in group_keys}

        for ids in range(len(group_keys)):
            group_sorted_ids = true_fagin_candidate_ind[ids]
            N = len(self.data)
            N_i = self.label_counts[test_target]
            m_i = self.cal_m_q(test_target, k, group_sorted_ids)
            # if rank == 0:
            #     print("group {} m_i = {}".format(group_keys[ids], m_i))
            mi_value = self.digamma(N) - self.digamma(N_i) + self.digamma(k) - self.digamma(m_i)

            # groups_mi_values.append(mi_value)
            groups_mi_values_dict[group_keys[ids]] = mi_value
        # return groups_mi_values
        return groups_mi_values_dict

    def cal_m_q(self, test_target, k, group_ids):
        cur_label_top_k_ids = []
        cur_label_count = 0
        all_label_count = 0
        for ids in group_ids:
            all_label_count += 1
            if self.targets[ids] == test_target:
                cur_label_top_k_ids.append(ids)
                cur_label_count += 1
                if cur_label_count == k:
                    break

        return all_label_count