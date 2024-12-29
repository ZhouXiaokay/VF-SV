import time
import math

import numpy as np

from utils.distance import square_euclidean_np
from utils.comm_op import sum_sqrt_all_reduce
import torch.distributed as dist

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


class AllReduceBatchTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets

        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))


    # the leader decides whether to stop fagin when find k-nearest
    # aggregates distance
    # the leader calculates the number of this label and all samples
    # calculate I for this label
    # calculate average I for all labels

    @staticmethod
    def digamma(x):
        return math.log(x, math.e) - 0.5 / x

    def find_top_k(self, test_data, test_target, k, group_keys):
        start_time = time.time()
        # print(">>> start find top-{} <<<".format(k))

        local_dist_start = time.time()
        # is_attend = group_flags[self.args.rank]
        local_dist = square_euclidean_np(self.data, test_data)
        # print("local distance shape: {}".format(local_dist.shape))
        # if is_attend == 0:
        #     local_dist = np.zeros_like(local_dist)
        # print("{} local distance: {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start
        rank = dist.get_rank()
        group_dist_list = []
        comm_start = time.time()
        for key in group_keys:
            group_flags = utility_key_to_groups(key, self.args.world_size)
            group_local_dist = group_flags[rank] * local_dist
            group_dist = sum_sqrt_all_reduce(group_local_dist)
            group_dist_list.append(group_dist)
        group_dist = np.array(group_dist_list)

        # print("{} global distance: {}".format(len(global_dist), global_dist[:10]))
        comm_time = time.time() - comm_start

        groups_sorted_ids = np.argsort(group_dist, axis=1)
        # if self.args.rank == 0:
        #     print("group_dist = ", group_dist[:, :10])
        #     print("groups_sorted_ids = ", groups_sorted_ids[:, :10])


        group_mi_values = []
        for ids in range(len(group_keys)):
            cur_group_id = groups_sorted_ids[ids]
            N = len(self.data)
            N_i = self.label_counts[test_target]
            m_i = self.cal_m_q(test_target, k, cur_group_id)
            if rank == 0:
                print("group {} m_i = {}".format(group_keys[ids], m_i))
            mi_value = self.digamma(N) - self.digamma(N_i) + self.digamma(k) - self.digamma(m_i)

            group_mi_values.append(mi_value)

        return np.array(group_mi_values)

    def cal_m_q(self, test_target, k, sorted_ids):
        cur_label_top_k_ids = []
        cur_label_count = 0
        all_label_count = 0
        for ids in sorted_ids:
            # candidate_id = group_ids[i]
            all_label_count += 1
            if self.targets[ids] == test_target:
                cur_label_top_k_ids.append(ids)
                cur_label_count += 1
                if cur_label_count == k:
                    break
        # if self.args.rank == 0:
        #     print("all_label_count = ", all_label_count)
        #     print("cur_label_top_k_ids = ", cur_label_top_k_ids)
        #     print("sorted_ids = ", sorted_ids[:10])
        return all_label_count