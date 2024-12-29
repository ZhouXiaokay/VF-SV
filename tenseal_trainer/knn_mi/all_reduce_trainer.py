import time
import math

import numpy as np

from utils.distance import square_euclidean_np
from utils.comm_op import sum_sqrt_all_reduce
from transmission.tenseal_shapley.tenseal_shapley_client import ShapleyClient



class AllReduceTrainer(object):

    def __init__(self, args, data, targets):
        self.args = args
        self.data = data
        self.targets = targets

        unique, counts = np.unique(self.targets, return_counts=True)
        self.label_counts = dict(zip(unique, counts))
        self.server_addr = args.a_server_address
        self.client = ShapleyClient(self.server_addr, args)

    def transmit(self, vector):
        summed_vector = self.client.transmit(vector, operator='sum_all_reduce')
        # print(summed_vector)
        return summed_vector



    @staticmethod
    def digamma(x):
        return math.log(x, math.e) - 0.5 / x

    def find_top_k(self, test_data, test_target, k, group_flags):
        start_time = time.time()
        # print(">>> start find top-{} <<<".format(k))

        local_dist_start = time.time()
        is_attend = group_flags[self.args.rank]
        local_dist = square_euclidean_np(self.data, test_data)
        # print("local distance shape: {}".format(local_dist.shape))
        if is_attend == 0:
            local_dist = np.zeros_like(local_dist)
        # print("{} local distance: {}".format(len(local_dist), local_dist[:10]))
        local_dist_time = time.time() - local_dist_start

        comm_start = time.time()
        global_dist = self.transmit(local_dist)
        # print("{} global distance: {}".format(len(global_dist), global_dist[:10]))
        comm_time = time.time() - comm_start

        # select_top_start = time.time()
        sorted_ids = np.argsort(global_dist)
        # top_k_ids = sorted_ids[:self.args.k]
        # top_k_dist = global_dist[top_k_ids]
        # select_top_time = time.time() - select_top_start

        N = len(self.data)
        N_i = self.label_counts[test_target]
        m_i = self.cal_m_q(test_target, k, sorted_ids)

        mi_value = self.digamma(N) - self.digamma(N_i) + self.digamma(k) - self.digamma(m_i)

        # if self.args.rank == 0:
        #     print("indices of k near neighbor = {}".format(top_k_ids))
        #     print("distance of k near neighbor = {}".format(top_k_dist))

        return mi_value

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
        return all_label_count