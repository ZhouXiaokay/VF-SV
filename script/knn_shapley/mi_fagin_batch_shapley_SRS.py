"""
Estimate the Shapley value of each participant based on the simple random sampling (SRS) method.
The sampled permutation is fixed before estimation.
# utf-8
# Python version: 3.6
# author: Kay
"""

import time
import sys
import random
import math
from conf import global_args_parser
import numpy as np
from torch.multiprocessing import Process

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

# sys.path.append("../../")
# from data_loader.data_partition import load_dummy_partition_with_label
from data_loader.load_data import (load_dummy_partition_with_label,choose_dataset, load_dependent_data,
                                   load_dummy_partition_by_correlation)
from trainer.knn_mi.mi_fagin_batch_trainer import FaginBatchTrainer
from utils.helpers import seed_torch
from sklearn.model_selection import train_test_split


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


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


def run(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    seed_torch()
    if args.rank == 0:
        print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank

    # file_name = "{}/{}_{}".format(args.root, rank, world_size)
    # print("read file {}".format(file_name))
    # dataset = choose_dataset('web')

    load_start = time.time()
    # data, targets = load_dummy_partition_with_label(dataset, args.num_clients, rank)
    data, targets = load_dependent_data(rank, args.num_clients, seed=args.seed)
    # data, targets = load_dummy_partition_by_correlation(dataset=dataset, client_rank=rank,
    #                                                     num_clients=args.num_clients)
    if args.rank == 0:
        print("load data part cost {} s".format(time.time() - load_start))
    n_data = len(data)
    if args.rank == 0:
        print("number of data = {}".format(n_data))

    # # shuffle the data to split train data and test data
    # shuffle_ind = np.arange(n_data)
    # np.random.shuffle(shuffle_ind)
    # if args.rank == 0:
    #     print("test data indices: {}".format(shuffle_ind[:args.n_test]))
    # data = data[shuffle_ind]
    # targets = targets[shuffle_ind]
    num_data = len(data)
    n_test = int(num_data * args.test_ratio)
    # n_test = 10
    train_data = data[n_test:]
    train_targets = targets[n_test:]
    test_data = data[:n_test]
    test_targets = targets[:n_test]

    # train_data, test_data, train_targets, test_targets = train_test_split(data, targets,
    #                                                                       test_size=args.test_ratio,
    #                                                                       random_state=args.seed)

    # MI of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    trainer = FaginBatchTrainer(args, train_data, train_targets)

    # cal utility of all group_keys, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    group_keys = [i for i in range(start_key, end_key + 1)]
    mi_values = np.zeros(len(group_keys))
    for i in range(n_test):
        if rank == 0:
            print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        # trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        cur_mi_values = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        mi_values += cur_mi_values
    utility_start = time.time()
    mi_values = mi_values / n_test

    for group_key in range(start_key, end_key + 1):
        g_mi_value = mi_values[group_key-1]

        utility_value[group_key] = g_mi_value
    n_utility_round += 1
    if args.rank == 0:
        print("calculate utility cost {:.2f} s, total round {}".format(time.time() - utility_start, n_utility_round))

    sample_group_keys = random.choices(group_keys, k=args.sample_size)
    sample_group_keys = sorted(sample_group_keys)

    # cal factorial
    factor = 1/math.factorial(args.world_size)

    # shapley value of all clients
    shapley_value = [0.0] * world_size
    n_shapley_round = 0
    # cal shapley value of each client
    shapley_start = time.time()
    for group_key in sample_group_keys:
        for i in range(world_size):
            group_flags = utility_key_to_groups(group_key, world_size)
            group_size = sum(group_flags)
            group_flags[i] = 1
            u_with = utility_value[group_key]
            u_without = 0.0
            if group_size >1:
                group_flags[i] = 0
                group_key = get_utility_key(group_flags)
                u_without = utility_value[group_key]
            score = u_with - u_without
            shapley_value[i] += score * factor
            n_shapley_round += 1


    if args.rank == 0:
        print("calculate shapley value cost {:.2f} s".format(time.time() - shapley_start))
        print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

    shapley_ind = np.argsort(np.array(shapley_value))
    if args.rank == 0:
        print("client ranking = {}".format(shapley_ind.tolist()[::-1]))


def init_processes(arg, fn):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend=arg.backend,
                            init_method=arg.init_method,
                            rank=arg.rank,
                            world_size=arg.world_size)
    fn(arg)


if __name__ == '__main__':

    processes = []
    # torch.multiprocessing.set_start_method("spawn")
    args = global_args_parser()
    # args.dataset = 'libsvm-a8a'
    # args.loss_total = 0.01
    # args.seed = 2023
    for r in range(args.world_size):
        args.rank = r
        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()




