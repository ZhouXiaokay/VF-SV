import time
import sys
import math
from conf import global_args_parser
import numpy as np
from torch.multiprocessing import Process

import torch
import torch.distributed as dist
# sys.path.append("../../")
from data_loader.load_data import load_dummy_partition_with_label, choose_dataset, load_dummy_partition_mutual_info
from trainer.knn_mi.fagin_batch_trainer_mci import FaginBatchTrainer
from utils.helpers import seed_torch, stochastic_greedy


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

def find_keys_with_position_one(world_size, position):
    keys_with_one_at_position = []
    for i in range(2 ** world_size):
        if i & (1 << position):
            keys_with_one_at_position.append(i)
    return keys_with_one_at_position


def run(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    seed_torch()
    print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank

    # duplicate data
    # data_r_list = [0, 2, 0, 3, 1, 1]
    # data_rank = data_r_list[rank]

    dataset = choose_dataset('adult')

    load_start = time.time()
    # data, targets = load_dummy_partition_with_label(dataset, args.num_clients, data_rank)
    data, targets = load_dummy_partition_mutual_info(dataset, args.num_clients, rank)
    targets = np.int64(targets)
    # print(data[0])
    if args.rank == 0:
        print("load data part cost {} s".format(time.time() - load_start))
    n_data = len(data)
    if args.rank == 0:
        print("number of data = {}".format(n_data))

    # shuffle the data to split train data and test data
    shuffle_ind = np.arange(n_data)
    np.random.shuffle(shuffle_ind)
    if args.rank == 0:
        print("test data indices: {}".format(shuffle_ind[:args.n_test]))
    data = data[shuffle_ind]
    targets = targets[shuffle_ind]

    num_data = len(data)
    n_test = int(num_data * args.test_ratio)

    train_data = data[n_test:]
    train_targets = targets[n_test:]
    test_data = data[:n_test]
    test_targets = targets[:n_test]

    # accuracy of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    # cal utility of all group_keys, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    group_key_ind = np.arange(start_key, end_key + 1)
    # group_keys = [3, 5, 6, 9, 10, 12, end_key]
    group_keys = [i for i in range(start_key, end_key + 1)]
    trainer = FaginBatchTrainer(args, train_data, train_targets)

    client_mi_values = []
    for i in range(args.n_test):
        # print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        cur_mi_values = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        client_mi_values.append(cur_mi_values)

    if args.rank == 0:

        summed_client_mi_values = [sum(x) for x in zip(*client_mi_values)]

        client_scores = cal_mci_value(world_size, summed_client_mi_values, group_keys)
        mi_sort_ind = np.argsort(np.array(client_scores))
        print(mi_sort_ind)

def max_marginal_contribution(rank, mi_values, world_size ,group_keys):
    max_contribution = float('-inf')
    best_subset_index = None

    for group_key in group_keys:
        group_flags = utility_key_to_groups(group_key, world_size)
        group_size = sum(group_flags)
        if group_flags[rank] == 1 and group_size > 1:
            group_list_id = group_key - 1
            u_with = mi_values[group_key-1]
            group_flags[rank] = 0
            group_key = get_utility_key(group_flags)
            u_without = mi_values[group_key-1]
            contribution = u_with - u_without
            if contribution > max_contribution:
                max_contribution = contribution
                best_subset_index = group_key
    return best_subset_index, max_contribution

def cal_mci_value(world_size, client_mi_values, group_keys):
    client_group = [0.] * world_size
    for client in range(world_size):
        best_subset_index, max_contribution = max_marginal_contribution(client, client_mi_values, world_size, group_keys)
        client_group[client] = max_contribution

    return client_group

def init_processes(arg, fn):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend=arg.backend,
                            init_method=arg.init_method,
                            rank=arg.rank,
                            world_size=arg.world_size)
    fn(arg)


if __name__ == '__main__':

    processes = []
    args = global_args_parser()
    for r in range(args.world_size):
        args.rank = r
        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
