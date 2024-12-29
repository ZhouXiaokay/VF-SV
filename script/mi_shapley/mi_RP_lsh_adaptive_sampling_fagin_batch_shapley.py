import time
import sys
import math
from conf import global_args_parser
import numpy as np
from torch.multiprocessing import Process
import copy
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

# sys.path.append("../../")
# from data_loader.data_partition import load_dummy_partition_with_label
from data_loader.load_data import (load_dummy_partition_with_label,choose_dataset, load_dependent_data,
                                   load_dummy_partition_by_correlation, load_dependent_features, load_and_split_dataset)
from trainer.knn_mi_RP.mi_lsh_adaptive_sampling_fagin_batch_trainer import LSHAdaptiveFaginBatchTrainer
from utils.helpers import seed_torch
from typing import List, Union
from utils.comm_op import np_all_gather


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



def random_projection(data, n_size):
    """
    Random projection
    :param data: input data
    :param n_size: the size of projected data
    :return: projected data
    """
    n_features = data.shape[1]
    random_matrix = np.random.normal(loc=0, scale=1, size=[n_features, n_size])
    random_matrix = np.random.randn(n_features, n_size)
    # project data
    projected_data = np.matmul(data,random_matrix)
    return projected_data

def create_hash_table(projected_data, depth):
    n_data = len(projected_data)
    hash_tables = {}
    hash_index = hash_func(projected_data)
    for i in range(n_data):
        for k in range(depth):
            key = hash_index[i][k]
            hash_tables.setdefault(key, set()).add(i)
    return hash_tables

def query_hash_table(query_data, hash_tables, depth):
    query_index = hash_func(query_data).ravel()
    candidates = set()
    for k in range(depth):
        key = query_index[k]
        query_set = hash_tables[key]
        candidates.update(query_set)
    return list(candidates)

def hash_func(projected_data):
    hash_val = np.floor(np.abs(projected_data))
    return hash_val


def run(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    seed_torch()
    if args.rank == 0:
        print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank


    load_start = time.time()
    dataset = args.dataset
    all_data = load_and_split_dataset(dataset)
    data = all_data[rank]
    targets = all_data['labels']

    if args.rank == 0:
        print("load data part cost {} s".format(time.time() - load_start))
    n_data = len(data)
    if args.rank == 0:
        print("number of data = {}".format(n_data))

    data = random_projection(data, args.proj_size)
    num_data = len(data)
    n_test = int(num_data * args.test_ratio)
    # n_test = 10
    train_data = data[n_test:]
    train_targets = targets[n_test:]
    test_data = data[:n_test]
    test_targets = targets[:n_test]
    depth = 5
    all_train_data = np_all_gather(train_data)
    all_test_data = np_all_gather(test_data)
    hash_table = create_hash_table(all_train_data, depth)



    # train_data, test_data, train_targets, test_targets = train_test_split(data, targets,
    #                                                                       test_size=args.test_ratio,
    #                                                                       random_state=args.seed)

    # MI of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    trainer = LSHAdaptiveFaginBatchTrainer(args, train_data, train_targets)

    # cal utility of all group_keys, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    group_keys = [i for i in range(start_key, end_key + 1)]
    mi_values = np.zeros(len(group_keys))
    all_mi_values_dict = {key: [] for key in group_keys}
    mi_groups_dict = {key: [] for key in group_keys}
    adaptive_keys = copy.copy(group_keys)
    for i in range(n_test):
        if rank == 0:
            print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        lsh_candidates = query_hash_table(all_test_data[i], hash_table,depth=1)
        if rank == 0:
            print(len(lsh_candidates))
        cur_mi_values = trainer.find_top_k(cur_test_data, cur_test_target, args.k, lsh_candidates ,adaptive_keys)
        for key in adaptive_keys:
            all_mi_values_dict[key].append(cur_mi_values[key])
            mi_groups_dict[key].append(np.mean(all_mi_values_dict[key]))
        if i>n_test*0.1 and i%100==0:
            for key in adaptive_keys:
                var_key = np.var(mi_groups_dict[key])
                if rank == 0:
                    print("key = {}, var = {}".format(key, var_key))
                if var_key < 1e-4:
                    adaptive_keys.remove(key)
                    if rank == 0:
                        print("remove group key = {}".format(key))
                        print("adaptive keys = {}".format(adaptive_keys))
        if len(adaptive_keys) == 0:
            break
        # mi_values += cur_mi_values
    utility_start = time.time()
    mi_groups_list = [mi_groups_dict[key][-1] for key in group_keys]
    mi_values = np.array(mi_groups_list)


    for group_key in range(start_key, end_key + 1):
        g_mi_value = mi_values[group_key-1]

        utility_value[group_key] = g_mi_value
    n_utility_round += 1
    if args.rank == 0:
        print("calculate utility cost {:.2f} s, total round {}".format(time.time() - utility_start, n_utility_round))

    group_mi_sum = [0 for _ in range(args.world_size)]
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        n_participant = sum(group_flags)
        group_mi_sum[n_participant - 1] += utility_value[group_key]
        if args.rank == 0:
            print("group {}, MI = {}".format(group_flags, utility_value[group_key]))
    if args.rank == 0:
        print("MI sum of different size: {}".format(group_mi_sum))

    # cal factorial
    factor = [1] * args.world_size
    for i in range(1, args.world_size):
        factor[i] = factor[i - 1] * i

    # shapley value of all clients
    shapley_value = [0.0] * world_size
    n_shapley_round = 0
    utility_value[0] = 0.0
    # cal shapley value of each
    shapley_start = time.time()
    for i in range(world_size):
        score = 0.0
        # loop all possible group_keys including the current client
        start_key = 1
        end_key = int(math.pow(2, world_size)) - 1
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            group_size = sum(group_flags)
            # the current client is in the group
            if group_flags[i] == 1:
                u_with = utility_value[group_key]
                group_flags[i] = 0
                group_key = get_utility_key(group_flags)
                u_without = utility_value[group_key]
                score += factor[group_size - 1] / float(factor[world_size - 1]) * (u_with - u_without)
        score /= float(math.pow(2, world_size - 1))
        shapley_value[i] = score
        n_shapley_round += 1
    if args.rank == 0:
        print("calculate shapley value cost {:.2f} s".format(time.time() - shapley_start))
        print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

    shapley_ind = np.argsort(np.array(shapley_value))
    if args.rank == 0:
        print("client ranking = {}".format(shapley_ind.tolist()[::-1]))

    if args.rank == 0:
        print("number of train data = {}".format(len(train_data)))
        print("number of test data = {}".format(len(trainer.n_candidates)))
        print("number of candidates = {}".format(np.mean(trainer.n_candidates)))
    if args.rank == 0:
        print("Time = {}".format(time.time() - load_start))



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


