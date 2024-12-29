import time
import sys
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
                                   load_dummy_partition_by_correlation, load_dependent_features)
from trainer.knn_mi.mi_lsh_fagin_batch_trainer import LSHFaginBatchTrainer
from utils.helpers import seed_torch
from sklearn.model_selection import train_test_split
from typing import List, Union

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

def initialize_hash_label(features, tables_num: int, a: int, depth: int):
    R = np.random.random([depth, tables_num])
    b = np.random.uniform(0, a, [1, tables_num])
    hash_tables = [dict() for i in range(tables_num)]
    # 将inputs转化为二维向量
    inputs = np.array(features)
    if len(inputs.shape) == 1:
        inputs = inputs.reshape([1, -1])

    hash_index = hash_func(features, R, b, a)
    ind = 0
    for inputs_one, indexs in zip(inputs, hash_index):
        for i, key in enumerate(indexs):
            # i代表第i个hash_table，key则为当前hash_table的索引位置
            # inputs_one代表当前向量
            # hash_tables[i].setdefault(key, []).append((tuple(inputs_one), ind))
            hash_tables[i].setdefault(key, []).append(ind)
        ind += 1
    return R, b, hash_tables

def hash_func(inputs: Union[List[List], np.ndarray], R, b, a):
    """
    将向量映射到对应的hash_table的索引
    :param inputs: 输入的单个或多个向量
    :return: 每一行代表一个向量输出的所有索引，每一列代表位于一个hash_table中的索引
    """
    # H(V) = |V·R + b| / a，R是一个随机向量，a是桶宽，b是一个在[0,a]之间均匀分布的随机变量
    hash_val = np.floor(np.abs(np.matmul(inputs, R) + b) / a)
    return hash_val

def query(inputs, hash_tables, R, b, a):
    """
    查询与inputs相似的向量，并输出相似度最高的nums个
    :param inputs: 输入向量
    :param nums:
    :return:
    """
    hash_val = hash_func(inputs,R,b,a).ravel()
    candidates = set()
    candidates_list = []
    # 将相同索引位置的向量添加到候选集中
    for i, key in enumerate(hash_val):
        candidates.update(hash_tables[i][key])

    return candidates

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
    features = load_dependent_features(num_partitions=args.num_clients,seed=args.seed)
    # initialize hash label
    tables_num = 5
    a = 3
    depth = features.shape[1]



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
    # train_data = data[n_test:]
    # train_targets = targets[n_test:]
    # test_data = data[:n_test]
    # test_targets = targets[:n_test]
    train_data = data[n_test:]
    train_targets = targets[n_test:]
    test_data = data[:n_test]
    test_targets = targets[:n_test]

    train_features = features[n_test:]
    R, b, hash_tables = initialize_hash_label(train_features, tables_num, a, depth)
    test_features = features[:n_test]
    # test_data_lsh_candidates = query(test_features, hash_tables, R, b, a)

    # train_data, test_data, train_targets, test_targets = train_test_split(data, targets,
    #                                                                       test_size=args.test_ratio,
    #                                                                       random_state=args.seed)

    # MI of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    trainer = LSHFaginBatchTrainer(args, train_data, train_targets)

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
        lsh_candidates = list(query(test_features[i], hash_tables, R, b, a))
        cur_mi_values = trainer.find_top_k(cur_test_data, cur_test_target, args.k, lsh_candidates, group_keys)
        mi_values += cur_mi_values
    utility_start = time.time()
    mi_values = mi_values / n_test

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
        print("number of test data = {}".format(len(trainer.n_candidates)))
        print("number of candidates = {}".format(np.mean(trainer.n_candidates)))


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


