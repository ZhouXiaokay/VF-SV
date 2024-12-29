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
from data_loader.load_data import (load_dummy_partition_with_label, choose_dataset, load_dummy_partition_mutual_info,
                                   load_dummy_partition_by_correlation, load_dependent_data)
from trainer.knn_mi.fagin_batch_trainer_avg import FaginBatchTrainer
from utils.helpers import seed_torch, stochastic_greedy
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


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
    print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank
    # duplicate data
    #data_r_list = [0, 2, 0, 3, 1, 1]
    # data_rank = data_r_list[rank]

    dataset = choose_dataset('web')

    load_start = time.time()
    # data, targets = load_dummy_partition_with_label(dataset, args.num_clients, rank)
    # data, targets = load_dummy_partition_by_correlation(dataset, args.num_clients, rank)
    # data, targets = load_dummy_partition_mutual_info(dataset, args.num_clients, rank)

    data, targets = load_dependent_data(rank, args.num_clients, seed=args.seed)
    mi_scores_with_labels = mutual_info_classif(data, targets).mean()
    print("Mutual Information Scores with Labels:", mi_scores_with_labels)
    targets = np.int64(targets)
    # print(data[0])
    if args.rank == 0:
        print("load data part cost {} s".format(time.time() - load_start))
    n_data = len(data)
    if args.rank == 0:
        print("number of data = {}".format(n_data))

    # shuffle the data to split train data and test data
    # shuffle_ind = np.arange(n_data)
    # np.random.shuffle(shuffle_ind)
    # if args.rank == 0:
    #     print("test data indices: {}".format(shuffle_ind[:args.n_test]))
    # data = data[shuffle_ind]
    # targets = targets[shuffle_ind]

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
    group_keys = [1,2,4,8]
    # check
    trainer = FaginBatchTrainer(args, train_data, train_targets)

    # trainer = FaginBatchTrainer(args, data, targets)

    true_targets = []
    client_mi_values = np.zeros(args.world_size)
    for i in range(n_test):
        # print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        true_targets.append(cur_test_target)
        cur_mi_values = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        client_mi_values += cur_mi_values
    client_mi_values = client_mi_values / n_test
    client_group = [0]*args.world_size
    mi_sort_ind = np.argsort(client_mi_values)[::-1]
    select_client_rank = mi_sort_ind[:args.select_clients]
    for i in select_client_rank:
        client_group[i] = 1
    if args.rank == 0:
        print("client mi values",list(client_mi_values))
        print("client rank", list(mi_sort_ind))
        print("client group",client_group)


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
