import time
import sys
code_path = '/home/zxk/codes/vfps_mi_diversity'
sys.path.append(code_path)
import math
from conf import global_args_parser
import numpy as np
from torch.multiprocessing import Process

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, roc_auc_score

# sys.path.append("../../")
# from data_loader.data_partition import load_dummy_partition_with_label
from data_loader.load_data import load_dummy_partition_with_label,choose_dataset, load_dependent_data, load_and_split_dataset
from trainer.knn_mi.mi_all_reduce_trainer import AllReduceTrainer
from utils.helpers import seed_torch
import logging

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


def run(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.DEBUG,
                        filename=code_path + '/logs/ablation_study/all_reduce.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)
    seed_torch()
    if args.rank == 0:
        print("device = {}".format(device))

    world_size = args.world_size
    rank = args.rank

    load_start = time.time()
    dataset = args.dataset
    if args.rank == 0:
        logger.info("dataset:{}".format(dataset))
    all_data = load_and_split_dataset(dataset)
    data = all_data[rank]
    targets = all_data['labels']
    num_data = len(data)

    if args.rank == 0:
        print("load data part cost {} s".format(time.time() - load_start))
        print("number of data = {}".format(num_data))

    data = random_projection(data, args.proj_size)

    n_test = int(num_data * args.test_ratio)
    n_train = num_data - n_test
    # n_test = 10
    train_data = data[n_test:]
    train_targets = targets[n_test:]
    test_data = data[:n_test]
    test_targets = targets[:n_test]

    # MI of a group of clients, key is binary encode of client attendance
    utility_value = dict()
    n_utility_round = 0

    trainer = AllReduceTrainer(args, train_data, train_targets)

    # cal utility of all group_keys, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, args.world_size)) - 1
    utility_start = time.time()
    for group_key in range(start_key, end_key + 1):
        group_flags = utility_key_to_groups(group_key, world_size)
        if args.rank == 0:
            print("--- compute utility of group : {} ---".format(group_flags))

        mi_values = []

        test_start = time.time()

        for i in range(n_test):
            # print(">>>>>> test[{}] <<<<<<".format(i))
            one_test_start = time.time()
            cur_test_data = test_data[i]
            cur_test_target = test_targets[i]
            mi_value = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_flags)

            mi_values.append(mi_value)
            one_test_time = time.time() - one_test_start

        g_mi_value = np.mean(mi_values)

        utility_value[group_key] = g_mi_value
        n_utility_round += 1
        if args.rank == 0:
            print("test {} data cost {} s".format(n_test, time.time() - test_start))
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
            if group_flags[i] == 1 and group_size > 1:
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
        logger.info("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

    shapley_ind = np.argsort(np.array(shapley_value))
    if args.rank == 0:
        print("client ranking = {}".format(shapley_ind.tolist()[::-1]))
        print("Time = {}".format(time.time() - load_start))
        logger.info("client ranking = {}".format(shapley_ind.tolist()[::-1]))
        logger.info("Time = {}".format(time.time() - load_start))



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


