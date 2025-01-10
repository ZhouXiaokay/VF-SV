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
from data_loader.load_data import (load_dummy_partition_with_label,choose_dataset, load_dependent_data,
                                   load_and_split_random_dataset, load_and_split_dataset)
from baselines.VF_PS.trainer.mi_fagin_batch_avg_trainer import FaginBatchTrainer
from utils.helpers import seed_torch, stochastic_greedy
import logging
import warnings

import random
warnings.filterwarnings('ignore')


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

    logging.basicConfig(level=logging.DEBUG,
                        filename=code_path + '/logs/baselines/VF_PS/VF-PS.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)

    world_size = args.world_size
    rank = args.rank

    dataset = args.dataset
    # all_data = load_and_split_dataset(dataset)
    all_data = load_and_split_random_dataset(dataset)
    data = all_data[rank]
    targets = all_data['labels']
    n_data = len(data)
    if args.rank == 0:
        print("number of data = {}".format(n_data))


    load_start = time.time()
    targets = np.int64(targets)
    # print(data[0])
    if args.rank == 0:
        logger.info("dataset:{}, seed:{}".format(dataset, args.seed))
        print("load data part cost {} s".format(time.time() - load_start))
        print("number of data = {}".format(n_data))


    # shuffle the data to split train data and test data

    n_test = int(n_data * args.test_ratio)

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
    group_keys = [3, 5, 6, 9, 10, 12]
    group_keys = [random.randint(start_key, end_key) for _ in range(6)]
    trainer = FaginBatchTrainer(args, train_data, train_targets)

    utility_start = time.time()
    client_mi_values = np.zeros(args.num_clients)
    test_start = time.time()
    for i in range(n_test):
        dist.barrier()
        if args.rank == 0:
            print(">>>>>> test[{}] <<<<<<".format(i))
        one_test_start = time.time()
        cur_test_data = test_data[i]
        cur_test_target = test_targets[i]
        cur_mi_values = trainer.find_top_k(cur_test_data, cur_test_target, args.k, group_keys)
        client_mi_values += cur_mi_values
        # if i % 1000 == 0:
        #     if args.rank == 0:
        #         print("Per 1k time cost is: {}".format(time.time() - test_start))
        #         logger.info("Per 1k time cost is: {}".format(time.time()-test_start))
        #     test_start = time.time()
        dist.barrier()
    client_mi_values = client_mi_values / n_test
    client_group = [0]*args.world_size
    mi_sort_ind = np.argsort(client_mi_values)[::-1]
    select_client_rank = mi_sort_ind[:args.select_clients]
    for i in select_client_rank:
        client_group[i] = 1
    if args.rank == 0:
        logger.info("client mi values:{}".format(client_mi_values.tolist()))
        logger.info("client rank:{}".format(mi_sort_ind))
        logger.info("time cost = {}".format(time.time() - load_start))
        print("client mi values",client_mi_values.tolist())
        print("client rank", mi_sort_ind)
        print("client group",client_group)
        print("time cost = {}".format(time.time() - load_start))


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
