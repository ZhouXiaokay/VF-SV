import time, math

import torch

from utils.helpers import seed_torch, get_utility_key, utility_key_to_groups

from conf import global_args_parser

global_args = global_args_parser()
SEED = global_args.seed
seed_torch()
import argparse
import torch.distributed as dist
from trainer.acc_shapley.lr_trainer import ShapleyLRTrainer
from data_loader.load_data import load_dummy_partition_with_label, load_dependent_data
from torch.multiprocessing import Process
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    run_start = time.time()

    num_clients = args.num_clients
    # rank 0 is master
    print("rank = {}, world size = {}, pre trained = {}".format(args.rank, args.world_size, args.load_flag))
    d_name = args.dataset
    args.save_path = args.save_path + d_name + '/lr_rank_{0}_seed_{1}.pth'.format(args.rank, SEED)
    rank = args.rank
    data, targets = load_dependent_data(rank, args.num_clients, seed=args.seed)
    num_data = len(data)
    n_test = int(num_data * args.test_ratio)
    world_size = args.world_size

    train_data = torch.tensor(data[n_test:],device=args.device)
    train_targets = torch.tensor(targets[n_test:],device=args.device, dtype=torch.float32)
    test_data = torch.tensor(data[:n_test],device=args.device)
    test_targets = torch.tensor(targets[:n_test], dtype=torch.float32)

    n_train = train_data.shape[0]
    n_f = train_data.shape[1]
    args.n_f = n_f

    batch_size = args.batch_size
    n_batches = n_train // batch_size
    n_epochs = args.n_epochs
    utility_value = dict()
    n_utility_round = 0

    # cal utility of all groups, group key = 1-(2^k-1)
    start_key = 1
    end_key = int(math.pow(2, world_size)) - 1
    utility_start = time.time()
    n_utility_epochs = 0  # total used epochs
    for group_key in range(start_key, end_key + 1):
        seed_torch()
        group_flags = utility_key_to_groups(group_key, world_size)
        if args.rank == 0:
            print("--- compute utility of group : {} ---".format(group_flags))

        group_start = time.time()
        trainer = ShapleyLRTrainer(args, group_flags)

        epoch_loss_lst = []
        # loss_tol = 0.1
        # epoch_tol = 3  # loss should decrease in ${epoch_tol} epochs
        epoch_tol = args.epoch_total
        loss_tol = args.loss_total
        start_id = args.start_id
        accuracy, auc = 0.0, 0.0
        for epoch_idx in range(n_epochs):

            epoch_start = time.time()
            epoch_loss = 0.

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = (batch_idx + 1) * batch_size if batch_idx < n_batches - 1 else n_train
                cur_train = train_data[start:end]
                cur_target = train_targets[start:end].unsqueeze(dim=1)
                batch_loss = trainer.one_iteration(cur_train, cur_target)
                epoch_loss += batch_loss
            epoch_train_time = time.time() - epoch_start
            test_start = time.time()
            pred_targets, pred_probs = trainer.predict(test_data)

            accuracy = accuracy_score(test_targets, pred_targets)
            auc = roc_auc_score(test_targets, np.array(pred_probs))
            epoch_test_time = time.time() - test_start
            if args.rank == 0:
                print(
                    ">>> epoch[{}] finish, train loss {:.6f}, cost {:.2f} s, train cost {:.2f} s, test cost {:.2f} s, "
                    "accuracy = {:.6f}, auc = {:.6f}"
                        .format(epoch_idx, epoch_loss, time.time() - epoch_start, epoch_train_time, epoch_test_time,
                                accuracy,
                                auc))
            epoch_loss_lst.append(epoch_loss)

            if epoch_idx >= start_id and len(epoch_loss_lst) > epoch_tol \
                    and min(epoch_loss_lst[:-epoch_tol]) - min(epoch_loss_lst[-epoch_tol:]) < loss_tol:
                if args.rank == 0:
                    print("!!! train loss does not decrease > {} in {} epochs, early stop !!!"
                          .format(loss_tol, epoch_tol))
                break
        n_utility_epochs += epoch_idx + 1

        utility_value[group_key] = accuracy
        n_utility_round += 1
        if args.rank == 0:
            print("compute utility of group {} cost {:.2f} s".format(group_flags, time.time() - group_start))
    if args.rank == 0:
        print("calculate utility cost {:.2f} s, total round {}, total epochs {}"
              .format(time.time() - utility_start, n_utility_round, n_utility_epochs))

    if args.rank == 0:
        group_acc_sum = [0 for _ in range(args.world_size)]
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            n_participant = sum(group_flags)
            group_acc_sum[n_participant - 1] += utility_value[group_key]
            print("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
        print("accuracy sum of different size: {}".format(group_acc_sum))

        # cal factorial
        factor = [1] * args.world_size
        for epoch_idx in range(1, args.world_size):
            factor[epoch_idx] = factor[epoch_idx - 1] * epoch_idx

        # shapley value of all clients
        shapley_value = [0.0] * world_size
        n_shapley_round = 0

        # cal shapley value of each
        shapley_start = time.time()
        for epoch_idx in range(world_size):
            score = 0.0
            # loop all possible groups including the current client
            start_key = 1
            end_key = int(math.pow(2, world_size)) - 1
            for group_key in range(start_key, end_key + 1):
                group_flags = utility_key_to_groups(group_key, world_size)
                group_size = sum(group_flags)
                # the current client is in the group
                if group_flags[epoch_idx] == 1 and group_size > 1:
                    u_with = utility_value[group_key]
                    group_flags[epoch_idx] = 0
                    group_key = get_utility_key(group_flags)
                    u_without = utility_value[group_key]
                    score += factor[group_size - 1] / float(factor[world_size - 1]) * (u_with - u_without)
            score /= float(math.pow(2, world_size - 1))
            shapley_value[epoch_idx] = score
            n_shapley_round += 1
        print("calculate shapley value cost {:.2f} s".format(time.time() - shapley_start))
        print("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))

        shapley_ind = np.argsort(np.array(shapley_value))
        print("client ranking = {}".format(shapley_ind.tolist()[::-1]))

TEST_RATIO = 0.1
N_CLIENTS = WORLD_SIZE = 4
# WORLD_SIZE = 8
N_EPOCHS = 100
BATCH_SIZE = 100
VALID_RATIO = 0.1
LOSS_TOTAL = 0.01
EPOCH_TOTAL = 3
START_ID = 9
HOME_PATH = '/home/zxk/codes/vfl_data_valuation/save/all_participate/lr/'


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-num_clients', type=int, default=N_CLIENTS,
                        help='Number of processes participating in the job.')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--world_size', type=int, default=WORLD_SIZE)
    parser.add_argument('--n_f', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--load_flag', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default=HOME_PATH)
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--start_id', type=int, default=START_ID)
    parser.add_argument('--epoch_total', type=int, default=EPOCH_TOTAL)
    parser.add_argument('--loss_total', type=float, default=LOSS_TOTAL)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO)
    parser.add_argument('--device', type=str, default='cuda:1')
    arg = parser.parse_args()

    return arg


def init_processes(arg, fn):
    rank = arg.rank
    size = arg.world_size
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='gloo',
                            init_method="tcp://127.0.0.1:23456",
                            rank=rank,
                            world_size=size)
    fn(arg)


if __name__ == "__main__":
    # init_processes(0, 2, run)
    processes = []
    # torch.multiprocessing.set_start_method("spawn")
    for r in range(WORLD_SIZE):
        args = args_parser()
        args.rank = r

        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
