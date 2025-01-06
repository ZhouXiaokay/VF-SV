import time
import sys
code_path = '/home/zxk/codes/vfps_mi_diversity'
sys.path.append(code_path)
from utils.helpers import seed_torch
from conf import global_args_parser
import torch
global_args = global_args_parser()
SEED = global_args.seed
seed_torch()
import torch.distributed as dist
from data_loader.load_data import (load_dummy_partition_with_label,choose_dataset, load_dependent_data,
                                   load_dummy_partition_by_correlation, load_dependent_features, load_and_split_dataset)
from torch.multiprocessing import Process
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from utils.comm_op import sum_all_gather_tensor
import random
from conf.args import global_args_parser

from baselines.VF_CE.trainer.mine_trainer import MINETrainer
from baselines.VF_CE.utils.helpers import gen_batches, load_data, split_data



def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    run_start = time.time()
    log_file = code_path + '/logs/baselines/VF_CE/VF_CE.log'

    print("rank = {}, world size = {}".format(args.rank, args.world_size))

    dataset = args.dataset

    file_name = 'optdigits'
    train_x, train_targets = load_data(file_name)
    train_shuffled_targets = train_targets.copy()
    random.shuffle(train_shuffled_targets)
    n_train = train_x.shape[0]
    n_f = train_x.shape[1]

    train_x = torch.from_numpy(train_x).float()
    train_targets = torch.from_numpy(train_targets).float()
    train_shuffle_targets = torch.from_numpy(train_shuffled_targets).float()


    batch_size = args.batch_size
    n_batches = n_train // batch_size
    batches = gen_batches(n_train, batch_size)

    epoch_loss_lst = []
    phi_list = []
    n_epochs = args.n_epochs
    epoch_tol = args.epoch_total
    loss_tol = args.loss_total
    start_id = args.start_id

    trainer = MINETrainer(args, n_f)
    for epoch_idx in range(n_epochs):
        if args.rank == 0:
            print(">>> epoch [{}] start".format(epoch_idx))
        epoch_loss = 0.
        for i in range(n_batches):
            batch_ids = batches[i]
            batch_x = train_x[batch_ids]
            batch_targets = train_targets[batch_ids]
            batch_shuffle_targets = train_shuffle_targets[batch_ids]
            batch_loss = trainer.one_iteration(batch_x, batch_targets, batch_shuffle_targets)
            epoch_loss += batch_loss
            if epoch_idx == n_epochs - 1:
                trainer.attention_weights.append(trainer.top_model.attn_weights)
        epoch_loss_lst.append(epoch_loss/n_batches)
        trainer.scheduler_step()
        if args.rank == 0:
            print(">>> epoch [{}] loss = {}".format(epoch_idx, epoch_loss))



        # if epoch_idx >= start_id and len(epoch_loss_lst) > epoch_tol \
        #         and min(epoch_loss_lst[:-epoch_tol]) - min(epoch_loss_lst[-epoch_tol:]) < loss_tol:
        #     print("!!! train loss does not decrease > {} in {} epochs, early stop !!!"
        #           .format(loss_tol, epoch_tol))
        #     break

    if args.rank == 0:
        print(">>> training cost {} s".format(time.time() - run_start))
        trainer.sava_attention_weights('../save/{}'.format(file_name))
def init_processes(arg, fn):
    rank = arg.rank
    size = arg.world_size
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='gloo',
                            init_method="tcp://127.0.0.1:12345",
                            rank=rank,
                            world_size=size)
    fn(args)


if __name__ == "__main__":
    args = global_args_parser()
    processes = []

    for r in range(args.world_size):
        args.rank = r
        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()