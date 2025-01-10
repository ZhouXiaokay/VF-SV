import time
import sys
code_path = '/home/zxk/codes/vfps_mi_diversity'
sys.path.append(code_path)
from utils.helpers import seed_torch
from conf import global_args_parser
import torch
import os
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
import logging


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    run_start = time.time()
    log_file = code_path + '/logs/baselines/VF_CE/VF_CE.log'
    logging.basicConfig(level=logging.DEBUG,
                        filename=log_file,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)


    print("rank = {}, world size = {}".format(args.rank, args.world_size))

    rank = args.rank
    dataset = args.dataset
    if args.rank == 0:
        print("dataset = {}".format(dataset))
        logger.info("dataset = {}".format(dataset))
    save_path = code_path + '/baselines/VF_CE/save/' + dataset

    if args.rank == 0:
        if not os.path.exists(save_path):
            # if not save path, create it
            os.makedirs(save_path)

    all_data = load_and_split_dataset(dataset)
    data = all_data[rank]
    targets = all_data['labels'].reshape(data.shape[0], 1)
    shuffled_targets = targets.copy()
    random.shuffle(shuffled_targets)

    train_x = torch.from_numpy(data).float()
    train_targets = torch.from_numpy(targets).float()
    train_shuffled_targets = torch.from_numpy(shuffled_targets).float()

    n_train = train_x.shape[0]
    n_f = train_x.shape[1]


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
            # print(batch_targets.shape)
            batch_shuffle_targets = train_shuffled_targets[batch_ids]
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
        #     for i in range(n_batches):
        #         batch_ids = batches[i]
        #         batch_x = train_x[batch_ids]
        #         batch_targets = train_targets[batch_ids]
        #         # print(batch_targets.shape)
        #         batch_shuffle_targets = train_shuffled_targets[batch_ids]
        #         batch_loss = trainer.one_iteration(batch_x, batch_targets, batch_shuffle_targets)
        #         epoch_loss += batch_loss
        #         trainer.attention_weights.append(trainer.top_model.attn_weights)
        #     print("!!! train loss does not decrease > {} in {} epochs, early stop !!!"
        #           .format(loss_tol, epoch_tol))
        #     break

    if args.rank == 0:
        print(">>> training cost {} s".format(time.time() - run_start))
        logger.info(">>> training cost {} s".format(time.time() - run_start))
        trainer.sava_attention_weights(save_path)
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