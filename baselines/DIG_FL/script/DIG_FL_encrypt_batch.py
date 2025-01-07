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
import argparse
import torch.distributed as dist
from baselines.DIG_FL.trainer.lr_trainer_encrypt import LRTrainer
from data_loader.load_data import (load_dummy_partition_with_label,choose_dataset, load_dependent_data,
                                   load_dummy_partition_by_correlation, load_dependent_features, load_and_split_dataset)
from torch.multiprocessing import Process
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from utils.comm_op import sum_all_gather_tensor
import logging
def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    run_start = time.time()

    log_file = code_path + '/logs/baselines/DIG_FL/DIG_FL.log'
    logging.basicConfig(level=logging.DEBUG,
                        filename= log_file,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)
    num_clients = args.num_clients
    # rank 0 is master
    print("rank = {}, world size = {}".format(args.rank, args.world_size))

    dataset = args.dataset
    if args.rank == 0:
        print(">>> dataset = {}".format(dataset))
        logger.info(">>> dataset = {}".format(dataset))

    rank = args.rank
    world_size = args.world_size
    all_data = load_and_split_dataset(dataset)
    data = all_data[rank]
    targets = all_data['labels']

    data = torch.from_numpy(data).float()
    targets = torch.from_numpy(targets).float()

    train_x, test_x, train_targets, test_targets = train_test_split(data, targets,
                                                                    train_size=1 - args.test_ratio,
                                                                    random_state=args.seed)
    train_x, val_x, train_targets, val_targets = train_test_split(train_x, train_targets,
                                                                  train_size=0.9,random_state=args.seed)
    n_train = train_x.shape[0]
    n_f = train_x.shape[1]
    print(">>> rank = {}, n_train = {}, n_f = {}".format(rank, n_train, n_f))

    n_f_tensor = torch.tensor(n_f).reshape(1,1)
    nf_shape_list = sum_all_gather_tensor(n_f_tensor).cpu().numpy().tolist()[0]

    args.n_f = n_f

    batch_size = args.batch_size
    n_batches = n_train // batch_size
    trainer = LRTrainer(args,nf_shape_list)

    train_start = time.time()
    epoch_loss_lst = []
    phi_list = []
    n_epochs = args.n_epochs
    epoch_tol = args.epoch_total
    loss_tol = args.loss_total
    start_id = args.start_id
    for epoch_idx in range(n_epochs):
        if args.rank == 0:
            print(">>> epoch [{}] start".format(epoch_idx))
        epoch_start = time.time()
        # epoch_loss, phi = trainer.one_epoch(train_x,train_targets.unsqueeze(dim=1), val_x, val_targets.unsqueeze(dim=1))
        epoch_loss = 0.

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size if batch_idx < n_batches - 1 else n_train
            cur_train = train_x[start:end]
            cur_target = train_targets[start:end].unsqueeze(dim=1)
            batch_loss, phi = trainer.one_iteration(cur_train, cur_target, val_x, val_targets.unsqueeze(dim=1))
            epoch_loss += batch_loss
            phi_list.append(phi)

        epoch_train_time = time.time() - epoch_start
        test_start = time.time()
        pred_targets, pred_probs = trainer.predict(test_x)

        accuracy = accuracy_score(test_targets, pred_targets)
        auc = roc_auc_score(test_targets, np.array(pred_probs))
        epoch_test_time = time.time() - test_start
        if args.rank == 0:
            logger.info(">>> epoch[{}] finish, train loss {:.6f}, cost {:.2f} s, train cost {:.2f} s, test cost {:.2f} s, "
                  "accuracy = {:.6f}, auc = {:.6f}"
                  .format(epoch_idx, epoch_loss, time.time() - epoch_start, epoch_train_time, epoch_test_time, accuracy,
                          auc))
            print(">>> epoch[{}] finish, train loss {:.6f}, cost {:.2f} s, train cost {:.2f} s, test cost {:.2f} s, "
                  "accuracy = {:.6f}, auc = {:.6f}"
                  .format(epoch_idx, epoch_loss, time.time() - epoch_start, epoch_train_time, epoch_test_time, accuracy,
                          auc))
        epoch_loss_lst.append(epoch_loss)

        if epoch_idx >= start_id and len(epoch_loss_lst) > epoch_tol \
                and min(epoch_loss_lst[:-epoch_tol]) - min(epoch_loss_lst[-epoch_tol:]) < loss_tol:
            print("!!! train loss does not decrease > {} in {} epochs, early stop !!!"
                  .format(loss_tol, epoch_tol))
            break
    # save_path = '../save/all_participate/credit/lr_rank_{0}.pth'.format(args.rank)
    # trainer.save(args.save_path)
    phi = sum(phi_list) / len(phi_list)
    phi_tensor = torch.tensor(phi).reshape(1,1)
    all_phi = sum_all_gather_tensor(phi_tensor).numpy().tolist()[0]
    print("client ranking = {}".format(np.argsort(all_phi)))
    print(">>> task finish, cost {:.2f} s".format(time.time() - run_start))
    if args.rank == 0:
        logger.info(">>> task finish, cost {:.2f} s".format(time.time() - run_start))
        logger.info("client ranking = {}".format(np.argsort(all_phi)))




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
