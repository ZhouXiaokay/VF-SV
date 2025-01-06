# use new group pred
import sys
code_path = '/home/zxk/codes/vfps_mi_diversity'
sys.path.append(code_path)
import os
import math
import time
from conf import global_args_parser
import torch

from baselines.VFDV_IM.tools import (get_weight_params_state_dict, get_weight_list,
                                     get_weight_top_model_params, get_output_model,
                                     get_init_concat_layer_params, append_concat_layer_params)
from utils.helpers import seed_torch, get_utility_key, utility_key_to_groups

seed_torch()
import torch.distributed as dist
from baselines.VFDV_IM.trainer.shapley_contribution_mlp_trainer import ShapleyContrMLPTrainer
from data_loader.load_data import (load_dummy_partition_with_label,choose_dataset, load_dependent_data,
                                   load_dummy_partition_by_correlation, load_dependent_features, load_and_split_dataset)
from torch.multiprocessing import Process
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import logging


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    run_start = time.time()
    log_file = code_path + '/logs/baselines/VFDV_IM/mlp_reweight.log'
    logging.basicConfig(level=logging.DEBUG,
                        filename= log_file,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)

    num_clients = args.num_clients
    # rank 0 is master
    print("rank = {}, world size = {}, pre trained = {}， avg trained = {}".format(args.rank, args.world_size,
                                                                                  args.load_flag, args.avg_flag))
    load_start = time.time()
    dataset = args.dataset
    args.save_path = code_path + '/baselines/VFDV_IM/save/mlp/' + dataset

    if args.rank == 0:
        if not os.path.exists(args.save_path):
            # if not save path, create it
            os.makedirs(args.save_path)

    rank = args.rank
    world_size = args.world_size
    all_data = load_and_split_dataset(dataset)
    data = all_data[rank]
    targets = all_data['labels']

    data = torch.from_numpy(data).float()
    targets = torch.from_numpy(targets).float()

    train_x, test_x, train_targets,test_targets = train_test_split(data,targets,
                                                                   train_size=1-args.test_ratio,random_state=args.seed)
    n_train = train_x.shape[0]
    n_f = train_x.shape[1]
    # args.n_f = n_f

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
    send_size_list = []
    recv_size_list = []
    comm_time_list = []

    # logging
    if args.rank == 0:
        logger.info('***********************************')
        flag_msg = "dataset:{}, pretrained:{}, average:{}, loss_total:{}, start_id:{}, seed:{}".format(dataset,
                                                                                                       args.load_flag,
                                                                                                       args.avg_flag,
                                                                                                       args.loss_total,
                                                                                                       args.start_id,
                                                                                                       args.seed)
        logger.info(flag_msg)

    # the first round, all participants train
    args.load_flag = False
    first_group_key = end_key
    first_group_flag = utility_key_to_groups(first_group_key, world_size)
    first_trainer = ShapleyContrMLPTrainer(args, first_group_flag,n_f)
    epoch_loss_lst = []
    epoch_tol = args.epoch_total
    loss_tol = args.loss_total
    start_id = args.start_id
    for epoch_idx in range(n_epochs):

        epoch_start = time.time()
        epoch_loss = 0.

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size if batch_idx < n_batches - 1 else n_train
            cur_train = train_x[start:end]
            cur_target = train_targets[start:end].long()
            batch_loss = first_trainer.one_iteration(cur_train, cur_target)
            epoch_loss += batch_loss
        epoch_train_time = time.time() - epoch_start
        test_start = time.time()
        pred_targets, pred_probs = first_trainer.predict(test_x)

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

    first_trainer.save(args.save_path)
    init_bottom_params = first_trainer.get_bottom_model_params()
    init_top_params = first_trainer.get_top_model_params()
    init_pred = first_trainer.predict_for_all(test_x, init_bottom_params, init_top_params)
    pred_targets = torch.softmax(init_pred, dim=1).max(dim=1).indices.detach().numpy()
    accuracy = accuracy_score(test_targets, pred_targets)
    utility_value[first_group_key] = accuracy
    args.load_flag = True

    # init the pre-trained model params
    top_key_list = list(init_top_params.keys())
    init_output_params = get_output_model(top_key_list, init_top_params)
    bottom_params_list = [init_bottom_params]
    output_params_list = [init_output_params]
    concat_layer_params_dict = get_init_concat_layer_params(init_top_params, args.n_bottom_out, num_clients)
    send_size, recv_size, comm_time = first_trainer.get_comm_cost()
    send_size_list.append(send_size)
    recv_size_list.append(recv_size)
    comm_time_list.append(comm_time)

    # init impact
    sim_list = [1.]
    top_sim_list = [1.]
    count = 0

    # for group_key in range(start_key, end_key):
    for group_key in range(start_key, end_key):
        seed_torch()
        group_flags = utility_key_to_groups(group_key, world_size)
        if args.rank == 0:
            print("--- compute utility of group : {} ---".format(group_flags))

        group_start = time.time()
        trainer = ShapleyContrMLPTrainer(args, group_flags,n_f)
        weight_list = get_weight_list(sim_list)
        top_weight_list = get_weight_list(top_sim_list)
        weight_bottom_params = get_weight_params_state_dict(bottom_params_list, weight_list)
        weight_top_params = get_weight_top_model_params(top_key_list, output_params_list, group_flags,
                                                        concat_layer_params_dict, top_weight_list)

        reweight_params_dict = {'bottom_model': weight_bottom_params, 'top_model': weight_top_params}
        if group_flags[args.rank] == 0:
            reweight_params_dict['bottom_model'] = init_bottom_params

        if args.load_flag: trainer.load(reweight_params_dict)

        epoch_loss_lst = []
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
                cur_train = train_x[start:end]
                cur_target = train_targets[start:end].long()
                batch_loss = trainer.one_iteration(cur_train, cur_target)
                epoch_loss += batch_loss
            epoch_train_time = time.time() - epoch_start
            test_start = time.time()
            pred_targets, pred_probs = trainer.predict(test_x)

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

        bottom_model_params = trainer.get_bottom_model_params()
        bottom_params_list.append(bottom_model_params)

        top_model_params = trainer.get_top_model_params()

        # print(top_model_params)
        append_concat_layer_params(top_model_params, concat_layer_params_dict, args.n_bottom_out, group_flags)
        output_params = get_output_model(top_key_list, top_model_params)
        output_params_list.append(output_params)
        group_pred = trainer.predict_for_current(test_x)
        send_size, recv_size, comm_time = trainer.get_comm_cost()
        send_size_list.append(send_size)
        recv_size_list.append(recv_size)
        comm_time_list.append(comm_time)
        if args.avg_flag:
            top_sim_list.append(1.)
            if group_flags[args.rank] == 1:
                sim_list.append(1.)
            else:
                sim_list.append(0.)
        else:
            sim = torch.cosine_similarity(init_pred, group_pred, dim=1).mean().item()
            top_sim_list.append(sim)
            if group_flags[args.rank] == 1:
                count += 1
                if sum(group_flags) != 1:
                    top_sim_list[0] = (8 - count) / end_key
                    sim_list[0] = (8 - count) / end_key
                sim_list.append(sim)
            else:
                sim_list.append(0.)

        if args.rank == 0:
            print("compute utility of group {} cost {:.2f} s".format(group_flags, time.time() - group_start))
            group_msg = "compute utility of group {} cost {:.2f} s  epoch {} ".format(group_flags,
                                                                                      time.time() - group_start,
                                                                                      epoch_idx)
            logger.info(group_msg)

    if args.rank == 0:
        print("calculate utility cost {:.2f} s, total round {}, total epochs {}"
              .format(time.time() - utility_start, n_utility_round, n_utility_epochs))
        result_msg = "calculate utility cost {:.2f} s, total round {}, total epochs {}".format(
            time.time() - utility_start,
            n_utility_round,
            n_utility_epochs)
        logger.info(result_msg)

    if args.rank == 0:
        group_acc_sum = [0 for _ in range(args.world_size)]
        for group_key in range(start_key, end_key + 1):
            group_flags = utility_key_to_groups(group_key, world_size)
            n_participant = sum(group_flags)
            group_acc_sum[n_participant - 1] += utility_value[group_key]
            print("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
            logger.info("group {}, accuracy = {}".format(group_flags, utility_value[group_key]))
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

        # logger.info("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))
        # logger.info("client ranking = {}".format(shapley_ind.tolist()[::-1]))
        print("send size", sum(send_size_list))
        print("recv size", sum(recv_size_list))
        print("comm time", sum(comm_time_list))
        print("Time = {}".format(time.time() - load_start))


        logger.info("shapley value of {} clients: {}".format(len(shapley_value), shapley_value))
        logger.info("client ranking = {}".format(shapley_ind.tolist()[::-1]))
        logger.info("send size = {}".format(sum(send_size_list)))
        logger.info("Time = {}".format(time.time() - load_start))



def init_processes(arg, fn):
    rank = arg.rank
    size = arg.world_size
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='gloo',
                            init_method="tcp://127.0.0.1:23366",
                            rank=rank,
                            world_size=size)
    fn(arg)


if __name__ == "__main__":
    # init_processes(0, 2, run)
    processes = []
    # torch.multiprocessing.set_start_method("spawn")
    args = global_args_parser()
    for r in range(args.world_size):
        args.rank = r

        p = Process(target=init_processes, args=(args, run))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
