from conf import global_args_parser

global_args = global_args_parser()
SEED = global_args.seed
print(SEED)
import random
import os
import numpy as np
import torch
import collections


def get_device():
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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


def seed_torch(seed=SEED):
    # print("seed: ", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def split_model_params(state_dict, n_bottom_out, num_clients):
    # client_params_list = [collections.OrderedDict()] * num_clients
    client_params_list = []
    shape_list = [n_bottom_out] * num_clients
    for rank in range(num_clients):
        order_dict = collections.OrderedDict()
        client_params_list.append(order_dict)
    for key, value in state_dict.items():
        print(value.shape)
        params_list = list(torch.split(value, shape_list, dim=-1))
        for rank in range(num_clients):
            client_params_list[rank][key] = params_list[rank]

    return client_params_list


#
def concat_model_params(attend_params_list):
    # obtain all key in state_dict
    key_list = list(attend_params_list[0].keys())
    params_dict = collections.OrderedDict()

    # for k in attend_params_list[0].keys():
    #     key_list.append(k)
    # concat all params in attend list
    for key in key_list:
        value_list = []
        for i in range(len(attend_params_list)):
            value_list.append(attend_params_list[i][key])
        concat_params = torch.concat(value_list, dim=1)
        params_dict[key] = concat_params
    return params_dict


#
def get_selected_model_params(client_params_list, num_clients, group_flags):
    attend_list = []
    for rank in range(num_clients):
        if group_flags[rank] == 1:
            attend_list.append(client_params_list[rank])
    concat_params_dict = concat_model_params(attend_list)
    return concat_params_dict


def split_concat_layer_params(state_dict, n_bottom_out, num_clients):
    client_params_list = []
    key_list = list(state_dict.keys())
    concat_layer_key = key_list[0]
    shape_list = [n_bottom_out] * num_clients
    for rank in range(num_clients):
        order_dict = collections.OrderedDict()
        client_params_list.append(order_dict)
    for key, value in state_dict.items():
        for rank in range(num_clients):
            client_params_list[rank][key] = value
    concat_layer_params_list = list(torch.split(state_dict[concat_layer_key], shape_list, dim=-1))
    for rank in range(num_clients):
        client_params_list[rank][concat_layer_key] = concat_layer_params_list[rank]

    return client_params_list


def concat_layer_params(attend_params_list):
    # obtain all key in state_dict
    key_list = list(attend_params_list[0].keys())
    concat_layer_key = key_list[0]
    params_dict = collections.OrderedDict()
    for key in key_list:
        params_dict[key] = attend_params_list[0][key]
    client_value_list = []
    for i in range(len(attend_params_list)):
        client_value_list.append(attend_params_list[i][concat_layer_key])
    concat_params = torch.concat(client_value_list, dim=1)

    params_dict[concat_layer_key] = concat_params
    return params_dict


def get_selected_model_params_concat_layer(client_params_list, num_clients, group_flags):
    attend_list = []
    for rank in range(num_clients):
        if group_flags[rank] == 1:
            attend_list.append(client_params_list[rank])
    concat_params_dict = concat_layer_params(attend_list)

    return concat_params_dict


def get_avg_params_state_dict(dict_list):
    key_list = list(dict_list[0].keys())
    avg_params_dict = collections.OrderedDict()
    for key in key_list:
        value_list = []
        for d in dict_list:
            value_list.append(d[key])
        if value_list[0].dtype == torch.int64:
            avg_params_dict[key] = value_list[0]
        else:
            avg_params_dict[key] = torch.stack(value_list, dim=0).mean(dim=0)
    return avg_params_dict


def get_concat_layer_params(state_dict, n_bottom_out, group_flags, concat_layer_params_dict):
    key_list = list(state_dict.keys())
    concat_layer_key = key_list[0]
    num_clients = len(group_flags)
    attend_num_clients = sum(group_flags)
    shape_list = [n_bottom_out] * attend_num_clients
    concat_layer_params_list = list(torch.split(state_dict[concat_layer_key], shape_list, dim=-1))

    start = 0
    for rank in range(num_clients):
        if group_flags[rank] == 1:
            concat_layer_params_dict[rank].append(concat_layer_params_list[start])
            start += 1


def get_avg_top_model_params(state_dict, group_flags, concat_layer_params_dict):
    key_list = list(state_dict.keys())
    concat_layer_key = key_list[0]
    avg_params_dict = collections.OrderedDict()
    avg_concat_layer_dict = {}
    attend_list = []
    for key, value in concat_layer_params_dict.items():
        avg_concat_layer_dict[key] = torch.stack(value, dim=0).mean(dim=0)
    for key in key_list:
        avg_params_dict[key] = state_dict[key]

    for rank in range(len(group_flags)):
        if group_flags[rank] == 1:
            attend_list.append(avg_concat_layer_dict[rank])
    avg_params_dict[concat_layer_key] = torch.cat(attend_list, dim=-1)

    return avg_params_dict


def get_avg_top_model_params(state_dict, group_flags, concat_layer_params_dict):
    key_list = list(state_dict.keys())
    concat_layer_key = key_list[0]
    avg_params_dict = collections.OrderedDict()
    avg_concat_layer_dict = {}
    attend_list = []
    for key, value in concat_layer_params_dict.items():
        avg_concat_layer_dict[key] = torch.stack(value, dim=0).mean(dim=0)
    for key in key_list:
        avg_params_dict[key] = state_dict[key]

    for rank in range(len(group_flags)):
        if group_flags[rank] == 1:
            attend_list.append(avg_concat_layer_dict[rank])
    avg_params_dict[concat_layer_key] = torch.cat(attend_list, dim=-1)

    return avg_params_dict


def get_init_concat_layer_params(state_dict, n_bottom_out, num_clients):
    key_list = list(state_dict.keys())
    shape_list = [n_bottom_out] * num_clients
    concat_layer_key = key_list[0]
    params_list = torch.split(state_dict[concat_layer_key], shape_list, dim=1)
    concat_layer_params_dict = {}
    for rank in range(num_clients):
        concat_layer_params_dict[rank] = [params_list[rank]]

    return concat_layer_params_dict


def append_concat_layer_params(state_dict, concat_layer_params_dict, n_bottom_out, group_flags):
    key_list = list(state_dict.keys())
    attend_num_clients = sum(group_flags)
    shape_list = [n_bottom_out] * attend_num_clients
    concat_layer_key = key_list[0]
    params_list = torch.split(state_dict[concat_layer_key], shape_list, dim=1)
    group_rank = 0
    for rank in range(len(group_flags)):
        if group_flags[rank] == 1:
            concat_layer_params_dict[rank].append(params_list[group_rank])
            group_rank += 1
        else:
            init_params = concat_layer_params_dict[rank][0]
            zero_params = torch.zeros_like(init_params)
            concat_layer_params_dict[rank].append(zero_params)


def cal_cosine_similarity(sim_dict, client_rank, grad_list):
    num_clients = len(grad_list)
    c_grad = grad_list[client_rank]
    for i in range(num_clients):
        if i == client_rank:
            sim_dict[i] = 0
            continue
        sim = torch.cosine_similarity(c_grad, grad_list[i], dim=1).mean().item()
        sim_dict[i] = sim_dict[i] + sim


def get_group_sim(group, sim_dict):
    group_sim = 0.
    for i in range(len(group)):
        if group[i]:
            # group_sim += abs(sim_dict[i])
            group_sim += sim_dict[i]
    return group_sim


def get_init_sim_list(sim_dict):
    sim_list = list(sim_dict.values())
    init_sim = sum(sim_list)

    return [init_sim]


# max-min normalize
# def get_weight_list(sim_list):
#     re_list = []
#     if len(sim_list) == 1:
#         re_list = [sim_list[0]]
#     else:
#         max_sim = max(sim_list)
#         min_sim = min(sim_list)
#         sub_diff = max_sim - min_sim
#         for s in sim_list:
#             re_w = (s - min_sim) / sub_diff
#             re_list.append(re_w)
#     # make sum is 1
#     re_list = list(np.divide(re_list, sum(re_list)))
#     return re_list


# def get_weight_list(sim_list):
#     if len(sim_list) == 1:
#         re_list = [sim_list[0]]
#     else:
#         sum_score = sum(sim_list)
#         re_list = list(np.divide(sim_list, sum_score))
#     return re_list
def get_weight_list(sim_list):
    sum_score = sum(sim_list)
    if sum_score == 0.:
        return sim_list
    re_list = list(np.divide(sim_list, sum_score))
    nor_list = [round(i,4) for i in re_list]
    # return re_list
    return nor_list


def get_weight_params_state_dict(dict_list, weight_list):
    key_list = list(dict_list[0].keys())
    weight_params_dict = collections.OrderedDict()
    for key in key_list:
        value_list = []
        index = 0
        for d in dict_list:
            value_list.append(d[key] * weight_list[index])
            index += 1
        if value_list[0].dtype == torch.int64:
            weight_params_dict[key] = value_list[0]
        else:
            weight_params_dict[key] = torch.stack(value_list, dim=0).sum(dim=0)
    return weight_params_dict


def get_output_model(key_list, top_model):
    output_params = collections.OrderedDict()
    for key in key_list[1:]:
        output_params[key] = top_model[key]
    return output_params


def get_weight_top_model_params(key_list, output_model_list, group_flags, concat_layer_params_dict, weight_list):
    concat_layer_key = key_list[0]
    weight_params = collections.OrderedDict()
    weight_concat_layer = {}
    attend_list = []
    weight_concat_layer_list = []
    for key, value in concat_layer_params_dict.items():
        for index in range(len(value)):
            weight_concat_layer_list.append(value[index] * weight_list[index])
        weight_concat_layer[key] = torch.stack(weight_concat_layer_list, dim=0).mean(dim=0)

    for rank in range(len(group_flags)):
        if group_flags[rank] == 1:
            attend_list.append(weight_concat_layer[rank])
    weight_params[concat_layer_key] = torch.cat(attend_list, dim=-1)
    weight_output = get_weight_params_state_dict(output_model_list, weight_list)

    for key in key_list[1:]:
        weight_params[key] = weight_output[key]

    return weight_params


def get_params_delete_client(rank, params, num_clients):
    # shape_list = [n_bottom_out] * num_clients
    deleted_params_dict = collections.OrderedDict()
    key_list = list(params.keys())
    for key in key_list:
        layer_params_list = list(torch.split(params[key], num_clients, dim=1))
        zero_params = torch.zeros_like(layer_params_list[rank])
        layer_params_list[rank] = zero_params
        layer_params = torch.cat(layer_params_list, dim=1)
        deleted_params_dict[key] = layer_params

    return deleted_params_dict


def get_params_delete_group(group, params, num_clients):
    deleted_params_dict = collections.OrderedDict()
    key_list = list(params.keys())
    for key in key_list:
        layer_params_list = list(torch.chunk(params[key], 4, dim=1))
        for rank in range(num_clients):
            if not group[rank]:
                zero_params = torch.zeros_like(layer_params_list[rank])
                layer_params_list[rank] = zero_params
        layer_params = torch.cat(layer_params_list, dim=1)
        deleted_params_dict[key] = layer_params

    return deleted_params_dict


def cal_group_sim(pred_list, group_pred):
    sim_list = []
    for pred in pred_list:
        sim = torch.cosine_similarity(pred, group_pred, dim=1).mean().item()
        sim_list.append(sim)
    return sim_list
