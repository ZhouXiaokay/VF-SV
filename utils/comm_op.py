import numpy as np

import torch
import torch.distributed as dist

def sum_all_reduce_tensor(tensor):
    """ sum square distance and calculate sqrt """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def sum_sqrt_all_reduce(dist_arr):
    """ sum square distance and calculate sqrt """
    dist_tensor = torch.from_numpy(dist_arr)
    dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
    return np.sqrt(dist_tensor.numpy())


def sum_all_reduce(np_arr):
    """ sum square distance (numpy)"""
    np_tensor = torch.from_numpy(np_arr)
    dist.all_reduce(np_tensor, op=dist.ReduceOp.SUM)
    return np_tensor.numpy()


def sum_all_reduce_tensor(t):
    """ sum square distance and calculate sqrt """
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def sum_sqrt_all_gather(dist_arr):
    dist_tensor = torch.from_numpy(dist_arr)
    tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, dist_tensor)
    sum_tensor = torch.stack(tensor_list, dim=0).sum(dim=0)
    return np.sqrt(sum_tensor.numpy())


def sum_all_gather_tensor(params_tensor):
    tensor_list = [torch.zeros_like(params_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, params_tensor)
    # print("tensor_list = ", tensor_list)
    gather_tensor = torch.concat(tensor_list, dim=-1)
    return gather_tensor

def all_gather_tensor_list(params_tensor):
    tensor_list = [torch.zeros_like(params_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, params_tensor)
    # print("tensor_list = ", tensor_list)
    return tensor_list

def all_gather_variable_tensors(local_tensor, group=None):
    """
    Perform all_gather on tensors of different sizes across ranks.

    Args:
        local_tensor (torch.Tensor): The tensor local to this rank.
        group (optional): The process group to work on (default: None).

    Returns:
        list[torch.Tensor]: List of tensors gathered from all ranks.
    """
    if group is None:
        group = dist.group.WORLD

    # Get the size of the local tensor
    local_size = torch.tensor([local_tensor.numel()], dtype=torch.long, device=local_tensor.device)

    # Gather all sizes
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size(group))]
    dist.all_gather(all_sizes, local_size, group=group)

    # Find the maximum size
    max_size = max([size.item() for size in all_sizes])

    # Pad the local tensor to the maximum size
    padded_tensor = torch.zeros(max_size, dtype=local_tensor.dtype, device=local_tensor.device)
    padded_tensor[:local_tensor.numel()] = local_tensor.view(-1)

    # Gather all padded tensors
    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered_tensors, padded_tensor, group=group)

    # Remove padding based on the original sizes
    result = []
    for i, size in enumerate(all_sizes):
        result.append(gathered_tensors[i][:size.item()].view(-1))

    gather_tensor = torch.concat(result, dim=-1)

    return gather_tensor


# def all_gather_list(params_list):
#     dist_tensor = torch.from_numpy(params_list)
#     tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
#
#     tensor_list = [torch.tensor(params) for params in params_list]
#     gather_list = [torch.zeros_like(params) for params in params_list]
#     dist.all_gather(gather_list, tensor_list)
#     return gather_list

def np_all_gather(np_params):
    params_tensor= torch.from_numpy(np_params)
    tensor_list = [torch.zeros_like(params_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, params_tensor)
    gather_tensor = torch.concat(tensor_list, dim=-1)
    return gather_tensor.numpy()

def all_gather(dist_arr):
    dist_tensor = torch.from_numpy(dist_arr)
    tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, dist_tensor)
    sum_tensor = torch.stack(tensor_list, dim=1)

    return sum_tensor.numpy()


def sum_gather(dist_arr):
    rank = dist.get_rank()
    dist_tensor = torch.from_numpy(dist_arr)
    if rank == 0:
        tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
        dist.gather(dist_tensor, gather_list=tensor_list)
        sum_tensor = torch.stack(tensor_list, dim=0).sum(dim=0)
        dist.barrier()
        return np.sqrt(sum_tensor.numpy())
    else:
        dist.gather(dist_tensor, gather_list=None)
        dist.barrier()
        return None


def gather(np_arr):
    rank = dist.get_rank()
    dist_tensor = torch.from_numpy(np_arr)
    if rank == 0:
        tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
        dist.gather(dist_tensor, gather_list=tensor_list)
        return [t.numpy() for t in tensor_list]
    else:
        dist.gather(dist_tensor, gather_list=None)
        return None


def gather_np(np_arr):
    rank = dist.get_rank()
    dist_tensor = torch.from_numpy(np_arr)
    if rank == 0:
        tensor_list = [torch.zeros_like(dist_tensor) for _ in range(dist.get_world_size())]
        dist.gather(dist_tensor, gather_list=tensor_list)
        return np.array([t.numpy() for t in tensor_list])
    else:
        dist.gather(dist_tensor, gather_list=None)
        return None


if __name__ == "__main__":
    a = np.asarray([1,2,3])
    b = np.asarray([4,5,6])
    t_list = [torch.from_numpy(a), torch.from_numpy(b)]
    t_sum = torch.stack(t_list, dim=0).sum(dim=0)
    print(t_sum)
