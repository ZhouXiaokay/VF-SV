import torch
from torch.nn import BCELoss
from model.LR import LR
import torch.optim as optim
from utils.comm_op import sum_all_reduce_tensor, all_gather_variable_tensors
from utils.helpers import seed_torch
import torch.distributed as dist


class LRTrainer(object):
    def __init__(self, args, nf_shape_list):
        self.nf_shape_list = nf_shape_list
        self.args = args
        self.n_f = self.args.n_f
        seed_torch()
        self.lr = LR(self.n_f).to(args.device)
        self.criterion = BCELoss()
        self.optimizer = optim.Adam(self.lr.parameters(), lr=1e-3)
        self.rank = args.rank


    def one_epoch(self, train_data, train_targets, val_data, val_targets):

        # compute the validation loss and grads
        val_z = self.lr(val_data)
        val_sum_z = sum_all_reduce_tensor(val_z)
        val_h = torch.sigmoid(val_sum_z)
        val_loss = self.criterion(val_h, val_targets)
        val_loss.backward()
        val_grads = self.lr.linear.weight.grad.squeeze()
        val_global_grads = all_gather_variable_tensors(val_grads)
        self.optimizer.zero_grad()

        # train model
        partial_z = self.lr(train_data)
        sum_z = sum_all_reduce_tensor(partial_z)
        h = torch.sigmoid(sum_z)
        loss = self.criterion(h, train_targets)
        self.optimizer.zero_grad()
        loss.backward()
        train_grads = self.lr.linear.weight.grad.squeeze()
        self.optimizer.step()

        global_train_grads = torch.zeros_like(val_global_grads)

        start_id = sum(self.nf_shape_list[:self.rank])
        self.assign_continuous_values(global_train_grads, train_grads, start_id)

        # print("rank = {}, global_train_grads = {}".format(self.rank, global_train_grads))
        # print("rank = {}, val_global_grads = {}".format(self.rank, val_global_grads))
        phi = torch.dot(global_train_grads, val_global_grads)
        dist.barrier()
        return loss.item(), phi

    def assign_continuous_values(self, target_tensor, source_tensor, start_idx, dim=0):
        """
        Assign values from source_tensor to a continuous region of target_tensor.

        Args:
            target_tensor (torch.Tensor): The tensor to be updated.
            source_tensor (torch.Tensor): The tensor whose values will be assigned.
            start_idx (int): The starting index for assignment.
            dim (int): The dimension along which the assignment will occur (default is 0).
        """
        # Ensure the assignment won't exceed target_tensor's bounds
        if start_idx + source_tensor.size(dim) > target_tensor.size(dim):
            raise ValueError("Source tensor size exceeds target tensor bounds at the specified start index.")

        # Perform the assignment
        target_tensor.narrow(dim, start_idx, source_tensor.size(dim)).copy_(source_tensor)

    def one_iteration(self, x, y):

        partial_z = self.lr(x)
        sum_z = sum_all_reduce_tensor(partial_z)
        h = torch.sigmoid(sum_z)
        loss = torch.zeros(1)

        loss = self.criterion(h, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        grads = self.lr.linear.weight.grad
        # global_loss = loss.detach()
        # global_loss = sum_all_reduce_tensor(global_loss)

        global_grads = all_gather_variable_tensors(grads)


        return loss.item(),grads

    def predict(self, x):
        partial_z = self.lr(x)
        sum_z = sum_all_reduce_tensor(partial_z)
        pos_prob = torch.sigmoid(sum_z).squeeze().detach().cpu().numpy()
        pred = (pos_prob > 0.5).astype(int)

        return pred, pos_prob

    def save(self, save_path):
        torch.save(self.lr.state_dict(), save_path)

    def load(self, save_path):
        self.lr.load_state_dict(torch.load(save_path))
