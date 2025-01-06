import sys
import time
import torch
from torch.nn.functional import cross_entropy
from baselines.VF_CE.trainer.model import MINEBottomModel,MINETopModel
import torch.optim as optim
from utils.comm_op import sum_all_gather_tensor, sum_all_reduce_tensor, all_gather_tensor_list
from utils.helpers import seed_torch


class MINETrainer(object):
    def __init__(self, args, n_features):
        # init params
        self.args = args
        self.rank = args.rank
        self.args = args
        # self.n_f = self.args.n_f
        self.n_bottom_out = self.args.n_bottom_out
        self.n_f = n_features
        self.bottom_model = MINEBottomModel(n_f_in=n_features, n_f_out=self.n_bottom_out)

        seed_torch()
        self.top_model = MINETopModel(input_size=self.n_bottom_out, num_clients=args.num_clients)
        self.bottom_optimizer = optim.SGD(self.bottom_model.parameters(), lr=15e-5)
        self.top_optimizer = optim.Adam(self.top_model.parameters(), lr=15e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.top_optimizer,
                                                                    T_max=args.n_epochs)  # * iters
        self.send_size_list = []
        self.recv_size_list = []
        self.comm_time_list = []
        self.attention_weights = []
        self.learning_rate = []


    def top_model_update(self, bottom_out, y, y_shuffled):
        bottom_out.retain_grad()
        y.requires_grad = True
        y_shuffled.requires_grad = True
        y.retain_grad()
        y_shuffled.retain_grad()
        self.top_optimizer.zero_grad()
        pred_xy = self.top_model(bottom_out, y)
        pred_x_y = self.top_model(bottom_out, y_shuffled)
        ret = torch.mean(pred_xy) - \
              torch.log(torch.mean(torch.exp(pred_x_y)))  # 将pred_xy和pred_x_y代入式（8-49）中，得到互信息ret。
        loss = - ret  # 最大化互信息：在训练过程中，因为需要将模型权重向着互信息最大的方向优化，所以对互信息取反，得到最终的loss值。
        loss.backward()  # 反向传播：在得到loss值之后，便可以进行反向传播并调用优化器进行模型优化。
        self.top_optimizer.step()
        self.learning_rate.append(self.scheduler.get_last_lr())
        return bottom_out.grad, loss

    def scheduler_step(self):
        self.scheduler.step()

    def one_iteration(self, x, y, y_shuffled):
        # bottom model forward
        partial_z = self.bottom_model(x)
        self.send_size_list.append(sys.getsizeof(partial_z))
        comm_start = time.time()
        # get all bottom outputs
        bottom_out_list = all_gather_tensor_list(partial_z)
        concat_bottom = torch.stack(bottom_out_list, dim=0)
        # print(concat_bottom.shape)
        concat_bottom.requires_grad = True
        concat_bottom.retain_grad()

        # concat_bottom = sum_all_gather_tensor(partial_z)
        # concat_bottom.requires_grad = True
        # top model update
        concat_bottom_grad, batch_loss = self.top_model_update(concat_bottom, y, y_shuffled)

        # bottom model backward
        bottom_grad = concat_bottom_grad[:, :][self.rank]
        self.recv_size_list.append(sys.getsizeof(bottom_grad))
        self.bottom_optimizer.zero_grad()
        partial_z.backward(bottom_grad)
        self.bottom_optimizer.step()
        comm_end = time.time()
        comm_time = comm_end - comm_start
        self.comm_time_list.append(comm_time)

        global_loss = batch_loss.detach()
        global_loss = sum_all_reduce_tensor(global_loss)
        return global_loss.item() / self.args.world_size

    def get_comm_cost(self):
        send_size = sum(self.send_size_list)
        recv_size = sum(self.recv_size_list)
        sum_comm_time = sum(self.comm_time_list)
        return send_size, recv_size, sum_comm_time

    def predict(self, x):
        partial_z = self.bottom_model(x)
        concat_bottom = sum_all_gather_tensor(partial_z)
        pred = self.top_model(concat_bottom)
        pred = sum_all_reduce_tensor(pred) / self.args.world_size
        pos_prob = torch.softmax(pred, dim=1).max(dim=1).values.detach().numpy()
        pred = torch.softmax(pred, dim=1).max(dim=1).indices.detach().numpy()

        return pred, pos_prob


    def save(self, save_path):
        torch.save(self.bottom_model.state_dict(),
                   save_path + '/bottom_model_{0}_seed_{1}.pth'.format(self.rank, self.args.seed))
        if  self.rank == 0:
            torch.save(self.top_model.state_dict(),
                       save_path + '/top_model_seed_{1}.pth'.format(self.rank, self.args.seed))

    def sava_attention_weights(self, save_path):

        torch.save(self.attention_weights,
                   save_path + '/ep_attn_weights_scalar.pt')
    def load(self, params_dict):
        bottom_model_state_dict = params_dict['bottom_model']
        top_model_state_dict = params_dict['top_model']
        self.bottom_model.load_state_dict(bottom_model_state_dict)
        self.top_model.load_state_dict(top_model_state_dict)

    def get_bottom_model_params(self):
        return self.bottom_model.state_dict()

    def get_top_model_params(self):
        return self.top_model.state_dict()
