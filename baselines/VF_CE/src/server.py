from typing import List
import numpy as np
import torch
from .client import Client

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Server(object):
    def __init__(self, Y_train, Y_train_shuffle, config) -> None:
        self.device = config['device']
        # Save label data
        self.Y_train = Y_train
        self.Y_train_shuffle = Y_train_shuffle

        # Extract config info
        self.client_num = config['client_num']
        self.epoch_num = config['epoch_num']
        self.batch_size = config['batch_size']
        self.lr = config['learning_rate']

        # Server节点这边的网络
        # self.server_net = Net()

        self.data_num = len(Y_train)

        # Empty list used to collect clients
        self.clients = list()

        # Model param of server
        # self.bias = np.zeros(self.class_num)
        self.embedding_data = np.zeros(shape=(self.client_num,
                                              config["feature_dim"], self.batch_size))

        self.batch_indexes = [0] * self.batch_size
        # # 创建并应用自注意力模型
        # self.attention_model = selfAttention(num_attention_heads=8,
        #                                      input_size=config["batch_size"],
        #                                      hidden_size=config["batch_size"]).to(self.device)


    # # 创建server节点这边的网络Net
    # def creat_server_net(self):
    #     self.server_net = Net()

    def attach_clients(self, clients: List[Client]):
        """ Attach clients to the server.
        The server can access the client by id.
        """
        self.clients = clients

    def update_embedding_data(self, client: Client, period_type='batch', save = 0):
        """ Call client to calculate embedding data and send it to server.
        Server will receive it and save it.
        """
        # if period_type == 'test':
        #     self.test_embedding_data[client.id] = client.get_embedding_data(period_type)
        if period_type == 'batch':
            self.embedding_data[client.id] = client.get_embedding_data(period_type)

    # 设置当前batch的索引
    def set_batch_indexes(self, batch_indexes):
        self.batch_indexes = batch_indexes

    # 发送
    def send_embedding_grads(self, client: Client, grads):
        self.clients[client.id].set_embedding_grads(grads)

    # 遍历每一个client 计算梯度，并同时更新server
    def my2_cal_batch_embedding_grads(self, model, optimizer, scheduler, lr, config):
        """ Calculate grads w.r.t. embedding data
            Update server model
        """
        loss = 0
        lr = lr
        # 记录所有client的梯度
        clients_grads = np.zeros(shape=(self.client_num, config['feature_dim'], self.batch_size))
        # c1_grads = np.zeros(shape=(config["feature_dim"], self.batch_size))
        # c2_grads = np.zeros(shape=(config["feature_dim"], self.batch_size))
        # c3_grads = np.zeros(shape=(config["feature_dim"], self.batch_size))
        # c4_grads = np.zeros(shape=(config["feature_dim"], self.batch_size))

        # 定义网络
        model = model
        # 定义优化器
        optimizer = optimizer
        # 余弦退火
        scheduler = scheduler

        # # 各个client的输入特征
        # c1 = torch.tensor(self.embedding_data[0], dtype=torch.float,
        #                          requires_grad=True).to(config['device'])
        # c2 = torch.tensor(self.embedding_data[1], dtype=torch.float,
        #                          requires_grad=True).to(config['device'])
        # c3 = torch.tensor(self.embedding_data[2], dtype=torch.float,
        #                          requires_grad=True).to(config['device'])
        # c4 = torch.tensor(self.embedding_data[3], dtype=torch.float,
        #                          requires_grad=True).to(config['device'])
        # 输入server端网络的所有clients的embedding_data
        clients_embedding_data = torch.tensor(self.embedding_data, dtype=torch.float,
                          requires_grad=True).to(config['device'])

        # Ground truth  根据index取数据集的label
        # 将标签y分成未打乱和已打乱的，作为joint_distribution 和 marginal_distribution
        y_joint = torch.tensor(self.Y_train[self.batch_indexes].astype(int),
                                dtype=torch.float, requires_grad=True).to(config['device'])
        y_marginal = torch.tensor(self.Y_train_shuffle[self.batch_indexes].astype(int),
                                 dtype=torch.float, requires_grad=True).to(config['device'])

        # # 转换维度 （batchsize放dim=0）
        # c1 = c1.transpose(0, 1)
        # c2 = c2.transpose(0, 1)
        # c3 = c3.transpose(0, 1)
        # c4 = c4.transpose(0, 1)
        #
        # c1.retain_grad()
        # c2.retain_grad()
        # c3.retain_grad()
        # c4.retain_grad()

        clients_embedding_data.retain_grad()
        y_joint.retain_grad()
        y_marginal.retain_grad()

        # 开始训练
        model.zero_grad()
        # 式(8-49）中的第一项联合分布的期望:将x_sample和y_sample放到模型中，得到联合概率（P(X,Y)=P(Y|X)P(X)）关于神经网络的期望值pred_xy。
        pred_xy = model(clients_embedding_data, y_joint.to(device))
        # 式(8-49)中的第二项边缘分布的期望:将x_sample和y_shuffle放到模型中，得到边缘概率关于神经网络的期望值pred_x_y 。
        pred_x_y = model(clients_embedding_data, y_marginal.to(device))

        ret = torch.mean(pred_xy) - \
              torch.log(torch.mean(torch.exp(pred_x_y)))  # 将pred_xy和pred_x_y代入式（8-49）中，得到互信息ret。
        loss = - ret  # 最大化互信息：在训练过程中，因为需要将模型权重向着互信息最大的方向优化，所以对互信息取反，得到最终的loss值。

        loss.backward()  # 反向传播：在得到loss值之后，便可以进行反向传播并调用优化器进行模型优化。

        clients_grads[:, :] = clients_embedding_data.grad.cpu()

        optimizer.step()  # 调用优化器 优化model
        lr.append(scheduler.get_last_lr())

        # Send it to n's clients
        for i in range(self.client_num):
            self.send_embedding_grads(self.clients[i], clients_grads[i])  # self.embedding_grads = grads
        # self.send_embedding_grads(self.clients[1], c2_grads)  # self.embedding_grads = grads
        # self.send_embedding_grads(self.clients[2], c3_grads)  # self.embedding_grads = grads
        # self.send_embedding_grads(self.clients[3], c4_grads)  # self.embedding_grads = grads

        return loss