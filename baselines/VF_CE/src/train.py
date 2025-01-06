from typing import List
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .client import Client
from .server import Server
from .util import *

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 设置随机数生成器的种子
seed = 42
set_seed(seed)


# 1.2 定义神经网络模型
# 注意力模块
class ItemLevelAttention(nn.Module):
    def __init__(self, input_dim, batch_size):
        super(ItemLevelAttention, self).__init__()
        # 定义hj和ht乘积
        self.hj_product = nn.Linear(input_dim, input_dim, bias=False)
        self.ht_product = nn.Linear(input_dim, input_dim, bias=False)
        # 定义激活后的乘积，将最终的结果映射为标量
        self.v_product = nn.Linear(input_dim, 1, bias=False)
        # # 添加偏置，使其shape与hj ht一致
        # self.bias = nn.Parameter(torch.Tensor(batch_size, input_dim))

    def forward(self, x, y):
        # 用没有bais的线性层模拟乘积操作

        # 计算hj
        # 获取输入张量的形状信息
        batch_size, client_num, input_dim = x.size()
        # 将输入张量的第二维度展平为(batch_size * client_num, input_dim)
        # (bs, feature_dim, client_num) -> (bs*client_num, feature_dim)
        x_flatten = x.contiguous().view(batch_size * client_num, input_dim)
        hj_flatten = self.hj_product(x_flatten)
        # 将输出张量的形状恢复为原始形状
        hj = hj_flatten.view(batch_size, client_num, input_dim)

        # 计算ht
        # 将输入张量的第二维度展平为(batch_size * client_num, input_dim)
        # (bs, feature_dim, client_num) -> (bs*client_num, feature_dim)
        y_flatten = y.view(batch_size * client_num, input_dim)
        ht_flatten = self.ht_product(y_flatten)
        # 将输出张量的形状恢复为原始形状
        ht = ht_flatten.view(batch_size, client_num, input_dim)

        # 对结果应用激活函数
        res1 = F.sigmoid(hj + ht).to(torch.float32)

        # 计算最终结果
        # 将输入张量的第二维度展平为(batch_size * client_num, input_dim)
        # (bs, feature_dim, client_num) -> (bs*client_num, feature_dim)
        res1_flatten = res1.view(batch_size * client_num, input_dim)
        res2_flatten = self.v_product(res1_flatten)
        # 将输出张量的形状恢复为原始形状
        res2 = res2_flatten.view(batch_size, client_num, 1)

        return res2

# 主mine网络
class Net(nn.Module):
    def __init__(self, input_dim=8, batch_size=2048, client_num=4):
        super(Net, self).__init__()
        # 自注意力模块
        self.item_level_attention = ItemLevelAttention(input_dim, batch_size)
        # 线性层
        self.fc0 = nn.Linear(1, input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, input_dim * 4)
        self.fc3 = nn.Linear(input_dim * 4, 1)

        # 定义额外的参数client_num
        self.client_num = torch.tensor(client_num)

    def forward(self, clients_embedding_data, y):
        # 将y的shape 变为与client的embedding_data的shape一致
        y = self.fc0(y)
        # 使用permute函数交换维度，(client_num, feature_dim, bs) -> (bs, client_num, feature_dim)
        hj = clients_embedding_data.permute(2, 0, 1)
        ht = torch.stack([y] * self.client_num, dim=1)
        # 注意力计算
        attn_res = self.item_level_attention(hj, ht)
        # 得到注意力值
        attn_weights = F.softmax(attn_res, dim=1).squeeze()
        # 保存注意力权重标量
        self.attn_weights = attn_weights.detach().cpu().numpy()

        # # 融合注意力模块的输出
        # x = (c1 * attn_weights[:, 0].unsqueeze(1) + c2 * attn_weights[:, 1].unsqueeze(1) +
        #      c3 * attn_weights[:, 2].unsqueeze(1) + c4 * attn_weights[:, 3].unsqueeze(1))

        # (client_num, feature_dim, bs) -> (client_num, bs, feature_dim)
        clients_embedding_data = clients_embedding_data.permute(0, 2, 1)
        attn_weights = attn_weights.transpose(0, 1).unsqueeze(-1)
        element_wise_result = clients_embedding_data * attn_weights
        x = torch.sum(element_wise_result, dim=0)

        h1 = F.relu(self.fc1(x + y))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3

# mine的纵向联邦学习过程实现
def vfl_lr_train(server: Server, clients: List[Client], config):
    device = config['device']
    # 创建一个数组收集最后一个epoch的所有迭代轮次 每个client的attn_weights:
    last_ep_attn_weights_scalar = []  # 存储每个迭代的数据
    # 创建一个数组收集所有epoch的loss
    plot_loss = []

    # 开始训练
    # 创建新的模型
    model = Net(config['feature_dim'], config['batch_size'], config['client_num']).float().to(device)

    # # 打印模型结构
    # print(model)
    # # 打印大网络中的参数
    # print("model 初始权重:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.data}")

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])  # 使用Adam优化器
    # 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=config['epoch_num'])  # * iters
    # 余弦退火的学习率
    lr = []

    batch_num = server.data_num // server.batch_size
    # Divide batches  # 取数据
    batches = gen_batches(server.data_num, server.batch_size)

    # 开始迭代
    for epoch in tqdm(range(config['epoch_num'])):
        # 每个epoch的平均loss
        epoch_loss = 0
        # Init clients
        # 这里就开始遍历每一个client 做各自的第一层处理
        for i in range(batch_num):
            # 取当前batch的数据对应的indexes
            batch_indexes = batches[i]
            server.set_batch_indexes(batch_indexes)
            for c in clients:
                c.set_batch_indexes(batch_indexes)
                # Step 1: server calls clients to send embedding data and receive
                # 得到第一层处理后的数据
                server.update_embedding_data(c)  # 客户端经过一层wx操作 将数据传给了服务器 (通过这一步操作，已经传过来了)
            loss = server.my2_cal_batch_embedding_grads(model, optimizer, scheduler, lr, config)
            # print(loss)
            epoch_loss += loss.data
            # Step 4: Clients update models
            for c in clients:
                c.update_weight()

            # 判断是否为最后一个epoch的最后一次迭代
            if epoch == (config["epoch_num"] - 1):
                # 保存att_weight
                last_ep_attn_weights_scalar.append(model.attn_weights)
        # 余弦退火 每个epoch 结束， 更新learning rate
        scheduler.step()
        epoch_loss /= (batch_num + 1)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        plot_loss.append(epoch_loss.item())  # 收集损失值

    plt.xlabel("epoch_num")
    plt.ylabel("mutal_information")
    plot_y1 = np.array(plot_loss).reshape(-1, )  # 可视化
    plt.plot(np.arange(len(plot_loss)), -plot_y1, 'r')  # 直接将|oss值取反，得到最大化互信息的值。
    # plot_y2 = np.array(plot_loss[1]).reshape(-1, )  # 可视化
    # plt.plot(np.arange(len(plot_loss[1])), -plot_y2, 'g', label="client2")  # 直接将|oss值取反，得到最大化互信息的值。
    # plot_y3 = np.array(plot_loss[2]).reshape(-1, )  # 可视化
    # plt.plot(np.arange(len(plot_loss[2])), -plot_y3, 'b', label="client3")  # 直接将|oss值取反，得到最大化互信息的值。
    # plt.legend()
    # plt.savefig('./runs/res_curve.pdf', bbox_inches='tight')
    # 在最后一个epoch结束后，保存所有数据
    torch.save(last_ep_attn_weights_scalar, 'last_ep_attn_weights_scalar.pt')
    # plt.show()