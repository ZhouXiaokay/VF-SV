import torch.nn as nn
import torch.nn.functional as F
import torch
class MINEBottomModel(nn.Module):
    def __init__(self, n_f_in, n_f_out):
        super().__init__()
        self.dense = nn.Linear(n_f_in, n_f_out, bias=False)

    def forward(self, x):
        x = self.dense(x)
        return x

class ItemLevelAttention(nn.Module):
    def __init__(self, input_size):
        super(ItemLevelAttention, self).__init__()
        # 定义hj和ht乘积
        self.hj_product = nn.Linear(input_size, input_size, bias=False)
        self.ht_product = nn.Linear(input_size, input_size, bias=False)
        # 定义激活后的乘积，将最终的结果映射为标量
        self.v_product = nn.Linear(input_size, 1, bias=False)
        # # 添加偏置，使其shape与hj ht一致
        # self.bias = nn.Parameter(torch.Tensor(batch_size, input_size))

    def forward(self, x, y):
        # 用没有bais的线性层模拟乘积操作

        # 计算hj
        # 获取输入张量的形状信息
        batch_size, num_clients, input_size = x.size()
        # 将输入张量的第二维度展平为(batch_size * num_clients, input_size)
        # (bs, feature_dim, num_clients) -> (bs*num_clients, feature_dim)
        x_flatten = x.contiguous().view(batch_size * num_clients, input_size)
        hj_flatten = self.hj_product(x_flatten)
        # 将输出张量的形状恢复为原始形状
        hj = hj_flatten.view(batch_size, num_clients, input_size)

        # 计算ht
        # 将输入张量的第二维度展平为(batch_size * num_clients, input_size)
        # (bs, feature_dim, num_clients) -> (bs*num_clients, feature_dim)
        y_flatten = y.view(batch_size * num_clients, input_size)
        ht_flatten = self.ht_product(y_flatten)
        # 将输出张量的形状恢复为原始形状
        ht = ht_flatten.view(batch_size, num_clients, input_size)

        # 对结果应用激活函数
        res1 = F.sigmoid(hj + ht).to(torch.float32)

        # 计算最终结果
        # 将输入张量的第二维度展平为(batch_size * num_clients, input_size)
        # (bs, feature_dim, num_clients) -> (bs*num_clients, feature_dim)
        res1_flatten = res1.view(batch_size * num_clients, input_size)
        res2_flatten = self.v_product(res1_flatten)
        # 将输出张量的形状恢复为原始形状
        res2 = res2_flatten.view(batch_size, num_clients, 1)

        return res2

# 主mine网络
class MINETopModel(nn.Module):
    def __init__(self, input_size=8, num_clients=4):
        super(MINETopModel, self).__init__()
        # 自注意力模块
        self.item_level_attention = ItemLevelAttention(input_size)
        # 线性层
        self.fc0 = nn.Linear(1, input_size)
        self.fc1 = nn.Linear(input_size, input_size * 2)
        self.fc2 = nn.Linear(input_size * 2, input_size * 4)
        self.fc3 = nn.Linear(input_size * 4, 1)

        # 定义额外的参数num_clients
        self.num_clients = torch.tensor(num_clients)

    def forward(self, clients_embedding_data, y):
        # 将y的shape 变为与client的embedding_data的shape一致
        y = self.fc0(y)
        # # 使用permute函数交换维度，(num_clients, feature_dim, bs) -> (bs, num_clients, feature_dim)
        # hj = clients_embedding_data.permute(2, 0, 1)
        # 使用permute函数交换维度，(num_clients, bs, feature_dim) -> (bs, num_clients, feature_dim)
        hj = clients_embedding_data.permute(1, 0, 2)
        ht = torch.stack([y] * self.num_clients, dim=1)
        # 注意力计算
        attn_res = self.item_level_attention(hj, ht)
        # 得到注意力值
        attn_weights = F.softmax(attn_res, dim=1).squeeze()
        # 保存注意力权重标量
        self.attn_weights = attn_weights.detach().cpu().numpy()

        # # 融合注意力模块的输出
        # x = (c1 * attn_weights[:, 0].unsqueeze(1) + c2 * attn_weights[:, 1].unsqueeze(1) +
        #      c3 * attn_weights[:, 2].unsqueeze(1) + c4 * attn_weights[:, 3].unsqueeze(1))

        # (num_clients, feature_dim, bs) -> (num_clients, bs, feature_dim)
        # clients_embedding_data = clients_embedding_data.permute(0, 2, 1)
        attn_weights = attn_weights.transpose(0, 1).unsqueeze(-1)
        element_wise_result = clients_embedding_data * attn_weights
        x = torch.sum(element_wise_result, dim=0)

        h1 = F.relu(self.fc1(x + y))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3