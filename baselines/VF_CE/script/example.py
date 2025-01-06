import random
import torch
from baselines.VF_CE.src import *

# 设置运行项目的设备
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

file_name = 'optdigits'
X_train, Y_train = load_data(file_name)
Y_train_shuffle = Y_train.copy()
random.shuffle(Y_train_shuffle)

# config
config = dict()
# 项目运行的设备
config['device'] = device
# client的数目
config['client_num'] = 4
# 数据集的特征个数
config['feature_num'] = 64
# 定义划分的份数
config['splits'] = [2, 8, 16, 38]
# 训练的超参数
config['epoch_num'] = 10000
config['batch_size'] = 2048
config['learning_rate'] = 15e-5
# client本地网络处理后输入中心模型的feature数
config['feature_dim'] = 8
# 设置网络参数的初始化随机种子
config['seed'] = 42

# Split Data  ->  For client
X_train_s = split_data(X_train, config['splits'], config['feature_num'])

# 添加自注意力层，在输入数据上执行注意力操作，并返回新的表示

# Init server
server = Server(Y_train, Y_train_shuffle, config)
# Init clients
clients = list()
# 创建客户端
for i in range(config['client_num']):
    c = Client(X_train_s[i], config)
    c.set_id(i)
    # # 将对应client数据进行onehot编码
    # c.encode_data_to_onehot()
    clients.append(c)

# 建立联系
server.attach_clients(clients)  # 直接变成自己的属性，实现数据的通信

# Train
vfl_lr_train(server, clients, config)