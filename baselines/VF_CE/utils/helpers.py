import numpy as np
import pandas as pd


# 模拟取数据集
def gen_batches(data_num, batch_size):
    # 计算一个epoch要迭代多少次
    batch_num = data_num // batch_size
    # n个[batch_size]
    size_list = [batch_size] * batch_num
    # 随机打乱所有数据
    indexes = np.arange(data_num)
    np.random.shuffle(indexes)
    res = list()
    b = 0
    # 为每个batch装填数据
    for size in size_list:
        res.append(indexes[b:b + size])
        b += size
    return res

def load_data(file_name):
    if file_name == 'optdigits':
        data = pd.read_csv('/home/zxk/codes/vfps_mi_diversity/baselines/VF_CE/data/optdigits/optdigits.csv')
        data = np.array(data)
        # 提取前十列数据到X_train变量
        X_train = data[:, :64]
        # 提取第11列数据到Y_train变量
        Y_train = data[:, 64]
        Y_train = Y_train.reshape(3822, 1)
        return X_train, Y_train

    else:
        raise Exception('this dataset is not supported yet')

def split_data(X_train, splits, feature_num):
    # # 设置随机数生成器的种子，以确保每次运行得到相同的结果
    # np.random.seed(13)
    # # 获取第二维度的索引顺序
    # original_indices_second_dim = np.arange(X_train.shape[1])
    # # 通过设置种子来确保每次生成相同的交换顺序
    # rng = np.random.default_rng(13)
    # new_indices_second_dim = rng.permutation(original_indices_second_dim)
    # # 将数据按照新的第二维度顺序重新排列
    # X_train = X_train[:, new_indices_second_dim]

    # 测试划分是否合理
    if(sum(splits) == feature_num):
        # 使用np.split函数按照第一个维度划分数组
        X_train_s = np.split(X_train, np.cumsum(splits)[:-1], axis=1)
    else:
        raise Exception('this split is not reasonable')

    return X_train_s