import numpy as np
import pandas as pd
from utils.helpers import seed_torch
from conf import global_args_parser

global_args = global_args_parser()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import FastICA
import os
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
def load_csv(f_path):
    """ Load data set """
    data = pd.read_csv(f_path)
    return data


def load_txt(txt_path):
    data = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.split()
            data.append(splits)
    np_data = np.array(data).astype(int)
    return np_data


def vertical_split(data, num_users):
    seed_torch()
    num_features = int(data['x'].shape[1] / num_users)
    f_id = [i for i in range(data['x'].shape[1])]
    split_result = {}
    split_f = []
    for i in range(num_users):
        if i == num_users - 1:
            leave_id = list(set(f_id) - set(sum(split_f, [])))
            split_f.append(list(leave_id))
        else:
            t = set(np.random.choice(f_id, num_features, replace=False))
            split_f.append(list(t))
            f_id = list(set(f_id) - t)
    client_rank = 0
    for item in split_f:
        x_sub = concat_split_result(item, data['x'])
        split_result[client_rank] = x_sub
        client_rank += 1

    return split_result


def concat_split_result(r_list, npx):
    x_list = []
    for i in r_list:
        x_list.append(npx[:, i:i + 1])
    return np.concatenate(tuple(x_list), axis=1)


def load_credit_data(csv_path='/home/zxk/data/credit/credit.csv'):
    data = load_csv(csv_path)
    data.rename(columns={'default.payment.next.month': 'def_pay'}, inplace=True)
    data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    x = data.drop(['def_pay', 'ID'], axis=1)
    x_std = MinMaxScaler().fit_transform(x)
    # x_std = StandardScaler().fit_transform(x)
    y = data.def_pay
    # dataset = {'id': data.index.values, 'x': x.values, 'y': y.values}
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_bank_data(csv_path='/home/zxk/data/bank/bank.csv'):
    data = load_csv(csv_path)
    data = data.drop(['customer_id'], axis=1)
    data['country'] = LabelEncoder().fit_transform(data['country'])
    data['gender'] = LabelEncoder().fit_transform(data['gender'])

    y = data['churn']
    x = data.copy()
    x.drop('churn', axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    # print(x_std)
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_mushroom_data(csv_path='/home/zxk/data/mushroom/mushrooms.csv'):
    data = load_csv(csv_path)
    data.drop(columns='veil-type', inplace=True)
    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])
    y = data['class']
    x = data.copy()
    x.drop('class', axis=1, inplace=True)
    x_std = MinMaxScaler().fit_transform(x)
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_covtype_data(data_path='/home/zxk/data/covtype/covtype.libsvm.binary.scale.bz2'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1] - 1
    x_std = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_g2_data(data_path='/home/zxk/data/g2/g2-128-10.txt'):
    x = load_txt(data_path)
    c_1 = np.zeros(1024)
    c_2 = np.ones(1024)
    y = np.concatenate((c_1, c_2))
    idx = np.arange(0, x.shape[0])
    dataset = {'id': idx, 'x': x, 'y': y}
    return dataset


def load_phishing_data(data_path='/home/zxk/data/phishing/phishing.txt'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_adult_data(data_path='/home/zxk/data/libsvm'):
    train_path = data_path + '/a8a.txt'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    train_X = train_X[:, :-1]

    test_path = data_path + '/a8a.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_web_data(data_path='/home/zxk/data/libsvm'):
    train_path = data_path + '/w8a.txt'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/w8a.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_ijcnn_data(data_path='/home/zxk/data/libsvm'):
    train_path = data_path + '/ijcnn1.tr'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/ijcnn1.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = MinMaxScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_splice_data(data_path='/home/zxk/data/libsvm/'):
    train_path = data_path + '/splice'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/splice.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = MinMaxScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_madelon_data(data_path='/home/zxk/data/madelon/'):
    train_path = data_path + '/madelon'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/madelon.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = MinMaxScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_gisette_data(data_path='/home/zxk/data/gisette/'):
    train_path = data_path + '/gisette_scale'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/gisette_scale.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    x_std = MinMaxScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_SUSY_data(data_path='/home/zxk/data/SUSY/SUSY'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_heart_data(data_path='/home/zxk/data/heart/heart.csv'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_spambase_data(csv_path="/home/zxk/data/spambase/spambase.data"):
    data = load_csv(csv_path)

    le = LabelEncoder()
    y = le.fit_transform(data.iloc[:, -1])

    x = data.iloc[:, :-1]
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y}

    return dataset


def load_HIGGS_data(data_path='/home/zxk/codes/data/HIGGS/HIGGS', sample_size=None, random_seed=None):
    data = load_svmlight_file(data_path)

    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0

    # 分别获取两个类别的索引
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # 计算每个类别的采样数量
    if sample_size is not None:
        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples_class_0 = int(sample_size / 2)
        num_samples_class_1 = sample_size - num_samples_class_0

        # 随机选择每个类别的索引
        sampled_indices_class_0 = np.random.choice(class_0_indices, num_samples_class_0, replace=False)
        sampled_indices_class_1 = np.random.choice(class_1_indices, num_samples_class_1, replace=False)

        # 合并两个类别的采样索引
        sampled_indices = np.concatenate([sampled_indices_class_0, sampled_indices_class_1])

        # 使用采样索引来选择数据
        x = x[sampled_indices]
        y = y[sampled_indices]

    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_realsim_data(data_path='/home/zxk/data/real-sim/real-sim', sample_size=None, random_seed=None):
    data = load_svmlight_file(data_path)

    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0

    # 分别获取两个类别的索引
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # 计算每个类别的采样数量
    if sample_size is not None:
        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples_class_0 = int(sample_size / 2)
        num_samples_class_1 = sample_size - num_samples_class_0

        # 随机选择每个类别的索引
        sampled_indices_class_0 = np.random.choice(class_0_indices, num_samples_class_0, replace=False)
        sampled_indices_class_1 = np.random.choice(class_1_indices, num_samples_class_1, replace=False)

        # 合并两个类别的采样索引
        sampled_indices = np.concatenate([sampled_indices_class_0, sampled_indices_class_1])

        # 使用采样索引来选择数据
        x = x[sampled_indices]
        y = y[sampled_indices]

    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_epsilon_data(data_path='/home/zxk/data/epsilon/', sample_size=None, random_seed=None):
    train_path = data_path + '/epsilon_normalized'
    train_data = load_svmlight_file(train_path)
    train_X, train_y = train_data[0].toarray(), train_data[1]

    # train_X = train_X[:, :-1]

    test_path = data_path + '/epsilon_normalized.t'
    test_data = load_svmlight_file(test_path)
    test_X, test_y = test_data[0].toarray(), test_data[1]

    x = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    y[y == -1] = 0

    # 分别获取两个类别的索引
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # 计算每个类别的采样数量
    if sample_size is not None:
        if random_seed is not None:
            np.random.seed(random_seed)

        num_samples_class_0 = int(sample_size / 2)
        num_samples_class_1 = sample_size - num_samples_class_0

        # 随机选择每个类别的索引
        sampled_indices_class_0 = np.random.choice(class_0_indices, num_samples_class_0, replace=False)
        sampled_indices_class_1 = np.random.choice(class_1_indices, num_samples_class_1, replace=False)

        # 合并两个类别的采样索引
        sampled_indices = np.concatenate([sampled_indices_class_0, sampled_indices_class_1])

        # 使用采样索引来选择数据
        x = x[sampled_indices]
        y = y[sampled_indices]

    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_magicGammaTelescope_data(csv_path='/home/zxk/data/magic+gamma+telescope/magic04.csv'):
    data = load_csv(csv_path)

    le = LabelEncoder()
    y = le.fit_transform(data.iloc[:, -1])
    y[y == 'g'] = 0
    y[y == 'h'] = 1

    x = data.iloc[:, :-1]
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y}

    return dataset


def load_smkdrk_data(csv_path='/home/zxk/data/SMK_DRK/smoking_driking_dataset_Ver01.csv'):
    data = load_csv(csv_path)

    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['DRK_YN']
    x = data.copy()
    x.drop('DRK_YN', axis=1, inplace=True)
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_nba_player_data(csv_path='/home/zxk/data/nba_players/nba-players.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['target_5yrs']
    x = data.copy()
    columns_to_drop = ['id', 'name', 'target_5yrs']
    x.drop(columns_to_drop, axis=1, inplace=True)
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_heart_disease_data(csv_path='/home/zxk/data/heart_disease/heart_disease_health_indicators.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['HeartDiseaseorAttack']
    x = data.copy()
    x.drop('HeartDiseaseorAttack', axis=1, inplace=True)
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_rice_data(csv_path='/home/zxk/data/rice/riceClassification.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['Class']
    x = data.copy()
    x.drop(['id', 'Class'], axis=1, inplace=True)
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_creditcard_data(csv_path='/home/zxk/data/creditcard/creditcard.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['class']
    x = data.copy()
    x.drop(['time', 'amount', 'class'], axis=1, inplace=True)
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_smoke_data(csv_path='/home/zxk/data/smoke/smoke_detection_iot.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['Fire Alarm']
    x = data.copy()
    x.drop(['Fire Alarm', 'ID'], axis=1, inplace=True)
    x_std = MinMaxScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_dummy_partition_with_label(dataset, num_clients, client_rank):
    split_x = vertical_split(dataset, num_clients)
    x = split_x[client_rank]
    y = dataset['y']
    return x, y

def load_dummy_partition_mutual_info(dataset, num_clients, client_rank):
    x = dataset['x']
    y = dataset['y']
    mutual_info = mutual_info_classif(x, y)
    mutual_info_series = pd.Series(mutual_info)
    slices = pd.qcut(mutual_info_series, num_clients, labels=False)


    selected_features = mutual_info_series[slices == client_rank].index
    split_x = x[:, selected_features]

    return split_x, y

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
# def vertical_split_based_correlation(features, num_users):
#
#     col_num = features.shape[1]
#
#     # 使用PCA尝试提取独立特征
#     pca = PCA(n_components=col_num)
#     pca_features = pca.fit_transform(features)
#
#     # 均匀分割PCA后的特征
#     split_data = np.array_split(pca_features, num_users, axis=1)
#
#     # 计算分割后的数据子集之间的互信息
#     # 由于互信息需要标量输出，我们可以简单地计算每个分割之间的每个特征的互信息，并求平均值
#     mi_scores = []
#     for i in range(len(split_data)):
#         for j in range(i + 1, len(split_data)):
#             # 计算互信息
#             mi = mutual_info_regression(split_data[i], split_data[j].mean(axis=1))
#             mi_scores.append(mi.mean())
#
#     # 输出互信息结果
#     print("互信息评分：", mi_scores)
from scipy.cluster.hierarchy import linkage, fcluster
def vertical_split_by_correlation(features, num_users):
    # 标准化数据以减少尺度影响
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # 计算特征的相关性矩阵
    df = pd.DataFrame(features_scaled)
    correlation_matrix = df.corr().abs()

    # 使用层次聚类根据相关性矩阵进行聚类
    linked = linkage(correlation_matrix, 'complete')

    # 根据最大集群数量确定每个特征的集群标签
    cluster_labels = fcluster(linked, t=num_users, criterion='maxclust')

    # 将特征按聚类结果分割
    partitions = [[] for _ in range(num_users)]
    for i, label in enumerate(cluster_labels):
        partitions[label - 1].append(df.columns[i])

    # 根据索引返回实际数据的子集
    subset_data = [features[:, indices] for indices in partitions]

    # print(partitions)

    return subset_data


def partition_features(X, y, num_partitions):
    """
    对特征矩阵X进行分区，使得每个分区内的特征间相关性较高，分区间的相关性较低，并且每个分区与标签y的互信息从小到大排序，且互信息均大于零。

    参数:
    X -- 特征矩阵，形状为(n_samples, n_features)
    y -- 标签数组，形状为(n_samples,)
    num_partitions -- 需要的分区数量

    返回:
    sorted_partitions -- 按互信息排序的分区列表，互信息均大于零
    """

    # 步骤1: 标准化特征矩阵
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 步骤2: 计算标准化后的特征矩阵的距离矩阵，并进行层次聚类
    dist_matrix = pdist(X_scaled)
    linked = linkage(dist_matrix, method='average')

    # 步骤3: 使用fcluster进行分区
    labels = fcluster(linked, t=num_partitions, criterion='maxclust')

    # 步骤4: 将特征按聚类结果分组
    partitions = {i: [] for i in range(1, num_partitions + 1)}
    for i, cluster_id in enumerate(labels):
        partitions[cluster_id].append(i)

    # 步骤5: 计算每个分区与标签的互信息，并过滤掉互信息为零的分区
    mutual_info_scores = []
    for k, indices in partitions.items():
        subset = X[:, indices]
        mi_score = mutual_info_classif(subset, y, discrete_features=False, random_state=42)
        avg_mi = np.mean(mi_score)
        if avg_mi > 0:
            mutual_info_scores.append((avg_mi, indices))

    # 步骤6: 按互信息从小到大排序分区
    sorted_partitions = sorted(mutual_info_scores, key=lambda x: x[0])

    return [indices for _, indices in sorted_partitions]

def load_dummy_partition_by_correlation(dataset, num_clients, client_rank):
    x = dataset['x']
    y = dataset['y']
    # x_split_list = vertical_split_by_correlation(x, num_clients)
    x_split_list = partition_features(x, y, num_clients)
    x_split = x_split_list[client_rank]
    return x_split, y


def generate_independent_data(num_partitions, num_samples=30000, num_features=20, seed=66):
    np.random.seed(seed)  # Global random seed
    labels = np.random.choice([0, 1], size=num_samples)  # Balanced labels
    data_partitions = []

    # Define different levels of label effects and shuffle them
    label_effects = np.linspace(1, 10, num_partitions)
    np.random.shuffle(label_effects)  # Shuffle to randomize the order of effects

    for i in range(num_partitions):
        np.random.seed(i)  # Different seed for each partition
        base_means = np.random.rand(num_features) * 1000
        base_vars = np.random.rand(num_features) * 100

        # Differentiate the impact based on label
        features = np.zeros((num_samples, num_features))
        for j in range(num_samples):
            effect = label_effects[i] if labels[j] == 1 else 0
            features[j, :] = np.random.normal(loc=base_means + effect, scale=base_vars, size=num_features)

        # Adding some noise to enhance diversity
        noise = np.random.normal(0, 1, size=(num_samples, num_features))
        features += noise * 0.2  # Control the amount of noise

        data_partitions.append(features)

    # Sort partitions by calculated mutual information with labels (for demonstration)
    data_partitions.sort(key=lambda x: np.mean([np.corrcoef(x[:, i], labels)[0, 1] for i in range(num_features)]))


    return data_partitions, labels


def load_dependent_data(client_rank, num_partitions, num_samples=30000, num_features=20, seed=66):
    data_partitions, labels = generate_independent_data(num_partitions,num_samples, num_features, seed)
    scaler = MinMaxScaler()
    scaled_partitions = [scaler.fit_transform(partition) for partition in data_partitions]
    # x_split = data_partitions[client_rank]
    x_split = scaled_partitions[client_rank]
    return x_split, labels
    # return {'x':x_split, 'y':labels, 'X':np.concatenate(data_partitions, axis=1)}

"""
return the concatenated data of all partitions
"""
def load_dependent_features(num_partitions, num_samples=30000, num_features=20, seed=66):
    data_partitions, labels = generate_independent_data(num_partitions, num_samples, num_features, seed)
    scaler = MinMaxScaler()
    scaled_partitions = [scaler.fit_transform(partition) for partition in data_partitions]
    return np.concatenate(scaled_partitions, axis=1)

def load_and_split_dataset(d_name):
    try:
        # Check if the file exists
        if os.path.exists("/home/zxk/codes/VF-SV/data_loader/mi_split_info.csv"):
            print("File exists")
            # Read the CSV file
            mi_split_info = pd.read_csv("/home/zxk/codes/VF-SV/data_loader/mi_split_info.csv",
                                        header=None)

            # Find the row corresponding to the dataset name
            row = mi_split_info[mi_split_info[0] == d_name]

            if not row.empty:
                # Extract the feature groups
                feature_groups = row.iloc[0, 1:].values

                # Load the dataset
                data = choose_dataset(d_name)
                if data == -1:
                    return None

                # Split the dataset based on the feature groups
                split_result = {}
                split_features = []
                for i, group in enumerate(feature_groups):
                    feature_indices = list(map(int, group.strip('[]').split(', ')))
                    # split_result[f"part_{i + 1}"] = data['x'][:, feature_indices]
                    split_result[i] = data['x'][:, feature_indices]
                    split_features.append(data['x'][:, feature_indices])

                all_features = np.concatenate(split_features, axis=1)
                split_result['labels'] = data['y']
                split_result['features'] = all_features
                return split_result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def choose_dataset(d_name):
    if d_name == 'credit':
        data = load_credit_data()
    elif d_name == 'bank':
        data = load_bank_data()
    elif d_name == 'mushroom':
        data = load_mushroom_data()
    elif d_name == 'covtype':
        data = load_covtype_data()
    elif d_name == 'adult':
        data = load_adult_data()
    elif d_name == 'web':
        data = load_web_data()
    elif d_name == 'phishing':
        data = load_phishing_data()
    elif d_name == 'ijcnn':
        data = load_ijcnn_data()
    elif d_name == 'splice':
        data = load_splice_data()
    elif d_name == 'SUSY':
        data = load_SUSY_data()
    elif d_name == 'heart':
        data = load_heart_data()
    elif d_name == 'HIGGS':
        data = load_HIGGS_data('/home/zxk/codes/data/HIGGS/HIGGS', 100000, 1)
    elif d_name == 'madelon':
        data = load_madelon_data()
    elif d_name == 'real-sim':
        data = load_realsim_data('/home/zxk/codes/data/real-sim/real-sim', 100000, 1)
    elif d_name == 'epsilon':
        data = load_epsilon_data('/home/zxk/codes/data/epsilon', 10000, 1)
    elif d_name == 'gisette':
        data = load_gisette_data()
    elif d_name == 'spambase':
        data = load_spambase_data()
    elif d_name == 'magicGammaTelescope':
        data = load_magicGammaTelescope_data()
    elif d_name == 'smk-drk':
        data = load_smkdrk_data()
    elif d_name == 'heart-disease':
        data = load_heart_disease_data()
    elif d_name == 'rice':
        data =load_rice_data()
    elif d_name == 'creditcard':
        data = load_creditcard_data()
    elif d_name == 'smoke':
        data = load_smoke_data()
    else:
        print("there's not this dataset")
        return -1
    return data

if __name__ == '__main__':
    data = load_and_split_dataset('credit')
    print(data[1])
    # x = data['x']
    # vertical_split_by_correlation(x,4)
    # x_1, y_1 = load_dummy_partition_with_label(data, 4, 0)
    # x_2, y_2 = load_dummy_partition_with_label(data, 4, 1)
    # x_3, y_3 = load_dummy_partition_with_label(data, 4, 2)
    # x_4, y_4 = load_dummy_partition_with_label(data, 4, 3)
    # x_l = [x_1, x_2, x_3, x_4]
    # mi_scores = []
    # for i in range(len(x_l)):
    #     for j in range(i + 1, len(x_l)):
    #         # 计算互信息
    #         mi = mutual_info_regression(x_l[i], x_l[j].mean(axis=1))
    #         mi_scores.append(mi.mean())
    #
    # # 输出互信息结果
    # print("互信息评分：", mi_scores)
