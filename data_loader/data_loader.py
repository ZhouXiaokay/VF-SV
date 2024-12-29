import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import mutual_info_classif
# from k_means_constrained import KMeansConstrained

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


def load_credit_data(csv_path='/home/xy_li/data/credit/credit.csv'):
    data = load_csv(csv_path)
    data.rename(columns={'default.payment.next.month': 'def_pay'}, inplace=True)
    data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
    x = data.drop(['def_pay', 'ID'], axis=1)
    x_std = StandardScaler().fit_transform(x)
    y = data.def_pay
    # dataset = {'id': data.index.values, 'x': x.values, 'y': y.values}
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_bank_data(csv_path='/home/xy_li/data/bank/bank.csv'):
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


def load_mushroom_data(csv_path='/home/xy_li/data/mushroom/mushrooms.csv'):
    data = load_csv(csv_path)
    data.drop(columns='veil-type', inplace=True)
    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])
    y = data['class']
    x = data.copy()
    x.drop('class', axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)
    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_covtype_data(data_path='/home/xy_li/data/covtype/covtype.libsvm.binary.scale.bz2'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1] - 1
    x_std = StandardScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_g2_data(data_path='/home/xy_li/data/g2/g2-128-10.txt'):
    x = load_txt(data_path)
    c_1 = np.zeros(1024)
    c_2 = np.ones(1024)
    y = np.concatenate((c_1, c_2))
    idx = np.arange(0, x.shape[0])
    dataset = {'id': idx, 'x': x, 'y': y}
    return dataset


def load_phishing_data(data_path='/home/xy_li/data/phishing/phishing.txt'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_adult_data(data_path='/home/xy_li/data/libsvm'):
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

    x_std = StandardScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_web_data(data_path='/home/xy_li/data/libsvm'):
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


def load_ijcnn_data(data_path='/home/xy_li/data/libsvm'):
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

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_splice_data(data_path='/home/xy_li/data/libsvm/'):
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

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_madelon_data(data_path='/home/xy_li/data/madelon/'):
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

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_gisette_data(data_path='/home/xy_li/data/gisette/'):
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

    x_std = StandardScaler().fit_transform(x)
    # x_max_min = MinMaxScaler().fit_transform(x)
    dataset = {'id': np.array(range(len(x))), 'x': x_std, 'y': y}

    return dataset


def load_SUSY_data(data_path='/home/xy_li/data/SUSY/SUSY', sample_size=None, random_seed=None):

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


def load_smkdrk_data(csv_path='/home/xy_li/data/SMK_DRK/smoking_driking_dataset_Ver01.csv', sample_size = None, random_seed=None):
    data = load_csv(csv_path)

    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['DRK_YN']
    y = y.values
    x = data.copy()
    x.drop('DRK_YN', axis=1, inplace=True)
    x = StandardScaler().fit_transform(x)
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

    dataset = {'id': data.index.values, 'x': x, 'y': y}
    return dataset

def load_heart_data(data_path='/home/xy_li/data/heart/heart'):
    data = load_svmlight_file(data_path)
    x = data[0].toarray()
    y = data[1]
    y[y == -1] = 0
    dataset = {'id': np.array(range(len(x))), 'x': x, 'y': y}

    return dataset


def load_spambase_data(csv_path="/home/xy_li/data/spambase/spambase.csv"):
    data = load_csv(csv_path)

    le = LabelEncoder()
    y = le.fit_transform(data.iloc[:, -1])

    x = data.iloc[:, :-1]
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y}

    return dataset


def load_HIGGS_data(data_path='/home/xy_li/data/HIGGS/HIGGS', sample_size=None, random_seed=None):
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


def load_realsim_data(data_path='/home/xy_li/data/real-sim/real-sim', sample_size=None, random_seed=None):
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


def load_epsilon_data(data_path='/home/xy_li/data/epsilon/', sample_size=None, random_seed=None):
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


def load_magicGammaTelescope_data(csv_path='/home/xy_li/data/magic+gamma+telescope/magic04.csv'):
    data = load_csv(csv_path)

    le = LabelEncoder()
    y = le.fit_transform(data.iloc[:, -1])
    y[y == 'g'] = 0
    y[y == 'h'] = 1

    x = data.iloc[:, :-1]
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y}

    return dataset


def load_nba_player_data(csv_path='/home/xy_li/data/nba_players/nba-players.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['target_5yrs']
    x = data.copy()
    columns_to_drop = ['id', 'name', 'target_5yrs']
    x.drop(columns_to_drop, axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_heart_disease_data(csv_path='/home/xy_li/data/heart_disease/heart_disease_health_indicators.csv', sample_size = None, random_seed=None):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['HeartDiseaseorAttack'].values
    x = data.copy()
    x.drop('HeartDiseaseorAttack', axis=1, inplace=True)
    x = StandardScaler().fit_transform(x)

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

    dataset = {'id': data.index.values, 'x': x, 'y': y}
    return dataset


def load_rice_data(csv_path='/home/xy_li/data/rice/riceClassification.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['Class']
    x = data.copy()
    x.drop(['id', 'Class'], axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_creditcard_data(csv_path='/home/xy_li/data/creditcard/creditcard.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['class']
    x = data.copy()
    x.drop(['time', 'amount', 'class'], axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset


def load_smoke_data(csv_path='/home/xy_li/data/smoke/smoke_detection_iot.csv'):
    data = load_csv(csv_path)

    for col in data.columns:
        # data = data.apply(le.fit_transform)
        data[col] = LabelEncoder().fit_transform(data[col])

    y = data['Fire Alarm']
    x = data.copy()
    x.drop(['Fire Alarm', 'ID'], axis=1, inplace=True)
    x_std = StandardScaler().fit_transform(x)

    dataset = {'id': data.index.values, 'x': x_std, 'y': y.values}
    return dataset

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
        data = load_HIGGS_data('/home/xy_li/data/HIGGS/HIGGS', 100000, 1)
    elif d_name == 'madelon':
        data = load_madelon_data()
    elif d_name == 'real-sim':
        data = load_realsim_data('/home/xy_li/data/real-sim/real-sim', 100000, 1)
    elif d_name == 'epsilon':
        data = load_epsilon_data('/home/xy_li/data/epsilon', 10000, 1)
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
""""
def split_dataset(d_name, num_parts=4):
    try:
        dataset = choose_dataset(d_name)
        X = pd.DataFrame(dataset['x'])
        y = dataset['y']

        # 计算互信息
        mi = mutual_info_classif(X, y, random_state=42)
        mi_info = pd.DataFrame({'feature': X.columns, 'mutual_information': mi})

        # 按互信息排序
        mi_info_sorted = mi_info.sort_values(by='mutual_information', ascending=False)
        mi_values = mi_info_sorted['mutual_information'].values.reshape(-1, 1)

        k = num_parts

        kmeans = KMeansConstrained(n_clusters=k, size_min= int(X.shape[1]/num_parts), random_state=42)
        kmeans.fit(mi_values)

        mi_info_sorted['cluster'] = kmeans.labels_

        dataset_dict = {}
        group_mi_with_labels = []

        for cluster_id in range(k):
            group_features = mi_info_sorted[mi_info_sorted['cluster'] == cluster_id]
            feature_names = group_features['feature'].values
            grouped_data = X[feature_names]
            group_mi_with_labels.append((group_features.iloc[0]['mutual_information'], cluster_id, grouped_data))

        group_mi_with_labels.sort(key=lambda x: x[0], reverse=True)

        for new_id, (_, original_id, grouped_data) in enumerate(group_mi_with_labels):
            dataset_dict[f"part_{new_id + 1}"] = grouped_data
            print(f"Group {new_id + 1} (original ID: {original_id}):")
            print(mi_info_sorted[mi_info_sorted['cluster'] == original_id][['feature', 'mutual_information']])

        dataset_dict['labels'] = pd.DataFrame(y, columns=['label'])

        # Create a new DataFrame with the d_name string as the first row
        d_name_df = pd.DataFrame({0: [d_name]})

        # Combine feature lists into a single string for each cluster
        feature_strings = []
        for new_id in range(k):
            features = list(mi_info_sorted[mi_info_sorted['cluster'] == group_mi_with_labels[new_id][1]]['feature'].index)
            feature_string = f"[{', '.join(map(str, features))}]"
            feature_strings.append(feature_string)

        # Create a single row DataFrame with the d_name and combined feature strings
        combined_row = pd.DataFrame([[d_name] + feature_strings])

        # Append the combined DataFrame to the CSV file without the header
        combined_row.to_csv("mi_split_info.csv", index=False, header=False, mode='a')
        print(f"Appending split mutual information info to mi_split_info.csv")

        print("Dataset splitting and storage in dictionary complete.")
        return dataset_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
"""

def load_and_split_dataset(d_name):
    try:
        # Check if the file exists
        if os.path.exists("mi_split_info.csv"):
            # Read the CSV file
            mi_split_info = pd.read_csv("mi_split_info.csv", header=None)

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
                for i, group in enumerate(feature_groups):
                    feature_indices = list(map(int, group.strip('[]').split(', ')))
                    split_result[f"part_{i + 1}"] = data['x'][:, feature_indices]

                split_result['labels'] = pd.DataFrame(data['y'], columns=['label'])
                return split_result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None