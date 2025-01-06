import random
import numpy as np

class Client(object):

    def __init__(self, X_train, config) -> None:
        self.device = config['device']
        self.X_train = X_train
        self.X_train_onehot = None

        # Extract config info
        self.batch_size = config['batch_size']
        self.lr = config['learning_rate']
        self.feature_num = X_train.shape[1]

        # Init client's params
        self.batch_indexes = [0] * self.batch_size

        # 设置随机数生成器的种子
        self.seed = config['seed']
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.weight = np.random.random(size=(config["feature_dim"],
                                             self.feature_num))
        self.embedding_grads = np.random.random(size=(config["feature_dim"],
                                                      self.batch_size))
        self.id = None
        # # 创建并应用自注意力模型
        # self.attention_model = selfAttention(num_attention_heads=8,
        #                                      input_size=config["batch_size"],
        #                                      hidden_size=config["batch_size"]).to(self.device

    def set_id(self, client_id):
        self.id = client_id

    def encode_data_to_onehot(self):
        # Define the number of classes for each feature
        num_classes_per_feature = [16, 16, 16,
                                   16]  # 15 classes for each of the 4 features

        # Initialize an empty tensor for one-hot encoding
        one_hot_encoding = np.zeros(
            (self.X_train.shape[0], sum(num_classes_per_feature)), dtype=int)

        # Compute the cumulative number of classes for each feature
        cumulative_classes_per_feature = np.cumsum(num_classes_per_feature)

        # Convert each feature to one-hot encoding and concatenate
        start_idx = 0
        for i, num_classes_i in enumerate(num_classes_per_feature):
            indices_i = self.X_train[:, i].astype(int)
            one_hot_i = np.eye(num_classes_i)[indices_i]
            end_idx = start_idx + num_classes_i

            # Concatenate the one-hot encodings for this feature
            one_hot_encoding[:, start_idx:end_idx] = one_hot_i
            start_idx = end_idx

        # 把编码后的结果赋值成client的属性
        self.X_train_onehot = one_hot_encoding

    def set_batch_indexes(self, batch_indexes):
        self.batch_indexes = batch_indexes

    def set_embedding_grads(self, embedding_grads):
        self.embedding_grads = embedding_grads

    # Update weight param of the model
    def update_weight(self):
        # 某个client的某个batch的数据
        X_batch = self.X_train[self.batch_indexes]
        # 求出某个batch的平均梯度
        grad = np.sum((np.dot(self.embedding_grads, X_batch))) / self.batch_size
        # SGD
        self.weight -= self.lr * grad

    # 得到本地网络的结果，以传送给server
    def get_embedding_data(self, period_type="batch"):
        """
        Return the embedding data, calculated on X.
            batch training - X_batch
            testing - X_test
        """
        if period_type == 'batch':

            # 根据index取数据
            X_batch = self.X_train[self.batch_indexes]
            # 客户端的第一层处理
            res = np.dot(self.weight, X_batch.T)  # (input_feature, bs) -> (config["feature_dim"], bs)
            # res = res[np.newaxis, :, :]
        return res