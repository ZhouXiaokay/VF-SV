import sys
code_path = '/home/zxk/codes/vfps_mi_diversity'
sys.path.append(code_path)
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from conf import global_args_parser

args = global_args_parser()

dataset = args.dataset
dir_path = code_path + '/baselines/VF_CE/save/' + dataset
client_num = 4



# 加载保存的weights
list_attn_weights = torch.load(dir_path + '/ep_attn_weights_scalar.pt', weights_only=False)
# 使用 concatenate 沿着第一个轴（axis=0）拼接数组
attn_weights = np.concatenate(list_attn_weights, axis=0)
print(attn_weights.shape)

fitted_means = {}
for i in range(client_num):
    data = attn_weights[:, i].squeeze()
    # 拟合高斯分布
    mean, std_dev = norm.fit(data)
    fitted_means[i] = mean
    # 生成拟合后的高斯分布曲线
    xmin, xmax = min(data), max(data)
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mean, std_dev)

    # 绘制直方图和拟合曲线
    plt.hist(data, bins=100, density=True, alpha=0.7, color='b', label='Histogram')
    plt.plot(x, p, 'k', linewidth=2, label='Fit results: mean=%.2f, std=%.2f' % (mean, std_dev))
    plt.legend()
    filename = f'{dir_path}/client_{i+1}_gaussian.pdf'
    plt.show()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    # # 添加额外的绘图，标出不符合高斯分布的数据
    # threshold = 0.02  # 阈值，可以根据实际情况调整
    # non_gaussian_data = data[abs(norm.pdf(data, mean, std_dev) - norm.pdf(data, mean, std_dev).max()) < threshold]
    # plt.scatter(non_gaussian_data, norm.pdf(non_gaussian_data, mean, std_dev), color='r', marker='x',
    #             label='Non-Gaussian Data')
    #
    # plt.legend()
    # plt.show()

# 按照值进行降序排序
sorted_fitted_means = dict(sorted(fitted_means.items(), key=lambda item: item[1], reverse=True))

# 将结果存入csv文件中
# 将新结果转为df
# 列名
columns1 = ['feature_index']
columns2 = ['attn_weights']
# 转换为 pandas 的 DataFrame
df1 = pd.DataFrame(list(sorted_fitted_means.keys()), columns=columns1)
df2 = pd.DataFrame(list(sorted_fitted_means.values()), columns=columns2)
# 合并2个 DataFrame
result_df = pd.concat([df1, df2], axis=1)
# 指定 CSV 文件路径
csv_file_path = dir_path + '/attn_weights.csv'
# 将合并后的 DataFrame 写入 CSV 文件
result_df.to_csv(csv_file_path, index=False)