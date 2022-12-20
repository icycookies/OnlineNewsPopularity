import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr


# def load_data(path):
#     df = pd.read_csv(path)
#     df = df[df[' n_unique_tokens'] <= 1]
#     data = df.to_numpy()[:, 2:].astype(float)
#     log_rows = [1, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
#     for i in range(data.shape[1]):
#         if i in log_rows:
#             data[:, i] = np.log(data[:, i] - np.min(data[:, i]) + 1)
#         data[:, i] = MinMaxScaler().fit_transform(data[:, i].reshape(-1, 1)).reshape(1, -1)[0]
#     return data[:, :-1], data[:, -1]


def load_data(path):
    df = pd.read_csv(path)
    df = df[df[' n_unique_tokens'] <= 1]
    data = df.to_numpy()[:, 2:].astype(float)
    labels = data[:, -1]
    data = data[:, :-1]

    log_rows = [1, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    for i in range(data.shape[1]):
        if i in log_rows:
            data[:, i] = np.log(data[:, i] - np.min(data[:, i]) + 1)
        data[:, i] = MinMaxScaler().fit_transform(data[:, i].reshape(-1, 1)).reshape(1, -1)[0]
    
    labels = np.where(labels >= 1400, 1, 0)
    # return data[:, 1:], labels
    return data, labels


# def random_split(x, y, train_ratio, val_ratio, test_ratio):
#     assert train_ratio + val_ratio + test_ratio == 1
#     num_train_val = int(len(y) * (train_ratio + val_ratio))
#     test_x, test_y = x[num_train_val:, :], y[num_train_val:]
#     perm = np.random.permutation(num_train_val)
#     num_train = int(len(y) * train_ratio)
#     train_x, train_y = x[perm[:num_train], :], y[perm[:num_train]]
#     val_x, val_y = x[perm[num_train:], :], y[perm[num_train:]]
#     return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def random_split(x, y, train_ratio, test_ratio):
    assert train_ratio + test_ratio <= 1
    num_train_val = int(len(y) * (1 - test_ratio))
    num_train = int(len(y) * train_ratio)

    perm = np.random.permutation(num_train_val)

    test_x, test_y = x[num_train_val:, :], y[num_train_val:]
    train_x, train_y = x[perm[:num_train]], y[perm[:num_train]]
    return (train_x, train_y), (test_x, test_y)


def split_by_time(x, y, train_ratio, test_ratio, reverse=False):
    assert train_ratio + test_ratio <= 1
    num_train_val = int(len(y) * (1 - test_ratio))
    num_train = int(len(y) * train_ratio)

    test_x, test_y = x[num_train_val:, :], y[num_train_val:]

    if reverse:
        train_x, train_y = x[num_train_val-num_train:num_train_val], y[num_train_val-num_train:num_train_val]
    else:
        train_x, train_y = x[:num_train], y[:num_train]
    return (train_x, train_y), (test_x, test_y)

def process_features(x, y, topk=10, reverse=False):
    pcc = []
    for i in range(x.shape(1)):
        pcc.append((pearsonr(x[:, i], y)[0], i))
    pcc = sorted(pcc)
    p = np.array([x[1] for x in pcc[:topk]])
    if reverse:
        p = np.flip(p)
    return x[:, p]


def analyze(weight, sample=None):
    raw_data = pd.read_csv("OnlineNewsPopularity.csv")
    cols = raw_data.columns.tolist()[2:-1]
    weight = weight.reshape(-1).tolist()
    if sample is not None:
        sample = sample.reshape(-1).tolist()
        pd.DataFrame({"Feature": cols, "Weight": weight, "Sample": sample}).to_csv("feature_weight.csv")
    else:
        pd.DataFrame({"Feature": cols, "Weight": weight}).to_csv("feature_weight.csv")


def plot_split():
    # split_ratio = [0.008, 0.04, 0.4, 0.8]
    split_ratio = [0.01, 0.05, 0.2, 0.5, 1.0]
    acc = [0.5965, 0.6273, 0.6367, 0.6395, 0.6401]
    auc = [0.5969, 0.6274, 0.6370, 0.6398, 0.6406]
    f1 = [0.5863, 0.6166, 0.6282, 0.6326, 0.6338]

    # plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
    # plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
    # plt.figure()
    for i, y in {"Accuracy": acc, "AUC": auc, "F1": f1}.items():
        plt.plot(np.arange(len(y)), y, marker="x", label=i)
    plt.legend()
    plt.xlabel("Split Ratio")
    plt.ylabel("Score")
    plt.title(r"Effect of Split Ratio on Model Performance")
    plt.xticks(np.arange(5), ['0.01', '0.05', '0.2', '0.5', '1.0'])
    # plt.show()
    plt.savefig("split_ratio.png")


if __name__ == "__main__":
    plot_split()
    # x, y = load_data('./OnlineNewsPopularity.csv')
    # print(x.shape, y.shape)
    # train_x, train_y, val_x, val_y, test_x, test_y = split(x, y, 0.8, 0.1, 0.1)
    # print(train_x.shape, train_y.shape, val_x.shape,val_y.shape, test_x.shape, test_y.shape)

