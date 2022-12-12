import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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
    train_x, train_y = x[perm[:num_train:]], y[perm[:num_train]]
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


if __name__ == "__main__":
    x, y = load_data('./OnlineNewsPopularity.csv')
    print(x.shape, y.shape)
    # train_x, train_y, val_x, val_y, test_x, test_y = split(x, y, 0.8, 0.1, 0.1)
    # print(train_x.shape, train_y.shape, val_x.shape,val_y.shape, test_x.shape, test_y.shape)