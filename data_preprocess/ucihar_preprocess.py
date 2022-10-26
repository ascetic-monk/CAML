# encoding=utf-8
"""
    Created on 10:38 2019/2/19
    @author: Hangwei Qian
"""

import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import torch
import pickle as cp
# from utils import get_sample_weights


def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    return X


# This is for parsing the Y data, you can ignore it if you do not need preprocessing
def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    YY = np.eye(6)[data]
    return YY


def load_data():
    str_folder = '../dataset_raw/' + 'UCI HAR Dataset/UCI HAR Dataset/'
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                       INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' + item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'

    X_train = format_data_x(str_train_files)[:, : :]#
    X_test = format_data_x(str_test_files)[:, :, :]#
    Y_train = np.argmax(format_data_y(str_train_y), 1)
    Y_test = np.argmax(format_data_y(str_test_y), 1)

    seg = [0]
    seg_cnt = 0
    for i in range(1, len(X_train)):
        if np.sum(X_train[i-1][64:] - X_train[i][:64]) == 0:
            seg.append(seg_cnt)
        else:
            seg_cnt += 1
            seg.append(seg_cnt)

    np.save('../dataset/ucil/' + 'segtrain.npy', seg)

    X_train, X_test = maxminnorm(X_train, X_test)

    # np.save('../dataset/uci/' + 'xtrain.npy', X_train)
    # np.save('../dataset/uci/' + 'ytrain.npy', Y_train)
    # np.save('../dataset/uci/' + 'xtest.npy', X_test)
    # np.save('../dataset/uci/' + 'ytest.npy', Y_test)


def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]


def maxminnorm(x_train, x_test):
    mean_x = np.mean(np.reshape(x_train, (-1, x_train.shape[2])), 0)
    std_x = np.std(np.reshape(x_train, (-1, x_train.shape[2])), 0)
    return (x_train - mean_x)/std_x, (x_test - mean_x)/std_x


if __name__ == '__main__':
    load_data()
