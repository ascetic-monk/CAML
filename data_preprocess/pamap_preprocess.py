# encoding=utf-8
"""
    Created on 10:38 2018/12/17
    @author: Hangwei Qian
    Adapted from: https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
"""
import numpy as np
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
# from utils import get_sample_weights, opp_sliding_window
from sliding_window import sliding_window

NUM_FEATURES = 52


def opp_sliding_window(data_x, data_y, data_t, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    data_t = np.asarray([[i[0]] for i in sliding_window(data_t, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8), data_t.reshape(len(data_y))


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.
    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001

    x /= std
    return x


def complete_HR(data):
    """Sampling rate for the heart rate is different from the other sensors. Missing
    measurements are filled
    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        HR channel data
    """

    pos_NaN = np.isnan(data)
    idx_NaN = np.where(pos_NaN == False)[0]
    data_no_NaN = data * 0
    for idx in range(idx_NaN.shape[0] - 1):
        data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = data[idx_NaN[idx]]

    data_no_NaN[idx_NaN[-1]:] = data[idx_NaN[-1]]

    return data_no_NaN


def divide_x_y(data):
    """Segments each sample into time, labels and sensor channels
    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Time and labels as arrays, sensor channels as matrix
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]

    return data_t, data_x, data_y


def adjust_idx_labels(data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function adjust the labels picking the labels
    for the protocol settings
    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    data_y[data_y == 24] = 0
    data_y[data_y == 12] = 8
    data_y[data_y == 13] = 9
    data_y[data_y == 16] = 10
    data_y[data_y == 17] = 11

    return data_y


def del_labels(data_t, data_x, data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function deletes the nonrelevant labels
    18 ->
    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 9)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 10)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 11)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 18)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 19)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 20)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)


def downsampling(data_t, data_x, data_y):
    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset
    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_t.shape[0], 3)

    return data_t[idx], data_x[idx], data_y[idx]


def process_dataset_file(data):
    """Function defined as a pipeline to process individual Pamap2 files
    :param data: numpy integer matrix
        channel data: samples in rows and sensor channels in columns
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into samples-channel measurements (x) and labels (y)
    """

    # Data is divided in time, sensor data and labels
    data_t, data_x, data_y = divide_x_y(data)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # nonrelevant labels are deleted
    data_t, data_x, data_y = del_labels(data_t, data_x, data_y)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # Labels are adjusted
    data_y = adjust_idx_labels(data_y)
    data_y = data_y.astype(int)

    if data_x.shape[0] != 0:
        HR_no_NaN = complete_HR(data_x[:, 0])
        data_x[:, 0] = HR_no_NaN

        data_x[np.isnan(data_x)] = 0

        data_x = normalize(data_x)

    else:
        data_x = data_x
        data_y = data_y
        data_t = data_t

        print("SIZE OF THE SEQUENCE IS CERO")

    data_t, data_x, data_y = downsampling(data_t, data_x, data_y)

    return data_t, data_x, data_y


def load_data_pamap2():
    # import os
    # data_dir = './data/'

    dataset = '../dataset_raw/'
    # File names of the files defining the PAMAP2 data.
    PAMAP2_DATA_FILES = ['PAMAP2_Dataset/Protocol/subject101.dat',  # 0
                         'PAMAP2_Dataset/Optional/subject101.dat',  # 1
                         'PAMAP2_Dataset/Protocol/subject102.dat',  # 2
                         'PAMAP2_Dataset/Protocol/subject103.dat',  # 3
                         'PAMAP2_Dataset/Protocol/subject104.dat',  # 4
                         'PAMAP2_Dataset/Protocol/subject107.dat',  # 5
                         'PAMAP2_Dataset/Protocol/subject108.dat',  # 6
                         'PAMAP2_Dataset/Optional/subject108.dat',  # 7
                         'PAMAP2_Dataset/Protocol/subject109.dat',  # 8
                         'PAMAP2_Dataset/Optional/subject109.dat',  # 9
                         'PAMAP2_Dataset/Protocol/subject105.dat',  # 10
                         'PAMAP2_Dataset/Optional/subject105.dat',  # 11
                         'PAMAP2_Dataset/Protocol/subject106.dat',  # 12
                         'PAMAP2_Dataset/Optional/subject106.dat',  # 13
                         ]

    SLIDING_WINDOW_LEN = 128
    SLIDING_WINDOW_STEP = 64
    X_train = np.empty((0, SLIDING_WINDOW_LEN, NUM_FEATURES))
    y_train = np.empty((0))
    seg_train = np.empty((0))

    X_test = np.empty((0, SLIDING_WINDOW_LEN, NUM_FEATURES))
    y_test = np.empty((0))

    counter_files = -1
    seg = -1

    print('Processing dataset files ...')
    for filename in PAMAP2_DATA_FILES:
        counter_files += 1
        print(counter_files)
        if counter_files < 12:
            # Train partition
            try:
                seg += 1
                seglst = []
                seglst.append(seg)

                print('Train... file {0}'.format(filename))
                data = np.loadtxt(dataset + filename)
                print('Train... data size {}'.format(data.shape))
                t, x, y = process_dataset_file(data)
                print(x.shape)
                print(y.shape)

                if x.shape[0] == 0:
                    continue

                for i in range(1, len(t)):
                    if not (0.02 < (t[i] - t[i - 1]) < 0.04):
                        seg += 1
                    seglst.append(seg)

                seglst = np.array(seglst)

                for s in range(min(seglst), max(seglst)+1):
                    if np.sum(seglst == s) < 128: continue
                    xsub, ysub, seglstsub = x[seglst == s],\
                                            y[seglst == s],\
                                            seglst[seglst == s]
                    xsub, ysub, seglstsub = opp_sliding_window(xsub,
                                                               ysub,
                                                               seglstsub,
                                                               SLIDING_WINDOW_LEN,
                                                               SLIDING_WINDOW_STEP)

                    X_train = np.vstack((X_train, xsub))
                    y_train = np.concatenate([y_train, ysub])
                    seg_train = np.concatenate([seg_train, seglstsub])

            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))

        else:
            # Testing partition
            try:
                print('Test... file {0}'.format(filename))
                data = np.loadtxt(dataset + filename)
                print('Test... data size {}'.format(data.shape))
                t, x, y = process_dataset_file(data)
                print(x.shape)
                print(y.shape)
                if x.shape[0] == 0:
                    continue

                x, y, _ = opp_sliding_window(x, y, t, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

                X_test = np.vstack((X_test, x))
                y_test = np.concatenate([y_test, y])
            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))

    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))

    # SLIDING_WINDOW_LEN = 128
    # SLIDING_WINDOW_STEP = 64
    # X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    # X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

    np.save('../dataset/pamapl/' + 'segtrain.npy', seg_train)
    np.save('../dataset/pamapl/' + 'xtrain.npy', X_train)
    np.save('../dataset/pamapl/' + 'ytrain.npy', y_train)
    np.save('../dataset/pamapl/' + 'xtest.npy', X_test)
    np.save('../dataset/pamapl/' + 'ytest.npy', y_test)
    # obj = [(X_train, y_train), (X_test, y_test)]
    # f = file(os.path.join(target_filename), 'wb')
    # return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    load_data_pamap2()