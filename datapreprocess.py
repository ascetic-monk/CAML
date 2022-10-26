# import tensorflow as tf
import argparse
import os
import shutil

import numpy as np
from pandas import Series
# from absl import app
from scipy import stats

from sliding_window import sliding_window

parser = argparse.ArgumentParser(description='PyTorch SALIENCE Pre-Process Implementation')
parser.add_argument('--dataset', type=str, default='pamap2', metavar='N',
                    help='opp or pamap')
args = parser.parse_args()

if args.dataset == 'opp':
    args.length = 300
    args.overlap = 30
else:
    args.length = 200
    args.overlap = 100


def downsampling(data_x, data_y):
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

    idx = np.arange(0, data_x.shape[0], 3)

    return data_x[idx], data_y[idx]


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


def preprocess_opportunity():
    dataset_path = 'dataset/opp/'
    channel_num = 113

    file_list = [['OpportunityUCIDataset/dataset/S1-Drill.dat',
                  'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                  'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                  'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                  'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                  'OpportunityUCIDataset/dataset/S1-ADL5.dat'],
                 ['OpportunityUCIDataset/dataset/S2-Drill.dat',
                  'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                  'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                  'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                  'OpportunityUCIDataset/dataset/S2-ADL4.dat',
                  'OpportunityUCIDataset/dataset/S2-ADL5.dat'],
                 ['OpportunityUCIDataset/dataset/S3-Drill.dat',
                  'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                  'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                  'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                  'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                  'OpportunityUCIDataset/dataset/S3-ADL5.dat'],
                 ['OpportunityUCIDataset/dataset/S4-Drill.dat',
                  'OpportunityUCIDataset/dataset/S4-ADL1.dat',
                  'OpportunityUCIDataset/dataset/S4-ADL2.dat',
                  'OpportunityUCIDataset/dataset/S4-ADL3.dat',
                  'OpportunityUCIDataset/dataset/S4-ADL4.dat',
                  'OpportunityUCIDataset/dataset/S4-ADL5.dat']]

    invalid_feature = np.arange(46, 50)
    invalid_feature = np.concatenate([invalid_feature, np.arange(59, 63)])
    invalid_feature = np.concatenate([invalid_feature, np.arange(72, 76)])
    invalid_feature = np.concatenate([invalid_feature, np.arange(85, 89)])
    invalid_feature = np.concatenate([invalid_feature, np.arange(98, 102)])
    invalid_feature = np.concatenate([invalid_feature, np.arange(134, 244)])
    invalid_feature = np.concatenate([invalid_feature, np.arange(245, 249)])

    lower_bound = np.array([3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                            3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                            3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                            3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                            3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                            3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                            3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                            3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                            3000, 3000, 3000, 10000, 10000, 10000, 1500, 1500, 1500,
                            250, 25, 200, 5000, 5000, 5000, 5000, 5000, 5000,
                            10000, 10000, 10000, 10000, 10000, 10000, 250, 250, 25,
                            200, 5000, 5000, 5000, 5000, 5000, 5000, 10000, 10000,
                            10000, 10000, 10000, 10000, 250, ])

    upper_bound = np.array([-3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                            -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                            -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                            -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000, -3000,
                            -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                            -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                            -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                            -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                            -3000, -3000, -3000, -10000, -10000, -10000, -1000, -1000, -1000,
                            -250, -100, -200, -5000, -5000, -5000, -5000, -5000, -5000,
                            -10000, -10000, -10000, -10000, -10000, -10000, -250, -250, -100,
                            -200, -5000, -5000, -5000, -5000, -5000, -5000, -10000, -10000,
                            -10000, -10000, -10000, -10000, -250, ])

    if os.path.exists(dataset_path + 'processed_data/'):
        shutil.rmtree(dataset_path + 'processed_data/')
    os.mkdir(dataset_path + 'processed_data/')

    for usr_idx in range(4):

        # import pdb; pdb.set_trace()
        print("process data... user{}".format(usr_idx))
        time_windows = np.empty([0, args.length, channel_num], dtype=np.float)
        act_labels = np.empty([0], dtype=np.int)

        for file_idx in range(6):
            filename = file_list[usr_idx][file_idx]

            file = dataset_path + filename
            signal = np.loadtxt(file)
            signal = np.delete(signal, invalid_feature, axis=1)

            data = signal[:, 1:114].astype(np.float)
            label = signal[:, 114].astype(np.int)

            label[label == 0] = -1

            label[label == 101] = 0
            label[label == 102] = 1
            label[label == 103] = 2
            label[label == 104] = 3
            label[label == 105] = 4


            # fill missing values using Linear Interpolation
            data = np.array([Series(i).interpolate(method='linear') for i in data.T]).T
            data[np.isnan(data)] = 0.

            # normalization
            diff = upper_bound - lower_bound
            data = (data - lower_bound) / diff

            data[data > 1] = 1.0
            data[data < 0] = 0.0

            # sliding window
            data = sliding_window(data, (args.length, channel_num), (args.overlap, 1))
            label = sliding_window(label, args.length, args.overlap)
            label = stats.mode(label, axis=1)[0][:, 0]

            # remove non-interested time windows (label==-1)
            invalid_idx = np.nonzero(label < 0)[0]
            data = np.delete(data, invalid_idx, axis=0)
            label = np.delete(label, invalid_idx, axis=0)

            time_windows = np.concatenate((time_windows, data), axis=0)
            act_labels = np.concatenate((act_labels, label), axis=0)

        np.save(dataset_path + 'processed_data/' + 'sub{}_features'.format(usr_idx), time_windows)
        np.save(dataset_path + 'processed_data/' + 'sub{}_labels'.format(usr_idx), act_labels)
        print("sub{} finished".format(usr_idx))


def preprocess_pamap2():
    dataset_path = 'dataset/pamap/'
    channel_num = 52

    if os.path.exists(dataset_path + 'processed_data/'):
        shutil.rmtree(dataset_path + 'processed_data/')
    os.mkdir(dataset_path + 'processed_data/')

    file_list = [
        'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
        'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat']

    for usr_idx in range(len(file_list)):
        file = dataset_path + 'PAMAP2_Dataset/Protocol/' + file_list[usr_idx]
        data = np.loadtxt(file)

        label = data[:, 1].astype(int)
        label[label == 0] = -1
        label[label == 1] = 0  # lying
        label[label == 2] = 1  # sitting
        label[label == 3] = 2  # standing
        label[label == 4] = 3  # walking
        label[label == 5] = 4  # running
        label[label == 6] = 5  # cycling
        label[label == 7] = 6  # nordic walking
        label[label == 12] = 7  # ascending stairs
        label[label == 13] = 8  # descending stairs
        label[label == 16] = 9  # vacuum cleaning
        label[label == 17] = 10  # ironing
        label[label == 24] = 11  # rope jumping

        # fill missing values
        # valid_idx = np.concatenate((np.arange(4, 16), np.arange(21, 33), np.arange(38, 50)), axis=0)
        data = data[:, 2:]
        data = np.array([Series(i).interpolate() for i in data.T]).T

        HR_no_NaN = complete_HR(data[:, 0])
        data[:, 0] = HR_no_NaN

        data[np.isnan(data)] = 0

        data = normalize(data)

        # # min-max normalization
        # diff = upperBound - lowerBound
        # data = 2 * (data - lowerBound) / diff - 1

        # data[data > 1] = 1.0
        # data[data < -1] = -1.0

        # sliding window
        data = sliding_window(data, (args.length, channel_num), (args.overlap, 1))
        label = sliding_window(label, args.length, args.overlap)
        label = stats.mode(label, axis=1)[0][:, 0]

        # remove non-interested time windows (label==-1)
        invalid_idx = np.nonzero(label < 0)[0]
        data = np.delete(data, invalid_idx, axis=0)
        label = np.delete(label, invalid_idx, axis=0)
        print(data.shape)
        np.save(dataset_path + 'processed_data/' + 'sub{}_features'.format(usr_idx), data)
        np.save(dataset_path + 'processed_data/' + 'sub{}_labels'.format(usr_idx), label)
        print("sub{} finished".format(usr_idx))


# def preprocess_ucihar():
#     dataset_path = 'dataset_raw/UCI HAR Dataset/'
#     channel_num = 9
#
#     if os.path.exists(dataset_path + 'processed_data/'):
#         shutil.rmtree(dataset_path + 'processed_data/')
#     os.mkdir(dataset_path + 'processed_data/')
#
#     # file_list = [
#     #     'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
#     #     'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat']
#     # file_list = [
#     #     'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
#     #     'subject107.dat', 'subject108.dat', 'subject109.dat', 'subject106.dat']
#
#     file_data = dataset_path + 'UCI HAR Dataset/train/X_train.txt'
#     file_label = dataset_path + 'UCI HAR Dataset/train/Y_train.txt'
#     data = np.loadtxt(file_data)
#     label = np.loadtxt(file_label) - 1
#
#     # fill missing values
#     # valid_idx = np.concatenate((np.arange(4, 16), np.arange(21, 33), np.arange(38, 50)), axis=0)
#     # data = data[:, valid_idx]
#     data = np.array([Series(i).interpolate() for i in data.T]).T
#
#     # sliding window
#     data = sliding_window(data, (args.length, channel_num), (args.overlap, 1))
#     label = sliding_window(label, args.length, args.overlap)
#     label = stats.mode(label, axis=1)[0][:, 0]
#
#     # remove non-interested time windows (label==-1)
#     # invalid_idx = np.nonzero(label < 0)[0]
#     # data = np.delete(data, invalid_idx, axis=0)
#     # label = np.delete(label, invalid_idx, axis=0)
#
#     np.save(dataset_path + 'processed_data/' + 'sub{}_features'.format(usr_idx), data)
#     np.save(dataset_path + 'processed_data/' + 'sub{}_labels'.format(usr_idx), label)
#     print("sub{} finished".format(usr_idx))


def preprocess_wisdm():
    lowerBound = np.array([-19.61, -19.61, -19.8])
    upperBound = np.array([19.95, 20.04, 19.61])

    dataset_path = 'dataset/wisdm/'

    if os.path.exists(dataset_path + 'processed_data/'):
        shutil.rmtree(dataset_path + 'processed_data/')
    os.mkdir(dataset_path + 'processed_data/')

    file = dataset_path + 'WISDM.npz'
    data = np.load(file)

    data, label, folds = data['X'], data['y'], data['folds']

    for usr_idx in range(len(folds)):
        data_sub = data[folds[usr_idx][1]][:, 0, ...]

        # min-max normalization
        diff = upperBound - lowerBound
        data_sub = 2 * (data_sub - lowerBound) / diff - 1

        data_sub[data_sub > 1] = 1.0
        data_sub[data_sub < -1] = -1.0

        label_sub = np.argmax(label[folds[usr_idx][1]], 1)

        np.save(dataset_path + 'processed_data/' + 'sub{}_features'.format(usr_idx), data_sub)
        np.save(dataset_path + 'processed_data/' + 'sub{}_labels'.format(usr_idx), label_sub)
        print("sub{} finished".format(usr_idx))


def preprocess_mhealth():
    lowerBound = np.array([ -22.438 ,  -20.188 ,  -18.401 ,   -8.6196,   -8.6196,  -22.146 ,
                            -19.619 ,  -19.373 ,   -1.7792,   -2.6604,   -2.6267, -357.56  ,
                            -364.57  , -282.39  ,  -22.345 ,  -18.972 ,  -18.238 ,   -1.1529,
                            -2.2567,   -1.1142, -319.03  , -355.85  , -702.57  ])
    upperBound = np.array([19.094, 20.927, 26.196, 8.5065, 8.5191, 20.024,
                           21.161, 25.015, 1.7106, 1.666, 1.5815, 339.22,
                           336.24, 272.56, 19.801, 21.965, 25.741, 1.4157,
                           1.1211, 1.528, 239.69, 335.25, 657.18])

    dataset_path = 'dataset/mhealth/'

    if os.path.exists(dataset_path + 'processed_data/'):
        shutil.rmtree(dataset_path + 'processed_data/')
    os.mkdir(dataset_path + 'processed_data/')

    file = dataset_path + 'MHEALTH.npz'
    data = np.load(file)

    data, label, folds = data['X'], data['y'], data['folds']

    for usr_idx in range(len(folds)):

        data_sub = data[folds[usr_idx][1]][:, 0, ...]

        # min-max normalization
        diff = upperBound - lowerBound
        data_sub = 2 * (data_sub - lowerBound) / diff - 1

        data_sub[data_sub > 1] = 1.0
        data_sub[data_sub < -1] = -1.0

        label_sub = np.argmax(label[folds[usr_idx][1]], 1)

        np.save(dataset_path + 'processed_data/' + 'sub{}_features'.format(usr_idx), data_sub)
        np.save(dataset_path + 'processed_data/' + 'sub{}_labels'.format(usr_idx), label_sub)
        print("sub{} finished".format(usr_idx))


def main():
    if args.dataset == 'opp':
        preprocess_opportunity()
    elif args.dataset == 'pamap2':
        preprocess_pamap2()
    elif args.dataset == 'mhealth':
        preprocess_mhealth()
    elif args.dataset == 'wisdm':
        preprocess_wisdm()


if __name__ == '__main__':
    main()
