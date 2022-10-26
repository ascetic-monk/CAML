import numpy as np
import pandas as pd
from sliding_window import sliding_window

from scipy import stats

##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = {
    'Downstairs': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 3,
    'Upstairs': 4,
    'Walking': 5
}

DATA_PATH = '../dataset_raw/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'

RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 90

# Model
N_CLASSES = 6
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

# Hyperparameters optimized
# SEGMENT_TIME_SIZE = 64#180
SEGMENT_TIME_SIZE = 180


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


if __name__ == '__main__':

    # LOAD DATA
    data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    data['z-axis'].replace({';': ''}, regex=True, inplace=True)
    # data = data.dropna()
    data = data.values

    datax = data[:, 3:].astype(np.float32)
    datat = data[:, 2].astype(np.float32)
    datax[np.isnan(datax)] = 0
    datax = normalize(datax)

    # DATA PREPROCESSING
    data_convoluted_train = []
    labels_train = []
    data_convoluted_test = []
    labels_test = []

    tslst, seglst = [], []

    seg = -1
    for user in range(1, 36, 1):
        idx = (data[:, 0] == user)
        seg += 1
        ts = 0
        data_user = datax[idx, :]
        label_user = data[idx, 1]
        for j in range(0, len(data_user) - SEGMENT_TIME_SIZE, TIME_STEP):
            if user in [33, 34, 35]:
                data_convoluted_test.append(data_user[j: j + SEGMENT_TIME_SIZE])
                label = stats.mode(label_user[j: j + SEGMENT_TIME_SIZE])[0][0]
                labels_test.append(label)
            else:
                data_convoluted_train.append(data_user[j: j + SEGMENT_TIME_SIZE])
                label = stats.mode(label_user[j: j + SEGMENT_TIME_SIZE])[0][0]
                labels_train.append(label)
                if j==0 or (j >= 1 and datat[j] - datat[j-1] < 500000000) or datat[j-1]==0 or datat[j]==0:
                    tslst.append(ts)
                    seglst.append(seg)
                    ts += 1
                else:
                    ts = 0
                    seg += 1
                    tslst.append(ts)
                    seglst.append(seg)


    # Convert to numpy
    X_train = np.asarray(data_convoluted_train, dtype=np.float32)
    X_test = np.asarray(data_convoluted_test, dtype=np.float32)

    # One-hot encoding
    y_train = np.asarray([LABELS[lbl] for lbl in labels_train], dtype=np.int64)
    y_test = np.asarray([LABELS[lbl] for lbl in labels_test], dtype=np.int64)

    np.save('../dataset/wisdml/' + 'segtrain.npy', np.array(seglst))
    np.save('../dataset/wisdml/' + 'xtrain.npy', X_train)
    np.save('../dataset/wisdml/' + 'ytrain.npy', y_train)
    np.save('../dataset/wisdml/' + 'xtest.npy', X_test)
    np.save('../dataset/wisdml/' + 'ytest.npy', y_test)

    # print("Convoluted data shape: ", data_convoluted_train.shape)
    # print("Labels shape:", labelss_train.shape)


