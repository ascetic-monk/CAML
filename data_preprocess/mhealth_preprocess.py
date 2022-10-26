import sys
sys.path.append('../')
from sliding_window import sliding_window
import pandas as pd
import numpy as np


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

ws = 128
ss = 64
NUM_FEATURES = 21

data_x_win_train = np.empty((0, ws, NUM_FEATURES))
data_y_win_train = np.empty((0))
seglst = np.empty((0))

data_x_win_test = np.empty((0, ws, NUM_FEATURES))
data_y_win_test = np.empty((0))
# Loop through and add all 10 subjects' sensor data to dataframe
cnt_seg = 0

#L1: Standing still (1 min) 
#L2: Sitting and relaxing (1 min) 
#L3: Lying down (1 min) 
#L4: Walking (1 min) 
#L5: Climbing stairs (1 min) 
#L6: Waist bends forward (20x) 
#L7: Frontal elevation of arms (20x)
#L8: Knees bending (crouching) (20x)
#L9: Cycling (1 min)
#L10: Jogging (1 min)
#L11: Running (1 min)
#L12: Jump front & back (20x)

list_cls = np.array([0, 1, 1, 0, 2, 5, 5, 5, 5, 4, 5, 3, 5])
#list_cls = np.array([0,0,1,2,3,4,5,6,7,8,9,10,11])
for i in range(1, 7):#9
    df = pd.read_csv(f'../dataset_raw/MHEALTHDATASET/MHEALTHDATASET/mHealth_subject{i}.log', header=None, sep='\t')
    # Note: Excluding the ECG data collected with the chest sensor
    df = df.loc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].rename(columns= {0: 'acc_ch_x', 1: 'acc_ch_y',         2: 'acc_ch_z',         5: 'acc_la_x',         6: 'acc_la_y',         7: 'acc_la_z',         8: 'gyr_la_x',         9: 'gyr_la_y',         10: 'gyr_la_z',        11: 'mag_la_x',         12: 'mag_la_y',        13: 'mag_la_z',        14: 'acc_rw_x',        15: 'acc_rw_y',        16: 'acc_rw_z',         17: 'gyr_rw_x',        18: 'gyr_rw_y',        19: 'gyr_rw_z',        20: 'mag_rw_x',         21: 'mag_rw_y',        22: 'mag_rw_z',        23: 'activity'})
    # df['subject'] = f'subject{i}'
    
    data_x, data_y = df.values[:, :-1], df.values[:, -1]
    data_t = np.arange(len(data_x))
    select = data_y>0
    data_x, data_y, data_t = data_x[select], data_y[select], data_t[select]
    #print(np.sum(data_y==3))
    data_y = list_cls[data_y.astype(np.int64)]
    #print(np.sum(data_y==0))
    ind = list(np.where(data_t[1:]-data_t[:-1]!=1)[0])
    ind = [0] + ind + [len(data_x)-1]
    for j in range(len(ind)-1):
        data_x_tmp, data_y_tmp = opp_sliding_window(data_x[ind[j]:ind[j+1]+1], data_y[ind[j]:ind[j+1]+1], ws, ss)
        #data_x, data_y = opp_sliding_window(data_x, data_y, ws, ss)

        data_x_win_train = np.concatenate([data_x_win_train, data_x_tmp], axis=0)
        data_y_win_train = np.concatenate([data_y_win_train, data_y_tmp], axis=0)
    
        seglst = np.concatenate([seglst, (cnt_seg) * np.ones_like(data_y_tmp)], axis=0)
        cnt_seg += 1
        #print(data_y_tmp)
    # combined_df = pd.concat([combined_df, df])

data_x_re = np.reshape(data_x_win_train, (-1, 21))
mean, std = np.mean(data_x_re, axis=0), np.std(data_x_re, axis=0)
mean, std = np.reshape(mean, (1, 1, 21)), np.reshape(std, (1, 1, 21))
data_x_win_train = (data_x_win_train - mean)/std

for i in range(7, 11):#9
    df = pd.read_csv(f'../dataset_raw/MHEALTHDATASET/MHEALTHDATASET/mHealth_subject{i}.log', header=None, sep='\t')
#    # Note: Excluding the ECG data collected with the chest sensor
    df = df.loc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]].rename(columns= {0: 'acc_ch_x', 1: 'acc_ch_y',         2: 'acc_ch_z',         5: 'acc_la_x',         6: 'acc_la_y',         7: 'acc_la_z',         8: 'gyr_la_x',         9: 'gyr_la_y',         10: 'gyr_la_z',        11: 'mag_la_x',         12: 'mag_la_y',        13: 'mag_la_z',        14: 'acc_rw_x',        15: 'acc_rw_y',        16: 'acc_rw_z',         17: 'gyr_rw_x',        18: 'gyr_rw_y',        19: 'gyr_rw_z',        20: 'mag_rw_x',         21: 'mag_rw_y',        22: 'mag_rw_z',        23: 'activity'})
#    # df['subject'] = f'subject{i}'
    
    data_x, data_y = df.values[:, :-1], df.values[:, -1]
    data_t = np.arange(len(data_x))
    data_x, data_y, data_t = data_x[data_y>0], data_y[data_y>0], data_t[data_y>0]
    data_y = list_cls[data_y.astype(np.int64)]
    ind = list(np.where(data_t[1:]-data_t[:-1]!=1)[0])
    ind = [0] + ind + [len(data_x)-1]
    for j in range(len(ind)-1):
        data_x_tmp, data_y_tmp = opp_sliding_window(data_x[ind[j]:ind[j+1]+1], data_y[ind[j]:ind[j+1]+1], ws, ss)
        #data_x, data_y = opp_sliding_window(data_x, data_y, ws, ss)

        data_x_win_test = np.concatenate([data_x_win_test, data_x_tmp], axis=0)
        data_y_win_test = np.concatenate([data_y_win_test, data_y_tmp], axis=0)

data_x_win_test = (data_x_win_test - mean)/std

for i in range(int(np.max(data_y_win_train))+1):
    print(i)
    print(np.sum(data_y_win_train==i))
    print(np.sum(data_y_win_test==i))

np.save('../dataset/mhealthl/' + 'segtrain.npy', seglst)
np.save('../dataset/mhealthl/' + 'xtrain.npy', data_x_win_train)
np.save('../dataset/mhealthl/' + 'ytrain.npy', data_y_win_train)
np.save('../dataset/mhealthl/' + 'xtest.npy', data_x_win_test)
np.save('../dataset/mhealthl/' + 'ytest.npy', data_y_win_test)
