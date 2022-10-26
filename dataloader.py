import numpy as np
import torch.utils.data as data
from sklearn import utils as skutils
import random
import math


def ratio_split(ratio, train_x, train_y, seed, act_num):
    sup_cnt = 0
    index_sup, index_unsup = [], []
    train_x_sup, train_y_sup, train_adj_sup, \
    train_x_unsup, train_y_unsup, train_adj_unsup = [], [], [], [], [], []
    for c in range(act_num):
        train_x_c, train_y_c, indexc = skutils.shuffle(train_x[train_y == c],
                                                       train_y[train_y == c],
                                                       np.where(train_y == c)[0],
                                                       random_state=seed)

        train_x_sup.append(train_x_c[:math.ceil(ratio * len(train_y_c))])
        train_y_sup.append(train_y_c[:math.ceil(ratio * len(train_y_c))])
        index_sup.append(indexc[:math.ceil(ratio * len(train_y_c))])
        train_x_unsup.append(train_x_c[math.ceil(ratio * len(train_y_c)):])
        train_y_unsup.append(train_y_c[math.ceil(ratio * len(train_y_c)):])
        index_unsup.append(indexc[math.ceil(ratio * len(train_y_c)):])
        sup_cnt += math.ceil(ratio * len(train_y_c))

    train_x_sup, train_y_sup, index_sup = skutils.shuffle(np.concatenate(train_x_sup, 0),
                                                          np.concatenate(train_y_sup, 0),
                                                          np.concatenate(index_sup, 0),
                                                          random_state=seed)
    train_x_unsup, train_y_unsup, index_unsup = skutils.shuffle(np.concatenate(train_x_unsup, 0),
                                                                np.concatenate(train_y_unsup, 0),
                                                                np.concatenate(index_unsup, 0),
                                                                random_state=seed)
    train_x, train_y, index = np.concatenate([train_x_sup, train_x_unsup], 0),\
                              np.concatenate([train_y_sup, train_y_unsup], 0), \
                              np.concatenate([index_sup, index_unsup], 0)
    return train_x, train_y, index, sup_cnt


class HARData(object):
    def __init__(self, dataset='pamap', args=None):
        print(args.lenset)
        if args.lenset == 'long':
            suf = 'l'
            print('long setting')
        else:
            suf = ''
            print('short setting')

        if dataset == 'pamap':
            self._path = 'dataset/pamap' + suf + '/'
            self._pthdis = './distance/pamap' + suf + '_sbd.npy'
            self._channel_num = 52
            self._length = 128
            self._act_num = 12
        elif dataset == 'uci':
            self._path = 'dataset/uci' + suf + '/'
            self._pthdis = './distance/uci' + suf + '_sbd.npy'
            self._channel_num = 9
            self._length = 128
            self._act_num = 6
        elif dataset == 'wisdm':
            self._path = 'dataset/wisdm' + suf + '/'
            self._pthdis = './distance/wisdm' + suf + '_sbd.npy'
            self._channel_num = 3
            self._length = 180
            self._act_num = 6
        elif dataset == 'afib':
            self._path = 'dataset/afib' + '/'
            self._pthdis = './distance/wisdm' + suf + '_sbd.npy'
            self._channel_num = 2
            self._length = 2500
            self._act_num = 4
        elif dataset == 'mhealth':
            self._path = 'dataset/mhealthl' + '/'
            self._pthdis = './distance/wisdm' + suf + '_sbd.npy'
            self._channel_num = 21
            self._length = 128
            self._act_num = 6
        elif dataset == 'mhealth12':
            self._path = 'dataset/mhealth12'+'/'
            self._pthdis = './distance/wisdm'+suf+'_sbd.npy'
            self._channel_num = 21
            self._length = 128
            self._act_num = 12
    def load(self, ratio, seed, adj=False, dist_enable=False, seg_enable=False, dist_mean=True):

        train_x = np.load(self._path + 'xtrain.npy')
        train_y = np.load(self._path + 'ytrain.npy')
        test_x = np.load(self._path + 'xtest.npy')
        test_y = np.load(self._path + 'ytest.npy')

        if dist_enable:
            if dist_mean:
                dist = np.load(self._pthdis).mean(2)
            else:
                dist = np.load(self._pthdis)

        if seg_enable:
            seg = np.load(self._path + 'segtrain.npy')

        # train_x, train_y, train_adj = ratio_split(ratio, train_x, train_y, train_adj, seed, self._act_num)
        train_x, train_y, index, sup_cnt = ratio_split(ratio,
                                                       train_x,
                                                       train_y,
                                                       seed,
                                                       self._act_num)
        test_x, test_y = skutils.shuffle(test_x, test_y, random_state=seed)

        # if adj:
        #     return train_x, train_y, test_x, test_y, train_adj
        # else:
        if dist_enable:
            return train_x, train_y, test_x, test_y, index, dist, sup_cnt
        elif seg_enable:
            return train_x, train_y, test_x, test_y, index, seg, sup_cnt
        else:
            return train_x, train_y, test_x, test_y, sup_cnt


class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, data, label, sup_cnt=100,
                 transform=None, target_transform=None, transform_suf=None,
                 mode='train', ind=None, seg=None, unsup_ratio=1.0, neighborlen=3):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label
        self.mode = mode
        # self.ratio = ratio
        self.transform_suf = transform_suf
        self.unsup_ratio = unsup_ratio
        self.neighborlen = neighborlen

        self.ind = ind
        if not (self.ind is None):
            self.ind_rev = np.zeros_like(self.ind)
            for cnt, i in enumerate(ind):
                self.ind_rev[i] = cnt
        self.seg = seg

        if 'train' in mode:
            idx_sup = sup_cnt
            self.data_sup = self.data[:idx_sup]
            self.labels_sup = self.labels[:idx_sup]
            self.data_unsup = self.data[idx_sup:]
            self.labels_unsup = self.labels[idx_sup:]
            if not (self.ind is None):
                self.ind_sup = self.ind[:idx_sup]
                self.ind_unsup = self.ind[idx_sup:]

        if mode == 'train_sup':
            self._data, self._labels = self.data_sup, self.labels_sup
            if not (self.ind is None):
                self._ind = self.ind_sup
        elif mode == 'train_unsup':
            self._data, self._labels = self.data_unsup[:int(unsup_ratio*len(self.data_unsup))],\
                                       self.labels_unsup[:int(unsup_ratio*len(self.data_unsup))]
            if not (self.ind is None):
                self._ind = self.ind_unsup
        elif mode == 'train' or mode == 'test':
            self._data, self._labels = self.data, self.labels
            if not (self.ind is None):
                self._ind = self.ind

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        if self.transform:
            data, target = self.transform(self._data[index]), self._labels[index]
        else:
            data, target = self._data[index], self._labels[index]

        if not (self.seg is None):
            prev, futu = self.findneighboor(index)
            return data, target, prev, futu

        if not (self.ind is None):
            ind = self._ind[index]
            return data, target, ind

        if self.transform_suf:
            return data, self.transform_suf(data), target
        else:
            return data, target

    def setmode(self, mode):
        self.mode = mode
        if mode == 'train_sup':
            self._data, self._labels = self.data_sup, self.labels_sup
            if not (self.ind is None):
                self._ind = self.ind_sup
        elif mode == 'train_unsup':
            self._data, self._labels = self.data_unsup, self.labels_unsup
            if not (self.ind is None):
                self._ind = self.ind_unsup
        elif mode == 'train' or mode == 'test':
            self._data, self._labels = self.data, self.labels
            if not (self.ind is None):
                self._ind = self.ind

    def modemem(self, mode):
        if mode == 'train_sup':
            self.data_sup, self.labels_sup = self._data, self._labels
        elif mode == 'train_unsup':
            self.data_unsup, self.labels_unsup = self._data, self._labels

    def adddata(self, data_persudo, labels_persudo):
        self._data = np.concatenate([self._data, np.array(data_persudo)], 0)
        self._labels = np.concatenate([self._labels, np.array(labels_persudo)], 0)

    def deletedata(self, idx):
        self._data = self._data[idx]
        self._labels = self._labels[idx]

    def findneighboor(self, i):
        i_r = self._ind[i]
        prev, futu = [self._data[i]], [self._data[i]]
        for idx in range(1, self.neighborlen):
            if (i_r - idx >= 0) and (self.seg[i_r - idx] == self.seg[i_r]):
                prev.append(self.data[self.ind_rev[i_r - idx]])
            else:
                prev.append(prev[-1])
            if (i_r + idx <= len(self.seg) - 1) and (self.seg[i_r + idx] == self.seg[i_r]):
                futu.append(self.data[self.ind_rev[i_r + idx]])
            else:
                futu.append(futu[-1])

        return prev[::-1], futu[::-1]

    def labelsup_stat(self):
        from scipy import stats
        labels_lst = []
        for i in range(int(max(self._labels))+1):
            labels_lst.append(sum(self._labels==i))
        return labels_lst

    def __len__(self):
        return len(self._data)


if __name__ == '__main__':
    pass
    # dataset = Opportunity()
    # dataset = PAMAP2()
    # train_x, train_y, test_x, test_y = dataset.load_data()
    # dataloader_train = Dataset(train_x, train_y)
    # dataloader_test = Dataset(test_x, test_y)
    #
    # for idx, (data, lbl) in enumerate(dataloader_train):
    #     pass
