import numpy as np


root = '../dataset/'

datasets = ['uci/', 'pamap/', 'wisdm/']
for dataset in datasets:
    xtrain = np.load(root+dataset+'xtrain.npy')
    xtest = np.load(root+dataset+'xtest.npy')
    print(dataset)
    print("train pair: %d" % xtrain.shape[0])
    print("length: %d" % xtrain.shape[1])
    print("channel: %d" % xtrain.shape[2])
    print("test pair: %d" % xtest.shape[0])
