import numpy as np
import argparse


def ERP(Xi, Xj, flag):
    dimension = Xi.shape[0]
    distance = 0
    if 0 == flag:
        for i in range(dimension):
            distance = distance + abs(Xi(i))
    else:
        D = np.zeros((dimension + 1, dimension + 1))
        D[0, :] = np.NaN
        D[:, 0] = np.NaN
        D[0, 0] = 0

        for i in range(1, dimension+1):
            for j in range(1, dimension+1):
                D[i, j] = min([D[i-1, j-1] + abs(Xi[i-1] - Xj[j-1]),
                               D[i, j-1] + abs(Xj[j-1]),
                               D[i-1, j] + abs(Xi[i-1])])
    distance = D[dimension, dimension]
    return distance


def ERP_multi(Xi, Xj):
    B1, T, C = Xi.shape
    B2, T, C = Xj.shape
    Kx = np.zeros((B1, B2, C))
    for c in range(C):
        for b1 in range(B1):
            Kx[b1, b1, c] = 1
            for b2 in range(b1+1, B2):
                tmp = ERP(Xi[b1, :, c], Xj[b2, :, c], 1)
                Kx[b1, b2, c] = tmp
                Kx[b2, b1, c] = tmp
            print("channel %d/%d, b %d/%d" %(c, C, b1, B1))
    return Kx


def NCCc(x,y):
    lenx = len(x)

    fftlength = int(np.power(2,np.ceil(np.log2(lenx))))
    r = np.fft.ifft( np.fft.fft(x,fftlength) * np.conj(np.fft.fft(y,fftlength)) )
    r = np.concatenate([r[-lenx+1:], r[0:lenx]], 0)

    cc_sequence = r/(np.linalg.norm(x=x, ord=2)*np.linalg.norm(x=y, ord=2))
    return 1-np.max(cc_sequence)


def SBD_multi(Xi, Xj):
    B1, T, C = Xi.shape
    B2, T, C = Xj.shape
    Kx = np.zeros((B1, B2, C))
    for c in range(C):
        for b1 in range(B1):
            Kx[b1, b1, c] = 1
            for b2 in range(b1+1, B2):
                tmp = NCCc(Xi[b1, :, c], Xj[b2, :, c])
                Kx[b1, b2, c] = tmp
                Kx[b2, b1, c] = tmp
            print("channel %d/%d, b %d/%d" %(c, C, b1, B1))
    return Kx


parser = argparse.ArgumentParser(description='Run SEMI')
parser.add_argument('--dataset', type=str, default='wisdm')
parser.add_argument('--metrics', type=str, default='sdb')
args = parser.parse_args()

if __name__ == '__main__':
    # Xi = np.ones((150, ))#np.array([2,1,2])#np.ones((150, ))
    # Xj = np.ones((150, ))#np.array([1,2,1])#np.ones((150, ))
    # print(NCCc(Xi, Xj))

    xtrain = np.load('../dataset/'+args.dataset+'/xtrain.npy')
    ytrain = np.load('../dataset/'+args.dataset+'/ytrain.npy')

    # Xi = np.random.randn(128,64,9)
    # print(ERP_multi(Xi, Xi))
    if args.metrics == 'sdb':
        np.save('../distance/'+args.dataset+'_sbd.npy', SBD_multi(xtrain, xtrain))
    else:
        np.save('../distance/'+args.dataset+'_erp.npy', ERP_multi(xtrain, xtrain))

    # import matplotlib.pyplot as plt
    # plt.imshow(sbd)
    # plt.savefig('a.jpg')
    # plt.show()
    # pass