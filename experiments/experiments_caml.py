import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader

from dataloader import Dataset, HARData
from network import model_select
from experiments.test import test
import utils.transforms as transforms
from utils.pytorchtools import EarlyStopping
import numpy as np
import sklearn.metrics as metrics



def train_caml(net, dataloader, criterion, optimizer, epo=60, args=None):
    trainloader_sup, trainloader_unsup, testloader = dataloader

    early_stopping = EarlyStopping(args.patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format('checkpoints'))

    net1, net2, sup_fw_head, trans_fw = net
    net1, net2, sup_fw_head, trans_fw = net1.train(),\
                                        net2.train(),\
                                        sup_fw_head.train(),\
                                        trans_fw.train()
    net1, net2, sup_fw_head, trans_fw = net1.cuda(), \
                                        net2.cuda(), \
                                        sup_fw_head.cuda(), \
                                        trans_fw.cuda()
    optimizer1, optimizer2 = optimizer
    weight_cls = torch.tensor(trainloader_sup.dataset.labelsup_stat()).cuda()
    print(weight_cls)
    weight_cls = weight_cls/ weight_cls.sum()

    for epoch in range(epo):  # loop over the dataset multiple times
        running_loss = 0.0
        iter_num = args.iter
        for i in range(iter_num):
            # get the inputs
            inputs_sup, labels_sup, prev_sup, futu_sup = next(iter(trainloader_sup))
            inputs_unsup, __, prev_unsup, futu_unsup = next(iter(trainloader_unsup))

            neig_sup = prev_sup[-(args.tgl+1):] + futu_sup[1:(args.tgl+1)]
            neig_unsup = prev_unsup[-(args.tgl+1):] + futu_unsup[1:(args.tgl+1)]

            inputs_sup, labels_sup, inputs_unsup, neig_sup, neig_unsup = inputs_sup.float().cuda(),\
                                                                         labels_sup.long().cuda(),\
                                                                         inputs_unsup.float().cuda(),\
                                                                         torch.cat(neig_sup).float().cuda(), \
                                                                         torch.cat(neig_unsup).float().cuda()

            # wrap them in Variable
            # zero the parameter gradients
            ratio = args.tgr
            long_window = torch.tensor([ratio**(abs(i)) for i in range(-args.tgl,(args.tgl+1))]).cuda().unsqueeze(1).unsqueeze(1)
            long_window = long_window/torch.sum(long_window)

            optimizer1.zero_grad()
            outputs1_sup = net1(inputs_sup)
            h2_s, _ = net2(neig_sup, seq_out=True)
            h2_s = F.normalize(h2_s.mean(2), 1)
            h2_s = (h2_s.view(2*args.tgl+1, -1, h2_s.shape[1])*\
                    long_window).sum(0)
            outputs2_sup = sup_fw_head(h2_s)

            outputs1_unsup = net1(inputs_unsup)
            h2_u, _ = net2(neig_unsup, seq_out=True)
            h2_u = F.normalize(h2_u.mean(2), 1)
            h2_u = (h2_u.view(2*args.tgl+1, -1, h2_u.shape[1])*\
                   long_window).sum(0)
            outputs2_unsup = sup_fw_head(h2_u)

            weight, _ = torch.max(F.softmax(outputs2_unsup, 1).detach(), 1)
            loss_div = -(weight_cls*torch.log(F.softmax(outputs1_unsup,1).mean(0))).sum()
            loss1 = criterion(outputs1_sup, labels_sup) + \
                    F.kl_div(F.log_softmax(outputs1_sup, 1),
                             F.softmax(outputs2_sup, 1),
                             reduction='batchmean') + \
                    (weight * F.kl_div(F.log_softmax(outputs1_unsup, 1),
                                       F.softmax(outputs2_unsup, 1),
                                       reduce=False).sum(1)).mean() + 5e-2*loss_div# + loss_ps
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            outputs1_sup = net1(inputs_sup)
            h2_s, _ = net2(neig_sup, seq_out=True)
            h2_s = F.normalize(h2_s.mean(2), 1)
            h2_s = (h2_s.view(2*args.tgl+1, -1, h2_s.shape[1])*\
                   long_window).sum(0)
            outputs2_sup = sup_fw_head(h2_s)

            outputs1_unsup = net1(inputs_unsup)
            h2_u, _ = net2(neig_unsup, seq_out=True)
            h2_u = F.normalize(h2_u.mean(2), 1)
            h2_u = (h2_u.view(2*args.tgl+1, -1, h2_u.shape[1])*\
                   long_window).sum(0)
            outputs2_unsup = sup_fw_head(h2_u)

            # if weight_kl_bool:
            weight, _ = torch.max(F.softmax(outputs1_unsup, 1).detach(), 1)
            loss_div = -(weight_cls*torch.log(F.softmax(outputs2_unsup,1).mean(0))).sum()
            loss2 = criterion(outputs2_sup, labels_sup) + \
                    F.kl_div(F.log_softmax(outputs2_sup, 1),
                             F.softmax(outputs1_sup, 1),
                             reduction='batchmean') + \
                    (weight * F.kl_div(F.log_softmax(outputs2_unsup, 1),
                                       F.softmax(outputs1_unsup, 1),
                                       reduce=False).sum(1)).mean() + 5e-2*loss_div # + loss_ps
            # (_ == __.cuda()).sum()
            loss2.backward()
            optimizer2.step()

            # print statistics
            running_loss += loss1.cpu().item()

        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / 100))
        if args.es:
            early_stopping(running_loss, net1)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if args.es:
        net1.load_state_dict(early_stopping.checkpoint)
    print('Finished Training')
    torch.save(net1.state_dict(), 'net_param_mutual.pth')

    return net1


def exp_caml(seed, dataset, args):
    # create networks
    net1 = model_select(args.model_name, args)  # .cuda()
    net2 = model_select(args.model_name, args)  # .cuda()

    sup_fw_head = nn.Linear(args.feature_size, args.act_num)
    trans_fw = nn.TransformerEncoderLayer(d_model=args.feature_size, nhead=2, dim_feedforward=args.feature_size)

    # create the optimizer and loss function
    optimizer1 = torch.optim.Adam([{'params': net1.parameters()}], lr=args.lr)
    optimizer2 = torch.optim.Adam([{'params': net2.parameters()},
                                   {'params': sup_fw_head.parameters()},
                                   {'params': trans_fw.parameters()}], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()  # weight=weights)

    # create dataset and dataloader
    if args.notransform:
        transform = transforms.Compose([transforms.MagnitudeWrap(sigma=0.3, knot=4, p=0.2),
                                        transforms.TimeWarp(sigma=0.2, knot=8, p=0.2)])
    else:
        transform = None
    train_x, train_y, test_x, test_y, index, seg, sup_cnt = dataset.load(ratio=args.ratio,
                                                                         seed=seed,
                                                                         seg_enable=True)
    trainset_sup = Dataset(train_x, train_y, ind=index, seg=seg,
                           sup_cnt=sup_cnt, transform=transform, neighborlen=args.tgl+1)
    trainset_sup.setmode('train_sup')
    trainset_unsup = Dataset(train_x, train_y, ind=index, seg=seg,
                             sup_cnt=sup_cnt, transform=transform, neighborlen=args.tgl+1)
    trainset_unsup.setmode('train_unsup')
    trainloader_sup = DataLoader(trainset_sup, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
    trainloader_unsup = DataLoader(trainset_unsup, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)
    testset = Dataset(test_x, test_y)
    testloader = DataLoader(testset, batch_size=args.batch_size,  # 9894
                            shuffle=False, num_workers=args.num_workers)

    net1 = train_caml(net=(net1, net2, sup_fw_head, trans_fw),
                     dataloader=(trainloader_sup,
                                 trainloader_unsup,
                                 testloader),
                     criterion=criterion,
                     optimizer=(optimizer1, optimizer2),
                     epo=args.epo,
                     args=args)

    # test
    f1_macro, acc, prec, recal, f1_macro_weight = test(net1, testloader, loadname=args.exp+args.DATASET+str(seed))
    return f1_macro, acc, prec, recal, f1_macro_weight
