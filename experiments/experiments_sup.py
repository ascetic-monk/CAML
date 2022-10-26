import numpy as np
import sklearn.metrics as metrics
import torch
# import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader

from dataloader import Dataset, HARData
from network import model_select
from experiments.test import test
import utils.transforms as transforms
from utils.pytorchtools import EarlyStopping


def train_sup(net, dataloader, criterion, optimizer, epo=60, args=None):
    trainloader, testloader = dataloader
    early_stopping = EarlyStopping(args.patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format('checkpoints'))

    net.train()
    net = net.cuda()
    criterion = criterion.cuda()
    trainloader.dataset.setmode('train_sup')
    for epoch in range(epo):  # loop over the dataset multiple times
        running_loss = 0.0
        iter_num = args.iter
        # for i, data in enumerate(trainloader):
        for i in range(iter_num):
            # get the inputs
            data = next(iter(trainloader))
            # print("length of trainloader is %d" % len(trainloader))
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.float().cuda(), labels.long().cuda()

            # wrap them in Variable
            # inputs, labels = inputs.cuda(), labels.cuda()
            # labels = labels.squeeze(1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.cpu().item()
        print('[%d, %5d] loss: %.6f' %
              (epoch + 1, i + 1, running_loss / 100))
        # running_loss = 0.0
        if args.es:
            early_stopping(running_loss, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if args.es:
        net.load_state_dict(early_stopping.checkpoint)
    print('Finished Training')
    torch.save(net.state_dict(), 'net_param_sup.pth')

    return net


def exp_sup(seed, dataset, args):

    # create networks
    net = model_select(args.model_name, args)  # .cuda()

    # create the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()  # weight=weights)

    # create dataset and dataloader
    if args.notransform:
        transform = transforms.Compose([transforms.MagnitudeWrap(sigma=0.3, knot=4, p=0.2),
                                        transforms.TimeWarp(sigma=0.2, knot=8, p=0.2)])
    else:
        print('no transform are utilized!')
        transform = None
    # transform = None
    train_x, train_y, test_x, test_y, sup_cnt = dataset.load(ratio=args.ratio,
                                                             seed=seed,
                                                             adj=False)

    trainset = Dataset(train_x, train_y, sup_cnt=sup_cnt, unsup_ratio=args.unratio, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    testset = Dataset(test_x, test_y)
    testloader = DataLoader(testset, batch_size=args.batch_size,  # 9894
                            shuffle=False, num_workers=args.num_workers)

    net = train_sup(net=net,
                     dataloader=(trainloader, testloader),
                     criterion=criterion,
                     optimizer=optimizer,
                     epo=args.epo,
                     args=args)

    # test

    f1_macro, acc, prec, recal, f1_macro_weight = test(net, testloader, loadname=args.exp+args.DATASET+str(seed))
    return f1_macro, acc, prec, recal, f1_macro_weight


def test_sup(seed, dataset, args):

    # create networks
    net = model_select(args.model_name, args)  # .cuda()

    # create the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()  # weight=weights)

    # create dataset and dataloader
    if args.notransform:
        transform = transforms.Compose([transforms.MagnitudeWrap(sigma=0.3, knot=4, p=0.2),
                                        transforms.TimeWarp(sigma=0.2, knot=8, p=0.2)])
    else:
        print('no transform are utilized!')
        transform = None
    # transform = None
    train_x, train_y, test_x, test_y, sup_cnt = dataset.load(ratio=args.ratio,
                                                             seed=seed,
                                                             adj=False)

    trainset = Dataset(train_x, train_y, sup_cnt=sup_cnt, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    testset = Dataset(test_x, test_y)
    testloader = DataLoader(testset, batch_size=args.batch_size,  # 9894
                            shuffle=False, num_workers=args.num_workers)

    net = train_sup(net=net,
                     dataloader=(trainloader, testloader),
                     criterion=criterion,
                     optimizer=optimizer,
                     epo=args.epo,
                     args=args)

    # test

    f1_macro, acc, prec, recal, f1_macro_weight = test(net, testloader)
    return f1_macro, acc, prec, recal, f1_macro_weight