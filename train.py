import os
import random

import numpy as np
import torch
from dataloader import HARData

from experiments.experiments_sup import exp_sup
from experiments.experiments_caml import exp_caml
import logging


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)

    logger.setLevel(level=logging.DEBUG)

    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Run SEMI')
    parser.add_argument('--exp', type=str, default='sup')
    parser.add_argument('--DATASET', type=str, default='uci')
    parser.add_argument('--model_name', type=str, default='Conv')#Conv
    parser.add_argument('--de_model', type=str, default='DeConv')
    parser.add_argument('--feature_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=15)
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--epo', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--es', action="store_false")
    parser.add_argument('--notransform', action="store_true")
    parser.add_argument('--iter', type=int, default=80)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lenset', type=str, default='long')
    parser.add_argument('--unratio', type=float, default=1.0)
    parser.add_argument('--logname', type=str, default='test')
    parser.add_argument('--tgr', type=float, default=0.8)
    parser.add_argument('--tgl', type=int, default=2)

    args = parser.parse_args()
    print(args.lenset)
    if args.DATASET == 'pamap':
        args.user_num = 9
        args.channel_num = 52
        args.act_num = 12
        dataset = HARData('pamap', args)
    elif args.DATASET == 'uci':
        args.user_num = 10
        args.channel_num = 9
        args.act_num = 6
        dataset = HARData('uci', args)
    elif args.DATASET == 'wisdm':
        args.user_num = 36
        args.channel_num = 3
        args.act_num = 6
        dataset = HARData('wisdm', args)
    elif args.DATASET == 'mhealth':
        args.user_num = 11
        args.channel_num = 21
        args.act_num = 6
        dataset = HARData('mhealth', args)
        
    logger = logger_config(log_path='./results/'+args.DATASET+args.exp+str(args.ratio).replace('.', '_')+'.txt',
                           logging_name=args.logname)
    logger_all = logger_config(log_path='./results/overall_results.txt',
                               logging_name=args.logname)

    seeds = np.arange(args.K)
    f1_macros, accs, precs, recals, f1_weights = [], [], [], [], []
    f1_macros2, accs2, precs2, recals2 = [], [], [], []
    logger.info('feature_s:%f, batch_s:%f, K:%d, epo:%d, lr:%f'
                % (args.feature_size, args.batch_size, args.K, args.epo, args.lr))

    for seed in seeds:
        print('======== SEED %d/%d ==========' % (seed, len(seeds)))
        print('=R:%f, exp:%s, data:%s=' % (args.ratio, args.exp, args.DATASET ))
        seed_torch(seed)
        if args.exp == 'sup':
            f1_macro, acc, prec, recal, f1_weight = exp_sup(seed=seed,
                                                             dataset=dataset,
                                                             args=args)
        elif args.exp == 'caml':
            f1_macro, acc, prec, recal, f1_weight = exp_caml(seed=seed,
                                                            dataset=dataset,
                                                            args=args)


        logger.info('f1:%f, acc:%f, f1w:%f' % (f1_macro, acc, f1_weight))
        f1_macros.append(f1_macro)
        f1_weights.append(f1_weight)
        accs.append(acc)
        precs.append(prec)
        recals.append(recal)
    f1_macros = np.array(f1_macros)
    f1_weights = np.array(f1_weights)
    accs = np.array(accs)
    precs = np.array(precs)
    recals = np.array(recals)
    logger.info('overall f1:%f(%f), acc:%f(%f)' % (np.mean(f1_macros), np.std(f1_macros),
                                                   np.mean(accs), np.std(accs)))

    logger_all.info('Dataset:%s|exp:%s|ratio:%f|K:%d|f1:%f(%f)|acc:%f(%f)|f1_weight:%f(%f)' % (args.DATASET, argexp,
                                                                                               args.ratio, args.K,
                                                                                               np.mean(f1_macros),
                                                                                               np.std(f1_macros),
                                                                                               np.mean(accs),
                                                                                               np.std(accs),
                                                                                               np.mean(f1_weights),
                                                                                               np.std(f1_weights)))

    print('===================================================')
    print("\t Macro - average Test fscore mean:\t{:.3f}, std:\t{:.3f} ".format(np.mean(f1_macros), np.std(f1_macros)))
    print("\t Average Test accuracy mean:\t{:.3f}, std:\t{:.3f} ".format(np.mean(accs), np.std(accs)))
    print("\t Average Test precision mean:\t{:.3f}, std:\t{:.3f} ".format(np.mean(precs), np.std(precs)))
    print("\t Average Test recall mean:\t{:.3f}, std:\t{:.3f} ".format(np.mean(recals), np.std(recals)))
    print('===================================================')

