from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import os
import sys
import datetime
sys.path.insert(0, os.getcwd())
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize
import collections
import torch.nn.functional as F

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from gcc import datasets
from gcc import models
from gcc.trainer_v2 import GCCTrainer
from gcc.evaluators import Evaluator, extract_features
from gcc.utils.data import IterLoader
from gcc.utils.data import transforms as T
from gcc.utils.data.sampler import RandomMultipleGallerySampler
from gcc.utils.data.preprocessor import Preprocessor
from gcc.utils.logging import Logger
from gcc.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from gcc.utils.rerank import compute_jaccard_dist
from sklearn.cluster import DBSCAN

best_mAP = 0

def get_data(name, data_dir):
    # root = osp.join(data_dir, name)
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters=800, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = dataset.train if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=2),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    arch = []
    arch.append('resnet50')
    
    model_list = [models.create(arch[i], num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters).cuda() for i in range(1)]
    model_ema_list = [models.create(arch[i], num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters).cuda() for i in range(1)]
    
    model_list = [nn.DataParallel(model_list[i]) for i in range(len(model_list))]
    model_ema_list = [nn.DataParallel(model_ema_list[i]) for i in range(len(model_ema_list))]

    if len(args.init_1) > 0:
        initial_weights = load_checkpoint(args.init_1)
        copy_state_dict(initial_weights['state_dict'], model_list[0])
        copy_state_dict(initial_weights['state_dict'], model_ema_list[0])
        model_ema_list[0].module.classifier.weight.data.copy_(model_list[0].module.classifier.weight.data)        

    for i in range(len(model_ema_list)):
        for param in model_ema_list[i].parameters():
            param.detach_()

    return model_list, model_ema_list

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    global best_mAP
    best_mAP = 0.
    cudnn.benchmark = True

    args.logs_dir += f'-lw{args.loss_weight}'
    if os.path.exists(args.logs_dir):
        print(f'there already is {args.logs_dir}'
              f'\n Press \'y\' or 1 to remove the file and continue. Press other keys to exit.')
        temp_input = input('y/n or 1/0')
        if temp_input == 'y' or temp_input == 1 or temp_input == '1':
            print(f'\rContinue training in {args.logs_dir}')
        else:
            os._exit(0)
    sys.stdout = Logger(osp.join(args.logs_dir, 'train.log'))
    BEGIN_TIME = datetime.datetime.now()
    print(BEGIN_TIME)
    # print("==========\nArgs:{}\n==========".format(args))
    print("==========")
    for k in args.__dict__:
        print(f'{k} : {args.__dict__[k]}')
    print("==========")

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model_list, model_ema_list = create_model(args)

    # Evaluator
    evaluator_ema_list = [Evaluator(model_ema_list[i]) for i in range(len(model_ema_list))]

    for epoch in range(args.start_epoch, args.epochs):
        cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)

        cf = []
        for i in range(len(model_ema_list)):
            if os.path.exists('pre-ex-cf-'+str(i)+'.npy') and (epoch == args.start_epoch):
                temp = input('load pre-extracted features? press 1 to continue, 0 to exit')
                if temp == 1 or temp == '1':
                    cf_i = np.load('pre-ex-cf-'+str(i)+'.npy')
                    print('\n load pre-extracted features \n')
                else:
                    os._exit(0)
            else:
                dict_f, _ = extract_features(model_ema_list[i], cluster_loader, print_freq=100)
                cf_i = torch.stack(list(dict_f.values()))
                np.save('cf-'+str(i)+'.npy', cf_i.numpy())
            cf.append(cf_i)

        print('k-means')
        cf_avg = cf[0]
        for i in range(1, len(cf)):
            cf_avg = torch.cat((cf_avg, cf[i]), dim=1)
        
        km = MiniBatchKMeans(n_clusters=args.num_clusters, max_iter=100, batch_size=100, init_size=1500).fit(cf_avg.numpy())
        for i in range(len(model_ema_list)):
            num_features = model_list[i].module.classifier.in_features
            model_list[i].module.classifier = nn.Linear(num_features, args.num_clusters, bias=False).cuda()
            model_list[i].module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_[:, i*2048:i*2048+2048], axis=1)).float().cuda())

            model_ema_list[i].module.classifier = nn.Linear(num_features, args.num_clusters, bias=False).cuda()
            model_ema_list[i].module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_[:, i*2048:i*2048+2048], axis=1)).float().cuda())

        target_label = km.labels_
        # change pseudo labels
        for i in range(len(dataset_target.train)):
            dataset_target.train[i] = list(dataset_target.train[i])
            dataset_target.train[i][1] = int(target_label[i])
            dataset_target.train[i] = tuple(dataset_target.train[i])

        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)
        print('\n Clustering into {} classes \n'.format(args.num_clusters))
        
        # Optimizer
        print('learning rate', args.lr)
        params = []
        for i in range(len(model_list)):
            for key, value in model_list[i].named_parameters():
                if not value.requires_grad:
                    continue
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)

        # Trainer
        trainer = GCCTrainer(model_list, model_ema_list, num_cluster=args.num_clusters, alpha=args.alpha)
        train_loader_target.new_epoch()
        trainer.train(epoch, train_loader_target, optimizer, print_freq=args.print_freq, 
                      train_iters=len(train_loader_target), loss_weight = args.loss_weight, k=args.k, beta=args.beta)

        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = []
            for i in range(len(evaluator_ema_list)):
                mAP_i = evaluator_ema_list[i].evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
                mAP.append(mAP_i)
            is_best = max(mAP) > best_mAP
            best_mAP = max(mAP + [best_mAP])
            for i in range(len(evaluator_ema_list)):
                save_model(model_ema_list[i], (is_best and (mAP[i]==best_mAP)), best_mAP, i)

            print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} best: {:5.1%}{}'.
                  format(epoch, mAP[0], best_mAP, ' *' if is_best else ''))
            print(datetime.datetime.now())
            print('Cost ' + str(datetime.datetime.now() - BEGIN_TIME))            
            print(args.logs_dir, end='\n')

    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model_ema_list[0].load_state_dict(checkpoint['state_dict'])
    evaluator_ema_list[0].evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
    print('\n' + str(datetime.datetime.now()))
    print('Cost ' + str(datetime.datetime.now() - BEGIN_TIME))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MEB-Net Training")
    # data
    parser.add_argument('-dt', '--dataset_target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--num_clusters', type=int, default=500)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num_instances', type=int, default=16)
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--k', type=int, default=12)

    # training configs
    parser.add_argument('--init_1', type=str, default='')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--scatter', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, default='/data/ceph_11015/ssd/xbinliu/reid/data/')
    parser.add_argument('--logs_dir', type=str, default=osp.join(os.getcwd(), 'logs'))

    main()
