##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import argparse
import os
import torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Parameters():
    def __init__(self):
        modes = ['train', 'test']
        encoder_names = ['resnet50', 'resnet101']
        decoder_names = ['attention']
        classifier_type = ['CE', 'OR', 'OHEM']
        inference_type = ['hard', 'soft']
        opt_names = ['sgd', 'adam', 'adagrad', 'amsgrad', 'adabound', 'amsbound']
        sch_names = ['step', 'poly', 'plateau', 'constant', 'cosine']
        dataset_names = ['nyu', 'kitti']

        parser = argparse.ArgumentParser(description='ACAN Pytorch Implementation.')
        
        parser.add_argument("--random-seed", type=int, default=2019,
                            help="random seed to have reproducible results.")
        # model settings
        parser.add_argument('--mode',            type=str,      default='train',          choices=modes,
                            help='mode: '+'|'.join(modes)+' (default: train)')
        parser.add_argument('--encoder',         type=str,      default='resnet50',       choices=encoder_names,
                            help='encoder: '+'|'.join(encoder_names)+' (default: resnet50)')
        parser.add_argument('--decoder',         type=str,      default='attention',      choices=decoder_names,
                            help='decoder: '+'|'.join(decoder_names)+" (default: ['attention'])")
        parser.add_argument('--classes',         type=int,      default=80,
                            help='number of discrete classes of detph (default: 80)')
        parser.add_argument('--resume',          type=str,      default='',
                            help='reloaded checkpoint, absolute path (str), given epoch number (int) or nn.Module class')
        parser.add_argument('--pretrain',        type=str2bool, default=False,
                            help='whether to initialize net from pretrained model')
        parser.add_argument('--retain',          type=str2bool, default=False,
                            help='whether to restore the model from breakpoint')
        # dataset settings
        parser.add_argument('--dataset',         type=str,      default='nyu',            choices=dataset_names,
                            help='dataset: '+'|'.join(dataset_names)+' (default: nyu)')
        parser.add_argument('--rgb-dir',         type=str,      required=True,
                            help='root to rgb')
        parser.add_argument('--dep-dir',         type=str,      required=True,
                           help='root to depth')
        parser.add_argument('--train-rgb',       type=str,      default='',
                            help='path to the rgb txt file of trainset')
        parser.add_argument('--train-dep',       type=str,      default='',
                            help='path to the depth txt file of trainset')
        parser.add_argument('--val-rgb',         type=str,      default='',
                             help='path to the rgb txt file of valset')
        parser.add_argument('--val-dep',         type=str,      default='',
                            help='path to the depth txt file of valset')
        parser.add_argument('--test-rgb',        type=str,      default='',
                            help='path to the rgb txt file of testset')
        parser.add_argument('--test-dep',        type=str,      default='',
                            help='path to the depth txt file of testset')
        # data augmentation settings
        parser.add_argument("--random-flip",     type=str2bool, default=False,
                            help="whether to randomly left-right flip the inputs during the training.")
        parser.add_argument("--random-scale",    type=str2bool, default=False,
                            help="whether to randomly scale the inputs during the training.")
        parser.add_argument("--random-rotate",   type=str2bool, default=False,
                            help="whether to randomly rotate the inputs during the training.")
        parser.add_argument("--random-jitter",   type=str2bool, default=False,
                            help="whether to apply color jitter to the inputs during the training.")
        parser.add_argument("--random-crop",     type=str2bool, default=False,
                            help="whether to randomly crop the inputs during the training.")
        # criterion settings
        parser.add_argument('--classifier',      type=str,      default='OR',           choices=classifier_type,
                            help='classifier: '+'|'.join(classifier_type)+' (default: OR)')
        parser.add_argument('--inference',       type=str,      default='soft',         choices=inference_type,
                            help='inference: '+'|'.join(inference_type)+' (default: soft)')
        parser.add_argument("--eps",             type=float,    default=0.0,
                            help="a small value of label smoothing regularization. (default: 0.0)")
        parser.add_argument("--prior",           type=str,      default='uniform',
                            help="prior distribution of label smoothing regularization. (default: uniform)")
        parser.add_argument("--use-weights",     type=str2bool, default=False,
                            help="whether to use the weights of classes (default: false).")
        parser.add_argument("--ohem-thres",      type=float,    default=0.6,
                            help="choose the samples with correct probability underthe threshold. (default: 0.6)")
        parser.add_argument("--ohem-thres1",     type=float,    default=0.8,
                            help="choose the threshold for easy samples. (default: 0.8)")
        parser.add_argument("--ohem-thres2",     type=float,    default=0.5,
                            help="choose the threshold for hard samples. (default: 0.5)")
        parser.add_argument("--ohem-keep",       type=int,      default=100000,
                            help="choose the samples with correct probability underthe threshold. (default: 100000)")
        parser.add_argument('--alpha',           type=float,    default=0.0,
                            help='coefficient of intermediate supervision loss (default: 0.0)')
        parser.add_argument('--beta',            type=float,    default=0.0,
                            help='coefficient of attention loss (default: 0.0)')
        # optimizer settings
        parser.add_argument('--optimizer',       type=str,      default='sgd',          choices=opt_names,
                            help='optimizer: '+'|'.join(opt_names)+' (default: sgd)')
        parser.add_argument('--momentum',        type=float,    default=0.9,
                            help='momentum of optimizer (default: 0.9)')
        parser.add_argument('--gamma',           type=float,    default=1e-3,
                            help='convergence speed term of AdaBound (default: 1e-3)')
        parser.add_argument('--beta1',           type=float,    default=0.9,
                            help='Adam coefficients beta1 (default: 0.9)')
        parser.add_argument('--beta2',           type=float,    default=0.95,
                            help='Adam coefficients beta2 (default: 0.95)')
        parser.add_argument('--weight-decay',    type=float,    default=5e-4,
                            help='initial learning rate (default: 5e-4)')
        # scheduler settings
        parser.add_argument('--scheduler',       type=str,      default='step',         choices=sch_names,
                            help='mode: '+'|'.join(sch_names)+' (default: step)')
        parser.add_argument('--lr',              type=float,    default=1e-4,
                            help='initial learning rate (default: 1e-4)')
        parser.add_argument('--final-lr',        type=float,    default=1e-2,
                            help='final learning rate of adabound lr-scheduler (default: 1e-2)')
        parser.add_argument('--min-lr',          type=float,    default=1e-8,
                            help='minimal learning rate of cosine annealing lr-scheduler (default: 1e-8)')
        parser.add_argument('--milestones',      type=str,      default='[10, 20]',
                            help='milestones of step lr-scheduler (default: [10, 20])')
        parser.add_argument('--T-max',           type=int,      default=5,
                            help='cycle of cosine annealing lr-scheduler(default: 5)')
        parser.add_argument("--patience",        type=int,      default=5,
                            help="patience of plateau lr-scheduler (default: 5)")
        parser.add_argument("--lr-decay",        type=float,    default=0.1,
                            help="decay rate of step or plateau lr-scheduler (default: 0.1)")
        parser.add_argument("--power",           type=float,    default=0.9,
                            help="decay rate of poly lr-scheduler (default: 0.9)")
        # common settings
        parser.add_argument('--batch',           type=int,      default=8,
                            help='batch size of trainset (default: 8)')
        parser.add_argument('--batch-val',       type=int,      default=8,
                            help='batch size of valset (default: 8)')
        parser.add_argument('--epochs',          type=int,      default=50,
                            help='number of epochs (default: 50)')
        parser.add_argument('--eval-freq',       type=int,      default=1,
                            help='number of evaluation interval during training (default: 1)')
        parser.add_argument('--gpu',             type=str2bool, default=True,
                            help='GPU or CPU (default: True)')
        parser.add_argument('--threads',         type=int,      default=4,
                            help='number of threads for data loading (default: 4)')
        # evaluation settings
        parser.add_argument("--use-flip",        type=str2bool, default=False,
                            help="whether to flip during test-stage.")
        parser.add_argument("--use-ms",          type=str2bool, default=False,
                            help="whether to multi-scale crop during test-stage.")    
        # workspace settings
        parser.add_argument('--workdir',         type=str,      default='../workspace/',
                            help='directory for storing training files')
        parser.add_argument('--logdir',          type=str,  
                            help='subdir of workdir, storing checkpoint and logfile (style: LOG_net_dataset_exp)')
        parser.add_argument('--resdir',          type=str,      default='res',
                            help='subdir of logdir, storing predicted results (default: res)')
        self.parser = parser


    def parse(self):
        args = self.parser.parse_args()

        if args.dataset == 'nyu':
            args.min_depth, args.max_depth = 0.72, 10.0 # 0.71
            args.height, args.width = 224, 304
        elif args.dataset == 'kitti':
            args.min_depth, args.max_depth = 1.98, 80.0 # 1.97
            args.height, args.width = 160, 640
        return args
