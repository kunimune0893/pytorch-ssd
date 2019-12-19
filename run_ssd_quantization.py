import argparse
import os, sys
import random
import logging
import itertools
import numpy as np
from functools import partial

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

script_dir = os.path.dirname("__file__")
module_path = os.path.abspath(os.path.join(script_dir, "..", "pytorch-distiller-new"))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
from distiller.data_loggers import *

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

# Params operation
parser.add_argument('--ope', default='calibration', type=str, help="calibration(default) or debug")

# Params dataset
parser.add_argument('--dataset_type', default='voc', type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--effective-train-size', '--etrs', type=float, default=1.,
                    help='Portion of training dataset to be used in each epoch. '
                         'NOTE: If --validation-split is set, then the value of this argument is applied '
                         'AFTER the train-validation split according to that argument')

parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--debug-dk', type=str, help='for debug')

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

# 再現を可能にする形態で動作させる
seed = 7
torch.manual_seed( seed )
random.seed( seed )
np.random.seed( seed )
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test(model, loader, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for ii, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        
        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
        
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        
        logging.info( "test(): ii={:4d}, running_loss/num={:6.4f}, running_regression_loss/num={:6.4f}, running_classification_loss/num={:6.4f}".format(
            ii, running_loss / num, running_regression_loss / num, running_classification_loss / num) )
    
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

if __name__ == '__main__':
    timer = Timer()
    
    logging.info(args)
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    # Transform：統計情報は学習用データで取得するが、Augumentation は不要 → テスト用の Transform を使う
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=test_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                 transform=test_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)
        
        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    
    train_dataset = ConcatDataset(datasets)
    
    if args.effective_train_size == 0:
        train_dataset = Subset(train_dataset, [0])
        logging.info("Train dataset size: {}".format(len(train_dataset)))
        
        train_loader = DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False)
    else:
        train_size = int( len(train_dataset) * args.effective_train_size )
        train_dataset, dmy_dataset = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        logging.info("Train dataset size: {}, {}".format(train_size, len(train_dataset)))
        
        train_loader = DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True)
    
    logging.info("Build network.")
    net = create_net(num_classes, is_test=True, debug_dk=args.debug_dk)
    min_loss = -10000.0
    last_epoch = -1
    
    # 学習はしない
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    
    net.to(DEVICE)
    
    if args.ope == "calibration":
        criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                 center_variance=0.1, size_variance=0.2, device=DEVICE)
        
        test_fn = partial( test, loader=train_loader, criterion=criterion, device=DEVICE )
        
        collect_quant_stats( net, test_fn, save_dir='./logs', classes=None, inplace_runtime_check=True,
                            disable_inplace_attrs=True)
    
    elif args.ope == "debug":
        net.eval()
        for ii, data in enumerate(train_loader):
            images, boxes, labels = data
            images = images.to(DEVICE)
            boxes  = boxes.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with torch.no_grad():
                print( images.shape )
                print( net )
                print( net.base_net )
                print( net.base_net[0][0] )
                
                
                tmp = net.base_net[0][0]( images )
                print( tmp.shape ) #=> torch.Size([1, 32, 150, 150])
                print( "tmp[0, 21, 63, 146]=", tmp[0, 21, 63, 146] ) #=> tensor(2.8134, device='cuda:0')
                
                inp = images.data.cpu().numpy()
                wei = net.base_net[0][0].weight.data.cpu().numpy()
                
                print( np.sum(inp[0, 0:3, 124:127, 290:293] * wei[21]) ) #=> 2.081288
                # 21 * 150 * 150 = 472500 (21枚目のフィルタにジャンプ)
                # 472500 + 150   = 472650 (先頭行はパディングがあるためスキップ)
                # 472650 / 16    = 29540.625
                # 29540 * 16     = 472640
                for ii in range(1, 150):
                    #print( np.sum(inp[0, 0:3, 1:4, (1 + 2 * ii):(4 + 2 * ii)] * wei[21]) ) # 21 * 150 * 150 + 150 = 472650, 472650 / 16 = 29540.625, 29540 * 16 = 472640
                    #print( np.sum(inp[0, 0:3, 1:4, (1 + 2 * ii):(4 + 2 * ii)] * wei[2]) ) # 2 * 150 * 150 + 150 = 45150, 45150 / 16 = 2821.875, 2821 * 16 = 45136
                    print( np.sum(inp[0, 0:3, (63 * 2 - 1):(63 * 2 - 1 + 3), (ii * 2 - 1):(ii * 2 - 1 + 3)] * wei[21]) )
