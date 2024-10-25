import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.loss import NMTCritierion, NMTNORMCritierion, KLDiscretLoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelName',
                        help='model name',
                        type=str,
                        default='resnet3D_pc')
    
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--lossType',
                        help='loss type',
                        type=str,
                        default='KLDiscretLoss')
    parser.add_argument('--netDepth',
                        help='Network Depth',
                        type=int,
                        default=18)
    parser.add_argument('--start_epoch',
                        help='Start Epoch',
                        type=int,
                        default=0)
    parser.add_argument('--end_epoch',
                        help='End Epoch',
                        type=int,
                        default=1000)
    parser.add_argument('--GPUS',
                        help='GPUs',
                        type=tuple,
                        default=(0,))
    parser.add_argument('--batch_size_per_gpu',
                        help='Batch Size per GPU',
                        type=int,
                        default=1)
    parser.add_argument('--output_dir',
                        help='Output Dir',
                        type=str,
                        default='./output/')
    parser.add_argument('--log_dir',
                        help='Log Dir',
                        type=str,
                        default='./log')
    parser.add_argument('--auto_resume',
                        help='Auto Resume',
                        type=bool,
                        default=True)
    parser.add_argument('--optimizer',
                        help='Train Optimizer',
                        type=str,
                        default='adam'),
    parser.add_argument('--sigma',
                        help='Sigma',
                        type=float,
                        default=6.0),
    parser.add_argument('--target_size',
                        help='Target Size',
                        type=tuple,
                        default=(512,256,256)),
    parser.add_argument('--data_dir',
                        help='Data Dir',
                        type=str,
                        default='/gpfs/share/home/2301213095/L3Loc/LocL3Dataset/')
    parser.add_argument('--split_ratio',
                        help='Split Ratio',
                        type=float,
                        default=1.0)
    parser.add_argument('--strong_aug',
                        help='Strong Augmentation',
                        type=bool,
                        default=False)
    parser.add_argument('--test_file',
                        help='Test File',
                        type=str,
                        default='./output/LocL3Dataset/')
    args = parser.parse_args()

    return args



def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(args, 'valid')

    logger.info(pprint.pformat(args))

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    model = eval('models.'+args.modelName+'.generate_model')(
        args
    )

    if args.test_file+args.modelName+'_01/final_state.pth':
        logger.info('=> loading model from {}'.format(args.test_file+args.modelName+'_01/final_state.pth'))
        model.load_state_dict(torch.load(args.test_file+args.modelName+'_01/final_state.pth'), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )               
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=args.GPUS).cuda()

    if args.lossType == 'NMTCritierion':
        criterion = NMTCritierion(label_smoothing=0.1).cuda()
    elif args.lossType == 'NMTNORMCritierion':
        criterion = NMTNORMCritierion(label_smoothing=0.1).cuda()
    elif args.lossType == 'KLDiscretLoss':
        criterion = KLDiscretLoss().cuda()            
    
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.5], std=[0.5]
    )

    valid_dataset = eval('dataset.'+'LocL3Dataset'+'.LocL3Dataset')(
        args, 'valid'
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=len(args.GPUS),
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    # evaluate on validation set
    validate(args, valid_loader, valid_dataset, model, criterion) 

if __name__ == '__main__':
    main()