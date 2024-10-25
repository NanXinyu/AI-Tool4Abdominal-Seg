import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.loss import NMTCritierion, NMTNORMCritierion, KLDiscretLoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
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
                        default='resnet3D_lc')
    
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
                        default=150)
    parser.add_argument('--GPUS',
                        help='GPUs',
                        type=tuple,
                        default=(0,1,))
    parser.add_argument('--batch_size_per_gpu',
                        help='Batch Size per GPU',
                        type=int,
                        default=8)
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
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(args, 'train')

    logger.info(pprint.pformat(args))

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    
    # TODO
    model = eval('models.'+args.modelName+'.generate_model')(
        args
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, './models', args.modelName + '.py'),
        final_output_dir)


    dump_input = torch.rand(
        (1, 3, args.target_size[0], args.target_size[1], args.target_size[2])
    )
    
    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total number of parameters: %d" % pytorch_total_params)

    model = torch.nn.DataParallel(model, device_ids=args.GPUS).cuda()

    # define loss function (criterion) and optimizer
    if args.lossType == 'NMTCritierion':
        criterion = NMTCritierion(label_smoothing=0.1).cuda()
    elif args.lossType == 'NMTNORMCritierion':
        criterion = NMTNORMCritierion(label_smoothing=0.1).cuda()
    elif args.lossType == 'KLDiscretLoss':
        criterion = KLDiscretLoss().cuda()        
  

    # TODO:Data loading code
    normalize = transforms.Normalize(
        mean=[0.5], std=[0.5]
    )
    train_dataset = eval('dataset.'+'LocL3Dataset'+'.LocL3Dataset')(
        args, 'train'
    )

    valid_dataset = eval('dataset.'+'LocL3Dataset'+'.LocL3Dataset')(
        args, 'valid'
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu*len(args.GPUS),
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=len(args.GPUS),
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    best_perf = 123456789.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(args, model)
    begin_epoch = args.start_epoch
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if args.auto_resume and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [90, 110], 0.1,
        last_epoch=last_epoch
    )        

    for epoch in range(begin_epoch, args.end_epoch):
        

        train(args, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()    
        cur_perf = validate(
            args, valid_loader, valid_dataset, model, criterion)   
        
        if cur_perf <= best_perf:
            best_perf = cur_perf
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.modelName,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': cur_perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
   


if __name__ == '__main__':
    main()