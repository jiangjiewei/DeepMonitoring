"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import PIL
import cv2
import timm.models as models
from shutil import copyfile
import pickle
from PIL import Image
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.rep_VGGNET import RepVGG_A1
from models import build_model
from models.convnext import convnext_base


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--data', metavar='DIR', default='./dataA_kz_1/',
#                     help='path to dataset')
parser.add_argument('--data', metavar='DIR', default='./',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='convnext_base',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate moinrtdel on validation set')
parser.add_argument('--pretrained', default='pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fine-tuning', default='True', action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU id to use.')  # default=3,
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    pre_model_keritits = 0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch)

    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            if args.arch.find('mobilenetv3_large_075') != -1:
                model = models.mobilenetv3_large_075(pretrained=True)
            elif args.arch.find('convnext_base') != -1:
                model = models.convnext_base(pretrained=True)
            elif args.arch.find('RepVGG_A1') != -1:
                model = RepVGG_A1()
                model.load_state_dict(torch.load("./"))
            elif args.arch.find('Transform_base') != -1:
                model = build_model("swin_base")
                model.load_state_dict(torch.load("./")["model"])
            else:
                print('### please check the args.arch for load model in training###')
                exit(-1)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

    if args.fine_tuning:
        print("=> transfer-learning mode + fine-tuning (train only the last FC layer)")
        # Freeze Previous Layers(now we are using them as features extractor)
        # jiangjiewei
        # for param in model.parameters():
        #    param.requires_grad = False

        # Fine Tuning the last Layer For the new task
        # juge network: alexnet, inception_v3, densennet, resnet50
        if args.arch.find('RepVGG_A1') != -1:
            inchannel = model.linear.in_features
            model.linear = nn.Linear(inchannel, 6)
        elif args.arch.find('mobilenetv3_large_075') != -1:
            inchannel = model.classifier.in_features
            model.classifier = nn.Linear(inchannel, 6)
        elif args.arch.find('Transform_base') != -1:
            inchannel = model.head.in_features
            model.head = nn.Linear(inchannel, 6)
        elif args.arch.find('convnext_base') != -1:
            inchannel = model.head.fc.in_features
            model.head.fc = nn.Linear(inchannel, 6)
        else:
            print("###Error: Fine-tuning is not supported on this architecture.###")
            exit(-1)
        print(model)
    else:
        parameters = model.parameters()


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet-1') or args.arch.startswith('vgg-1'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.arch.find('RepVGG_A1') != -1:
        fine_tune_parameters = model.linear.parameters()
    elif args.arch.find('Transform_base') != -1:
        fine_tune_parameters = model.head.parameters()
    elif args.arch.find('mobilenetv3_large_075') != -1:
        fine_tune_parameters = model.classifier.parameters()
    elif args.arch.find('convnext_base') != -1:
        fine_tune_parameters = model.head.fc.parameters()

    else:
        print('### please check the ignored params ###')
        exit(-1)

    ignored_params = list(map(id, fine_tune_parameters))

    if args.arch.find('alexnet') != -1:
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())
    else:
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())

    # optimizer = torch.optim.SGD([{'params': base_params},  #model.parameters()
    #                           {'params': fine_tune_parameters, 'lr': args.lr}],
    #                          lr=args.lr,
    #                          momentum=args.momentum,
    #                          weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,betas=(0.9,0.999), eps=1e-08,weight_decay=1E-2)# weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam([{'params': base_params},
    #                                {'params': fine_tune_parameters, 'lr': 10*args.lr}],
    #                              lr=0.001, weight_decay=5E-5)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train1')
    valdir = os.path.join(args.data, 'val1')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.5817834, 0.459701, 0.43193635],
                                         std=[0.26537213, 0.27368215, 0.27420396])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
    else:
        image_size = args.image_size

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.Resize((256, 256), interpolation=PIL.Image.BICUBIC),
            # transforms.Resize((224, 224)),
            transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
            # transforms.RandomResizedCrop((image_size, image_size) ),  #RandomRotation scale=(0.9, 1.0)
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print('classes:', train_dataset.classes)
    # Get number of labels
    labels_length = len(train_dataset.classes)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop((image_size,image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res = validate(val_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return
    # times = []
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(args.start_epoch, args.epochs):
        # time1 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        losses = train(train_loader, model, criterion, optimizer, epoch, args)
        train_loss_list.append(losses)
        epoch_list.append(epoch + 1)

        # evaluate on validation set
        acc1, test_losses = validate(val_loader, model, criterion, args)
        test_loss_list.append(test_losses)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.arch.find('RepVGG_A1') != -1:
            pre_name = './RepVGG_A1'
        elif args.arch.find('Transform_base') != -1:
            pre_name = './Transform_base'
        elif args.arch.find('mobilenetv3_large_075') != -1:
            pre_name = './mobilenetv3_large_075'
        elif args.arch.find('convnext_base') != -1:
            pre_name = './convnext_base'
        else:
            print('### please check the args.arch for pre_name###')
            exit(-1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, pre_name)
    print('Finished Training best_acc: ', best_acc1)

    with open(args.arch + "_loss.pkl", "wb") as f:
        pickle.dump(train_loss_list, f)
        pickle.dump(test_loss_list, f)
        pickle.dump(epoch_list, f)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    return losses


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            # torch.cuda.synchronize()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses


def save_checkpoint(state, is_best, pre_filename='my_checkpoint.pth.tar'):
    check_filename = pre_filename + '_checkpoint.pth.tar'
    des_best_filename = pre_filename + '_model_best.pth.tar'
    torch.save(state, check_filename)
    if is_best:
        shutil.copyfile(check_filename, des_best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 20))
    lr_decay = 0.1 ** (epoch // 20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay
        # param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def DeepMonitoring_test():
    args = parser.parse_args()

    dataset_subdir = [
        # 'XM Data 0424',
        #   'train1',
        #  'test1',
        #  'val1',
        # 'External Data2_0512'
        'VIVO Data 0514 数据集',
    ]
    model_names = [
        # 'RepVGG_A1',
        # 'Transform_base',
        'mobilenetv3_large_075',
        # 'convnext_base',
    ]

    for index_name in range(0, len(model_names)):
        args.arch = model_names[index_name]
        for i in range(0, len(dataset_subdir)):
            dataset_dir = '/' + dataset_subdir[i]
            resultset_dir = './' + dataset_subdir[i]
            args, model, val_transforms = load_modle_trained(args)
            mk_result_dir(args, resultset_dir)
            DeepMonitoring_test_exec(args, model, val_transforms, dataset_dir, resultset_dir)
            # fundus_test_exec_external(args, model, val_transforms, dataset_dir, resultset_dir)


def load_modle_trained(args):
    # normalize = transforms.Normalize(mean = [0.5765036, 0.34929818, 0.2401832], std = [0.2179051, 0.19200659, 0.17808074])# old
    normalize = transforms.Normalize(mean=[0.5817834, 0.459701, 0.43193635], std=[0.26537213, 0.27368215, 0.27420396])
    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
        # transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    print("=> loading checkpoint### ", args.arch)
    if args.arch.find('RepVGG_A1') != -1:
        pre_name = './RepVGG_A1'
    elif args.arch.find('Transform_base') != -1:
        pre_name = './Transform_base'
    elif args.arch.find('mobilenetv3_large_075') != -1:
        pre_name = './mobilenetv3_large_075'
    elif args.arch.find('convnext_base') != -1:
        pre_name = './convnext_base'
    else:
        print('### please check the args.arch###')
        exit(-1)
    PATH = pre_name + '_model_best.pth.tar'
    # PATH = './densenet_nodatanocost_best.ckpt'

    # PATH = pre_name + '_checkpoint.pth.tar'

    if args.arch.find('RepVGG_A1') != -1:
        model = RepVGG_A1()
        num_ftrs = model.linear.in_features
        model.linear = nn.Linear(num_ftrs, 6)
    elif args.arch.find('Transform_base') != -1:
        model = build_model('swin_base')
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 6)
    elif args.arch.find('mobilenetv3_large_075') != -1:
        model = models.mobilenetv3_large_075()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 6)
    elif args.arch.find('convnext_base') != -1:
        model = models.convnext_base()
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, 6)
    else:
        print('### please check the args.arch for load model in testing###')
        exit(-1)

    print(model)
    """
    if args.arch.find('alexnet') == -1:
        model = torch.nn.DataParallel(model).cuda()  #for modles trained by multi GPUs: densenet inception_v3 resnet50
    """
    checkpoint = torch.load(PATH)
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    if args.arch.find('alexnet') != -1:
        model = torch.nn.DataParallel(model).cuda()  # for models trained by single GPU: Alexnet
    start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    print('best_epoch and best_acc1 is: ', start_epoch, best_acc1)
    return args, model, val_transforms


def mk_result_dir(args, testdata_dir='./data/val1'):
    testdatadir = testdata_dir
    model_name = args.arch
    result_dir = testdatadir + '/' + model_name
    grade1_grade2 = 'Defocused image_High-quality image'
    grade1_grade3 = 'Defocused image_Image of not fully opened eye'
    grade1_grade4 = 'Defocused image_Image of poor cornea position'
    grade1_grade5 = 'Defocused image_Overexposed images'
    grade1_grade6 = 'Defocused image_Underexposed images'

    grade2_grade1 = 'High-quality image_Defocused image'
    grade2_grade3 = 'High-quality image_Image of not fully opened eye'
    grade2_grade4 = 'High-quality image_Image of poor cornea position'
    grade2_grade5 = 'High-quality image_Overexposed images'
    grade2_grade6 = 'High-quality image_Underexposed images'

    grade3_grade1 = 'Image of not fully opened eye_Defocused image'
    grade3_grade2 = 'Image of not fully opened eye_High-quality image'
    grade3_grade4 = 'Image of not fully opened eye_Image of poor cornea position'
    grade3_grade5 = 'Image of not fully opened eye_Overexposed images'
    grade3_grade6 = 'Image of not fully opened eye_Underexposed images'

    grade4_grade1 = "Image of poor cornea position_Defocused image"
    grade4_grade2 = "Image of poor cornea position_High-quality image"
    grade4_grade3 = "Image of poor cornea position_Image of not fully opened eye"
    grade4_grade5 = "Image of poor cornea position_Overexposed images"
    grade4_grade6 = "Image of poor cornea position_Underexposed images"

    grade5_grade1 = "Overexposed images_Defocused image"
    grade5_grade2 = "Overexposed images_High-quality image"
    grade5_grade3 = "Overexposed images_Image of not fully opened eye"
    grade5_grade4 = "Overexposed images_Image of poor cornea position"
    grade5_grade6 = "Overexposed images_Underexposed images"

    grade6_grade1 = "Underexposed images_Defocused image"
    grade6_grade2 = "Underexposed images_High-quality image"
    grade6_grade3 = "Underexposed images_Image of not fully opened eye"
    grade6_grade4 = "Underexposed images_Image of poor cornea position"
    grade6_grade5 = "Underexposed images_Overexposed images"

    grade1_grade1 = 'Defocused image_Defocused image'
    grade2_grade2 = 'High-quality image_High-quality image'
    grade3_grade3 = 'Image of not fully opened eye_Image of not fully opened eye'
    grade4_grade4 = 'Image of poor cornea position_Image of poor cornea position'
    grade5_grade5 = 'Overexposed images_Overexposed images'
    grade6_grade6 = 'Underexposed images_Underexposed images'

    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
        os.makedirs(result_dir + '/' + grade1_grade2)
        os.makedirs(result_dir + '/' + grade1_grade3)
        os.makedirs(result_dir + '/' + grade1_grade4)
        os.makedirs(result_dir + '/' + grade1_grade5)
        os.makedirs(result_dir + '/' + grade1_grade6)

        os.makedirs(result_dir + '/' + grade2_grade1)
        os.makedirs(result_dir + '/' + grade2_grade3)
        os.makedirs(result_dir + '/' + grade2_grade4)
        os.makedirs(result_dir + '/' + grade2_grade5)
        os.makedirs(result_dir + '/' + grade2_grade6)

        os.makedirs(result_dir + '/' + grade3_grade1)
        os.makedirs(result_dir + '/' + grade3_grade2)
        os.makedirs(result_dir + '/' + grade3_grade4)
        os.makedirs(result_dir + '/' + grade3_grade5)
        os.makedirs(result_dir + '/' + grade3_grade6)

        os.makedirs(result_dir + '/' + grade4_grade1)
        os.makedirs(result_dir + '/' + grade4_grade2)
        os.makedirs(result_dir + '/' + grade4_grade3)
        os.makedirs(result_dir + '/' + grade4_grade5)
        os.makedirs(result_dir + '/' + grade4_grade6)

        os.makedirs(result_dir + '/' + grade5_grade1)
        os.makedirs(result_dir + '/' + grade5_grade2)
        os.makedirs(result_dir + '/' + grade5_grade3)
        os.makedirs(result_dir + '/' + grade5_grade4)
        os.makedirs(result_dir + '/' + grade5_grade6)

        os.makedirs(result_dir + '/' + grade6_grade1)
        os.makedirs(result_dir + '/' + grade6_grade2)
        os.makedirs(result_dir + '/' + grade6_grade3)
        os.makedirs(result_dir + '/' + grade6_grade4)
        os.makedirs(result_dir + '/' + grade6_grade5)

        os.makedirs(result_dir + '/' + grade1_grade1)
        os.makedirs(result_dir + '/' + grade2_grade2)
        os.makedirs(result_dir + '/' + grade3_grade3)
        os.makedirs(result_dir + '/' + grade4_grade4)
        os.makedirs(result_dir + '/' + grade5_grade5)
        os.makedirs(result_dir + '/' + grade6_grade6)


def DeepMonitoring_test_exec(args, model, val_transforms, testdata_dir='./data/val1', resultset_dir='./data/val1'):
    # switch to evaluate mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testdatadir = testdata_dir
    desdatadir = resultset_dir + '/' + args.arch
    model.eval()
    str_end = ('.jpg', '.JPG', '.bmp', '.jpeg', '.JPEG', '.BMP', '.tif', '.TIF', '.png', '.PNG')
    grade1_grade2 = 'Defocused image_High-quality image'
    grade1_grade3 = 'Defocused image_Image of not fully opened eye'
    grade1_grade4 = 'Defocused image_Image of poor cornea position'
    grade1_grade5 = 'Defocused image_Overexposed images'
    grade1_grade6 = 'Defocused image_Underexposed images'

    grade2_grade1 = 'High-quality image_Defocused image'
    grade2_grade3 = 'High-quality image_Image of not fully opened eye'
    grade2_grade4 = 'High-quality image_Image of poor cornea position'
    grade2_grade5 = 'High-quality image_Overexposed images'
    grade2_grade6 = 'High-quality image_Underexposed images'

    grade3_grade1 = 'Image of not fully opened eye_Defocused image'
    grade3_grade2 = 'Image of not fully opened eye_High-quality image'
    grade3_grade4 = 'Image of not fully opened eye_Image of poor cornea position'
    grade3_grade5 = 'Image of not fully opened eye_Overexposed images'
    grade3_grade6 = 'Image of not fully opened eye_Underexposed images'

    grade4_grade1 = "Image of poor cornea position_Defocused image"
    grade4_grade2 = "Image of poor cornea position_High-quality image"
    grade4_grade3 = "Image of poor cornea position_Image of not fully opened eye"
    grade4_grade5 = "Image of poor cornea position_Overexposed images"
    grade4_grade6 = "Image of poor cornea position_Underexposed images"

    grade5_grade1 = "Overexposed images_Defocused image"
    grade5_grade2 = "Overexposed images_High-quality image"
    grade5_grade3 = "Overexposed images_Image of not fully opened eye"
    grade5_grade4 = "Overexposed images_Image of poor cornea position"
    grade5_grade6 = "Overexposed images_Underexposed images"

    grade6_grade1 = "Underexposed images_Defocused image"
    grade6_grade2 = "Underexposed images_High-quality image"
    grade6_grade3 = "Underexposed images_Image of not fully opened eye"
    grade6_grade4 = "Underexposed images_Image of poor cornea position"
    grade6_grade5 = "Underexposed images_Overexposed images"

    grade1_grade1 = 'Defocused image'
    grade2_grade2 = 'High-quality image'
    grade3_grade3 = 'Image of not fully opened eye'
    grade4_grade4 = 'Image of poor cornea position'
    grade5_grade5 = 'Overexposed images'
    grade6_grade6 = 'Underexposed images'

    with torch.no_grad():
        grade1_num = 0
        grade1_grade1_num = 0
        grade1_grade2_num = 0
        grade1_grade3_num = 0
        grade1_grade4_num = 0
        grade1_grade5_num = 0
        grade1_grade6_num = 0
        list_grade1_grade2 = [grade1_grade2]
        list_grade1_grade3 = [grade1_grade3]
        list_grade1_grade4 = [grade1_grade4]
        list_grade1_grade5 = [grade1_grade5]
        list_grade1_grade6 = [grade1_grade6]

        grade1_1 = [grade1_grade1]
        grade1_2 = [grade1_grade2]
        grade1_3 = [grade1_grade3]
        grade1_4 = [grade1_grade4]
        grade1_5 = [grade1_grade5]
        grade1_6 = [grade1_grade6]

        root = testdatadir + '/Defocused image'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)
            # img_PIL_Tensor = img_PIL_Tensor.cuda(args.gpu, non_blocking=True)
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]

            pred = torch.argmax(prob, dim=1)

            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            # print(prob_list)
            # print(pred_0)

            grade1_num = grade1_num + 1
            # print(grade1_num)
            if pred_0 == 0:
                # print('ok to ok')
                grade1_grade1_num = grade1_grade1_num + 1
                grade1_1.append(prob_list)
                file_new_1 = desdatadir + '/Defocused image_Defocused image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                # print('ok to location')
                grade1_grade2_num = grade1_grade2_num + 1
                list_grade1_grade2.append(img)
                grade1_2.append(prob_list)
                file_new_1 = desdatadir + '/Defocused image_High-quality image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                # print('ok to location')
                grade1_grade3_num = grade1_grade3_num + 1
                list_grade1_grade3.append(img)
                grade1_3.append(prob_list)
                file_new_1 = desdatadir + '/Defocused image_Image of not fully opened eye' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                # print('ok to location')
                grade1_grade4_num = grade1_grade4_num + 1
                list_grade1_grade4.append(img)
                grade1_4.append(prob_list)
                file_new_1 = desdatadir + '/Defocused image_Image of poor cornea position' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                # print('ok to location')
                grade1_grade5_num = grade1_grade5_num + 1
                list_grade1_grade5.append(img)
                grade1_5.append(prob_list)
                file_new_1 = desdatadir + '/Defocused image_Overexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('quality to quality')
                grade1_grade6_num = grade1_grade6_num + 1
                list_grade1_grade6.append(img)
                grade1_6.append(prob_list)
                file_new_1 = desdatadir + '/Defocused image_Underexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        #     print(grade1_grade1_num, grade1_grade2_num, grade1_grade3_num,grade1_grade4_num)

        grade2_grade1_num = 0
        grade2_grade2_num = 0
        grade2_grade3_num = 0
        grade2_grade4_num = 0
        grade2_grade5_num = 0
        grade2_grade6_num = 0
        grade2_num = 0
        list_grade2_grade1 = [grade2_grade1]
        list_grade2_grade3 = [grade2_grade3]
        list_grade2_grade4 = [grade2_grade4]
        list_grade2_grade5 = [grade2_grade5]
        list_grade2_grade6 = [grade2_grade6]

        grade2_1 = [grade2_grade1]
        grade2_3 = [grade2_grade3]
        grade2_2 = [grade2_grade2]
        grade2_4 = [grade2_grade4]
        grade2_5 = [grade2_grade5]
        grade2_6 = [grade2_grade6]
        root = testdatadir + '/High-quality image'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            # img_PIL.show()  # 原始图片
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            # image = cv2.imread(os.path.join(root, img))  # image = image.unsqueeze(0) # PIL_image = Image.fromarray(image)
            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            grade2_num = grade2_num + 1
            if pred_0 == 0:
                # print('location to ok')
                grade2_grade1_num = grade2_grade1_num + 1
                list_grade2_grade1.append(img)
                grade2_1.append(prob_list)
                file_new_1 = desdatadir + '/High-quality image_Defocused image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                # print('location to location')
                grade2_grade2_num = grade2_grade2_num + 1
                grade2_2.append(prob_list)
                file_new_1 = desdatadir + '/High-quality image_High-quality image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                # print('location to quality')
                grade2_grade3_num = grade2_grade3_num + 1
                list_grade2_grade3.append(img)
                grade2_3.append(prob_list)
                file_new_1 = desdatadir + '/High-quality image_Image of not fully opened eye' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                # print('location to quality')
                grade2_grade4_num = grade2_grade4_num + 1
                list_grade2_grade4.append(img)
                grade2_4.append(prob_list)
                file_new_1 = desdatadir + '/High-quality image_Image of poor cornea position' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                # print('location to quality')
                grade2_grade5_num = grade2_grade5_num + 1
                list_grade2_grade5.append(img)
                grade2_5.append(prob_list)
                file_new_1 = desdatadir + '/High-quality image_Overexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                # print('location to quality')
                grade2_grade6_num = grade2_grade6_num + 1
                list_grade2_grade6.append(img)
                grade2_6.append(prob_list)
                file_new_1 = desdatadir + '/High-quality image_Underexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
        #    print(grade2_grade1_num, grade2_grade2_num, grade2_grade3_num, grade2_grade4_num)

        grade3_grade1_num = 0
        grade3_grade3_num = 0
        grade3_grade2_num = 0
        grade3_grade4_num = 0
        grade3_grade5_num = 0
        grade3_grade6_num = 0
        grade3_num = 0
        list_grade3_grade1 = [grade3_grade1]
        list_grade3_grade4 = [grade3_grade4]
        list_grade3_grade2 = [grade3_grade2]
        list_grade3_grade5 = [grade3_grade5]
        list_grade3_grade6 = [grade3_grade6]

        grade3_1 = [grade3_grade1]
        grade3_2 = [grade3_grade2]
        grade3_3 = [grade3_grade3]
        grade3_4 = [grade3_grade4]
        grade3_5 = [grade3_grade5]
        grade3_6 = [grade3_grade6]

        root = testdatadir + '/Image of not fully opened eye'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            grade3_num = grade3_num + 1
            if pred_0 == 0:
                grade3_grade1_num = grade3_grade1_num + 1
                list_grade3_grade1.append(img)
                grade3_1.append(prob_list)
                file_new_1 = desdatadir + '/Image of not fully opened eye_Defocused image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                grade3_grade2_num = grade3_grade2_num + 1
                list_grade3_grade2.append(img)
                grade3_2.append(prob_list)
                file_new_1 = desdatadir + '/Image of not fully opened eye_High-quality image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                grade3_grade3_num = grade3_grade3_num + 1
                grade3_3.append(prob_list)
                file_new_1 = desdatadir + '/Image of not fully opened eye_Image of not fully opened eye' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                grade3_grade4_num = grade3_grade4_num + 1
                list_grade3_grade4.append(img)
                grade3_4.append(prob_list)
                file_new_1 = desdatadir + '/Image of not fully opened eye_Image of poor cornea position' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                grade3_grade5_num = grade3_grade5_num + 1
                list_grade3_grade5.append(img)
                grade3_5.append(prob_list)
                file_new_1 = desdatadir + '/Image of not fully opened eye_Overexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                grade3_grade6_num = grade3_grade6_num + 1
                list_grade3_grade6.append(img)
                grade3_6.append(prob_list)
                file_new_1 = desdatadir + '/Image of not fully opened eye_Underexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)

        #   print(grade3_grade1_num, grade3_grade2_num, grade3_grade3_num, grade3_grade4_num)

        grade4_grade1_num = 0
        grade4_grade3_num = 0
        grade4_grade2_num = 0
        grade4_grade4_num = 0
        grade4_grade5_num = 0
        grade4_grade6_num = 0

        grade4_num = 0
        list_grade4_grade1 = [grade4_grade1]
        list_grade4_grade3 = [grade4_grade3]
        list_grade4_grade2 = [grade4_grade2]
        list_grade4_grade5 = [grade4_grade5]
        list_grade4_grade6 = [grade4_grade6]

        grade4_1 = [grade4_grade1]
        grade4_2 = [grade4_grade2]
        grade4_3 = [grade4_grade3]
        grade4_4 = [grade4_grade4]
        grade4_5 = [grade4_grade5]
        grade4_6 = [grade4_grade6]

        root = testdatadir + '/Image of poor cornea position'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            grade4_num = grade4_num + 1
            if pred_0 == 0:
                grade4_grade1_num = grade4_grade1_num + 1
                list_grade4_grade1.append(img)
                grade4_1.append(prob_list)
                file_new_1 = desdatadir + '/Image of poor cornea position_Defocused image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                grade4_grade2_num = grade4_grade2_num + 1
                list_grade4_grade2.append(img)
                grade4_2.append(prob_list)
                file_new_1 = desdatadir + '/Image of poor cornea position_High-quality image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                grade4_grade3_num = grade4_grade3_num + 1
                list_grade4_grade3.append(img)
                grade4_3.append(prob_list)
                file_new_1 = desdatadir + '/Image of poor cornea position_Image of not fully opened eye' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                grade4_grade4_num = grade4_grade4_num + 1
                grade4_4.append(prob_list)
                file_new_1 = desdatadir + '/Image of poor cornea position_Image of poor cornea position' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                grade4_grade5_num = grade4_grade5_num + 1
                list_grade4_grade5.append(img)
                grade4_5.append(prob_list)
                file_new_1 = desdatadir + '/Image of poor cornea position_Overexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                grade4_grade6_num = grade4_grade6_num + 1
                list_grade4_grade6.append(img)
                grade4_6.append(prob_list)
                file_new_1 = desdatadir + '/Image of poor cornea position_Underexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)

        #     print(grade4_grade1_num, grade4_grade2_num, grade4_grade3_num, grade4_grade4_num)

        grade5_grade1_num = 0
        grade5_grade3_num = 0
        grade5_grade2_num = 0
        grade5_grade4_num = 0
        grade5_grade5_num = 0
        grade5_grade6_num = 0

        grade5_num = 0
        list_grade5_grade1 = [grade5_grade1]
        list_grade5_grade3 = [grade5_grade3]
        list_grade5_grade2 = [grade5_grade2]
        list_grade5_grade4 = [grade5_grade4]
        list_grade5_grade6 = [grade5_grade6]

        grade5_1 = [grade5_grade1]
        grade5_2 = [grade5_grade2]
        grade5_3 = [grade5_grade3]
        grade5_4 = [grade5_grade4]
        grade5_5 = [grade5_grade5]
        grade5_6 = [grade5_grade6]

        root = testdatadir + '/Overexposed images'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            grade5_num = grade5_num + 1
            if pred_0 == 0:
                grade5_grade1_num = grade5_grade1_num + 1
                list_grade5_grade1.append(img)
                grade5_1.append(prob_list)
                file_new_1 = desdatadir + '/Overexposed images_Defocused image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                grade5_grade2_num = grade5_grade2_num + 1
                list_grade5_grade2.append(img)
                grade5_2.append(prob_list)
                file_new_1 = desdatadir + '/Overexposed images_High-quality image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                grade5_grade3_num = grade5_grade3_num + 1
                list_grade5_grade3.append(img)
                grade5_3.append(prob_list)
                file_new_1 = desdatadir + '/Overexposed images_Image of not fully opened eye' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                grade5_grade4_num = grade5_grade4_num + 1
                list_grade5_grade4.append(img)
                grade5_4.append(prob_list)
                file_new_1 = desdatadir + '/Overexposed images_Image of poor cornea position' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                grade5_grade5_num = grade5_grade5_num + 1
                grade5_5.append(prob_list)
                file_new_1 = desdatadir + '/Overexposed images_Overexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                grade5_grade6_num = grade5_grade6_num + 1
                list_grade5_grade6.append(img)
                grade5_6.append(prob_list)
                file_new_1 = desdatadir + '/Overexposed images_Underexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)

        #     print(grade4_grade1_num, grade4_grade2_num, grade4_grade3_num, grade4_grade4_num)

        grade6_grade1_num = 0
        grade6_grade3_num = 0
        grade6_grade2_num = 0
        grade6_grade4_num = 0
        grade6_grade5_num = 0
        grade6_grade6_num = 0

        grade6_num = 0
        list_grade6_grade1 = [grade6_grade1]
        list_grade6_grade3 = [grade6_grade3]
        list_grade6_grade2 = [grade6_grade2]
        list_grade6_grade4 = [grade6_grade4]
        list_grade6_grade5 = [grade6_grade5]

        grade6_1 = [grade6_grade1]
        grade6_2 = [grade6_grade2]
        grade6_3 = [grade6_grade3]
        grade6_4 = [grade6_grade4]
        grade6_5 = [grade6_grade5]
        grade6_6 = [grade6_grade6]

        root = testdatadir + '/Underexposed images'
        img_list = [f for f in os.listdir(root) if f.endswith(str_end)]
        print("img_list_len: ", len(img_list))
        for img in img_list:
            img_PIL = Image.open(os.path.join(root, img)).convert('RGB')
            img_PIL_Tensor = val_transforms(img_PIL)
            img_PIL_Tensor = img_PIL_Tensor.unsqueeze(0)
            img_PIL_Tensor = img_PIL_Tensor.to(device)

            ouput = model(img_PIL_Tensor)
            prob = F.softmax(ouput, dim=1)
            prob_list = prob.cpu().numpy()
            prob_list = prob_list.tolist()[0]
            pred = torch.argmax(prob, dim=1)
            pred = pred.cpu().numpy()
            pred_0 = pred[0]

            grade6_num = grade6_num + 1
            if pred_0 == 0:
                grade6_grade1_num = grade6_grade1_num + 1
                list_grade6_grade1.append(img)
                grade6_1.append(prob_list)
                file_new_1 = desdatadir + '/Underexposed images_Defocused image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 1:
                grade6_grade2_num = grade6_grade2_num + 1
                list_grade6_grade2.append(img)
                grade6_2.append(prob_list)
                file_new_1 = desdatadir + '/Underexposed images_High-quality image' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 2:
                grade6_grade3_num = grade6_grade3_num + 1
                list_grade6_grade3.append(img)
                grade6_3.append(prob_list)
                file_new_1 = desdatadir + '/Underexposed images_Image of not fully opened eye' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 3:
                grade6_grade4_num = grade6_grade4_num + 1
                list_grade6_grade4.append(img)
                grade6_4.append(prob_list)
                file_new_1 = desdatadir + '/Underexposed images_Image of poor cornea position' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            elif pred_0 == 4:
                grade6_grade5_num = grade6_grade5_num + 1
                list_grade6_grade5.append(img)
                grade6_5.append(prob_list)
                file_new_1 = desdatadir + '/Underexposed images_Overexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)
            else:
                grade6_grade6_num = grade6_grade6_num + 1
                grade6_6.append(prob_list)
                file_new_1 = desdatadir + '/Underexposed images_Underexposed images' + '/' + img
                copyfile(os.path.join(root, img), file_new_1)

    #     print(grade4_grade1_num, grade4_grade2_num, grade4_grade3_num, grade4_grade4_num)
    confusion_matrix = [[grade1_grade1_num, grade1_grade2_num, grade1_grade3_num, grade1_grade4_num, grade1_grade5_num,
                         grade1_grade6_num],
                        [grade2_grade1_num, grade2_grade2_num, grade2_grade3_num, grade2_grade4_num, grade2_grade5_num,
                         grade2_grade6_num],
                        [grade3_grade1_num, grade3_grade2_num, grade3_grade3_num, grade3_grade4_num, grade3_grade5_num,
                         grade3_grade6_num],
                        [grade4_grade1_num, grade4_grade2_num, grade4_grade3_num, grade4_grade4_num, grade4_grade5_num,
                         grade4_grade6_num],
                        [grade5_grade1_num, grade5_grade2_num, grade5_grade3_num, grade5_grade4_num, grade5_grade5_num,
                         grade5_grade6_num],
                        [grade6_grade1_num, grade6_grade2_num, grade6_grade3_num, grade6_grade4_num, grade6_grade5_num,
                         grade6_grade6_num]]
    print('confusion_matrix:')
    print(confusion_matrix)

    result_confusion_file = args.arch + '_1.txt'
    result_pro_file = args.arch + '_2.txt'
    result_value_bin = args.arch + '_3.txt'

    with open(desdatadir + '/' + result_confusion_file, "w") as file_object:
        for i in confusion_matrix:
            file_object.writelines(str(i) + '\n')
        file_object.writelines('ERROR_images\n')
        for i in list_grade1_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade5:
            file_object.writelines(str(i) + '\n')
        for i in list_grade1_grade6:
            file_object.writelines(str(i) + '\n')

        for i in list_grade2_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade5:
            file_object.writelines(str(i) + '\n')
        for i in list_grade2_grade6:
            file_object.writelines(str(i) + '\n')

        for i in list_grade3_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade5:
            file_object.writelines(str(i) + '\n')
        for i in list_grade3_grade6:
            file_object.writelines(str(i) + '\n')

        for i in list_grade4_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade4_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade4_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade4_grade5:
            file_object.writelines(str(i) + '\n')
        for i in list_grade4_grade6:
            file_object.writelines(str(i) + '\n')

        for i in list_grade5_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade5_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade5_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade5_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade5_grade6:
            file_object.writelines(str(i) + '\n')

        for i in list_grade6_grade1:
            file_object.writelines(str(i) + '\n')
        for i in list_grade6_grade2:
            file_object.writelines(str(i) + '\n')
        for i in list_grade6_grade3:
            file_object.writelines(str(i) + '\n')
        for i in list_grade6_grade4:
            file_object.writelines(str(i) + '\n')
        for i in list_grade6_grade5:
            file_object.writelines(str(i) + '\n')

        file_object.close()

    with open(desdatadir + '/' + result_pro_file, "w") as file_object:
        for i in grade1_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade1_6:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade2_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade2_6:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade3_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade3_6:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade4_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade4_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade4_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade4_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade4_6:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade5_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade5_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade5_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade5_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade5_6:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        for i in grade6_1:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade6_2:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade6_3:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade6_4:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')
        for i in grade6_5:
            file_object.writelines(str(i) + '\t')
            file_object.writelines('\n')

        file_object.close()

    with open(desdatadir + '/' + result_value_bin, "wb") as file_object:
        pickle.dump(confusion_matrix, file_object)  # 顺序存入变量
        pickle.dump(grade1_1, file_object)
        pickle.dump(grade1_2, file_object)
        pickle.dump(grade1_3, file_object)
        pickle.dump(grade1_4, file_object)
        pickle.dump(grade1_5, file_object)
        pickle.dump(grade1_6, file_object)

        pickle.dump(grade2_1, file_object)
        pickle.dump(grade2_2, file_object)
        pickle.dump(grade2_3, file_object)
        pickle.dump(grade2_4, file_object)
        pickle.dump(grade2_5, file_object)
        pickle.dump(grade2_6, file_object)

        pickle.dump(grade3_1, file_object)
        pickle.dump(grade3_2, file_object)
        pickle.dump(grade3_3, file_object)
        pickle.dump(grade3_4, file_object)
        pickle.dump(grade3_5, file_object)
        pickle.dump(grade3_6, file_object)

        pickle.dump(grade4_1, file_object)
        pickle.dump(grade4_2, file_object)
        pickle.dump(grade4_3, file_object)
        pickle.dump(grade4_4, file_object)
        pickle.dump(grade4_5, file_object)
        pickle.dump(grade4_6, file_object)

        pickle.dump(grade5_1, file_object)
        pickle.dump(grade5_2, file_object)
        pickle.dump(grade5_3, file_object)
        pickle.dump(grade5_4, file_object)
        pickle.dump(grade5_5, file_object)
        pickle.dump(grade5_6, file_object)

        pickle.dump(grade6_1, file_object)
        pickle.dump(grade6_2, file_object)
        pickle.dump(grade6_3, file_object)
        pickle.dump(grade6_4, file_object)
        pickle.dump(grade6_5, file_object)
        pickle.dump(grade6_6, file_object)

        file_object.close()

#
if __name__ == '__main__':
    main()


