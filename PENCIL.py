import argparse
import os
import os.path
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

from utils.cifar10_subset import Cifar10Dataset, TransformsC10
from utils.mypreactresnet import resnet32


# from utils.datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from utils_amdim import weight_init, test_model, flatten
import mixed_precision
from model import Model
from task_classifiers import train_classifiers
from checkpoint import Checkpointer

# for remote debugging
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

#PENCIL

noise_options = ['clean','symmetric','asymmetric']
model_names = ['preact_resnet32','resnet34','resnet50']
# dataset_names = ['cifar10','cifar100','CUB200_2011','Clothing1M']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='preact_resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--lr2', '--learning-rate2', default=0.2, type=float,
                    metavar='H-P', help='initial learning rate of stage3')
parser.add_argument('--alpha', default=0.1, type=float,
                    metavar='H-P', help='the coefficient of Compatibility Loss')
parser.add_argument('--beta', default=0.8, type=float,
                    metavar='H-P', help='the coefficient of Entropy Loss')
parser.add_argument('--lambda1', default=200, type=int,
                    metavar='H-P', help='the value of lambda')
parser.add_argument('--stage1', default=5, type=int,
                    metavar='H-P', help='number of epochs utill stage1')
parser.add_argument('--stage2', default=135, type=int,
                    metavar='H-P', help='number of epochs utill stage2')
parser.add_argument('--epochs', default=255, type=int, metavar='H-P',
                    help='number of total epochs to run')
parser.add_argument('--datanum', default=45000, type=int,
                    metavar='H-P', help='number of train dataset samples')
parser.add_argument('--classnum', default=10, type=int,
                    metavar='H-P', help='number of train dataset classes')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False,dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', dest='gpu', default='0', type=str,
                    help='select gpu')
parser.add_argument('--dir', dest='dir', default="/content/drive/My Drive/torchTest/checkpoints/", 

parser.add_argument('--noise', default='clean', type=str, choices= noise_options,  help='Select noise for labels')
parser.add_argument('--noise_rate', default=0.0, type=float, help='Select noise rate for labels') 
parser.add_argument('--dataset', default='cifar10', type=str, help='define dataset(used for lambda1 parameter)') 
parser.add_argument('--run-without-validation', default=False, action='store_true', help='Use validation set') 
parser.add_argument('--lambda2', default=200, type=int, metavar='H-P', help='temp var to keep value of lambda1')
parser.add_argument('--lambda_epoch', default=150, type=int, metavar='H-P', help='the value of lambda')

#AMDIM parameters
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--classifiers', action='store_true', default=False)
parser.add_argument('--use_bn', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='./runs')
parser.add_argument('--input_dir', type=str, default='/mnt/imagenet') 
parser.add_argument('--cpt_load_path', type=str, default='./amdim_cpt.pth')
parser.add_argument('--cpt_name', type=str, default='amdim_cpt_classifier.pth')


best_prec1 = 0

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

    
def create_sets(dataset):

    global args

    if(dataset == 'cifar10'):
        trainset = Cifar10Dataset(root='./', train=0, transform=Cifar10Dataset.transform1,noise = args.noise, rate = args.noise_rate)
        testset = Cifar10Dataset(root='./', train=1,transform=Cifar10Dataset.transform2, noise = args.noise, rate = args.noise_rate)
        valset = Cifar10Dataset(root='./', train=2, transform=Cifar10Dataset.transform2, noise = args.noise, rate = args.noise_rate)

    elif(dataset == 'cifar100'):
        trainset = Cifar100Dataset(root='./', train=0, transform=Cifar100Dataset.transform1, noise = args.noise , rate = args.noise_rate)
        testset = Cifar100Dataset(root='./', train=1,transform=Cifar100Dataset.transform2, noise = args.noise, rate = args.noise_rate)
        valset = Cifar100Dataset(root='./', train=2, transform=Cifar100Dataset.transform2, noise = args.noise, rate = args.noise_rate)

    elif(dataset == 'CUB200_2011'):
        trainset = Cub200_2011Dataset(root='./', train=0, transform=Cub200_2011Dataset.transform1, noise = args.noise , rate = args.noise_rate)
        testset = Cub200_2011Dataset(root='./', train=1,transform=Cub200_2011Dataset.transform2)
        valset = Cub200_2011Dataset(root='./', train=2, transform=Cub200_2011Dataset.transform2)

    elif(dataset == 'clothing1M'):
        trainset = clothing1MDataset(root='./', train=0, transform=clothing1MDataset.transform1, noise = args.noise , rate = args.noise_rate)
        testset = clothing1MDataset(root='./', train=1,transform=clothing1MDataset.transform2)
        valset = clothing1MDataset(root='./', train=2, transform=clothing1MDataset.transform2)
        
    else:
        print('not supported datset')
        
    return (trainset,testset,valset)

def build_dataset(dataset, batch_size, input_dir=None, labeled_only=False, noise = 'clean', rate = 0.0):

    if dataset == 'cifar10':
        num_classes = 10
        train_transform = TransformsC10()
        test_transform = train_transform.test_transform
        dataset_path = os.path.join(os.path.expanduser('~'),'data/cifar10/')
        train_dataset = Cifar10Dataset(root=dataset_path,
                                         train=True,
                                         transform=train_transform,
                                         download=False,noise = noise, rate = rate)
        test_dataset = Cifar10Dataset(root=dataset_path,
                                        train=False,
                                        transform=test_transform,
                                        download=False,noise = noise, rate = rate)

    train_loader = \
        torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=False,
                                    num_workers=1)#16
    test_loader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=False,
                                    num_workers=1)#16

    return train_loader, test_loader, len(train_dataset)

def main():

    global args, best_prec1 , device

    ################################################################################################

    args = parser.parse_args(sys.argv[1:])
    #change default parameters
    # args.noise = 'clean'
    # args.noise_rate = 0.0 
    args.arch = "amdim"
    # args.dataset = "cifar10"
    args.datanum = 45000 
    # args.classnum = 10
    # args.run_without_validation = False
 
    #args.dir = os.path.join(os.getcwd(), 'checkpoints')
    args.dir = os.path.join(os.getcwd(), args.noise+str(args.noise_rate)+'checkpoints')
    #args.dir = os.path.join(os.getcwd(), args.dir)
    # args.output_dir = args.dir
    # args.amp = True
    if args.amp:
        mixed_precision.enable_mixed_precision()

    # set the RNG seeds (probably more hidden elsewhere...)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    y_file = os.path.join(args.dir,"y.npy")
    np.random.seed(2020)
    args.lambda2 = args.lambda1

    if os.path.exists(args.dir):
        print(args.dir+' folder exists already')
    else:
      os.makedirs(args.dir)
    
    if os.path.exists(os.path.join(args.dir,'record')):
        print(os.path.join(args.dir,'record')+' folder exists already')
    else:
      os.makedirs(os.path.join(args.dir,'record'))

    with open(os.path.join(args.dir,'commandline_args.txt'), 'a+') as f:
        print(vars(args), file=f)
    ################################################################################################

    if(args.arch == "preact_resnet32"):
        model = resnet32()
    elif(args.arch == "resnet34"):
        model =  torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    elif(args.arch == "resnet50"):
        model =  torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    elif(args.arch =="amdim"):
        checkpointer = Checkpointer(args.dir, args.cpt_name)
        if args.cpt_load_path:
            model = checkpointer.restore_model_from_checkpoint(args.cpt_load_path, training_classifier=args.classifiers)
        else:
            print("Not supported model !")

    if(args.dataset == "cifar100"):
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            model.maxpool = nn.Identity()
    
    if(args.dataset != "cifar10"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.classnum)


    # #data parallelism when >1 GPUs are available
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    checkpoint_dir = os.path.join(args.dir , "checkpoint.pth.tar")
    modelbest_dir = os.path.join(args.dir , "model_best.pth.tar")

    statistics_hist = {'train_loss_hist':[],'train_acc_hist':[], 'val_loss_hist':[], 'val_acc_hist':[],'test_loss_hist':[],'test_acc_hist':[], 'current_epoch':0}

    # optionally resume from a checkpoint
    if os.path.isfile(checkpoint_dir):
        print("=> loading checkpoint '{}'".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        args.start_epoch = checkpoint['epoch']
        # args.start_epoch = 0
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        statistics_hist = checkpoint['current_statistics_hist']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_dir, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))

    cudnn.benchmark = True

    #create datasets
    trainloader, testloader , datanum = build_dataset(dataset='cifar10', batch_size=args.batch_size, input_dir=args.input_dir, labeled_only=args.classifiers,noise = args.noise,rate = args.noise_rate)


    args.datanum = datanum

    if args.evaluate:
        validate(testloader, model, criterion)
        return


    # retrain the evaluation classifiers using the trained feature encoder
    for mod in model.class_modules:
        # reset params in the evaluation classifiers
        mod.apply(weight_init)
    mods_to_opt = [m for m in model.class_modules]

    optimizer = torch.optim.SGD([{'params': mod.parameters(), 'lr': args.lr} for mod in mods_to_opt],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # If mixed precision is on, will add the necessary hooks into the model and
    # optimizer for half precision conversions
    model, optimizer = mixed_precision.initialize(model, optimizer)

    for param in model.encoder.parameters():
        param.requires_grad = False

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # load y_tilde
        if os.path.isfile(y_file):
            y = np.load(y_file)
        else:
            y = []

        current_epoch_train_acc, training_loss = train(trainloader, model, criterion, optimizer, epoch, y)
        ######################################################################
        #save train statistics history
        statistics_hist['train_loss_hist'].append(training_loss)
        statistics_hist['train_acc_hist'].append(current_epoch_train_acc.item())
        ######################################################################
        # evaluate on validation set
        if(args.run_without_validation == False):
            prec1, valid_loss  = validate(valloader, model, criterion)
            ######################################################################
            statistics_hist['val_loss_hist'].append(valid_loss)
            statistics_hist['val_acc_hist'].append(prec1.item())
        ######################################################################
        print('after this is testloader:')
        current_epoch_test_acc, test_loss = validate(testloader, model, criterion)
        ######################################################################
        statistics_hist['test_loss_hist'].append(test_loss)
        statistics_hist['test_acc_hist'].append(current_epoch_test_acc.item())
        if(args.run_without_validation == True):
            prec1 = current_epoch_test_acc
        ######################################################################

        print('end of testloader')
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        is_best = False 
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            'current_statistics_hist': statistics_hist,
        }, is_best,filename=checkpoint_dir,modelbest=modelbest_dir)
        ##############################################################################
        # if ((epoch + 1) == args.stage1):
        #     #save end of stage 1 model
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #         'optimizer' : optimizer.state_dict(),
        #         'current_statistics_hist': statistics_hist,
        #     }, False,filename=os.path.join(args.dir,args.dataset+args.noise+str(args.noise_rate)+"end_st1.pth.tar"),modelbest=modelbest_dir)
        ##############################################################################
        ##############################################################################
        # if ((epoch + 1) == args.stage2):
        #     #save end of stage 1 model
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #         'optimizer' : optimizer.state_dict(),
        #         'current_statistics_hist': statistics_hist,
        #     }, False,filename=os.path.join(args.dir,args.dataset+args.noise+str(args.noise_rate)+"end_st2.pth.tar"),modelbest=modelbest_dir)
        ##############################################################################
    
    #save only stats
    only_stats = torch.load(os.path.join(args.dir,"checkpoint.pth.tar"),map_location=torch.device('cpu'))
    only_stats = only_stats['current_statistics_hist']
    torch.save(only_stats, os.path.join(args.dir , "checkpoint.pth.tar"))
    
    return statistics_hist


def train(train_loader, model, criterion, optimizer, epoch, y):

    global device

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    

    # switch to train mode
    model.train()

    end = time.time()

    # new y is y_tilde after updating
    new_y = np.zeros([args.datanum,args.classnum])

    for i, ((input,input2), target, index) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        index = index.numpy()
        input = input.to(device)
        target = target.to(device)
   
        # compute output
        # output = model(input)
        res_dict = model(x1=input, x2=input, class_only=True)
        _ , output = res_dict['class']#lgt_glb_mlp

        logsoftmax = nn.LogSoftmax(dim=1).to(device)#cuda()
        softmax = nn.Softmax(dim=1).to(device)#cuda()
        if epoch < args.stage1:
            # lc is classification loss
            lc = criterion(output, target)
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(target.size(0), args.classnum,device = device).scatter_(1, target.view(-1, 1), 10.0)
            onehot = onehot.to(torch.device("cpu")).numpy()
            new_y[index, :] = onehot
        else:
            yy = y
            yy = yy[index,:] 
            yy = torch.tensor(yy,requires_grad = True, device = device)#torch.FloatTensor(yy)

            # obtain label distributions (y_hat)
            last_y_var = softmax(yy)
            lc = torch.mean(softmax(output)*(logsoftmax(output)-torch.log((last_y_var))))
            # lo is compatibility loss
            lo = criterion(last_y_var, target)
        # le is entropy loss
        le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

        if epoch < args.stage1:
            loss = lc
        elif epoch < args.stage2:
            loss = lc + args.alpha * lo + args.beta * le
        else:
            loss = lc

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #loss.backward()
        mixed_precision.backward(loss, optimizer)  # special mixed precision stuff
        optimizer.step()
        if epoch >= args.stage1 and epoch < args.stage2:
            lambda1 = args.lambda1
            # update y_tilde by back-propagation
            yy.data.sub_(lambda1*yy.grad.data)
            new_y[index,:] = yy.data.cpu().numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    if(args.dataset=='cifar10'):
        #linearly decrease lambda1
        if(args.noise == 'asymmetric' and args.noise_rate >= 0.4 ):
            if epoch >= args.stage1 and epoch < args.stage2:
                if epoch < args.lambda_epoch:
                    print('lambda1 is:')
                    print(args.lambda1)
                    args.lambda1 = args.lambda1 - (args.lambda2/(args.lambda_epoch-5))#(args.lambda2/(130-args.lambda_epoch))
                    if(args.lambda1 <0):
                        print("lamda1 is zero now")
                        args.lambda1 = 0
                else:
                    args.lambda1 = 0

    elif(args.dataset=='clothing1M'):
        if(epoch>4 and epoch<10):
            args.lambda1 = 3000
        elif(epoch>9 and epoch<15):
            args.lambda1 = 500
    
    if epoch < args.stage2:
        # save y_tilde
        y = new_y
        y_file = os.path.join(args.dir,"y.npy")#args.dir + "y.npy"
        np.save(y_file,y)
        y_record = os.path.join(args.dir,'record')+"/y_%03d.npy" % epoch #os.path.join(args.dir,"record/y_%03d.npy")#args.dir + "record/y_%03d.npy" % epoch
        np.save(y_record,y)
    
    ################################################################################################################
    #save y for stages
    if ((epoch + 1) == args.stage1):
        y_file = os.path.join(args.dir,args.dataset+args.noise+str(args.noise_rate)+"y_end_st1.npy")#os.path.join(args.dir,"y_stage_2.npy")
        np.save(y_file,y)
    ################################################################################################################
    ################################################################################################################
    #save y for stages
    if ((epoch + 1) == args.stage2):
        y_file = os.path.join(args.dir,args.dataset+args.noise+str(args.noise_rate)+"y_end_st2.npy")#os.path.join(args.dir,"y_stage_2.npy")
        np.save(y_file,y)
    ################################################################################################################

    return top1.avg, losses.avg

def validate(val_loader, model, criterion):

    global device

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target,index) in enumerate(val_loader):

        input = input.to(device)
        target = target.to(device)#cuda(async=True)

        # compute output
        # output = model(input)

        with torch.no_grad():
            res_dict = model(x1=input, x2=input, class_only=True)
            _, output = res_dict['class']

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg , losses.avg


def save_checkpoint(state, is_best, filename='', modelbest = ''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if epoch < args.stage2 :#<200 stage2
        lr = args.lr
    elif epoch < (args.epochs - args.stage2)/3 + args.stage2:#<240
        lr = args.lr2
    elif epoch < 2 * (args.epochs - args.stage2)/3 + args.stage2:#<280
        lr = args.lr2/10
    else:#eos 320
        lr = args.lr2/100

    if(args.dataset=='clothing1M'):
        if epoch < args.stage1 :#<100 stage2
            lr = args.lr
        elif epoch < args.stage2 and epoch >= args.stage1:
            lr = 8e-4
        elif epoch < 25 and epoch >= args.stage2:
            lr = args.lr2
        elif epoch >=19:#<140
            lr = args.lr2/10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _ , pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    statistics_hist = main()


    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(args.epochs),statistics_hist['train_loss_hist'], label='Training Loss')
    plt.plot(range(args.epochs),statistics_hist['test_loss_hist'],label='test Loss')

    # find position of lowest validation loss
    #minposs = valid_loss.index(min(valid_loss))+1 
    #plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.autoscale()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(args.dir, 'loss_plot.png'), bbox_inches='tight')
