import os
import argparse

import torch

import mixed_precision
from stats import AverageMeterSet
from datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from model import Model
from checkpoint import Checkpointer
from utils_amdim import test_model, test_model_2

from cifar10 import Cifar10Dataset

import multiprocessing

parser = argparse.ArgumentParser(description='Infomax Representations - Testing Script')
# parameters for general training stuff
parser.add_argument('--checkpoint_path', type=str,
                    help='path from which to load checkpoint', default = './runs/amdim_cpt_classifier.pth' )
parser.add_argument('--dataset', type=str, default='C10')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size (default: 200)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enables automatic mixed precision')
parser.add_argument('--input_dir', type=str, default='/mnt/imagenet',
                    help="Input directory for the dataset. Not needed For C10,"
                    " C100 or STL10 as the data will be automatically downloaded.")
parser.add_argument('--run_name', type=str, default='default_run',
                    help='name to use for the tensorbaord summary for this run')
args = parser.parse_args()


def test(model, test_loader, device, stats):
    test_model_2(model, test_loader, device, stats)


def main():


    #for debugging with vscode
    # multiprocessing.set_start_method('spawn')
    
    # enable mixed-precision computation if desired
    if args.amp:
        mixed_precision.enable_mixed_precision()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get the dataset
    dataset = get_dataset(args.dataset)

    _, test_loader, _ = build_dataset(dataset=dataset,
                            batch_size=args.batch_size,
                            input_dir=args.input_dir)

    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpointer = Checkpointer()
   
    model = checkpointer.restore_model_from_checkpoint(args.checkpoint_path)
    model = model.to(torch_device)
    model, _ = mixed_precision.initialize(model, None)

    test_stats = AverageMeterSet()
    test(model, test_loader, torch_device, test_stats)



if __name__ == "__main__":
    print(args)
    main()