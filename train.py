import os
import argparse

import torch

import mixed_precision
from stats import StatTracker
from datasets import Dataset, build_dataset, get_dataset, get_encoder_size
from model import Model
from checkpoint import Checkpointer
from task_self_supervised import train_self_supervised
from task_classifiers import train_classifiers

from cifar10 import Cifar10Dataset

import multiprocessing

parser = argparse.ArgumentParser(description='Infomax Representations - Training Script')
# parameters for general training stuff

# parameters for model and training objective
parser.add_argument('--classifiers', action='store_true', default=False,
                    help="Wether to run self-supervised encoder or"
                    "classifier training task")
parser.add_argument('--ndf', type=int, default=256,
                    help='feature width for encoder')
parser.add_argument('--n_rkhs', type=int, default=1024,
                    help='number of dimensions in fake RKHS embeddings')
parser.add_argument('--tclip', type=float, default=20.0,
                    help='soft clipping range for NCE scores')
parser.add_argument('--n_depth', type=int, default=10)
parser.add_argument('--use_bn', type=int, default=0)

# parameters for output, logging, checkpointing, etc
parser.add_argument('--output_dir', type=str, default='./runs',
                    help='directory where tensorboard events and checkpoints will be stored')
parser.add_argument('--input_dir', type=str, default='/mnt/imagenet',
                    help="Input directory for the dataset. Not needed For C10,"
                    " C100 or STL10 as the data will be automatically downloaded.")
# to path xrisimopoite mono gia to load, meta mporo na allakso to onoma pou kanei save to mdelo
# alla den tha douleuei to resume xoris na allakso ksana to path me to xeri(den to ilopoiisa)
parser.add_argument('--cpt_load_path', type=str, default='./runs/amdim_cpt.pth',#path to load pretrained model
                    help='path from which to load checkpoint (if available)')
# mono gia tous classifiers allazo to onoma pou sozete to model, gia unsupervised pat kai name einai idia
# afou treksei mia fora o classifier kai sosei to modelo, meta an stamatisei prepei na allaksi kai to path
# auto den to exo ilopoiisi
parser.add_argument('--cpt_name', type=str, default='amdim_cpt_classifier.pth',
                    help='name to use for storing checkpoints during training')
parser.add_argument('--run_name', type=str, default='default_run',
                    help='name to use for the tensorbaord summary for this run')
# ...
args = parser.parse_args()

# gia otan to trexo apo vscode xoris script
# args.classifiers = True
# args.amp = True

def main():



#########################################################################################################


    # get a helper object for tensorboard logging
    log_dir = os.path.join(args.output_dir, args.run_name)
    stat_tracker = StatTracker(log_dir=log_dir)


    # select which type of training to do
    task = train_classifiers if args.classifiers else train_self_supervised
    task(model, args.learning_rate, dataset, train_loader,
         test_loader, stat_tracker, checkpointer, args.output_dir, torch_device)


# if __name__ == "__main__":
#     print(args)
#     main()
