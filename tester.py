###### Importing libraries
import time
import tqdm
import torch
import argparse
import numpy as np
from model import *
from epoch import *
from icgd import icgdLoss
from parser import parse
from dataloader import dataLoader

###### Argument parser
args = parse()

###### Dataset
train_dataLoader, test_dataLoader = dataLoader(args)

if(args.dataset == 'soli'):
    input_dim = 32
    embed_dim = 32
    patch_size = (5,5,5)
    T = 40
    H = 32
    W = 32
    C = 4
    d_model = 32
    dff = 128
    num_heads = 16
    rate = 0.3
    G = 11
    I = 10

###### Model
if(args.model == 'res3dViViT'):
    model = res3dViViT(input_dim,
                       embed_dim,
                       patch_size,
                       T,
                       H,
                       W,
                       C,
                       d_model,
                       num_heads,
                       dff,
                       rate,
                       G,
                       I)
    
###### Evaluation

##### Defining essentials
if(args.multi_gpu == False):
    device = torch.device(args.device)
