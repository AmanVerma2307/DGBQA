###### Importing libraries
import random
import torch
import numpy as np
from model import *
from epoch import *
from icgd import icgdLoss, icgdLossIterator
from parser import parse
from dataloader import dataLoader
from utils.summary import *

seed = 10

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

###### Argument parser
args = parse()
wandb.init(project='dgbqaCodebase',name=args.exp_name)

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
def init_weights(m): 
    if (isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv3d)):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

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
    
model.apply(init_weights)
    
###### Training and validation

##### Defining essentials
if(args.multi_gpu == False):
    device = torch.device(args.device)
criterion_hgr = torch.nn.CrossEntropyLoss()
criterion_id = torch.nn.CrossEntropyLoss()
criterion_icgd = icgdLossIterator(G,I)
criterion_icgd.requires_grad_ = False
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-7)

#print_model_summary(model, (C,T,H,W))
total_params = sum(p.numel() for p in model.parameters())
print('Total parameters: '+str(total_params))

model = model.to(device)
wandb.watch(model,criterion_id,log="all",log_freq=1)

##### Training and validation loop
train_metrics, val_metrics = train_val(train_dataLoader,
                                       test_dataLoader,
                                       model,
                                       optimizer,
                                       criterion_hgr,
                                       criterion_id,
                                       criterion_icgd,
                                       args)

##### Saving
np.savez_compressed('./model history/'+args.exp_name+'_trainMetrics.npz',np.array(train_metrics.cpu()))
np.savez_compressed('./model history/'+args.exp_name+'_valMetrics.npz',np.array(val_metrics.cpu()))
 