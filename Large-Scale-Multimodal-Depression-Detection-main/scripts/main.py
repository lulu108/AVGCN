#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import logging
logging.getLogger("speechbrain").setLevel(logging.WARNING)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np
import random
import yaml
try:
    import wandb
except ImportError:
    wandb = None
import torch
import gc 
from tqdm import tqdm
import sys
sys.path.append('../')
from models import MultiModalDepDet
from datasets_process import get_dvlog_dataloader, get_lmvd_dataloader
from train_eval.utils import EarlyStopping, LOG_INFO, adjust_learning_rate
from train_eval.train_val import train_epoch, val
from train_eval.losses import CombinedLoss

# Seed 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)

CONFIG_PATH = "../configs/config.yaml"

def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model."
    )

    ## arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument(
        "-m", "--model", type=str,
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument('--optimizer', default='Adam', type=str, 
                        help='Adam or AdamW or SGD or RMSProp')
    parser.add_argument('--lr_scheduler', default='cos', type=str, 
                        help='cos or StepLR or Plateau')
    parser.add_argument('--amsgrad', default=0, type=int, 
                        help='Adam amsgrad')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[100, 200, 300, 400, 550, 700], type=float, nargs="+", metavar='LRSteps',
                        help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-g", "--gpu", type=str)
    parser.add_argument("-wdb", "--if_wandb", type=bool)
    parser.add_argument("-tqdm", "--tqdm_able", type=bool)
    parser.add_argument("-tr", "--train", type=bool, 
                        help='Whether you want to training or not!')
    parser.add_argument("--cross_infer", default=False, type=bool,
                        help="Exchange the dataset name and model")
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.add_argument('-n_h', '--num_heads', default=1, type=int, 
                        help='number of heads, in the paper 1 or 4')
    parser.add_argument('-fus', '--fusion', default='ia', type=str, 
                        help='fusion type: lt | it | ia | MT')
    parser.add_argument('--begin_epoch', default=1, type=int,
                        help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    
    parser.set_defaults(**config)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    # prepare the data
    if args.dataset=='dvlog-dataset':
        train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
        val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
        test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )

    elif args.dataset=='lmvd-dataset':
        train_loader = get_lmvd_dataloader(
            args.data_dir, "train", args.batch_size, args.train_gender
        )
        val_loader = get_lmvd_dataloader(
            args.data_dir, "valid", args.batch_size, args.test_gender
        )
        test_loader = get_lmvd_dataloader(
            args.data_dir, "test", args.batch_size, args.test_gender
        )

    if args.if_wandb and wandb is None:
        raise ImportError("wandb is not installed. Set --if_wandb False or install wandb.")
    if args.if_wandb:
        wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"
        wandb.init(
            project="Multi-Modal Depression Model", config=args, name=wandb_run_name,
        )
        args = wandb.config
    print(args)

    if args.cross_infer:
        # Automatically switch dataset
        if args.dataset == "dvlog-dataset":
            args.dataset = "lmvd-dataset"
        elif args.dataset == "lmvd-dataset":
            args.dataset = "dvlog-dataset"
    
    # Build Save Dir
    os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}", exist_ok=True)
    os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/samples", exist_ok=True)
    os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints", exist_ok=True)

    # construct the model
    if args.model == "DepMamba":
        try:
            from models import DepMamba
        except Exception as e:
            raise ImportError(
                "DepMamba requires optional dependencies (speechbrain, mamba-ssm). "
                "Install them or use --model MultiModalDepDet."
            ) from e
        if args.dataset=='lmvd-dataset':
            net = DepMamba(**args.mmmamba_lmvd)# mmmamba_lmvd mmmamba
        elif args.dataset=='dvlog-dataset':
            net = DepMamba(**args.mmmamba)# mmmamba_lmvd mmmamba
    elif args.model == "MultiModalDepDet":
        if args.dataset=='lmvd-dataset':
            net = MultiModalDepDet(**args.lmvd, fusion=args.fusion, num_heads=args.num_heads)
            # net = MultiModalDepDet(**args.dvlog, fusion=args.fusion, num_heads=args.num_heads)
        elif args.dataset=='dvlog-dataset':
            net = MultiModalDepDet(**args.dvlog, fusion=args.fusion, num_heads=args.num_heads)
            # net = MultiModalDepDet(**args.lmvd, fusion=args.fusion, num_heads=args.num_heads)
    else:
        raise NotImplementedError(f"The {args.model} method has not been implemented by this repo")
    
    ### model check
    if args.device[0] != 'cpu':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Final choice of computing device: {args.device}')
    net = net.to(args.device)
    if len(args.device) > 1:
        # net = torch.nn.DataParallel(net, device_ids=args.device)
        net = torch.nn.DataParallel(net, device_ids=None)

        pytorch_total_params = sum(p.numel() for p in net.parameters() if
                                p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        pytorch_total_params_ = sum(p.numel() for p in net.parameters())
        print("Total number of parameters: ", pytorch_total_params_)

    # set other training components
    loss_fn = CombinedLoss(lambda_reg=1e-5, 
                            focal_weight=0.5, 
                            l2_weight=0.5
                        )
    
    # Setting the optimizer for model training
    assert args.optimizer in ["RMSprop", "SGD", "Adam", "AdamW"]
    if args.optimizer=="SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),     # Parameters passed into the model
            lr=args.learning_rate,                                   # Learning rate
            momentum=args.momentum,                                  # Momentum factor (optional)
            dampening=args.dampening,
            weight_decay=args.weight_decay,                          # Weight decay (L2 regularization)
            nesterov=False)
    elif args.optimizer=="Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),  
            lr=args.learning_rate,  
            betas=(0.9,0.999), 
            weight_decay=args.weight_decay,  
            amsgrad=args.amsgrad
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),  
            lr=args.learning_rate, 
            betas=(0.9, 0.999),  
            weight_decay=args.weight_decay,  
            amsgrad=args.amsgrad  
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, net.parameters()),  
            lr=args.learning_rate, 
            alpha=0.99,  
            eps=1e-8,  
            weight_decay=args.weight_decay,  
            momentum=0.9,  
        )
    
    ## learning scheluder
    if args.lr_scheduler == "cos":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs // 5, eta_min=args.learning_rate / 20
        )
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.00001
        )
    elif args.lr_scheduler == "Plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience
        )
    else:
        lr_scheduler = None

    early_stopping = EarlyStopping(patience = 5, 
                                    verbose = True, 
                                    save_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
                    )
    
    best_val_acc = -1.0
    best_test_acc = -1.0

    # Check for fold-specific best model
    fold_best_model_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
    print(fold_best_model_path)
    if os.path.exists(fold_best_model_path) and not args.resume_path:  # Only check fold-specific if no resume_path
        print(f"Resuming from fold-specific checkpoint: {fold_best_model_path}")
        checkpoint = torch.load(fold_best_model_path, map_location=args.device, weights_only=False)
        # checkpoint = torch.load(fold_best_model_path, weights_only=False)
        assert args.model == checkpoint['arch']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded Model Best Val Acc: {best_val_acc}")
        args.begin_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        print(f"Ended Epoch: {checkpoint['epoch']} and Begining Epoch: {args.begin_epoch}")
        # print(checkpoint['state_dict'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path, weights_only=False)
        assert args.model == checkpoint['arch']
        best_val_acc = checkpoint['best_val_acc']
        print("Loaded Model Best Val Acc: {best_val_acc}")
        args.begin_epoch = checkpoint['epoch'] + 1
        print(f"Ended Epoch: {checkpoint['epoch']} and Begining Epoch: {args.begin_epoch}")
        net.load_state_dict(checkpoint['state_dict'])
    else:
        args.begin_epoch = 1
        best_val_acc = -1.0

    print(f"Training: {args.train}")
    if args.train:
        for epoch in range(args.begin_epoch, args.epochs+1):
            adjust_learning_rate(optimizer, epoch, args)

            train_results = train_epoch(
                net, train_loader, loss_fn, optimizer, lr_scheduler,
                args.device, epoch, args.epochs, args.tqdm_able
            )
            val_results = val(net, val_loader, loss_fn, args.device, args.tqdm_able, msg='additional metrics', cross_infer=args.cross_infer)

            # val_acc = (val_results["acc"] + val_results["precision"]+ val_results["recall"]+ val_results["f1"])/4.0
            val_acc = val_results["acc"] 
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                state = {
                    'epoch': epoch,
                    'arch': args.model,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'model': net
                }
                save_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
                torch.save(state, save_path)
                LOG_INFO(f"[{args.model}_{args.fusion}]: Model saved at epoch {state['epoch']}: {save_path}  | best_val_acc: {best_val_acc}", 'green')

            if early_stopping.early_stop: ## Early stop when it found increase loss or satuated
                LOG_INFO("Early stopping triggered", 'red')
                break

            if args.if_wandb:
                wandb.log({
                    "loss/train": train_results["loss"],
                    "acc/train": train_results["acc"],
                    "loss/val": val_results["loss"],
                    "acc/val": val_results["acc"],
                    "precision/val": val_results["precision"],
                    "recall/val": val_results["recall"],
                    "f1/val": val_results["f1"]
                })
        
    # print(f"resume_path: {args.resume_path}")

    # upload the best model to wandb website
    # load the best model for testing
    with torch.no_grad():
        if not args.resume_path:
            # print("not resume_path")
            best_state_path = f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt"
            LOG_INFO(f"best_state_path: {best_state_path}")
            checkpoint = torch.load(best_state_path, map_location=args.device, weights_only=False)
            net.load_state_dict(
                checkpoint['state_dict']
            )

        net.eval()
        test_results = val(net, test_loader, loss_fn, args.device,args.tqdm_able, msg='additional metrics', cross_infer=args.cross_infer)
        LOG_INFO(f"[{args.dataset}_{args.model}_{args.fusion}] Test results:")
        LOG_INFO(f"Test Result: {test_results}", "magenta")
        color_map = {
            "loss": "red",
            "acc": "cyan",
            "precision": "magenta",
            "recall": "yellow",
            "f1": "green",
            "weighted_accuracy": "blue",
            "unweighted_accuracy": "light_blue",
            "weighted_precision": "light_magenta",
            "unweighted_precision": "light_yellow",
            "weighted_recall": "light_cyan",
            "unweighted_recall": "light_green",
            "weighted_f1": "white",
            "unweighted_f1": "light_grey",
        }
        for key, value in test_results.items():
            color = color_map.get(key, 'blue')  # Default to blue if key not found
            LOG_INFO(f"{key}: {value:.4f}", color) # Format float values

        with open(f'../results/{args.dataset}_{args.model}_{args.fusion}.txt','w') as f:    
            test_result_str = f'Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]},\
                    Avg:{(test_results["acc"] + test_results["precision"]+ test_results["recall"]+ test_results["f1"])/4.0},\
                    WA:{test_results["weighted_accuracy"]}, UA:{test_results["unweighted_accuracy"]},\
                            WP:{test_results["weighted_precision"]}, UP:{test_results["unweighted_precision"]},\
                            WR:{test_results["weighted_recall"]}, UR:{test_results["unweighted_recall"]},\
                                WF:{test_results["weighted_f1"]}, UF:{test_results["unweighted_f1"]}'
            f.write(test_result_str)         

    if args.if_wandb:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(f"{args.save_dir}/{args.dataset}_{args.model}_{args.fusion}/checkpoints/best_model.pt")

        wandb.run.summary["acc/best_val_acc"] = best_val_acc
        wandb.log_artifact(artifact)
        wandb.run.summary["acc/test_acc"] = test_results["acc"]
        wandb.run.summary["loss/test_loss"] = test_results["loss"]
        wandb.run.summary["precision/test_precision"] = test_results["precision"]
        wandb.run.summary["recall/test_recall"] = test_results["recall"]
        wandb.run.summary["f1/test_f1"] = test_results["f1"]

        wandb.finish()

    del net 
    del optimizer
    del lr_scheduler
    del early_stopping
    del loss_fn 

    del train_loader
    del val_loader
    del test_loader

    gc.collect
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    

if __name__ == '__main__':
    main()