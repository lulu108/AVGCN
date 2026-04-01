#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from termcolor import colored
except ImportError:
    def colored(msg, *_args, **_kwargs):
        # Fallback when termcolor is unavailable.
        return str(msg)

def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG :", 'green') + colored(msg, mcolor))

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=15, delta=0.0, verbose=False, save_path=None, mode='min'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        assert mode in ['min', 'max'], "mode must be 'min' for loss or 'max' for accuracy"
        self.mode = mode

    def __call__(self, val_metric, model):
        if self.best_score is None:
            self.best_score = val_metric
            self.save_checkpoint(val_metric, model)
        elif (self.mode == 'min' and val_metric > self.best_score + self.delta) or \
             (self.mode == 'max' and val_metric < self.best_score - self.delta):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"{'Loss' if self.mode=='min' else 'Accuracy'} improved to {val_metric:.6f}. Saving model...")
        self.best_model = model.state_dict()

    def load_best_model(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)

def collate_fn(batch):
    # Assuming x is the data and y is the label
    data, labels = zip(*batch)
    
    # Determine max length in batch
    max_length = max(d.shape[0] for d in data)
    
    # Pad data efficiently
    padded_data = np.array([
        np.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant')
        for d in data
    ])  # Now it's a single NumPy array

    # Convert to PyTorch tensors
    return torch.from_numpy(padded_data).float(), torch.tensor(labels, dtype=torch.long)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr_new = args.learning_rate * (0.1 ** (sum(epoch >= np.array(args.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        # param_group['lr'] = opt.learning_rate


# -------------------------------
# Define the ShapeAdapter module
# -------------------------------
class ShapeAdapter(nn.Module):
    def __init__(self, lmvd_dim=264, dvlog_dim=161, lmvd_time=2086, dvlog_time=1443):
        super().__init__()
        self.lmvd_dim = lmvd_dim
        self.dvlog_dim = dvlog_dim
        self.lmvd_time = lmvd_time
        self.dvlog_time = dvlog_time

        # Define both directions
        self.lmvd_to_dvlog = nn.Linear(lmvd_dim, dvlog_dim)
        self.dvlog_to_lmvd = nn.Linear(dvlog_dim, lmvd_dim)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        if D == self.lmvd_dim:
            # Convert LMVD → DVLOG
            x = F.interpolate(x.permute(0, 2, 1), size=T, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # [B, T_out, D]
            x = self.lmvd_to_dvlog(x)  # [B, 1443, 161]

        elif D == self.dvlog_dim:
            # Convert DVLOG → LMVD
            x = F.interpolate(x.permute(0, 2, 1), size=T, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # [B, T_out, D]
            x = self.dvlog_to_lmvd(x)  # [B, 2086, 264]

        else:
            raise ValueError(f"Unsupported input dim: {D}, expected {self.lmvd_dim} or {self.dvlog_dim}")

        return x

if __name__ == '__main__':
    # -------------------------------
    # Simulate Input
    # -------------------------------
    
    adapter = ShapeAdapter()

    # LMVD to DVLOG
    x_lmvd = torch.randn(16, 2086, 264)
    x_dvlog = adapter(x_lmvd)
    print(f"LMVD to DVLOG: {x_dvlog.shape}")  # [16, 1443, 161]

    # DVLOG to LMVD
    x_dvlog_input = torch.randn(16, 1443, 161)
    x_lmvd_out = adapter(x_dvlog_input)
    print(f"DVLOG to LMVD: {x_lmvd_out.shape}")  # [16, 2086, 264]

