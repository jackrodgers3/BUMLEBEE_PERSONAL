"""
This is a single script that defines, trains, and tests Bumblebee,
a transformer for particle physics event reconstruction.

Draft 1 completed: Oct. 15, 2022

@version: 1.0
@author: AJ Wildridge Ethan Colbert,
modified by Jack P. Rodgers
"""
import wandb
import matplotlib
import numpy as np
import sklearn
import scipy
import torch
import ipykernel
import awkward as ak
import uproot
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models import make_model
from hist import Hist, axis
import hist
import argparse
import mplhep as hep
from tqdm import tqdm
import pickle
import dataloaders
from torch.utils.data import DataLoader
import itertools
import sys

hep.style.use(hep.style.CMS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_valid_test_split(full_dataset, tvt_split):
    """
    This is a function to split a dataset into training, validation, and
    testing sets. It simply makes cuts at the relevant places in the array.
    Note: if the sum of the three proportions is not 1, they will each be
    divided by their sum to normalize them.

    Parameters
    ----------
    full_dataset : array-like
        The dataset to be split.
    train_portion : float, optional
        The proportion of the data to be used for training. The default is 0.7.
    valid_portion : float, optional
        The proportion of the data to be used for validation. The default is 0.2.
    test_portion : float, optional
        The proportion of the data to be used for testing. The default is 0.1.

    Returns
    -------
    training_set : array-like
        The subset of full_dataset to use for training.
    validation_set : array-like
        The subset of full_dataset to use for validation.
    testing_set : array-like
        The subset of full_dataset to use for testing.

    """
    train_portion, valid_portion, test_portion = tvt_split
    total_portion = train_portion + valid_portion + test_portion
    if (total_portion != 1.0):
        train_portion /= total_portion
        valid_portion /= total_portion
        test_portion /= total_portion

    training_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                              tvt_split,
                                                                               generator = torch.Generator().manual_seed(23))
    
    return training_dataset, validation_dataset, test_dataset


def standardize_dataset(data):
    """
    Standardizes a dataset to a mean of 0 and standard deviation of 1.
    NOTE: THIS FUNCTION IS UNTESTED as of 10/13/2022.

    Parameters
    ----------
    data : array-like
        The dataset to be normalized.

    Returns
    -------
    array-like
        A normalized version of data.

    """
    mean = torch.mean(data, axis=0)
    stdev = torch.std(data, 0, True)
    return (data - mean) / stdev

# This class comse straight from the Jupyter notebook, it's a wrapper for the
# PyTorch optimizer that implements the learning rate scheduling we want.
class NoamOpt:
    def __init__(self, model_size, lr_mult, warmup, optimizer):
        self.model_size = model_size
        self.optimizer = optimizer
        self.warmup = warmup
        self.lr_mult = lr_mult
        self.steps = 0

    def step_and_update(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        d_model = self.model_size
        step, warmup = self.steps, self.warmup
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** (-1.5))

    def get_cur_lr(self):
        clr = None
        for p in self.optimizer.param_groups:
            clr = p['lr']
        return clr

    def update_learning_rate(self):
        self.steps += 1
        lr = self.lr_mult * self.get_lr_scale()
        for p in self.optimizer.param_groups:
            p['lr'] = lr


##############################
### THE START OF EXECUTION ###
##############################

# Accept hyperparameters from command line.
parser = argparse.ArgumentParser()
parser.add_argument('--d_model', help='model dimensionality', type=int)
parser.add_argument('--dropout', help='dropout rate', type=float)
parser.add_argument('--batch_size', help='size of minibatch', type=int)
parser.add_argument('--n_epochs', help = 'number of full runs over dataset', type=int)
parser.add_argument('--only_gen', help='use gen (T) or genreco (F)', type=bool)
parser.add_argument('--mask_probability', help='input mask probability', type=float)
parser.add_argument('--standardize', help='whether to standardize dataset', type=bool)
parser.add_argument('--tvt_split', help='train/valid/test split', type=tuple)
parser.add_argument('--lossf', help='L1 (T) or MSE (F)', type = bool)
parser.add_argument('--N', help='number of E/D layers', type=int)
parser.add_argument('--h', help='number of attention heads per MHA', type=int)
parser.add_argument('--warmup', help='number of warmup steps for lr scheduler', type=int)
parser.add_argument('--save_dir', help='directory to be saved on cluster', type=str)
parser.add_argument('--vpe', help='validations per epoch', type=int)
parser.set_defaults(d_model=256, dropout = 0.1,
                    batch_size = 16, n_epochs = 1, only_gen = False,
                    mask_prob = 0.09, standardize = True,
                    tvt_split = (0.7, 0.15, 0.15), lossf = True,
                    N = 3, h = 8, warmup=25000,
                    save_dir = r'C:\Users\jackm\PycharmProjects\BUMLEBEE_PERSONAL\saved_data/',
                    vpe = 4)

args = parser.parse_args()


# dataset creation and preparation

# background data creation
background_base_dir = r'D:\Data\Research\BUMBLEBEE\Reconstruction/'
channels = ['ee', 'emu', 'mumu']
years = ['2016ULpreVFP', '2016ULpostVFP', '2017UL', '2018UL']

background_filenames = {
    f'{background_base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': 'ttBar_treeVariables_step8' for
    channel, year in itertools.product(channels, years)
}


background_events = uproot.concatenate(
    background_filenames,
    [
        'l_pt', 'l_eta', 'l_phi', 'l_mass', 'l_pdgid',
        'lbar_pt', 'lbar_eta', 'lbar_phi', 'lbar_mass', 'lbar_pdgid',
        'b_pt', 'b_eta', 'b_phi', 'b_mass',
        'bbar_pt', 'bbar_eta', 'bbar_phi', 'bbar_mass',
        'met_pt', 'met_phi',
        'gen_l_pt', 'gen_l_eta', 'gen_l_phi', 'gen_l_mass', 'gen_l_pdgid',
        'gen_lbar_pt', 'gen_lbar_eta', 'gen_lbar_phi', 'gen_lbar_mass', 'gen_lbar_pdgid',
        'gen_b_pt', 'gen_b_eta', 'gen_b_phi', 'gen_b_mass',
        'gen_bbar_pt', 'gen_bbar_eta', 'gen_bbar_phi', 'gen_bbar_mass',
        'gen_nu_pt', 'gen_nu_eta', 'gen_nu_phi',
        'gen_nubar_pt', 'gen_nubar_eta', 'gen_nubar_phi', 'isTopGen'
    ],
    library='numpy'
)

# signal data creation
signal_base_dir = r'D:\Data\Research\BUMBLEBEE\Discrimination/'
channels = ['ee', 'emu', 'mumu']
years = ['2017', '2018']

signal_filenames = {
    f'{signal_base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_boundstate_{year}UL.root': 'ttBar_treeVariables_step8' for
    channel, year in itertools.product(channels, years)
}


signal_events = uproot.concatenate(
    signal_filenames,
    [
        'l_pt', 'l_eta', 'l_phi', 'l_mass', 'l_pdgid',
        'lbar_pt', 'lbar_eta', 'lbar_phi', 'lbar_mass', 'lbar_pdgid',
        'b_pt', 'b_eta', 'b_phi', 'b_mass',
        'bbar_pt', 'bbar_eta', 'bbar_phi', 'bbar_mass',
        'met_pt', 'met_phi',
        'gen_l_pt', 'gen_l_eta', 'gen_l_phi', 'gen_l_mass', 'gen_l_pdgid',
        'gen_lbar_pt', 'gen_lbar_eta', 'gen_lbar_phi', 'gen_lbar_mass', 'gen_lbar_pdgid',
        'gen_b_pt', 'gen_b_eta', 'gen_b_phi', 'gen_b_mass',
        'gen_bbar_pt', 'gen_bbar_eta', 'gen_bbar_phi', 'gen_bbar_mass',
        'gen_nu_pt', 'gen_nu_eta', 'gen_nu_phi',
        'gen_nubar_pt', 'gen_nubar_eta', 'gen_nubar_phi'
    ],
    library='numpy'
)

if args.only_gen:
    gen_reco_ids = torch.Tensor([0, 1, 1, 1, 1, 1, 1, 1])
    dataset_background = dataloaders.GenRecoDataset(background_events, gen_reco_ids,
                                                    standardize=args.standardize,
                                                    pretraining=False,
                                                    mask_probability=args.mask_prob)
    dataset_signal = dataloaders.GenRecoDataset(signal_events, gen_reco_ids,
                                                standardize=args.standardize,
                                                pretraining=False,
                                                mask_probability=args.mask_prob)
else:
    gen_reco_ids = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    dataset_background = dataloaders.GenRecoDataset(background_events, gen_reco_ids,
                                               standardize=args.standardize,
                                               pretraining=False,
                                               mask_probability=args.mask_prob)
    dataset_signal = dataloaders.GenRecoDataset(signal_events, gen_reco_ids,
                                               standardize=args.standardize,
                                               pretraining=False,
                                               mask_probability=args.mask_prob)

train_data_m, valid_data_m, test_data_m = train_valid_test_split(dataset_background, args.tvt_split)
dataset_background = dataloaders.DiscDataset(0.0, test_data_m)
dataset_signal = dataloaders.DiscDataset(1.0, dataset_signal)

full_dataset = torch.utils.data.ConcatDataset([dataset_background, dataset_signal])

train_dataset, valid_dataset, test_dataset = train_valid_test_split(full_dataset, args.tvt_split)

# weighted sampling
num_zeros = 0
num_ones = 0
ind_sample_weight = []
for i in tqdm(range(len(train_dataset)), desc="Creating class weights"):
    if train_dataset[i][1].item() == 1.0:
        num_ones += 1
    elif train_dataset[i][1].item() == 0.0:
        num_zeros += 1

samples_weight = np.array([1. / num_zeros, 1. / num_ones])
for i in tqdm(range(len(train_dataset)), desc="Making sampler"):
    if train_dataset[i][1].item() == 1.0:
        ind_sample_weight.append(samples_weight[1])
    elif train_dataset[i][1].item() == 0.0:
        ind_sample_weight.append(samples_weight[0])
ind_sample_weight = torch.tensor(ind_sample_weight)
weighted_sampler = torch.utils.data.WeightedRandomSampler(ind_sample_weight.type('torch.DoubleTensor'), len(ind_sample_weight))

print('DATA CREATED')
# Prepare the model.
criterion = nn.BCELoss(reduction='mean')

model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model,
                   d_ff= int(4 * args.d_model), h = args.h, dropout= args.dropout)
model = model.to(device)
model.load_state_dict(torch.load(args.save_dir + 'bumblebee2.pt'))
optimizer = torch.optim.Adam(model.parameters(),
                             lr=3e-4, betas=(0.9, 0.98), eps=1e-9,
                             weight_decay=0.0)
scheduler = NoamOpt(args.d_model, lr_mult=1.0, warmup=args.warmup, optimizer=optimizer)

# Create data loaders
# Data Loader
data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=weighted_sampler)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
print('DATALOADERS CREATED')
wandb_config = {
    "epochs": args.n_epochs,
    "d_model": args.d_model,
    "num_layers": args.N,
    "num_heads": args.h,
    "dropout": args.dropout,
    "batch_size": args.batch_size,
    "mask_probability": args.mask_prob,
    "warmup steps": args.warmup
}

# wandb initialization
wandb.init(
    project='BUMBLEBEE_DISC_TEST',
    entity='bumblebee_team',
    config=wandb_config
)


# Loss Arrays
training_losses = []

validation_losses = []

best_validation_loss = -1
num_steps = int(math.ceil(len(train_dataset) / args.batch_size))

test_losses = []
test_predictions = []
test_truths = []
print('TRAINING STARTED')
# Train the model.
for t in tqdm(range(args.n_epochs)):
    model.train()
    for batch_id, (x, target) in enumerate(data_loader):
        # Training Set
        x = x.to(device)
        target = target.to(device)
        scheduler.zero_grad()
        wandb.log({'lr': scheduler.get_cur_lr()})
        y_pred = model(x)
        etas = y_pred[:, :, 1]

        phis = 2 * torch.atan(y_pred[:, :, 2])

        corrected_pred = torch.cat((y_pred[:, :, 0][:, :, None], etas[:, :, None],
                                    phis[:, :, None], y_pred[:, :, 3][:, :, None]), axis=2)

        loss = criterion(torch.sigmoid(corrected_pred[:, 0, 0]), target.float())

        loss.backward()
        if batch_id % 5 == 0:
            wandb.log({'loss': loss.item()})

        # append to lists and detach
        training_losses.append(loss.detach().cpu().numpy())

        scheduler.step_and_update()
    # Validate
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        valid_step_number = 0
        avg_valid_loss = 0

        for valid_batch_id, (x_valid, target_valid) in enumerate(valid_data_loader):
            x_valid = x_valid.to(device)
            target_valid = target_valid.to(device)
            target_pred_valid = model(x_valid)

            # ask ML model to predict tan(theta) then convert to eta
            # valid_etas = -1 * torch.log(torch.abs(y_valid[:, :, 1]) + 0.01)
            valid_etas = target_pred_valid[:, :, 1]

            # similarly ask ML model to predict tan(phi) then convert to phi
            valid_phis = 2 * torch.atan(target_pred_valid[:, :, 2])

            corrected_y_valid = torch.cat((target_pred_valid[:, :, 0][:, :, None], valid_etas[:, :, None], valid_phis[:, :, None],
                                           target_pred_valid[:, :, 3][:, :, None]), axis=2)

            valid_loss = criterion(torch.sigmoid(corrected_y_valid[:, 0, 0]), target_valid.float())

            avg_valid_loss += valid_loss.item()

            wandb.log({"valid_loss": valid_loss.item()})

        avg_valid_loss /= valid_batch_id
        wandb.log({'average valid loss': avg_valid_loss})

        validation_losses.append(valid_loss.detach().cpu().numpy())

        if valid_loss.detach().cpu().numpy() < best_validation_loss or best_validation_loss == -1:
            best_validation_loss = valid_loss.detach().cpu().numpy()
            torch.save(model.state_dict(), args.save_dir + 'bumblebee2_cls.pt')


# There's a bunch of stuff related to plotting here in the notebook. I'm
# omitting it here, we'll have to figure out exactly how to handle output.

# Model testing
# Also copied from the notebook.
best_model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model,
                   d_ff= int(4 * args.d_model), h = args.h, dropout= args.dropout)
best_model = best_model.to(device)
best_model.load_state_dict(torch.load(args.save_dir + 'bumblebee2_cls.pt'))

best_model.eval()

for test_batch_id, (x_test, target_test) in enumerate(test_data_loader):
    x_test = x_test.to(device)
    target_test = target_test.to(device)
    target_pred_test = best_model(x_test)

    test_etas = target_pred_test[:, :, 1]

    test_phis = 2 * torch.atan(target_pred_test[:, :, 2])

    corrected_y_test = torch.cat(
        (target_pred_test[:, :, 0][:, :, None], test_etas[:, :, None], test_phis[:, :, None],
         target_pred_test[:, :, 3][:, :, None]), axis=2)


    test_loss = criterion(torch.sigmoid(corrected_y_test[:, 0, 0]), target_test.float())
    wandb.log({'test_loss': test_loss.item()})

    test_losses.append(test_loss.detach().cpu().numpy())
    test_predictions.append(torch.sigmoid(corrected_y_test[:, 0, 0]).detach().cpu().numpy())
    test_truths.append(target_test.detach().cpu().numpy())


# Analysis (also copied)
'''
# Record losses
transformer_loss_l = np.sum(np.mean(np.array(test_loss_components)[:, 7, :], axis=0))
wandb.log({'trans_loss_l': transformer_loss_l})
transformer_loss_lbar = np.sum(np.mean(np.array(test_loss_components)[:, 8, :], axis=0))
wandb.log({'trans_loss_lbar': transformer_loss_lbar})
transformer_loss_b = np.sum(np.mean(np.array(test_loss_components)[:, 9, :], axis=0))
wandb.log({'trans_loss_b': transformer_loss_b})
transformer_loss_bbar = np.sum(np.mean(np.array(test_loss_components)[:, 10, :], axis=0))
wandb.log({'trans_loss_bbar': transformer_loss_bbar})
transformer_loss_nu = np.sum(np.mean(np.array(test_loss_components)[:, 11, :], axis=0))
wandb.log({'trans_loss_nu': transformer_loss_nu})
transformer_loss_nubar = np.sum(np.mean(np.array(test_loss_components)[:, 12, :], axis=0))
wandb.log({'trans_loss_nubar': transformer_loss_nubar})

# Record errors
transformer_error_nu = ((((np.std(np.array(test_loss_components)[:, 11, :], axis=0)) / (
            (len(test_loss_components)) ** (0.5)))[0] ** 2) +
                        (((np.std(np.array(test_loss_components)[:, 11, :], axis=0)) / (
                                    (len(test_loss_components)) ** (0.5)))[1] ** 2) +
                        (((np.std(np.array(test_loss_components)[:, 11, :], axis=0)) / (
                                    (len(test_loss_components)) ** (0.5)))[2] ** 2) +
                        (((np.std(np.array(test_loss_components)[:, 11, :], axis=0)) / (
                                    (len(test_loss_components)) ** (0.5)))[3] ** 2)) ** (0.5)
wandb.log({'trans_error_nu': transformer_error_nu})
transformer_error_nubar = ((((np.std(np.array(test_loss_components)[:, 12, :], axis=0)) / (
            (len(test_loss_components)) ** (0.5)))[0] ** 2) +
                           (((np.std(np.array(test_loss_components)[:, 12, :], axis=0)) / (
                                       (len(test_loss_components)) ** (0.5)))[1] ** 2) +
                           (((np.std(np.array(test_loss_components)[:, 12, :], axis=0)) / (
                                       (len(test_loss_components)) ** (0.5)))[2] ** 2) +
                           (((np.std(np.array(test_loss_components)[:, 12, :], axis=0)) / (
                                       (len(test_loss_components)) ** (0.5)))[3] ** 2)) ** (0.5)
wandb.log({'trans_error_nubar': transformer_error_nubar})
transformer_error_l = ((((np.std(np.array(test_loss_components)[:, 7, :], axis=0)) / (
            (len(test_loss_components)) ** (0.5)))[0] ** 2) +
                       (((np.std(np.array(test_loss_components)[:, 7, :], axis=0)) / (
                                   (len(test_loss_components)) ** (0.5)))[1] ** 2) +
                       (((np.std(np.array(test_loss_components)[:, 7, :], axis=0)) / (
                                   (len(test_loss_components)) ** (0.5)))[2] ** 2) +
                       (((np.std(np.array(test_loss_components)[:, 7, :], axis=0)) / (
                                   (len(test_loss_components)) ** (0.5)))[3] ** 2)) ** (0.5)
wandb.log({'trans_error_l': transformer_error_l})
transformer_error_lbar = ((((np.std(np.array(test_loss_components)[:, 8, :], axis=0)) / (
            (len(test_loss_components)) ** (0.5)))[0] ** 2) +
                          (((np.std(np.array(test_loss_components)[:, 8, :], axis=0)) / (
                                      (len(test_loss_components)) ** (0.5)))[1] ** 2) +
                          (((np.std(np.array(test_loss_components)[:, 8, :], axis=0)) / (
                                      (len(test_loss_components)) ** (0.5)))[2] ** 2) +
                          (((np.std(np.array(test_loss_components)[:, 8, :], axis=0)) / (
                                      (len(test_loss_components)) ** (0.5)))[3] ** 2)) ** (0.5)
wandb.log({'trans_error_lbar': transformer_error_lbar})
transformer_error_b = ((((np.std(np.array(test_loss_components)[:, 9, :], axis=0)) / (
            (len(test_loss_components)) ** (0.5)))[0] ** 2) +
                       (((np.std(np.array(test_loss_components)[:, 9, :], axis=0)) / (
                                   (len(test_loss_components)) ** (0.5)))[1] ** 2) +
                       (((np.std(np.array(test_loss_components)[:, 9, :], axis=0)) / (
                                   (len(test_loss_components)) ** (0.5)))[2] ** 2) +
                       (((np.std(np.array(test_loss_components)[:, 9, :], axis=0)) / (
                                   (len(test_loss_components)) ** (0.5)))[3] ** 2)) ** (0.5)
wandb.log({'trans_error_b': transformer_error_b})
transformer_error_bbar = ((((np.std(np.array(test_loss_components)[:, 10, :], axis=0)) / (
            (len(test_loss_components)) ** (0.5)))[0] ** 2) +
                          (((np.std(np.array(test_loss_components)[:, 10, :], axis=0)) / (
                                      (len(test_loss_components)) ** (0.5)))[1] ** 2) +
                          (((np.std(np.array(test_loss_components)[:, 10, :], axis=0)) / (
                                      (len(test_loss_components)) ** (0.5)))[2] ** 2) +
                          (((np.std(np.array(test_loss_components)[:, 10, :], axis=0)) / (
                                      (len(test_loss_components)) ** (0.5)))[3] ** 2)) ** (0.5)
wandb.log({'trans_error_bbar': transformer_error_bbar})
# I'm omitting the "Comparisons" section, since we aren't including the other
# models. If we want anything from there, we can add it.
'''

wandb.finish()
# Save the data. This section will likely need some augmentation.

# Test
torch.save(np.array(test_losses), args.save_dir + 'test_losses2.pt')
torch.save(np.array(test_predictions), args.save_dir + 'predictions2.pt')
torch.save(np.array(test_truths), args.save_dir + 'truths2.pt')

### AND WE'RE DONE!!!
