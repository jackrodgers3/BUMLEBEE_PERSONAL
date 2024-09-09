"""
This is a single script that defines, trains, and tests Bumblebee,
a transformer for particle physics event reconstruction.

Draft 1 completed: Oct. 15, 2022

@version: 1.0
@author: AJ Wildridge Ethan Colbert,
modified by Jack P. Rodgers
"""

"""
updates from V4: mask fixes and implementing llbar phi, eta unmasking
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
import optuna
from sklearn.metrics import auc, roc_curve
G = torch.Generator()
G.manual_seed(23)
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


def merge_bs(dict1, dict2):
    ret_dict = {}
    for key in dict1.keys():
        temp = []
        for i in range(len(dict1[key])):
            temp.append(dict1[key][i])
        for j in range(len(dict2[key])):
            temp.append(dict2[key][j])
        ret_dict[key] = np.array(temp)
    return ret_dict


def train_valid_test(total_events, tvt_split, use_generator = False, subset = None):
    if subset is not None:
        n = int(len(total_events['l_pt']) * subset)
    else:
        n = len(total_events['l_pt'])
    if use_generator:
        total_indices = torch.randperm(n=n, generator=G).tolist()
    else:
        total_indices = torch.randperm(n=n).tolist()
    cut1 = int(np.ceil(tvt_split[0] * n))
    train_indices = total_indices[:cut1]
    cut2 = int(np.ceil((tvt_split[0] + tvt_split[1]) * n))
    valid_indices = total_indices[cut1:cut2]
    test_indices = total_indices[cut2:]

    keys = [key for key in total_events]
    train_values = [total_events[key][train_indices] for key in total_events]
    valid_values = [total_events[key][valid_indices] for key in total_events]
    test_values = [total_events[key][test_indices] for key in total_events]

    train_events = dict(zip(keys, train_values))
    valid_events = dict(zip(keys, valid_values))
    test_events = dict(zip(keys, test_values))

    return train_events, valid_events, test_events


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

class CombinedLoss(nn.Module):
    def __init__(self, reduction = 'none', device=torch.device('cpu')):
        super(CombinedLoss, self).__init__()
        self.da_mask = None
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, targets):
        #inputs[inputs > math.pi] = math.pi
        #inputs[inputs < -math.pi] = -math.pi
        self.da_mask = torch.ones(targets.shape, dtype=torch.bool)
        self.da_mask = self.da_mask.to(device)
        self.da_mask[:, :, 2] = torch.zeros(self.da_mask[:, :, 2].shape)
        ret = (self.da_mask * torch.pow(input=(inputs-targets), exponent=2)) + (2. - (2. * torch.cos(input=~self.da_mask * (inputs - targets))))
        if self.reduction == 'none':
            pass
        elif self.reduction == 'sum':
            ret = torch.sum(ret)
        elif self.reduction == 'mean':
            ret = torch.mean(ret)
        return ret

    def get_da_mask(self):
        return self.da_mask


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
parser.add_argument('--tvt_split', help='train/valid/test split', type=list)
parser.add_argument('--lossf', help='L1 or MSE or CL', type = str, choices = ['L1', 'MSE', 'CL'])
parser.add_argument('--N', help='number of E/D layers', type=int)
parser.add_argument('--h', help='number of attention heads per MHA', type=int)
parser.add_argument('--warmup', help='number of warmup steps for lr scheduler', type=int)
parser.add_argument('--reco_data_dir', help='directory where data reconstruction data is stored', type=str)
parser.add_argument('--disc_data_dir', help='directory where data discrimination data is stored', type=str)
parser.add_argument('--save_dir', help='directory to be saved on cluster', type=str)
parser.add_argument('--act_fn', help='activation function', type=str)
parser.add_argument('--weight_decay', help = 'coefficient for L1 regularization', type=float)
parser.set_defaults(d_model=512, dropout = 0.1,
                    batch_size = 32, n_epochs = 1, only_gen = False,
                    mask_prob = 0.09, standardize = True,
                    tvt_split = [0.7, 0.15, 0.15], lossf = 'CL',
                    N = 8, h = 8, warmup=30000,
                    reco_data_dir = r'/depot/cms/top/jprodger/Bumblebee/src/reco_data/',
                    disc_data_dir = r'/depot/cms/top/jprodger/Bumblebee/src/disc_data/',
                    save_dir = r'/depot/cms/top/jprodger/Bumblebee/src/disc_study/tune_output/',
                    act_fn = 'gelu', weight_decay = 1e-3)


def train(trial):
    args = parser.parse_args()
    args.d_model = trial.suggest_categorical("d_model", [128, 256, 512, 1024])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.95)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    args.N = trial.suggest_int("N", 3, 10)
    args.h = trial.suggest_categorical("h", [4, 8, 16, 32, 64])
    args.act_fn = trial.suggest_categorical("act_fn", ["relu", "leaky_relu", "gelu"])
    args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # dataset creation and preparation
    # constants
    selected_keys = [
            'l_pt', 'l_eta', 'l_phi', 'l_mass', 'l_pdgid',
            'lbar_pt', 'lbar_eta', 'lbar_phi', 'lbar_mass', 'lbar_pdgid',
            'b_pt', 'b_eta', 'b_phi', 'b_mass',
            'bbar_pt', 'bbar_eta', 'bbar_phi', 'bbar_mass',
            'top_pt', 'top_phi', 'top_eta', 'top_mass',
            'tbar_pt', 'tbar_phi', 'tbar_eta', 'tbar_mass',
            'met_pt', 'met_phi',
            'gen_l_pt', 'gen_l_eta', 'gen_l_phi', 'gen_l_mass', 'gen_l_pdgid',
            'gen_lbar_pt', 'gen_lbar_eta', 'gen_lbar_phi', 'gen_lbar_mass', 'gen_lbar_pdgid',
            'gen_b_pt', 'gen_b_eta', 'gen_b_phi', 'gen_b_mass',
            'gen_bbar_pt', 'gen_bbar_eta', 'gen_bbar_phi', 'gen_bbar_mass',
            'gen_nu_pt', 'gen_nu_eta', 'gen_nu_phi',
            'gen_nubar_pt', 'gen_nubar_eta', 'gen_nubar_phi'
        ]
    
    channels = ['ee', 'emu', 'mumu']
    
    # background data
    base_dir = args.reco_data_dir
    years = ['2016ULpreVFP', '2016ULpostVFP', '2017UL', '2018UL']
    #years = ['2018UL']
    notau_filenames = {
        f'{base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': 'ttBar_treeVariables_step8' for
        channel, year in itertools.product(channels, years)
    }
    
    background_total_events = uproot.concatenate(
        notau_filenames,
        selected_keys,
        library='numpy'
    )
    
    #disc data
    base_dir2 = args.disc_data_dir
    years1 = ['2017', '2018']
    notau_filenames1 = {
        f'{base_dir2}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_boundstate_{year}UL*': 'ttBar_treeVariables_step8' for
        channel, year in itertools.product(channels, years1)
    }
    
    years2 = ['2016preVFP', '2016postVFP']
    notau_filenames2 = {
        f'{base_dir2}/{year}/{channel}/{channel}_ttbarsignalplustau_{year[4:]}_fromDilepton_boundstate_{year[:4]}UL*': 'ttBar_treeVariables_step8' for
        channel, year in itertools.product(channels, years2)
    }
    
    signal_total_events = uproot.concatenate(
        [notau_filenames1, notau_filenames2],
        selected_keys,
        library='numpy'
    )
    
    # assigning bs
    background_total_events['bs'] = np.array([0.0 for _ in range(len(background_total_events['l_pt']))])
    signal_total_events['bs'] = np.array([1.0 for _ in range(len(signal_total_events['l_pt']))])
    
    reco_train_events, reco_valid_events, reco_test_events = train_valid_test(background_total_events, args.tvt_split, use_generator = False)
    print("DATA PRESENT. COUNTS:")
    print(f"Counts | background: {len(background_total_events['l_pt'])} | signal: {len(signal_total_events['l_pt'])}")
    
    
    if args.only_gen:
        gen_reco_ids = torch.Tensor([0, 1, 1, 1, 1, 1, 1, 1])
        dataset_train = dataloaders.GenRecoDataset(reco_train_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=True,
                                                   mask_probability=args.mask_prob)
        dataset_valid = dataloaders.GenRecoDataset(reco_valid_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=False,
                                                   mask_probability=args.mask_prob)
    else:
        gen_reco_ids = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        dataset_train = dataloaders.GenRecoDataset(reco_train_events, gen_reco_ids,
                                             standardize=args.standardize,
                                             pretraining=True,
                                             mask_probability=args.mask_prob)
        dataset_valid = dataloaders.GenRecoDataset(reco_valid_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=False,
                                                   mask_probability=args.mask_prob)
    print('DATA CREATED')
    # Prepare the model.
    if args.lossf == 'L1':
        criterion = nn.L1Loss(reduction='mean')
    elif args.lossf == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
    elif args.lossf == 'CL':
        criterion = CombinedLoss(reduction='mean', device = device)
    
    model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model,
                       d_ff= int(4 * args.d_model), h = args.h, dropout= args.dropout, act_fn = args.act_fn)
    if torch.cuda.device_count() > 1:
        print(f"USING {torch.cuda.device_count()} GPUS")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-4, betas=(0.9, 0.98), eps=1e-9,
                                 weight_decay=args.weight_decay)
    scheduler = NoamOpt(args.d_model, lr_mult=1.0, warmup=args.warmup, optimizer=optimizer)
    
    # Create data loaders
    # Data Loader
    data_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    valid_data_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False)
    print('DATALOADERS CREATED')
    
    
    wandb_config = {
        "epochs": args.n_epochs,
        "d_model": args.d_model,
        "num_layers": args.N,
        "num_heads": args.h,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "mask_probability": args.mask_prob,
        "warmup steps": args.warmup,
        "weight_decay": args.weight_decay,
        "trial_number": trial.number
    }
    
    # wandb initialization
    wandb.init(
        project='BUMBLEBEE_DISCRIMINATION_TUNING',
        entity='bumblebee_team',
        config=wandb_config
    )
    
    
    # Loss Arrays
    training_loss_components = []
    training_losses = []
    masked_training_losses = []
    unmasked_training_losses = []
    
    validation_losses = []
    masked_validation_losses = []
    unmasked_validation_losses = []
    validation_loss_components = []
    
    best_validation_loss = -1
    
    test_losses = []
    masked_test_losses = []
    test_loss_components = []
    test_predictions = []
    test_truths = []
    
    print('TRAINING STARTED')
    # Train the model
    for t in range(args.n_epochs):
        model.train()
        print(f"EPOCH {t}:")
        for batch_id, (x, target) in tqdm(enumerate(data_loader), desc="TRAINING"):
            # Training Set
            x = x.to(device)
            target = target.to(device)
    
            scheduler.zero_grad()
            wandb.log({'lr': scheduler.get_cur_lr()})
            y_pred = model(x)
    
            etas = y_pred[:, :, 1]
    
            phis = y_pred[:, :, 2]
    
            corrected_pred = torch.cat((y_pred[:, :, 0][:, :, None], etas[:, :, None],
                                        phis[:, :, None], y_pred[:, :, 3][:, :, None]), axis=2)
    
            out = criterion(corrected_pred, target.float())
    
            zerod_mask = x[:, :, -1]
            four_vector_mask = zerod_mask[:, :, None].repeat(1, 1, 4)
            four_vector_mask[:, -3:-1, 3] = torch.ones(four_vector_mask[:, -3:-1, 3].shape) # unmasking neutrino mass
            four_vector_mask[:, 7:9, 1:3] = torch.ones(four_vector_mask[:, 7:9, 1:3].shape) # unmasking llbar eta, phi
    
            # don't include losses which were 0 because of being unmasked in masked loss average
            masked_out = out * ~four_vector_mask.type(torch.bool).to(device)
            masked_out = torch.sum(torch.sum(masked_out, axis=1), axis=1)
            masked_out = masked_out[masked_out != 0]
    
            # don't include losses which were 0 because of being masked in unmasked loss average
            unmasked_out = out * four_vector_mask.type(torch.bool).to(device)
            unmasked_out = torch.sum(torch.sum(unmasked_out, axis=1), axis=1)
            unmasked_out = unmasked_out[unmasked_out != 0]
    
            # look to see if training objective on masking is being learned
            masked_loss = torch.mean(masked_out, dim=0)
            unmasked_loss = torch.mean(unmasked_out, dim=0)
    
            # see how well we are learning each of the particle components
            component_loss = torch.sum(out, axis=0) / torch.sum(four_vector_mask, axis=0).to(device)
    
            # see how well we are learning overall
            loss = torch.sum(component_loss)
            weighted_loss = (unmasked_loss + (1.0 / args.mask_prob)) * masked_loss
            masked_loss.backward()
    
            if batch_id % 15 == 0:
                wandb.log({'masked loss': masked_loss.item()})
    
            # append to lists and detach
            training_losses.append(loss.detach().cpu().numpy())
            masked_training_losses.append(masked_loss.detach().cpu().numpy())
            unmasked_training_losses.append(unmasked_loss.detach().cpu().numpy())
            training_loss_components.append(component_loss.detach().cpu().numpy())
    
            scheduler.step_and_update()
        # Validate
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            unmasked_valid_loss = 0
            masked_valid_loss = 0
            avg_masked_valid_loss = 0
            valid_step_number = 0
            valid_loss_components = torch.zeros(component_loss.shape).to(device)
            if args.standardize:
                max_pt = 10
            else:
                max_pt = 1000
            h_gen_nu_pt = hist.Hist(hist.axis.Regular(100, -10, max_pt))
            h_gen_nu_eta = hist.Hist(hist.axis.Regular(100, -10, 10))
            h_gen_nu_phi = hist.Hist(hist.axis.Regular(100, -3.14, 3.14))
            h_unmasked_reco_nu_pt = hist.Hist(hist.axis.Regular(100, -10, max_pt))
            h_unmasked_reco_nu_eta = hist.Hist(hist.axis.Regular(100, -10, 10))
            h_unmasked_reco_nu_phi = hist.Hist(hist.axis.Regular(100, -3.14, 3.14))
            h_masked_reco_nu_pt = hist.Hist(hist.axis.Regular(100, -10, max_pt))
            h_masked_reco_nu_eta = hist.Hist(hist.axis.Regular(100, -10, 10))
            h_masked_reco_nu_phi = hist.Hist(hist.axis.Regular(100, -3.14, 3.14))
    
            for valid_batch_id, (x_valid, target_valid) in tqdm(enumerate(valid_data_loader), desc="VALIDATING"):
                x_valid = x_valid.to(device)
                target_valid = target_valid.to(device)
                target_pred_valid = model(x_valid)
    
                # ask ML model to predict tan(theta) then convert to eta
                # valid_etas = -1 * torch.log(torch.abs(y_valid[:, :, 1]) + 0.01)
                valid_etas = target_pred_valid[:, :, 1]
    
                # similarly ask ML model to predict tan(phi) then convert to phi
                valid_phis = target_pred_valid[:, :, 2]
    
                corrected_y_valid = torch.cat((target_pred_valid[:, :, 0][:, :, None], valid_etas[:, :, None], valid_phis[:, :, None],
                                               target_pred_valid[:, :, 3][:, :, None]), axis=2)
    
                valid_out = criterion(corrected_y_valid, target_valid.float())
    
                zerod_mask_valid = x_valid[:, :, -1]
                four_vector_mask_valid = zerod_mask_valid[:, :, None].repeat(1, 1, 4)
                four_vector_mask_valid[:, -3:-1, 3] = torch.ones(four_vector_mask_valid[:, -3:-1, 3].shape)  # unmasking neutrino mass
                four_vector_mask_valid[:, 7:9, 1:3] = torch.ones(four_vector_mask_valid[:, 7:9, 1:3].shape)  # unmasking llbar eta, phi
    
                # don't include losses which were 0 because of being unmasked in masked loss average
                masked_valid_out = valid_out * ~four_vector_mask_valid.type(torch.bool).to(device)
                masked_valid_out = torch.sum(torch.sum(masked_valid_out, axis=1), axis=1)
                masked_valid_out = masked_valid_out[masked_valid_out != 0]
    
                # don't include losses which were 0 because of being masked in unmasked loss average
                unmasked_valid_out = valid_out * four_vector_mask_valid.type(torch.bool).to(device)
                unmasked_valid_out = torch.sum(torch.sum(unmasked_valid_out, axis=1), axis=1)
                unmasked_valid_out = unmasked_valid_out[unmasked_valid_out != 0]
    
                # look to see if training objective on masking is being learned
                masked_valid_loss = torch.mean(masked_valid_out, dim=0)
                unmasked_valid_loss = torch.mean(unmasked_valid_out, dim=0)
                valid_loss = torch.mean(torch.sum(valid_out), dim=0)
                avg_masked_valid_loss += masked_valid_loss.item()
    
                masked_validation_losses.append(masked_valid_loss.detach().cpu().numpy())
                unmasked_validation_losses.append(unmasked_valid_loss.detach().cpu().numpy())
                validation_losses.append(valid_loss.detach().cpu().numpy())
    
                valid_loss_components += torch.mean(valid_out, dim=0)
                if valid_batch_id % 5 == 0:
                    wandb.log({"masked_validation_loss": masked_valid_loss.item()})
    
                nunubar_validation_data = target_valid[:, -3:-1, :-1]
                nunubar_masks = four_vector_mask_valid[:, -3:-1, :-1].to(device)
    
                nu_pt = nunubar_validation_data[:, :, 0]
                nu_eta = nunubar_validation_data[:, :, 1]
                nu_phi = nunubar_validation_data[:, :, 2]
    
                masked_nu_pt = corrected_y_valid[:, -3:-1, 0] * ~nunubar_masks.type(torch.bool)[:, :, 0]
                masked_nu_pt = masked_nu_pt[masked_nu_pt != 0]
                masked_nu_eta = corrected_y_valid[:, -3:-1, 1] * ~nunubar_masks.type(torch.bool)[:, :, 1]
                masked_nu_eta = masked_nu_eta[masked_nu_eta != 0]
                masked_nu_phi = corrected_y_valid[:, -3:-1, 2] * ~nunubar_masks.type(torch.bool)[:, :, 2]
                masked_nu_phi = masked_nu_phi[masked_nu_phi != 0]
    
                unmasked_nu_pt = corrected_y_valid[:, -3:-1, 0] * nunubar_masks.type(torch.bool)[:, :, 0]
                unmasked_nu_pt = unmasked_nu_pt[unmasked_nu_pt != 0]
                unmasked_nu_eta = corrected_y_valid[:, -3:-1, 1] * nunubar_masks.type(torch.bool)[:, :, 1]
                unmasked_nu_eta = unmasked_nu_eta[unmasked_nu_eta != 0]
                unmasked_nu_phi = corrected_y_valid[:, -3:-1, 2] * nunubar_masks.type(torch.bool)[:, :, 2]
                unmasked_nu_phi = unmasked_nu_phi[unmasked_nu_phi != 0]
    
                h_gen_nu_pt.fill(nu_pt.detach().cpu().numpy().flatten())
                h_gen_nu_eta.fill(nu_eta.detach().cpu().numpy().flatten())
                h_gen_nu_phi.fill(nu_phi.detach().cpu().numpy().flatten())
                h_masked_reco_nu_pt.fill(masked_nu_pt.detach().cpu().numpy().flatten())
                h_masked_reco_nu_eta.fill(masked_nu_eta.detach().cpu().numpy().flatten())
                h_masked_reco_nu_phi.fill(masked_nu_phi.detach().cpu().numpy().flatten())
                h_unmasked_reco_nu_pt.fill(unmasked_nu_pt.detach().cpu().numpy().flatten())
                h_unmasked_reco_nu_eta.fill(unmasked_nu_eta.detach().cpu().numpy().flatten())
                h_unmasked_reco_nu_phi.fill(unmasked_nu_phi.detach().cpu().numpy().flatten())
    
            validation_loss_components.append(valid_loss_components.detach().cpu().numpy())
    
            avg_masked_valid_loss /= (valid_batch_id+1)
            wandb.log({"avg_masked_valid_loss": avg_masked_valid_loss})
    
            # early stopping
            if avg_masked_valid_loss >= best_validation_loss != -1:
                break
    
            if avg_masked_valid_loss < best_validation_loss or best_validation_loss == -1:
                best_validation_loss = avg_masked_valid_loss
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), args.save_dir + f'bumblebee{trial.number}.pt')
                else:
                    torch.save(model.state_dict(), args.save_dir + f'bumblebee{trial.number}.pt')
    
            nunubar_validation_loss = np.sum(np.sum(np.array(validation_loss_components)[:, -3:-1, :3], axis=1), axis=1)
            bbar_validation_loss = np.sum(np.sum(np.array(validation_loss_components)[:, -5:-3, :], axis=1), axis=1)
            llbar_validation_loss = np.sum(np.sum(np.array(validation_loss_components)[:, -7:-5, :], axis=1), axis=1)
    
            nu_pt_validation_loss = np.array(validation_loss_components)[:,-2,0]
            nu_eta_validation_loss = np.array(validation_loss_components)[:,-2,1]
            nu_phi_validation_loss = np.array(validation_loss_components)[:, -2, 2]
            nubar_pt_validation_loss = np.array(validation_loss_components)[:, -3, 0]
            nubar_eta_validation_loss = np.array(validation_loss_components)[:, -3, 1]
            nubar_phi_validation_loss = np.array(validation_loss_components)[:, -3, 2]
    
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            hep.style.use([hep.style.CMS, {'figure.figsize': (9, 27)}])
            hep.cms.text("Work In Progress", loc=0)
    
            hep.histplot([h_masked_reco_nu_pt, h_unmasked_reco_nu_pt],
                         label=[r'Bumblebee masked $p_{T,\nu}$', r'Bumblebee unmasked $p_{T,\nu}$'], histtype='fill',
                         linewidth=2, edgecolor='black', ax=ax1, yerr=False, sort='yield', stack=True)
            hep.histplot([h_gen_nu_pt], label=[r'Gen $p_{T,\nu}$'], histtype='step', linewidth=2, ax=ax1, yerr=False)
            ax1.set_yscale('log')
            ax1.legend()
            ax1.set_xlabel(r'$p_{T,\nu}$')
    
            hep.histplot([h_masked_reco_nu_eta, h_unmasked_reco_nu_eta],
                         label=[r'Bumblebee masked $\eta_{\nu}$', r'Bumblebee unmasked $\eta_{\nu}$'], histtype='fill',
                         linewidth=2, edgecolor='black', ax=ax2, yerr=False, sort='yield', stack=True)
            hep.histplot([h_gen_nu_eta], label=[r'Gen $\eta_{\nu}$'], histtype='step', linewidth=2, ax=ax2, yerr=False)
            ax2.set_yscale('log')
            ax2.legend()
            ax2.set_xlabel(r'$\eta_{\nu}$')
    
            hep.histplot([h_masked_reco_nu_phi, h_unmasked_reco_nu_phi],
                         label=[r'Bumblebee masked $\phi_{\nu}$', r'Bumblebee unmasked $\phi_{\nu}$'], histtype='fill',
                         linewidth=2, edgecolor='black', ax=ax3, yerr=False, sort='yield', stack=True)
            hep.histplot([h_gen_nu_phi], label=[r'Gen $\phi_{\nu}$'], histtype='step', linewidth=2, ax=ax3, yerr=False)
            ax3.legend()
            ax3.set_yscale('log')
            ax3.set_xlabel(r'$\phi_{\nu}$')
            # plt.show()
            plt.savefig(args.save_dir + f'hist_plots{trial.number}@{t}.pdf')
            plt.clf()
            plt.cla()
    
    
    print("RECONSTRUCTION FINISHED; DISCRIMINATION STARTED")
    num_negatives = len(reco_test_events['l_pt'])
    num_positives = len(signal_total_events['l_pt'])
    print(f"class counts | 0: {num_negatives} | 1: {num_positives}")
    
    disc_events = merge_bs(reco_test_events, signal_total_events)
    
    
    disc_train_events, disc_valid_events, disc_test_events = train_valid_test(disc_events, args.tvt_split, use_generator = False)
    
    if args.only_gen:
        gen_reco_ids = torch.Tensor([0, 1, 1, 1, 1, 1, 1, 1])
        dataset_train = dataloaders.GenRecoDataset_Toponium(disc_train_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=True,
                                                   mask_probability=args.mask_prob)
        dataset_valid = dataloaders.GenRecoDataset_Toponium(disc_valid_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=False,
                                                   mask_probability=args.mask_prob)
        dataset_test = dataloaders.GenRecoDataset_Toponium(disc_test_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=False,
                                                   mask_probability=args.mask_prob)
    else:
        gen_reco_ids = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        dataset_train = dataloaders.GenRecoDataset_Toponium(disc_train_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=True,
                                                   mask_probability=args.mask_prob)
        dataset_valid = dataloaders.GenRecoDataset_Toponium(disc_valid_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=False,
                                                   mask_probability=args.mask_prob)
        dataset_test = dataloaders.GenRecoDataset_Toponium(disc_test_events, gen_reco_ids,
                                                   standardize=args.standardize,
                                                   pretraining=False,
                                                   mask_probability=args.mask_prob)
    print('DATA CREATED')
    # Prepare the model.
    #weight calculation
    total_num_samples = num_negatives + num_positives
    class_0_weight = total_num_samples / (num_negatives * 2)
    class_1_weight = total_num_samples / (num_positives * 2)
    weight = torch.tensor([class_0_weight, class_1_weight])
    pos_weight = torch.tensor([class_1_weight / class_0_weight])
    pos_weight = pos_weight.to(device)
    #criterion = nn.BCEWithLogitsLoss(weight=weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #criterion = nn.BCEWithLogitsLoss()
    
    
    model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model,
                       d_ff= int(4 * args.d_model), h = args.h, dropout= args.dropout, act_fn = args.act_fn)
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.save_dir + f'bumblebee{trial.number}.pt'))
    else:
        model.load_state_dict(torch.load(args.save_dir + f'bumblebee{trial.number}.pt'))
    if torch.cuda.device_count() > 1:
        print(f"USING {torch.cuda.device_count()} GPUS")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-4, betas=(0.9, 0.98), eps=1e-9,
                                 weight_decay=args.weight_decay)
    scheduler = NoamOpt(args.d_model, lr_mult=1.0, warmup=args.warmup, optimizer=optimizer)
    
    # Create data loaders
    # Data Loader
    data_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    print('DATALOADERS CREATED')
    
    # Loss Arrays
    training_loss_components = []
    training_losses = []
    masked_training_losses = []
    unmasked_training_losses = []
    
    validation_losses = []
    masked_validation_losses = []
    unmasked_validation_losses = []
    validation_loss_components = []
    
    best_validation_loss = -1
    
    test_losses = []
    masked_test_losses = []
    test_loss_components = []
    test_predictions = []
    test_truths = []
    
    print('TRAINING STARTED')
    # Train the model
    for t in range(5):
        model.train()
        print(f"EPOCH {t}:")
        for batch_id, (x, target) in tqdm(enumerate(data_loader), desc="TRAINING"):
            # Training Set
            x = x.to(device)
            target = target.to(device)
    
            scheduler.zero_grad()
            wandb.log({'disc_lr': scheduler.get_cur_lr()})
            y_pred = model(x)
    
            etas = y_pred[:, :, 1]
    
            phis = y_pred[:, :, 2]
    
            corrected_pred = torch.cat((y_pred[:, :, 0][:, :, None], etas[:, :, None],
                                        phis[:, :, None], y_pred[:, :, 3][:, :, None]), axis=2)
    
            disc_loss = criterion(corrected_pred[:, 0, 0], target.float())
            
            disc_loss.backward()
            
            if batch_id % 15 == 0:
                wandb.log({'disc_loss': disc_loss.item()})
    
            # append to lists and detach
            training_losses.append(disc_loss.detach().cpu().numpy())
            scheduler.step_and_update()
        # Validate
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            unmasked_valid_loss = 0
            masked_valid_loss = 0
            avg_valid_loss = 0
            valid_step_number = 0
            valid_loss_components = torch.zeros(component_loss.shape).to(device)
            if args.standardize:
                max_pt = 10
            else:
                max_pt = 1000
            h_gen_nu_pt = hist.Hist(hist.axis.Regular(100, -10, max_pt))
            h_gen_nu_eta = hist.Hist(hist.axis.Regular(100, -10, 10))
            h_gen_nu_phi = hist.Hist(hist.axis.Regular(100, -3.14, 3.14))
            h_unmasked_reco_nu_pt = hist.Hist(hist.axis.Regular(100, -10, max_pt))
            h_unmasked_reco_nu_eta = hist.Hist(hist.axis.Regular(100, -10, 10))
            h_unmasked_reco_nu_phi = hist.Hist(hist.axis.Regular(100, -3.14, 3.14))
            h_masked_reco_nu_pt = hist.Hist(hist.axis.Regular(100, -10, max_pt))
            h_masked_reco_nu_eta = hist.Hist(hist.axis.Regular(100, -10, 10))
            h_masked_reco_nu_phi = hist.Hist(hist.axis.Regular(100, -3.14, 3.14))
    
            for valid_batch_id, (x_valid, target_valid) in tqdm(enumerate(valid_data_loader), desc="VALIDATING"):
                x_valid = x_valid.to(device)
                target_valid = target_valid.to(device)
                target_pred_valid = model(x_valid)
    
                # ask ML model to predict tan(theta) then convert to eta
                # valid_etas = -1 * torch.log(torch.abs(y_valid[:, :, 1]) + 0.01)
                valid_etas = target_pred_valid[:, :, 1]
    
                # similarly ask ML model to predict tan(phi) then convert to phi
                valid_phis = target_pred_valid[:, :, 2]
    
                corrected_y_valid = torch.cat((target_pred_valid[:, :, 0][:, :, None], valid_etas[:, :, None], valid_phis[:, :, None],
                                               target_pred_valid[:, :, 3][:, :, None]), axis=2)
    
                valid_disc_loss = criterion(corrected_y_valid[:, 0, 0], target_valid.float())
                avg_valid_loss += valid_disc_loss.item()
                if valid_batch_id % 5 == 0:
                    wandb.log({"disc_validation_loss": valid_disc_loss.item()})
    
            avg_valid_loss /= (valid_batch_id+1)
            wandb.log({"disc_avg_valid_loss": avg_valid_loss})
    
            # early stopping
            if avg_valid_loss >= best_validation_loss != -1:
                break
    
            if avg_valid_loss < best_validation_loss or best_validation_loss == -1:
                best_validation_loss = avg_valid_loss
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), args.save_dir + f'bumblebee_cls{trial.number}.pt')
                else:
                    torch.save(model.state_dict(), args.save_dir + f'bumblebee_cls{trial.number}.pt')
    
            
    # There's a bunch of stuff related to plotting here in the notebook. I'm
    # omitting it here, we'll have to figure out exactly how to handle output.
    
    # Model testing
    # Also copied from the notebook.
    best_model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model, d_ff= int(4 * args.d_model),
                            h = args.h, dropout= args.dropout, act_fn=args.act_fn)
    if torch.cuda.device_count() > 1:
        best_model = nn.DataParallel(best_model)
    best_model = best_model.to(device)
    if torch.cuda.device_count() > 1:
        best_model.module.load_state_dict(torch.load(args.save_dir + f'bumblebee_cls{trial.number}.pt'))
    else:
        best_model.load_state_dict(torch.load(args.save_dir + f'bumblebee_cls{trial.number}.pt'))
    
    best_model.eval()
    avg_test_loss = 0
    for test_batch_id, (x_test, target_test) in tqdm(enumerate(test_data_loader), desc="TESTING"):
        x_test = x_test.to(device)
        target_test = target_test.to(device)
        target_pred_test = best_model(x_test)
    
        test_etas = target_pred_test[:, :, 1]
    
        test_phis = target_pred_test[:, :, 2]
    
        corrected_y_test = torch.cat(
            (target_pred_test[:, :, 0][:, :, None], test_etas[:, :, None], test_phis[:, :, None],
             target_pred_test[:, :, 3][:, :, None]), axis=2)
    
    
        test_loss = criterion(corrected_y_test[:, 0, 0], target_test.float())
    
        avg_test_loss += test_loss.item()
        wandb.log({'disc_test_loss': test_loss})
    
        test_losses.append(test_loss.detach().cpu().numpy())
        test_predictions.append(torch.sigmoid(corrected_y_test[:, 0, 0]).detach().cpu().numpy())
        test_truths.append(target_test.detach().cpu().numpy())
    
    avg_test_loss /= (test_batch_id + 1)
    wandb.log({'disc_avg_test_loss': avg_test_loss})
    
    
    # Save the data. This section will likely need some augmentation.
    
    # Test
    torch.save(np.array(test_losses), args.save_dir + f'test_losses{trial.number}.pt')
    torch.save(np.array(test_predictions), args.save_dir + f'predictions{trial.number}.pt')
    torch.save(np.array(test_truths), args.save_dir + f'truths{trial.number}.pt')
    
    # PLOTTING
    predictions = np.array(test_predictions)
    truths = np.array(test_truths)
    total_predictions = []
    total_truths = []
    ttbar_predictions = []
    toponium_predictions = []
    
    for i in tqdm(range(len(predictions))):
        for j in range(len(predictions[i])):
            total_predictions.append(predictions[i][j])
            total_truths.append(truths[i][j])
    
    for i in range(len(total_truths)):
        if total_truths[i] == 0.0:
            ttbar_predictions.append(total_predictions[i])
        elif total_truths[i] == 1.0:
            toponium_predictions.append(total_predictions[i])
    
    n_bins = 30
    fig = plt.figure(figsize=(10, 8))
    plt.hist(x=[ttbar_predictions, toponium_predictions], bins=n_bins, color=['red', 'blue'], histtype='step', label=['ttbar', 'toponium'])
    plt.legend()
    plt.yscale('log')
    plt.savefig(args.save_dir + f'score_histogram{trial.number}.png')
    plt.cla()
    plt.clf()
    plt.close(fig)
    
    
    fpr, tpr, thresholds = roc_curve(y_true=total_truths, y_score=total_predictions)
    roc_auc = auc(fpr, tpr)
    wandb.log({"AUROC": roc_auc})
    fig = plt.figure(figsize = (10, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(args.save_dir + f'ROC_curve{trial.number}.png')
    plt.cla()
    plt.clf()
    plt.close(fig)
    
    wandb.finish()
    ### AND WE'RE DONE!!!
    return roc_auc

if __name__ == '__main__':
    study = optuna.create_study(study_name='Bumblebee Tuning', direction = 'maximize')
    study.optimize(train, n_trials=50)