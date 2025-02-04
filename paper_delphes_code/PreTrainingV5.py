"""
This is a single script that defines, trains, and tests Bumblebee,
a transformer for particle physics event reconstruction.

Draft 1 completed: Oct. 15, 2022

@version: 1.0
@author: AJ Wildridge Ethan Colbert,
modified by Jack P. Rodgers
"""

import os
import wandb
import matplotlib
import numpy as np
import sklearn
import scipy
import torch
import awkward as ak
import uproot
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models import make_model
import argparse
import mplhep as hep
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, Dataset
import itertools
import sys
import random
torch.set_printoptions(threshold=10000)
G = torch.Generator()
G.manual_seed(23)
hep.style.use(hep.style.CMS)
e_mass = 0.0005109989461
mu_mass = 0.105658375

#wandb.login(key='8b998ffdd7e214fa724dd5cf67eafb36b111d2a7')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
START_SEQ_VEC = [1, 0, 0, 0]
SEPARATOR_SEQ_VEC = [0, 1, 0, 0]
END_SEQ_VEC = [0, 0, 1, 0]

def merge_events(events):
    total_events = {}
    for k in events[0].keys():
        total_events[k] = np.concatenate(list(e[k] for e in events))
    return total_events


def get_stats_2d_jagged(arr):
    flat_arr = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            flat_arr.append(arr[i][j])
    flat_arr = np.array(flat_arr)
    return np.mean(flat_arr), np.std(flat_arr)

def get_grad_norm(model):
    parameters = model.parameters()
    norm_type = 2
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach().cpu(), norm_type) for p in parameters]), norm_type)
    return total_norm.item()


class MiniDataset(Dataset):
    def __init__(self, group, mask_prob, pretraining):
        self.group = group
        self.mask_prob = mask_prob
        self.pretraining = pretraining

    def __len__(self):
        return len(self.group)

    def __getitem__(self, item):
        R1, C1 = self.group[item][2].shape
        R2, C2 = self.group[item][3].shape
        self.mask_prob = 1.0 / (R1+R2)
        four_vector = torch.cat((
            self.group[item][2],
            self.group[item][3]
        ), dim=0)
        if self.pretraining:
            task_roll = torch.rand(1)
            if task_roll < 0.5:
                dice_rolls = torch.rand(R1+R2)
                zerod_mask = ~(dice_rolls < self.mask_prob)
                zerod_mask = torch.cat(
                    (zerod_mask[:R1], zerod_mask[R1:]),
                    dim=0
                )
            elif 0.5 <= task_roll < 0.75: # mask all gen
                dice_roll = torch.rand(1)
                if dice_roll < 0.5:
                    zerod_mask = torch.cat((torch.ones(R1), torch.zeros(R2)))
                else:
                    zerod_mask = torch.ones(R1+R2)
                #zerod_mask = torch.cat((torch.ones(R1), torch.zeros(R2)))
            elif 0.75 <= task_roll < 1.0: # mask all reco
                
                dice_roll = torch.rand(1)
                if dice_roll < 0.5:
                    zerod_mask = torch.cat((torch.zeros(R1), torch.ones(R2)))
                else:
                    zerod_mask = torch.ones(R1+R2)
                #zerod_mask = torch.cat((torch.zeros(R1), torch.ones(R2)))
        else:
            zerod_mask = torch.cat((torch.ones(R1), torch.zeros(R2)))
        four_vector_mask = zerod_mask[:, None].repeat(1, 4)
        # MET / Neutrino masking
        four_vector_mask[2, 1] = torch.ones(four_vector_mask[2, 1].shape)
        four_vector_mask [2, 3] = torch.ones(four_vector_mask[2, 3].shape)
        four_vector_mask [-2:0, 1] = torch.ones(four_vector_mask[-2:0, 1].shape)
        four_vector_mask [-2:0, 3] = torch.ones(four_vector_mask[-2:0, 3].shape)
        masked_four_vector = (four_vector * four_vector_mask)
        return torch.cat((self.group[item][0][:, None], self.group[item][1][:, None], masked_four_vector, zerod_mask[:, None]), dim=1), four_vector


def create_data(events, standardize, pretraining, mask_probability, batch_size, tag):
    # set our seed
    torch.manual_seed(42)

    # Define PDG/RECO IDs
    MET_ID = 40
    START_SEQ_ID = 50
    SEPARATOR_ID = 51
    END_SEQ_ID = 52
    b_PDG_ID = 5
    non_b_jet_PDG_ID = 41
    bbar_PDG_ID = -5

    # making base reco and gen 4 vectors as normal
    num_events = len(events['l_pt'])
    
    reco_four_vectors = torch.cat(
        (
            torch.cat(
                (
                    torch.from_numpy(events['l_pt'])[:, None],
                    torch.from_numpy(events['lbar_pt'])[:, None],
                    torch.from_numpy(events['met_pt'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_eta'])[:, None],
                    torch.from_numpy(events['lbar_eta'])[:, None],
                    torch.zeros(num_events)[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_phi'])[:, None],
                    torch.from_numpy(events['lbar_phi'])[:, None],
                    torch.from_numpy(events['met_phi'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_mass'])[:, None],
                    torch.from_numpy(events['lbar_mass'])[:, None],
                    torch.zeros(num_events)[:, None]
                ),
                dim=1
            )[:, :, None]
        ),
        dim=2
    )
    # NOTE: gen llbar eta, phi = reco llbar eta, phi
    gen_four_vectors = torch.cat(
        (
            torch.cat(
                (
                    torch.from_numpy(events['gen_l_pt'])[:, None],
                    torch.from_numpy(events['gen_lbar_pt'])[:, None],
                    torch.from_numpy(events['gen_b_pt'])[:, None],
                    torch.from_numpy(events['gen_bbar_pt'])[:, None],
                    torch.from_numpy(events['gen_nu_pt'])[:, None],
                    torch.from_numpy(events['gen_nubar_pt'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['gen_l_eta'])[:, None],
                    torch.from_numpy(events['gen_lbar_eta'])[:, None],
                    torch.from_numpy(events['gen_b_eta'])[:, None],
                    torch.from_numpy(events['gen_bbar_eta'])[:, None],
                    torch.from_numpy(events['gen_nu_eta'])[:, None],
                    torch.from_numpy(events['gen_nubar_eta'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['gen_l_phi'])[:, None],
                    torch.from_numpy(events['gen_lbar_phi'])[:, None],
                    torch.from_numpy(events['gen_b_phi'])[:, None],
                    torch.from_numpy(events['gen_bbar_phi'])[:, None],
                    torch.from_numpy(events['gen_nu_phi'])[:, None],
                    torch.from_numpy(events['gen_nubar_phi'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['gen_l_mass'])[:, None],
                    torch.from_numpy(events['gen_lbar_mass'])[:, None],
                    torch.from_numpy(events['gen_b_mass'])[:, None],
                    torch.from_numpy(events['gen_bbar_mass'])[:, None],
                    torch.zeros(num_events)[:, None],
                    torch.zeros(num_events)[:, None]
                ),
                dim=1
            )[:, :, None]
        ),
        dim=2
    )

    ids = torch.cat(
        (
            torch.from_numpy(events['l_pdgid'][:, None]),
            torch.from_numpy(events['lbar_pdgid'][:, None]),
            torch.Tensor([MET_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(events['gen_l_pdgid'][:, None]),
            torch.from_numpy(events['gen_lbar_pdgid'][:, None]),
            torch.Tensor([b_PDG_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([bbar_PDG_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(-1 * events['gen_l_pdgid'][:, None] - 1),  # corresponding antineutrino
            torch.from_numpy(-1 * events['gen_lbar_pdgid'][:, None] + 1)  # corresponding neutrino
        ),
        dim=1
    )

    if standardize:
        reco_means = torch.mean(reco_four_vectors, dim=0)
        reco_means[:, 2] = torch.zeros(reco_means[:, 2].shape)
        reco_means[:, 0] = torch.zeros(reco_means[:, 0].shape)
        reco_stdevs = torch.std(reco_four_vectors, 0, True)
        reco_stdevs[:, 2] = torch.ones(reco_stdevs[:, 2].shape)
        reco_stdevs[:, 0] = torch.ones(reco_stdevs[:, 0].shape)
        gen_means = torch.mean(gen_four_vectors, dim=0)
        gen_means[:, 2] = torch.zeros(gen_means[:, 2].shape)
        gen_means[:, 0] = torch.zeros(gen_means[:, 0].shape)
        gen_stdevs = torch.std(gen_four_vectors, 0, True)
        gen_stdevs[:, 2] = torch.ones(gen_stdevs[:, 2].shape)
        gen_stdevs[:, 0] = torch.ones(gen_stdevs[:, 0].shape)
        reco_four_vectors = (reco_four_vectors - reco_means) / reco_stdevs
        gen_four_vectors = (gen_four_vectors - gen_means) / gen_stdevs
        reco_four_vectors[:, :, 0] = torch.log(reco_four_vectors[:, :, 0])
        gen_four_vectors[:, :, 0] = torch.log(gen_four_vectors[:, :, 0])
        # standardize variable jets
        j_pt_mean, j_pt_std = get_stats_2d_jagged(events['jet_pt'])
        j_eta_mean, j_eta_std = get_stats_2d_jagged(events['jet_eta'])
        j_phi_mean, j_phi_std = get_stats_2d_jagged(events['jet_phi'])
        j_mass_mean, j_mass_std = get_stats_2d_jagged(events['jet_mass'])
        jet_mean_tensor = torch.tensor([0.0, j_eta_mean, 0.0, j_mass_mean])
        jet_std_tensor = torch.tensor([1.0, j_eta_std, 1.0, j_mass_std])
        # Entries set to 0 (MET mass/eta) and values with 0 stdev go to nan
        reco_four_vectors = torch.nan_to_num(reco_four_vectors)
        gen_four_vectors = torch.nan_to_num(gen_four_vectors)

    if tag == 'test':
        torch.save(np.array(reco_means), args.save_dir + 'reco_means.pt')
        torch.save(np.array(reco_stdevs), args.save_dir + 'reco_stdevs.pt')
        torch.save(np.array(gen_means), args.save_dir + 'gen_means.pt')
        torch.save(np.array(gen_stdevs), args.save_dir + 'gen_stdevs.pt')
        torch.save(np.array(jet_mean_tensor), args.save_dir + 'jet_means.pt')
        torch.save(np.array(jet_std_tensor), args.save_dir + 'jet_stdevs.pt')

    for i in range(len(events['jet_pt'])):
        if len(events['jet_btag'][i]) != len(events['jet_pt'][i]):
            print('ahhh')
    # creating jet tensor
    B, R, C = reco_four_vectors.shape
    recos_w_jets = []
    ids_w_jets = []
    gen_reco_ids_w_jets = []
    for i in tqdm(range(B), desc="EDITING FOR JETS"):
        # editing gen_reco ids
        gen_reco_id = [0, 0, 0, 1, 1, 1, 1, 1, 1]
        add_gen_reco_id = [0 for _ in range(len(events['jet_pt'][i]))]
        gen_reco_id = add_gen_reco_id + gen_reco_id
        gen_reco_ids_w_jets.append(torch.tensor(gen_reco_id, dtype=torch.int))
        # editing reco 4 vectors by adding jet tensor
        extra_jet_pt = torch.log(torch.from_numpy(events['jet_pt'][i]))[:, None]
        extra_jet_eta = torch.from_numpy(events['jet_eta'][i])[:, None]
        extra_jet_phi = torch.from_numpy(events['jet_phi'][i])[:, None]
        extra_jet_mass = torch.from_numpy(events['jet_mass'][i])[:, None]

        jet_tensor = torch.cat(
            (extra_jet_pt, extra_jet_eta, extra_jet_phi, extra_jet_mass), dim=1)
        jet_cur_mean_tensor = jet_mean_tensor.expand_as(jet_tensor)
        jet_cur_std_tensor = jet_std_tensor.expand_as(jet_tensor)
        jet_tensor = (jet_tensor - jet_cur_mean_tensor) / jet_cur_std_tensor
        recos_w_jets.append(torch.cat(
            (reco_four_vectors[i], jet_tensor), dim=0))
        # editing ids
        jet_ids = events['jet_btag'][i]
        jet_ids[jet_ids == 1] = b_PDG_ID
        jet_ids[jet_ids == 0] = non_b_jet_PDG_ID
        new_id = torch.cat((ids[i][:4], torch.tensor(jet_ids), ids[i][4:]), dim=0)
        new_id += 40
        ids_w_jets.append(new_id.type(torch.int))
    # putting data together
    unique_sizes = [[] for _ in range(100)]
    for i in tqdm(range(B), desc="CONCAT AND MASKING"):
        D, _ = recos_w_jets[i].shape
        unique_sizes[D-5].append([ids_w_jets[i], gen_reco_ids_w_jets[i], recos_w_jets[i], gen_four_vectors[i]])
    t_dataloaders = []
    unique_sizes = [x for x in unique_sizes if x != []]
    for i in tqdm(range(len(unique_sizes)), desc="MAKING DATASETS"):
        t_dataset = MiniDataset(unique_sizes[i], mask_prob=mask_probability, pretraining=pretraining)
        t_dataloaders.append(DataLoader(t_dataset, batch_size=batch_size, shuffle=True))
    return t_dataloaders


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
class BERTOpt:
    def __init__(self, start_lr, peak_lr, warmup, optimizer, lr_mult = 1.0):
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.warmup = warmup
        self.optimizer = optimizer
        self.lr_mult = lr_mult
        self.steps = 0
        self.warmup_factor = math.pow((peak_lr/start_lr), (1/warmup))

    def step_and_update(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        step = self.steps
        if step < self.warmup:
            return self.warmup_factor
        elif self.start_lr < self.get_cur_lr() < self.peak_lr and step > self.warmup:
            return 0.99999
        else:
            return 1.

    def get_cur_lr(self):
        clr = None
        for p in self.optimizer.param_groups:
            clr = p['lr']
        return clr

    def update_learning_rate(self):
        self.steps += 1
        lr = self.get_cur_lr() * self.lr_mult * self.get_lr_scale()
        for p in self.optimizer.param_groups:
            p['lr'] = lr


class CombinedLoss(nn.Module):
    def __init__(self, reduction = 'none', device=torch.device('cpu')):
        super(CombinedLoss, self).__init__()
        self.da_mask = None
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, targets):
        self.da_mask = torch.ones(targets.shape, dtype=torch.bool)
        self.da_mask = self.da_mask.to(device)
        self.da_mask[:, :, 2] = torch.zeros(self.da_mask[:, :, 2].shape)
        ret = (self.da_mask * torch.pow(input=(inputs-targets), exponent=2)) + (~self.da_mask * (2. - (2. * torch.cos(input=(inputs - targets)))))
        if self.reduction == 'none':
            pass
        elif self.reduction == 'sum':
            ret = torch.sum(ret)
        elif self.reduction == 'mean':
            ret = torch.mean(ret)
        return ret

    def get_da_mask(self):
        return self.da_mask


def exclude_negatives(events):
    keys = list(events.keys())
    extras = events['jet_pt']
    excludes = [i for i in range(len(extras)) if np.min(extras[i]) <= 0.0]
    for key in keys:
        events[key] = np.delete(events[key], excludes)
    return events


def exclude_taus(events):
    keys = list(events.keys())
    taulist = events['notau_flag']
    excludes = [i for i in range(len(taulist)) if not taulist[i]]
    for key in keys:
        events[key] = np.delete(events[key], excludes)
    return events


def correct_leptons(events):
    l_pdgs = events['l_pdgid']
    lbar_pdgs = events['lbar_pdgid']
    l_mass = []
    lbar_mass = []
    for i in tqdm(range(len(l_pdgs)),desc="Correcting Lepton Masses"):
        if l_pdgs[i] == 11:
            l_mass.append(e_mass)
        elif l_pdgs[i] == 13:
            l_mass.append(mu_mass)
        else:
            print(l_pdgs[i])
        if lbar_pdgs[i] == -11:
            l_mass.append(e_mass)
        elif lbar_pdgs[i] == -13:
            lbar_mass.append(mu_mass)
        else:
            print(lbar_pdgs[i])
    l_mass = np.array(l_mass)
    lbar_mass = np.array(lbar_mass)
    events['l_mass'] = l_mass
    events['lbar_mass'] = lbar_mass
    return events


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
parser.add_argument('--data_dir', help='directory where data reconstruction data is stored', type=str)
parser.add_argument('--save_dir', help='directory to be saved on cluster', type=str)
parser.add_argument('--act_fn', help='activation function', type=str)
parser.add_argument('--weight_decay', help = 'coefficient for L2 regularization', type=float)
parser.add_argument('--epsilon', help = 'epsilon for Adam', type=float)
parser.set_defaults(d_model=256, dropout = 0.1,
                    batch_size = 128, n_epochs = 5, only_gen = False,
                    mask_prob = 0.09, standardize = True,
                    tvt_split = [0.7, 0.15, 0.15], lossf = 'CL',
                    N = 8, h = 16, warmup=10000,
                    data_dir = r'/depot/cms/top/colberte/DelphesProduction/Processing/delphes_ttbar_sample_old2/',
                    save_dir = r'/depot/cms/top/jprodger/Bumblebee/src/Paper_Delphes_Stuff/pretraining/output/',
                    act_fn = 'gelu', weight_decay = 0, epsilon = 1e-6)

args = parser.parse_args()

# dataset creation and preparation


notau_filenames1 = {
    f'{args.data_dir}/delphes_ttbar_minitree_*': f'Step7;1'
}

selected_keys = [
        'l_pt', 'l_eta', 'l_phi', 'l_mass', 'l_pdgid',
        'lbar_pt', 'lbar_eta', 'lbar_phi', 'lbar_mass', 'lbar_pdgid',
        'jet_pt', 'jet_eta', 'jet_phi', 'jet_mass',
        'met_pt', 'met_phi',
        'gen_l_pt', 'gen_l_eta', 'gen_l_phi', 'gen_l_mass', 'gen_l_pdgid',
        'gen_lbar_pt', 'gen_lbar_eta', 'gen_lbar_phi', 'gen_lbar_mass', 'gen_lbar_pdgid',
        'gen_b_pt', 'gen_b_eta', 'gen_b_phi', 'gen_b_mass',
        'gen_bbar_pt', 'gen_bbar_eta', 'gen_bbar_phi', 'gen_bbar_mass',
        'gen_nu_pt', 'gen_nu_eta', 'gen_nu_phi',
        'gen_nubar_pt', 'gen_nubar_eta', 'gen_nubar_phi',
        'jet_btag', 'gluon_initiated', 'notau_flag'
    ]
events1 = uproot.concatenate(
    notau_filenames1,
    selected_keys,
    library='numpy',
    cut='notau_flag == True'
)

#events1 = exclude_negatives(events1)
events = [events1]

total_events = merge_events(events)
print('DATA CREATED')

train_events, valid_events, test_events = train_valid_test(total_events, args.tvt_split, use_generator = False)

torch.save(test_events, args.save_dir + 'test_data.pt')

print("DATA PRESENT")
print("Counts: ", len(train_events['l_pt']), len(valid_events['l_pt']), len(test_events['l_pt']))

valid_data_loaders = create_data(events=valid_events, standardize=args.standardize,
            pretraining=False, mask_probability=args.mask_prob, batch_size=args.batch_size, tag = 'valid')
test_data_loaders = create_data(events=test_events, standardize=args.standardize,
            pretraining=False, mask_probability=args.mask_prob, batch_size=args.batch_size, tag = 'test')
print('DATALOADERS CREATED')
# Prepare the model.
if args.lossf == 'L1':
    criterion = nn.L1Loss(reduction='mean')
elif args.lossf == 'MSE':
    criterion = nn.MSELoss(reduction='mean')
elif args.lossf == 'CL':
    criterion = CombinedLoss(reduction='mean', device = device)

model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model,
                   d_ff= int(4 * args.d_model), h = args.h, dropout= args.dropout, act_fn = args.act_fn)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Param count: {pytorch_total_params}")
if torch.cuda.device_count() > 1:
    print(f"USING {torch.cuda.device_count()} GPUS")
    model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-6, betas=(0.9, 0.999), eps=args.epsilon,
                             weight_decay=args.weight_decay)
scheduler = BERTOpt(start_lr=1e-6, peak_lr=1e-4, warmup=args.warmup, optimizer=optimizer)

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
    "epsilon": args.epsilon
}

# wandb initialization
wandb.init(
    project='Bumblebee_DP_pretraining',
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

best_validation_loss = -1

test_losses = []
masked_test_losses = []
test_loss_components = []
test_predictions = []
test_truths = []
test_reco_tops = []

train_steps = 0
valid_steps = 0
test_steps = 0


print('TRAINING STARTED')
# Train the model
for t in range(args.n_epochs):
    train_data_loaders = create_data(events=train_events, standardize=args.standardize,
            pretraining=True, mask_probability=args.mask_prob, batch_size=args.batch_size, tag = 'train')
    model.train()
    print(f"EPOCH {t}:")
    train_steps = 0
    random.shuffle(train_data_loaders)
    for data_loader in train_data_loaders:
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
                                        phis[:, :, None], y_pred[:, :, 3][:, :, None]), dim=2)

            out = criterion(corrected_pred, target.float())

            zerod_mask = x[:, :, -1]
            four_vector_mask = zerod_mask[:, :, None].repeat(1, 1, 4)

            # MET / Neutrino masking
            four_vector_mask[:, 2, 1] = torch.ones(four_vector_mask[:, 2, 1].shape)
            four_vector_mask[:, 2, 3] = torch.ones(four_vector_mask[:, 2, 3].shape)
            four_vector_mask[:, -2:0, 1] = torch.ones(four_vector_mask[:, -2:0, 1].shape)
            four_vector_mask[:, -2:0, 3] = torch.ones(four_vector_mask[:, -2:0, 3].shape)

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
            train_steps += 1
    valid_steps = 0
    # Validate
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        unmasked_valid_loss = 0
        masked_valid_loss = 0
        avg_masked_valid_loss = 0
        random.shuffle(valid_data_loaders)
        for valid_data_loader in valid_data_loaders:
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

                # MET / Neutrino masking
                four_vector_mask_valid[:, 2, 1] = torch.ones(four_vector_mask_valid[:, 2, 1].shape)
                four_vector_mask_valid[:, 2, 3] = torch.ones(four_vector_mask_valid[:, 2, 3].shape)
                four_vector_mask_valid[:, -2:0, 1] = torch.ones(four_vector_mask_valid[:, -2:0, 1].shape)
                four_vector_mask_valid[:, -2:0, 3] = torch.ones(four_vector_mask_valid[:, -2:0, 3].shape)

                # don't include losses which were 0 because of being unmasked in masked loss average
                masked_valid_out = valid_out * ~four_vector_mask_valid.type(torch.bool).to(device)
                masked_valid_out = torch.sum(torch.sum(masked_valid_out, dim=1), dim=1)
                masked_valid_out = masked_valid_out[masked_valid_out != 0]

                # don't include losses which were 0 because of being masked in unmasked loss average
                unmasked_valid_out = valid_out * four_vector_mask_valid.type(torch.bool).to(device)
                unmasked_valid_out = torch.sum(torch.sum(unmasked_valid_out, dim=1), dim=1)
                unmasked_valid_out = unmasked_valid_out[unmasked_valid_out != 0]

                # look to see if training objective on masking is being learned
                masked_valid_loss = torch.mean(masked_valid_out, dim=0)
                unmasked_valid_loss = torch.mean(unmasked_valid_out, dim=0)
                valid_loss = torch.mean(torch.sum(valid_out), dim=0)
                avg_masked_valid_loss += masked_valid_loss.item()

                masked_validation_losses.append(masked_valid_loss.detach().cpu().numpy())
                unmasked_validation_losses.append(unmasked_valid_loss.detach().cpu().numpy())
                validation_losses.append(valid_loss.detach().cpu().numpy())

                valid_steps += 1
                if valid_batch_id % 5 == 0:
                    wandb.log({"masked_validation_loss": masked_valid_loss.item()})

        avg_masked_valid_loss /= valid_steps
        wandb.log({"avg_masked_valid_loss": avg_masked_valid_loss})

        # early stopping
        if avg_masked_valid_loss >= best_validation_loss != -1:
            break

        if avg_masked_valid_loss < best_validation_loss or best_validation_loss == -1:
            best_validation_loss = avg_masked_valid_loss
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), args.save_dir + 'bumblebee.pt')
            else:
                torch.save(model.state_dict(), args.save_dir + 'bumblebee.pt')

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
    best_model.module.load_state_dict(torch.load(args.save_dir + 'bumblebee.pt'))
else:
    best_model.load_state_dict(torch.load(args.save_dir + 'bumblebee.pt'))

best_model.eval()
avg_masked_test_loss = 0
random.shuffle(test_data_loaders)
for test_data_loader in test_data_loaders:
    for test_batch_id, (x_test, target_test) in tqdm(enumerate(test_data_loader), desc="TESTING"):
        x_test = x_test.to(device)
        target_test = target_test.to(device)
        target_pred_test = best_model(x_test)

        test_etas = target_pred_test[:, :, 1]

        test_phis = target_pred_test[:, :, 2]

        corrected_y_test = torch.cat(
            (target_pred_test[:, :, 0][:, :, None], test_etas[:, :, None], test_phis[:, :, None],
             target_pred_test[:, :, 3][:, :, None]), axis=2)

        test_out = criterion(corrected_y_test, target_test.float())

        zerod_mask_test = x_test[:, :, -1]
        four_vector_mask_test = zerod_mask_test[:, :, None].repeat(1, 1, 4)

        # MET / Neutrino masking
        four_vector_mask_test[:, 2, 1] = torch.ones(four_vector_mask_test[:, 2, 1].shape)
        four_vector_mask_test[:, 2, 3] = torch.ones(four_vector_mask_test[:, 2, 3].shape)
        four_vector_mask_test[:, -2:0, 1] = torch.ones(four_vector_mask_test[:, -2:0, 1].shape)
        four_vector_mask_test[:, -2:0, 3] = torch.ones(four_vector_mask_test[:, -2:0, 3].shape)

        masked_test_out = test_out * ~four_vector_mask_test.type(torch.bool).to(device)
        masked_test_out = torch.sum(torch.sum(masked_test_out, axis=1), axis=1)
        masked_test_out = masked_test_out[masked_test_out != 0]

        masked_test_loss = torch.mean(masked_test_out, dim=0)
        test_loss = torch.mean(torch.sum(test_out), dim=0)
        avg_masked_test_loss += masked_test_loss.item()
        wandb.log({'masked_test_loss': masked_test_loss})

        test_loss_components.append(torch.mean(test_out, dim=0).detach().cpu().numpy())
        test_losses.append(test_loss.detach().cpu().numpy())
        masked_test_losses.append(masked_test_loss.detach().cpu().numpy())
        test_predictions.append(corrected_y_test.detach().cpu().numpy())
        test_truths.append(target_test.detach().cpu().numpy())

        test_steps += 1

avg_masked_test_loss /= test_steps
wandb.log({'avg_masked_test_loss': avg_masked_test_loss})



# Save the data. This section will likely need some augmentation.
# Training
#torch.save(np.array(masked_training_losses), args.save_dir + 'used_training_losses.pt')
#torch.save(np.array(training_losses), args.save_dir + 'net_training_losses.pt')
#torch.save(np.array(training_loss_components), args.save_dir + 'training_loss_components.pt')

# Validation
#torch.save(np.array(validation_losses), args.save_dir + 'validation_losses.pt')
#torch.save(np.array(masked_validation_losses), args.save_dir + 'masked_validation_losses.pt')

# Test
#torch.save(np.array(test_losses), args.save_dir + 'test_losses.pt')
#torch.save(np.array(masked_test_losses), args.save_dir + 'masked_test_losses.pt')
#torch.save(np.array(test_loss_components), args.save_dir + 'test_loss_components.pt')
torch.save(np.array(test_predictions, dtype="object"), args.save_dir + 'predictions.pt')
torch.save(np.array(test_truths, dtype="object"), args.save_dir + 'truths.pt')
#torch.save(np.array(test_reco_tops, dtype="object"), args.save_dir + 'reco_tops.pt')

wandb.finish()
### AND WE'RE DONE!!!
