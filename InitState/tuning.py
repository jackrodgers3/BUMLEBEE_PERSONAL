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
from torch.utils.data import DataLoader, Dataset
import itertools
import sys
import random
import optuna
from sklearn.metrics import auc, roc_curve
G = torch.Generator()
G.manual_seed(23)
hep.style.use(hep.style.CMS)

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


class MiniDataset_Discrimination(Dataset):
    def __init__(self, group):
        self.group = group

    def __len__(self):
        return len(self.group)

    def __getitem__(self, item):
        R1, C1 = self.group[item][2].shape
        R2, C2 = self.group[item][3].shape
        self.mask_prob = 1.0 / (R1+R2)
        four_vector = torch.cat((
            torch.Tensor([START_SEQ_VEC]),
            self.group[item][2],
            torch.Tensor([SEPARATOR_SEQ_VEC]),
            self.group[item][3],
            torch.Tensor([END_SEQ_VEC])
        ), dim=0)
        
        zerod_mask = torch.ones(R1+R2+3)
        
        return torch.cat((self.group[item][0][:, None], self.group[item][1][:, None], four_vector, zerod_mask[:, None]), dim=1), self.group[item][4]
        
def create_data_disc(events, standardize, batch_size, tag):
    # set our seed
    torch.manual_seed(42)

    # Define PDG/RECO IDs
    MET_ID = 40
    START_SEQ_ID = 50
    SEPARATOR_ID = 51
    END_SEQ_ID = 52
    b_PDG_ID = 5
    other_jet_PDG_ID = 6
    bbar_PDG_ID = -5

    MEDIUM_B_TAG_KEY = {'2016pre': 0.2598, '2016post': 0.2489, '2017': 0.3040, '2018': 0.2783}

    # making base reco and gen 4 vectors as normal
    num_events = len(events['l_pt'])
    
    reco_top_info = torch.cat(
        (
            torch.cat(
                (
                    torch.from_numpy(events['top_pt'])[:, None],
                    torch.from_numpy(events['tbar_pt'])[:, None]
                    ),
                    dim=1
                )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['top_eta'])[:, None],
                    torch.from_numpy(events['tbar_eta'])[:, None]
                    ),
                    dim=1
                )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['top_phi'])[:, None],
                    torch.from_numpy(events['tbar_phi'])[:, None]
                    ),
                    dim=1
                )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['top_mass'])[:, None],
                    torch.from_numpy(events['tbar_mass'])[:, None]
                    ),
                    dim=1
                )[:, :, None]
            ),
        dim=2
        )
    
    reco_four_vectors = torch.cat(
        (
            torch.cat(
                (
                    torch.from_numpy(events['l_pt'])[:, None],
                    torch.from_numpy(events['lbar_pt'])[:, None],
                    torch.from_numpy(events['b_pt'])[:, None],
                    torch.from_numpy(events['bbar_pt'])[:, None],
                    torch.from_numpy(events['met_pt'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_eta'])[:, None],
                    torch.from_numpy(events['lbar_eta'])[:, None],
                    torch.from_numpy(events['b_eta'])[:, None],
                    torch.from_numpy(events['bbar_eta'])[:, None],
                    torch.zeros(num_events)[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_phi'])[:, None],
                    torch.from_numpy(events['lbar_phi'])[:, None],
                    torch.from_numpy(events['b_phi'])[:, None],
                    torch.from_numpy(events['bbar_phi'])[:, None],
                    torch.from_numpy(events['met_phi'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_mass'])[:, None],
                    torch.from_numpy(events['lbar_mass'])[:, None],
                    torch.from_numpy(events['b_mass'])[:, None],
                    torch.from_numpy(events['bbar_mass'])[:, None],
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
            torch.Tensor([START_SEQ_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(events['l_pdgid'][:, None]),
            torch.from_numpy(events['lbar_pdgid'][:, None]),
            torch.Tensor([b_PDG_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([bbar_PDG_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([MET_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([SEPARATOR_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(events['gen_l_pdgid'][:, None]),
            torch.from_numpy(events['gen_lbar_pdgid'][:, None]),
            torch.Tensor([b_PDG_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([bbar_PDG_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(-1 * events['gen_l_pdgid'][:, None] - 1),  # corresponding antineutrino
            torch.from_numpy(-1 * events['gen_lbar_pdgid'][:, None] + 1),  # corresponding neutrino
            torch.Tensor([END_SEQ_ID])[None, :].repeat(num_events, 1)
        ),
        dim=1
    )

    if standardize:
        reco_means = torch.mean(reco_four_vectors, dim=0)
        reco_means[:, 2] = torch.zeros(reco_means[:, 2].shape)
        reco_stdevs = torch.std(reco_four_vectors, 0, True)
        reco_stdevs[:, 2] = torch.ones(reco_stdevs[:, 2].shape)
        gen_means = torch.mean(gen_four_vectors, dim=0)
        gen_means[:, 2] = torch.zeros(gen_means[:, 2].shape)
        gen_stdevs = torch.std(gen_four_vectors, 0, True)
        gen_stdevs[:, 2] = torch.ones(gen_stdevs[:, 2].shape)
        reco_four_vectors = (reco_four_vectors - reco_means) / reco_stdevs
        gen_four_vectors = (gen_four_vectors - gen_means) / gen_stdevs
        # standardize variable jets
        j_pt_mean, j_pt_std = get_stats_2d_jagged(events['extra_jet_pt'])
        j_eta_mean, j_eta_std = get_stats_2d_jagged(events['extra_jet_eta'])
        j_phi_mean, j_phi_std = get_stats_2d_jagged(events['extra_jet_phi'])
        j_mass_mean, j_mass_std = get_stats_2d_jagged(events['extra_jet_mass'])
        jet_mean_tensor = torch.tensor([j_pt_mean, j_eta_mean, 0.0, j_mass_mean])
        jet_std_tensor = torch.tensor([j_pt_std, j_eta_std, 1.0, j_mass_std])
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

    for i in range(len(events['extra_jet_pt'])):
        if len(events['extra_jet_b_tag'][i]) != len(events['extra_jet_pt'][i]):
            print('ahhh')
    # creating jet tensor
    B, R, C = reco_four_vectors.shape
    recos_w_jets = []
    ids_w_jets = []
    gen_reco_ids_w_jets = []
    top_prod_modes = []
    for i in tqdm(range(B), desc="EDITING FOR JETS"):
        # editing gen_reco ids
        gen_reco_id = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        add_gen_reco_id = [0 for _ in range(len(events['extra_jet_pt'][i]))]
        gen_reco_id = add_gen_reco_id + gen_reco_id
        gen_reco_ids_w_jets.append(torch.tensor(gen_reco_id, dtype=torch.int))
        # editing reco 4 vectors by adding jet tensor
        extra_jet_pt = torch.from_numpy(events['extra_jet_pt'][i])[:, None]
        extra_jet_eta = torch.from_numpy(events['extra_jet_eta'][i])[:, None]
        extra_jet_phi = torch.from_numpy(events['extra_jet_phi'][i])[:, None]
        extra_jet_mass = torch.from_numpy(events['extra_jet_mass'][i])[:, None]
        
        top_prod_modes.append(events['TopProductionMode'][i])

        jet_tensor = torch.cat(
            (extra_jet_pt, extra_jet_eta, extra_jet_phi, extra_jet_mass), dim=1)
        jet_cur_mean_tensor = jet_mean_tensor.expand_as(jet_tensor)
        jet_cur_std_tensor = jet_std_tensor.expand_as(jet_tensor)
        jet_tensor = (jet_tensor - jet_cur_mean_tensor) / jet_cur_std_tensor
        recos_w_jets.append(torch.cat(
            (reco_four_vectors[i, :4, :], jet_tensor, reco_four_vectors[i, 4:, :]), dim=0))
        # editing ids
        jet_ids = events['extra_jet_b_tag'][i]
        jet_ids[jet_ids > MEDIUM_B_TAG_KEY[events['b_key'][i]]] = b_PDG_ID
        jet_ids[jet_ids <= MEDIUM_B_TAG_KEY[events['b_key'][i]]] = other_jet_PDG_ID
        new_id = torch.cat((ids[i][:5], torch.tensor(jet_ids), ids[i][5:]), dim=0)
        new_id += 40
        ids_w_jets.append(new_id.type(torch.int))
    # putting data together
    top_prod_modes = torch.from_numpy(np.array(top_prod_modes))
    unique_sizes = [[] for _ in range(100)]
    for i in tqdm(range(B), desc="CONCAT AND MASKING"):
        D, _ = recos_w_jets[i].shape
        unique_sizes[D-5].append([ids_w_jets[i], gen_reco_ids_w_jets[i], recos_w_jets[i], gen_four_vectors[i], top_prod_modes[i]])
    t_dataloaders = []
    unique_sizes = [x for x in unique_sizes if x != []]
    for i in tqdm(range(len(unique_sizes)), desc="MAKING DATASETS"):
        t_dataset = MiniDataset_Discrimination(unique_sizes[i])
        t_dataloaders.append(DataLoader(t_dataset, batch_size=batch_size, shuffle=True))
    return t_dataloaders


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
            torch.Tensor([START_SEQ_VEC]),
            self.group[item][2],
            torch.Tensor([SEPARATOR_SEQ_VEC]),
            self.group[item][3],
            torch.Tensor([END_SEQ_VEC])
        ), dim=0)
        if self.pretraining:
            task_roll = torch.rand(1)
            if task_roll < 0.5:
                dice_rolls = torch.rand(R1+R2)
                zerod_mask = ~(dice_rolls < self.mask_prob)
                zerod_mask = torch.cat(
                    (torch.ones(1), zerod_mask[:R1], torch.ones(1), zerod_mask[R1:], torch.ones(1)),
                    dim=0
                )
            elif 0.5 <= task_roll < 0.75: # mask all gen
                dice_roll = torch.rand(1)
                if dice_roll < 0.5:
                    zerod_mask = torch.cat((torch.ones(R1+2), torch.zeros(R2), torch.ones(1)))
                else:
                    zerod_mask = torch.ones(R1+R2+3)
            elif 0.75 <= task_roll < 1.0: # mask all reco
                dice_roll = torch.rand(1)
                if dice_roll < 0.5:
                    zerod_mask = torch.cat((torch.ones(1), torch.zeros(R1), torch.ones(R2+1), torch.ones(1)))
                else:
                    zerod_mask = torch.ones(R1+R2+3)
        else:
            zerod_mask = torch.cat((torch.ones(R1+2), torch.zeros(R2), torch.ones(1)))
        four_vector_mask = zerod_mask[:, None].repeat(1, 4)
        # MET / Neutrino masking
        four_vector_mask[-9, 1] = torch.ones(four_vector_mask[-9, 1].shape)
        four_vector_mask [-9, 3] = torch.ones(four_vector_mask[-9, 3].shape)
        four_vector_mask [-3:-1, 1] = torch.ones(four_vector_mask[-3:-1, 1].shape)
        four_vector_mask [-3:-1, 3] = torch.ones(four_vector_mask[-3:-1, 3].shape)
        masked_four_vector = (four_vector * four_vector_mask)
        return torch.cat((self.group[item][0][:, None], self.group[item][1][:, None], masked_four_vector, zerod_mask[:, None]), dim=1), four_vector, self.group[item][4]


def create_data(events, standardize, pretraining, mask_probability, batch_size, tag):
    # set our seed
    torch.manual_seed(42)

    # Define PDG/RECO IDs
    MET_ID = 40
    START_SEQ_ID = 50
    SEPARATOR_ID = 51
    END_SEQ_ID = 52
    b_PDG_ID = 5
    other_jet_PDG_ID = 6
    bbar_PDG_ID = -5

    MEDIUM_B_TAG_KEY = {'2016pre': 0.2598, '2016post': 0.2489, '2017': 0.3040, '2018': 0.2783}

    # making base reco and gen 4 vectors as normal
    num_events = len(events['l_pt'])
    
    reco_top_info = torch.cat(
        (
            torch.cat(
                (
                    torch.from_numpy(events['top_pt'])[:, None],
                    torch.from_numpy(events['tbar_pt'])[:, None]
                    ),
                    dim=1
                )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['top_eta'])[:, None],
                    torch.from_numpy(events['tbar_eta'])[:, None]
                    ),
                    dim=1
                )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['top_phi'])[:, None],
                    torch.from_numpy(events['tbar_phi'])[:, None]
                    ),
                    dim=1
                )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['top_mass'])[:, None],
                    torch.from_numpy(events['tbar_mass'])[:, None]
                    ),
                    dim=1
                )[:, :, None]
            ),
        dim=2
        )
    
    reco_four_vectors = torch.cat(
        (
            torch.cat(
                (
                    torch.from_numpy(events['l_pt'])[:, None],
                    torch.from_numpy(events['lbar_pt'])[:, None],
                    torch.from_numpy(events['b_pt'])[:, None],
                    torch.from_numpy(events['bbar_pt'])[:, None],
                    torch.from_numpy(events['met_pt'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_eta'])[:, None],
                    torch.from_numpy(events['lbar_eta'])[:, None],
                    torch.from_numpy(events['b_eta'])[:, None],
                    torch.from_numpy(events['bbar_eta'])[:, None],
                    torch.zeros(num_events)[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_phi'])[:, None],
                    torch.from_numpy(events['lbar_phi'])[:, None],
                    torch.from_numpy(events['b_phi'])[:, None],
                    torch.from_numpy(events['bbar_phi'])[:, None],
                    torch.from_numpy(events['met_phi'])[:, None]
                ),
                dim=1
            )[:, :, None],
            torch.cat(
                (
                    torch.from_numpy(events['l_mass'])[:, None],
                    torch.from_numpy(events['lbar_mass'])[:, None],
                    torch.from_numpy(events['b_mass'])[:, None],
                    torch.from_numpy(events['bbar_mass'])[:, None],
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
            torch.Tensor([START_SEQ_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(events['l_pdgid'][:, None]),
            torch.from_numpy(events['lbar_pdgid'][:, None]),
            torch.Tensor([b_PDG_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([bbar_PDG_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([MET_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([SEPARATOR_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(events['gen_l_pdgid'][:, None]),
            torch.from_numpy(events['gen_lbar_pdgid'][:, None]),
            torch.Tensor([b_PDG_ID])[None, :].repeat(num_events, 1),
            torch.Tensor([bbar_PDG_ID])[None, :].repeat(num_events, 1),
            torch.from_numpy(-1 * events['gen_l_pdgid'][:, None] - 1),  # corresponding antineutrino
            torch.from_numpy(-1 * events['gen_lbar_pdgid'][:, None] + 1),  # corresponding neutrino
            torch.Tensor([END_SEQ_ID])[None, :].repeat(num_events, 1)
        ),
        dim=1
    )

    if standardize:
        reco_means = torch.mean(reco_four_vectors, dim=0)
        reco_means[:, 2] = torch.zeros(reco_means[:, 2].shape)
        reco_stdevs = torch.std(reco_four_vectors, 0, True)
        reco_stdevs[:, 2] = torch.ones(reco_stdevs[:, 2].shape)
        gen_means = torch.mean(gen_four_vectors, dim=0)
        gen_means[:, 2] = torch.zeros(gen_means[:, 2].shape)
        gen_stdevs = torch.std(gen_four_vectors, 0, True)
        gen_stdevs[:, 2] = torch.ones(gen_stdevs[:, 2].shape)
        reco_four_vectors = (reco_four_vectors - reco_means) / reco_stdevs
        gen_four_vectors = (gen_four_vectors - gen_means) / gen_stdevs
        # standardize variable jets
        j_pt_mean, j_pt_std = get_stats_2d_jagged(events['extra_jet_pt'])
        j_eta_mean, j_eta_std = get_stats_2d_jagged(events['extra_jet_eta'])
        j_phi_mean, j_phi_std = get_stats_2d_jagged(events['extra_jet_phi'])
        j_mass_mean, j_mass_std = get_stats_2d_jagged(events['extra_jet_mass'])
        jet_mean_tensor = torch.tensor([j_pt_mean, j_eta_mean, 0.0, j_mass_mean])
        jet_std_tensor = torch.tensor([j_pt_std, j_eta_std, 1.0, j_mass_std])
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

    for i in range(len(events['extra_jet_pt'])):
        if len(events['extra_jet_b_tag'][i]) != len(events['extra_jet_pt'][i]):
            print('ahhh')
    # creating jet tensor
    B, R, C = reco_four_vectors.shape
    recos_w_jets = []
    ids_w_jets = []
    gen_reco_ids_w_jets = []
    for i in tqdm(range(B), desc="EDITING FOR JETS"):
        # editing gen_reco ids
        gen_reco_id = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        add_gen_reco_id = [0 for _ in range(len(events['extra_jet_pt'][i]))]
        gen_reco_id = add_gen_reco_id + gen_reco_id
        gen_reco_ids_w_jets.append(torch.tensor(gen_reco_id, dtype=torch.int))
        # editing reco 4 vectors by adding jet tensor
        extra_jet_pt = torch.from_numpy(events['extra_jet_pt'][i])[:, None]
        extra_jet_eta = torch.from_numpy(events['extra_jet_eta'][i])[:, None]
        extra_jet_phi = torch.from_numpy(events['extra_jet_phi'][i])[:, None]
        extra_jet_mass = torch.from_numpy(events['extra_jet_mass'][i])[:, None]

        jet_tensor = torch.cat(
            (extra_jet_pt, extra_jet_eta, extra_jet_phi, extra_jet_mass), dim=1)
        jet_cur_mean_tensor = jet_mean_tensor.expand_as(jet_tensor)
        jet_cur_std_tensor = jet_std_tensor.expand_as(jet_tensor)
        jet_tensor = (jet_tensor - jet_cur_mean_tensor) / jet_cur_std_tensor
        recos_w_jets.append(torch.cat(
            (reco_four_vectors[i, :4, :], jet_tensor, reco_four_vectors[i, 4:, :]), dim=0))
        # editing ids
        jet_ids = events['extra_jet_b_tag'][i]
        jet_ids[jet_ids > MEDIUM_B_TAG_KEY[events['b_key'][i]]] = b_PDG_ID
        jet_ids[jet_ids <= MEDIUM_B_TAG_KEY[events['b_key'][i]]] = other_jet_PDG_ID
        new_id = torch.cat((ids[i][:5], torch.tensor(jet_ids), ids[i][5:]), dim=0)
        new_id += 40
        ids_w_jets.append(new_id.type(torch.int))
    # putting data together
    unique_sizes = [[] for _ in range(100)]
    for i in tqdm(range(B), desc="CONCAT AND MASKING"):
        D, _ = recos_w_jets[i].shape
        unique_sizes[D-5].append([ids_w_jets[i], gen_reco_ids_w_jets[i], recos_w_jets[i], gen_four_vectors[i], reco_top_info[i]])
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


class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, reduction="none", device = torch.device('cpu')):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


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
parser.add_argument('--lossf', help='L1 or MSE or CL', type = str, choices = ['L1', 'MSE', 'CL', 'FL'])
parser.add_argument('--N', help='number of E/D layers', type=int)
parser.add_argument('--h', help='number of attention heads per MHA', type=int)
parser.add_argument('--warmup', help='number of warmup steps for lr scheduler', type=int)
parser.add_argument('--data_dir', help='directory where data reconstruction data is stored', type=str)
parser.add_argument('--load_dir', help='directory where model parameters are loaded from', type=str)
parser.add_argument('--save_dir', help='directory to be saved on cluster', type=str)
parser.add_argument('--act_fn', help='activation function', type=str)
parser.add_argument('--weight_decay', help = 'coefficient for L2 regularization', type=float)
parser.add_argument('--epsilon', help = 'epsilon for Adam', type=float)
parser.add_argument('--lr_mult', help='lr_mult', type=float)
parser.set_defaults(d_model=512, dropout = 0.1,
                    batch_size = 64, n_epochs = 3, only_gen = False,
                    mask_prob = 0.09, standardize = True,
                    tvt_split = [0.7, 0.15, 0.15], lossf = 'CL',
                    N = 8, h = 16, warmup=10000,
                    data_dir = r'/depot/cms/top/jprodger/Bumblebee/src/reco_data2/',
                    load_dir = r'/depot/cms/top/jprodger/Bumblebee/src/new_features_transfer/tune_output/',
                    save_dir = r'/depot/cms/top/jprodger/Bumblebee/src/initial_state_discrimination2/fl_tune_output/',
                    act_fn = 'gelu', weight_decay = 1e-3, epsilon = 1e-6, lr_mult = 1e-1)

args = parser.parse_args()

simple_train = True

# dataset creation and preparation
base_dir = args.data_dir
channels = ['ee', 'emu', 'mumu']
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
        'gen_nubar_pt', 'gen_nubar_eta', 'gen_nubar_phi',
        'extra_jet_pt', 'extra_jet_eta', 'extra_jet_phi',
        'extra_jet_mass', 'extra_jet_b_tag', 'TopProductionMode'
    ]
    
if simple_train:
    years1 = ['2017UL']
    notau_filenames1 = {
    f'{base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': f'ttBar_treeVariables_step8' for
    channel, year in itertools.product(channels, years1)
    }
    events1 = uproot.concatenate(
    notau_filenames1,
    selected_keys,
    cut = 'TopProductionMode < 2',
    library='numpy'
    )
    events1['b_key'] = ['2017' for _ in range(len(events1['l_pt']))]
    events = [events1]
else:
    years1 = ['2016ULpreVFP']
    years2 = ['2016ULpostVFP']
    years3 = ['2017UL']
    years4 = ['2018UL']
    notau_filenames1 = {
        f'{base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': f'ttBar_treeVariables_step8' for
        channel, year in itertools.product(channels, years1)
    }
    notau_filenames2 = {
        f'{base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': f'ttBar_treeVariables_step8' for
        channel, year in itertools.product(channels, years2)
    }
    notau_filenames3 = {
        f'{base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': f'ttBar_treeVariables_step8' for
        channel, year in itertools.product(channels, years3)
    }
    notau_filenames4 = {
        f'{base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': f'ttBar_treeVariables_step8' for
        channel, year in itertools.product(channels, years4)
    }
    
    
    events1 = uproot.concatenate(
        notau_filenames1,
        selected_keys,
        cut = 'TopProductionMode < 2',
        library='numpy'
    )
    events2 = uproot.concatenate(
        notau_filenames2,
        selected_keys,
        cut = 'TopProductionMode < 2',
        library='numpy'
    )
    events3 = uproot.concatenate(
        notau_filenames3,
        selected_keys,
        cut = 'TopProductionMode < 2',
        library='numpy'
    )
    events4 = uproot.concatenate(
        notau_filenames4,
        selected_keys,
        cut = 'TopProductionMode < 2',
        library='numpy'
    )
    events1['b_key'] = ['2016pre' for _ in range(len(events1['l_pt']))]
    events2['b_key'] = ['2016post' for _ in range(len(events2['l_pt']))]
    events3['b_key'] = ['2017' for _ in range(len(events3['l_pt']))]
    events4['b_key'] = ['2018' for _ in range(len(events4['l_pt']))]
    events = [events1, events2, events3, events4]

total_events = merge_events(events)
print('DATA CREATED')

def train(trial_number):
    # load model parameters
    file = open(args.load_dir + f"bumblebee{trial_number}_config.txt", "r")
    m_params = file.read()
    m_params = m_params.split("\n")
    file.close()
    args.batch_size = int(m_params[0])
    args.d_model = int(m_params[1])
    args.dropout = float(m_params[2])
    args.epsilon = float(m_params[3])
    args.h = int(m_params[4])
    args.N = int(m_params[5])
    args.warmup = int(m_params[6])
    args.weight_decay = float(m_params[7])
    args.lr_mult = float(m_params[8])

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
    "epsilon": args.epsilon,
    "lr_mult": args.lr_mult
    }

    # wandb initialization
    wandb.init(
        project='BUMBLEBEE_INIT_STATE_W_NEW_FEATURES_NEWSTUFF',
        entity='bumblebee_team',
        config=wandb_config
    )
    
    
    # NOW DOING DISCRIMINATION
    disc_model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model, d_ff= int(4 * args.d_model),
                            h = args.h, dropout= args.dropout, act_fn=args.act_fn)
    if torch.cuda.device_count() > 1:
        disc_model = nn.DataParallel(disc_model)
    disc_model = disc_model.to(device)
    if torch.cuda.device_count() > 1:
        disc_model.module.load_state_dict(torch.load(args.load_dir + f'bumblebee{trial_number}.pt'))
    else:
        disc_model.load_state_dict(torch.load(args.load_dir + f'bumblebee{trial_number}.pt'))
    
    d_train_events, d_valid_events, d_test_events = train_valid_test(total_events, args.tvt_split, use_generator = False)
    
    train_data_loaders = create_data_disc(events=d_train_events, standardize=args.standardize,
                batch_size=args.batch_size, tag = 'train')
    valid_data_loaders = create_data_disc(events=d_valid_events, standardize=args.standardize,
                batch_size=args.batch_size, tag = 'valid')
    test_data_loaders = create_data_disc(events=d_test_events, standardize=args.standardize,
                batch_size=args.batch_size, tag = 'test')
                
    event_targets = d_train_events['TopProductionMode']
    
    num_negatives = len(event_targets[event_targets == 0.0])
    num_positives = len(event_targets[event_targets == 1.0])
    print(f"0: {num_negatives}, 1: {num_positives}")
    total_num_samples = num_negatives + num_positives
    class_0_weight = total_num_samples / (num_negatives * 2)
    class_1_weight = total_num_samples / (num_positives * 2)
    pos_weight = torch.tensor([class_0_weight / class_1_weight])
    pos_weight = pos_weight.to(device)
    alpha = class_0_weight / class_1_weight
    
    #criterion = FocalLoss(alpha = alpha, gamma = 1, reduction = "mean", device = device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(disc_model.parameters(),
                                 lr=1e-6, betas=(0.9, 0.999), eps=args.epsilon,
                                 weight_decay=args.weight_decay)
    scheduler = NoamOpt(args.d_model, lr_mult=1.0, warmup=args.warmup, optimizer=optimizer)
    
    avg_valid_loss = 0.0
    avg_test_loss = 0.0
    
    train_steps = 0
    valid_steps = 0
    test_steps = 0
    
    disc_truths = []
    disc_predictions = []
    
    best_valid_loss = -1
    
    print('TRAINING STARTED')
    # Train the model
    for t in range(args.n_epochs):
        disc_model.train()
        print(f"EPOCH {t}:")
        train_steps = 0
        random.shuffle(train_data_loaders)
        for data_loader in train_data_loaders:
            for batch_id, (x, target) in tqdm(enumerate(data_loader), desc="TRAINING"):
                # Training Set
                x = x.to(device)
                target = target.to(device)
    
                scheduler.zero_grad()
                wandb.log({'disc_lr': scheduler.get_cur_lr()})
                y_pred = disc_model(x)
    
                etas = y_pred[:, :, 1]
    
                phis = y_pred[:, :, 2]
    
                corrected_pred = torch.cat((y_pred[:, :, 0][:, :, None], etas[:, :, None],
                                            phis[:, :, None], y_pred[:, :, 3][:, :, None]), dim=2)
    
                loss = criterion(corrected_pred[:, 0, 0], target.float())
                
                if batch_id % 10 == 0:
                    wandb.log({'disc_loss': loss.item()})
    
                loss.backward()
    
                scheduler.step_and_update()
                train_steps += 1
        disc_model.eval()
        with torch.no_grad():
            for valid_data_loader in valid_data_loaders:
                for valid_batch_id, (x_valid, target_valid) in tqdm(enumerate(valid_data_loader), desc = "VALIDATING"):
                    x_valid = x_valid.to(device)
                    target_valid = target_valid.to(device)
                    
                    y_pred_valid = disc_model(x_valid)
                    
                    etas = y_pred_valid[:, :, 1]
    
                    phis = y_pred_valid[:, :, 2]
    
                    corrected_pred_valid = torch.cat((y_pred_valid[:, :, 0][:, :, None], etas[:, :, None],
                                            phis[:, :, None], y_pred_valid[:, :, 3][:, :, None]), dim=2)
    
                    loss_valid = criterion(corrected_pred_valid[:, 0, 0], target_valid.float())
                    
                    if valid_batch_id % 5 == 0:
                        wandb.log({'disc_valid_loss': loss_valid.item()})
                    
                    avg_valid_loss += loss_valid.item()
        
                    valid_steps += 1
        
        avg_valid_loss /= valid_steps
        
        wandb.log({'avg_disc_valid_loss': avg_valid_loss})
        
        if avg_valid_loss >= best_valid_loss != -1:
                break
    
        if avg_valid_loss < best_valid_loss or best_valid_loss == -1:
            best_validation_loss = avg_valid_loss
            if torch.cuda.device_count() > 1:
                torch.save(disc_model.module.state_dict(), args.save_dir + f'bumblebee_disc{trial_number}.pt')
            else:
                torch.save(disc_model.state_dict(), args.save_dir + f'bumblebee_disc{trial_number}.pt')
    
    disc_best_model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model, d_ff= int(4 * args.d_model),
                            h = args.h, dropout= args.dropout, act_fn=args.act_fn)
    if torch.cuda.device_count() > 1:
        disc_best_model = nn.DataParallel(disc_best_model)
    disc_best_model = disc_best_model.to(device)
    if torch.cuda.device_count() > 1:
        disc_best_model.module.load_state_dict(torch.load(args.save_dir + f'bumblebee_disc{trial_number}.pt'))
    else:
        disc_best_model.load_state_dict(torch.load(args.save_dir + f'bumblebee_disc{trial_number}.pt'))
        
    disc_best_model.eval()
    with torch.no_grad():
        for test_data_loader in test_data_loaders:
            for test_batch_id, (x_test, target_test) in tqdm(enumerate(test_data_loader), desc = "TESTING"):
                x_test = x_test.to(device)
                target_test = target_test.to(device)
                
                y_pred_test = disc_best_model(x_test)
                
                etas = y_pred_test[:, :, 1]
                phis = y_pred_test[:, :, 2]
    
                corrected_pred_test = torch.cat((y_pred_test[:, :, 0][:, :, None], etas[:, :, None],
                                        phis[:, :, None], y_pred_test[:, :, 3][:, :, None]), dim=2)
                                        
                loss_test = criterion(corrected_pred_test[:, 0, 0], target_test.float())
                
                if test_batch_id % 5 ==0:
                    wandb.log({'disc_test_loss': loss_test.item()})
                
                disc_predictions.append(torch.sigmoid(corrected_pred_test[:, 0, 0]).detach().cpu().numpy())
                disc_truths.append(target_test.detach().cpu().numpy())
                    
                avg_test_loss += loss_test.item()
        
                test_steps += 1
        
    avg_test_loss /= test_steps
    
    wandb.log({'avg_disc_test_loss': avg_test_loss})
    
    with open(args.save_dir + f"predictions{trial_number}.pkl", "wb") as f:
        pickle.dump(disc_predictions, f)
    f.close()
    
    with open(args.save_dir + f"truths{trial_number}.pkl", "wb") as f:
        pickle.dump(disc_truths, f)
    f.close()
    
    
    # PLOTTING
    predictions = np.array(disc_predictions)
    truths = np.array(disc_truths)
    total_predictions = []
    total_truths = []
    gg_predictions = []
    qqbar_predictions = []
    
    for i in tqdm(range(len(predictions))):
        for j in range(len(predictions[i])):
            total_predictions.append(predictions[i][j])
            total_truths.append(truths[i][j])
    
    for i in range(len(total_truths)):
        if total_truths[i] == 0.0:
            gg_predictions.append(total_predictions[i])
        elif total_truths[i] == 1.0:
            qqbar_predictions.append(total_predictions[i])
    
    n_bins = 30
    fig = plt.figure(figsize=(10, 8))
    plt.hist(x=[gg_predictions, qqbar_predictions], bins=n_bins, color=['red', 'blue'], histtype='step', label=['gg', 'qqbar'])
    plt.legend()
    plt.yscale('log')
    plt.savefig(args.save_dir + f'score_histogram{trial_number}.png')
    plt.close(fig)
    
    
    fpr, tpr, thresholds = roc_curve(y_true=total_truths, y_score=total_predictions)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(args.save_dir + f'ROC_curve{trial_number}.png')
    plt.close(fig)
    wandb.log({"test_auroc": roc_auc})
    wandb.finish()
    

train(2)

