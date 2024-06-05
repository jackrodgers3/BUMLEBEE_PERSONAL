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

    training_data, validation_data, test_data = torch.utils.data.random_split(full_dataset,
                                                                              tvt_split,
                                                                              generator = torch.Generator().manual_seed(23))

    return training_data, validation_data, test_data


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
parser.set_defaults(d_model=256, dropout = 0.1,
                    batch_size = 16, n_epochs = 1, only_gen = False,
                    mask_prob = 0.09, standardize = True,
                    tvt_split = (0.7, 0.15, 0.15), lossf = False,
                    N = 3, h = 8, warmup=100000,
                    save_dir = r'C:\Users\jackm\PycharmProjects\BUMLEBEE_PERSONAL\saved_data/')

args = parser.parse_args()


# dataset creation and preparation
base_dir = r'D:\Data\Research\BUMBLEBEE\Reconstruction/'
channels = ['ee', 'emu', 'mumu']
years = ['2016ULpreVFP', '2016ULpostVFP', '2017UL', '2018UL']

notau_filenames = {
    f'{base_dir}/{year}/{channel}/{channel}_ttbarsignalplustau_fromDilepton_{year}_*': 'ttBar_treeVariables_step8' for
    channel, year in itertools.product(channels, years)
}


events = uproot.concatenate(
    notau_filenames,
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
    dataset_nm = dataloaders.GenDataset(events, gen_reco_ids,
                                     standardize=args.standardize,
                                     pretraining=True,
                                     mask_probability=args.mask_prob)
    dataset_m = dataloaders.GenDataset(events, gen_reco_ids,
                                           standardize=args.standardize,
                                           pretraining=False,
                                           mask_probability=args.mask_prob)
else:
    gen_reco_ids = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    dataset_nm = dataloaders.GenRecoDataset(events, gen_reco_ids,
                                         standardize=args.standardize,
                                         pretraining=True,
                                         mask_probability=args.mask_prob)
    dataset_m = dataloaders.GenRecoDataset(events, gen_reco_ids,
                                               standardize=args.standardize,
                                               pretraining=False,
                                               mask_probability=args.mask_prob)

train_data_nm, _, _ = train_valid_test_split(dataset_nm, args.tvt_split)
_, valid_data_m, test_data_m = train_valid_test_split(dataset_m, args.tvt_split)


print('DATA CREATED')
# Prepare the model.
if args.lossf:
    criterion = nn.L1Loss(reduction='mean')
else:
    criterion = nn.MSELoss(reduction='mean')

model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model,
                   d_ff= int(4 * args.d_model), h = args.h, dropout= args.dropout)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=3e-4, betas=(0.9, 0.98), eps=1e-9,
                             weight_decay=0.0)
scheduler = NoamOpt(args.d_model, lr_mult=1.0, warmup=args.warmup, optimizer=optimizer)

# Create data loaders
# Data Loader
data_loader = DataLoader(train_data_nm, batch_size=args.batch_size, shuffle=True)
valid_data_loader = DataLoader(valid_data_m, batch_size=args.batch_size, shuffle=True)
test_data_loader = DataLoader(test_data_m, batch_size=args.batch_size, shuffle=True)
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
    project='BUMBLEBEE_BEST_MODEL_TRAINING',
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
test_loss_components = []
test_predictions = []
test_truths = []
print('TRAINING STARTED')
# Train the model.
for t in range(args.n_epochs):
    model.train()
    for batch_id, (x, target) in tqdm(enumerate(data_loader)):
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

        out = criterion(corrected_pred, target.float())

        zerod_mask = x[:, :, -1]
        four_vector_mask = zerod_mask[:, :, None].repeat(1, 1, 4)
        four_vector_mask[-3:-1, 3] = torch.ones(four_vector_mask[-3:-1, 3].shape)

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
            wandb.log({'masked loss': masked_loss.item(), 'unmasked loss': unmasked_loss.item()})

        # append to lists and detach
        training_losses.append(loss.detach().cpu().numpy())
        masked_training_losses.append(masked_loss.detach().cpu().numpy())
        unmasked_training_losses.append(unmasked_loss.detach().cpu().numpy())
        training_loss_components.append(component_loss.detach().cpu().numpy())

        scheduler.step_and_update()
        if batch_id == 20000:
            # Validate
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                unmasked_valid_loss = 0
                masked_valid_loss = 0
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

                    valid_out = criterion(corrected_y_valid, target_valid.float())

                    zerod_mask_valid = x_valid[:, :, -1]
                    four_vector_mask_valid = zerod_mask_valid[:, :, None].repeat(1, 1, 4)
                    four_vector_mask_valid[:, -3:-1, 3] = torch.ones(four_vector_mask_valid[:, -3:-1, 3].shape)

                    # don't include losses which were 0 because of being unmasked in masked loss average
                    masked_valid_out = valid_out * ~four_vector_mask_valid.type(torch.bool).to(device)
                    masked_valid_out = torch.sum(torch.sum(masked_valid_out, axis=1), axis=1)
                    masked_valid_out = masked_valid_out[masked_valid_out != 0]

                    # don't include losses which were 0 because of being masked in unmasked loss average
                    unmasked_valid_out = valid_out * four_vector_mask_valid.type(torch.bool).to(device)
                    unmasked_valid_out = torch.sum(torch.sum(unmasked_valid_out, axis=1), axis=1)
                    unmasked_valid_out = unmasked_valid_out[unmasked_valid_out != 0]

                    # look to see if training objective on masking is being learned
                    masked_valid_loss += torch.mean(masked_valid_out, dim=0)
                    unmasked_valid_loss += torch.mean(unmasked_valid_out, dim=0)

                    valid_loss_components += torch.mean(valid_out, dim=0)
                    valid_loss += torch.sum(torch.mean(valid_out, dim=0))

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

                valid_loss /= valid_batch_id
                unmasked_valid_loss /= valid_batch_id
                masked_valid_loss /= valid_batch_id
                wandb.log({'unmasked valid loss': unmasked_valid_loss,
                           'masked valid loss': masked_valid_loss})

                valid_loss_components /= valid_batch_id
                validation_losses.append(valid_loss.detach().cpu().numpy())
                validation_loss_components.append(valid_loss_components.detach().cpu().numpy())
                masked_validation_losses.append(masked_valid_loss.detach().cpu().numpy())
                unmasked_validation_losses.append(unmasked_valid_loss.detach().cpu().numpy())

                if valid_loss.detach().cpu().numpy() < best_validation_loss or best_validation_loss == -1:
                    best_validation_loss = valid_loss.detach().cpu().numpy()
                    torch.save(model.state_dict(), args.save_dir + 'bumblebee2.pt')

                nunubar_validation_loss = np.sum(np.sum(np.array(validation_loss_components)[:, -3:-1, :3], axis=1), axis=1)
                bbar_validation_loss = np.sum(np.sum(np.array(validation_loss_components)[:, -5:-3, :], axis=1), axis=1)
                llbar_validation_loss = np.sum(np.sum(np.array(validation_loss_components)[:, -7:-5, :], axis=1), axis=1)

                nu_pt_validation_loss = np.array(validation_loss_components)[:,-2,0]
                nu_eta_validation_loss = np.array(validation_loss_components)[:,-2,1]
                nu_phi_validation_loss = np.array(validation_loss_components)[:, -2, 2]
                nubar_pt_validation_loss = np.array(validation_loss_components)[:, -3, 0]
                nubar_eta_validation_loss = np.array(validation_loss_components)[:, -3, 1]
                nubar_phi_validation_loss = np.array(validation_loss_components)[:, -3, 2]

                wandb.log({'nunubar_valid_loss': nunubar_validation_loss,
                           'bbar_valid_loss': bbar_validation_loss,
                           'llbar_valid_loss': llbar_validation_loss,
                           'nu_pt_valid_loss': nu_pt_validation_loss,
                           'nu_eta_valid_loss': nu_eta_validation_loss,
                           'nu_phi_valid_loss': nu_phi_validation_loss,
                           'nubar_pt_valid_loss': nubar_pt_validation_loss,
                           'nubar_eta_valid_loss': nubar_eta_validation_loss,
                           'nubar_phi_valid_loss': nubar_phi_validation_loss})

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
                plt.savefig(args.save_dir + f'plots/hist_plots@{t}.pdf')
                plt.clf()
                plt.cla()


# There's a bunch of stuff related to plotting here in the notebook. I'm
# omitting it here, we'll have to figure out exactly how to handle output.

# Model testing
# Also copied from the notebook.
best_model = make_model(particle_dimensionality=4, N=args.N, d_model=args.d_model,
                   d_ff= int(4 * args.d_model), h = args.h, dropout= args.dropout)
best_model = best_model.to(device)
best_model.load_state_dict(torch.load(args.save_dir + 'bumblebee2.pt'))

best_model.eval()

for test_batch_id, (x_test, target_test) in enumerate(test_data_loader):
    x_test = x_test.to(device)
    target_test = target_test.to(device)
    target_pred_test = best_model(x_test)

    zerod_mask_test = x_test[:, :, -1]

    valid_etas = target_pred_test[:, :, 1]

    valid_phis = 2 * torch.atan(target_pred_test[:, :, 2])

    corrected_y_test = torch.cat(
        (target_pred_test[:, :, 0][:, :, None], valid_etas[:, :, None], valid_phis[:, :, None],
         target_pred_test[:, :, 3][:, :, None]), axis=2)


    test_out = criterion(corrected_y_test, target_test.float())
    test_loss = torch.sum(torch.mean(test_out * ~zerod_mask_test.type(torch.bool), dim=0))
    wandb.log({'test_loss': test_loss})

    test_loss_components.append(torch.mean(test_out, dim=0).detach().cpu().numpy())
    test_losses.append(test_loss.detach().cpu().numpy())
    test_predictions.append(corrected_y_test.detach().cpu().numpy())
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

# Save the data. This section will likely need some augmentation.
# Training
torch.save(np.array(masked_training_losses), args.save_dir + 'make_travis_plots/used_training_losses.pt')
torch.save(np.array(training_losses), args.save_dir + 'make_travis_plots/net_training_losses.pt')
torch.save(np.array(training_loss_components), args.save_dir + 'make_travis_plots/training_loss_components.pt')

# Validation
torch.save(np.array(validation_losses), args.save_dir + 'make_travis_plots/validation_losses.pt')
torch.save(np.array(validation_loss_components), args.save_dir + 'make_travis_plots/validation_loss_components.pt')


# Test
torch.save(np.array(test_losses), args.save_dir + 'make_travis_plots/test_losses.pt')
torch.save(np.array(test_loss_components), args.save_dir + 'make_travis_plots/test_loss_components.pt')
torch.save(np.array(test_predictions), args.save_dir + 'make_travis_plots/predictions.pt')
torch.save(np.array(test_truths), args.save_dir + 'make_travis_plots/truths.pt')


wandb.finish()

### AND WE'RE DONE!!!