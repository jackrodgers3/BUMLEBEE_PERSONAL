import sys
import numpy as np
import uproot
import mplhep as hep
from matplotlib import pyplot as plt
import awkward as ak
import torch
from tqdm import tqdm
import vector
import torch.nn as nn
import os
hep.style.use(hep.style.CMS)
hep.cms.label(label='Work in Progress')

def compute_bias_and_resolution(residuals, gen_tt_mass, bins):
    """
    Computes the median residual (bias) and the resolution based on the
    14th and 86th percentiles of the residuals.

    Parameters:
    residuals (numpy.ndarray): Array of residual values.

    Returns:
    tuple: A tuple containing the bias and resolution.
    """
    bias = np.array([])
    p14 = np.array([])
    p86 = np.array([])

    if len(bins) > 1:
        for i in range(len(bins)):
            if i == len(bins) - 1:
                break
            masked_residual = residuals[(bins[i] < gen_tt_mass) & (gen_tt_mass < bins[i + 1])]

            if len(masked_residual) == 0:
                _bias = np.array([0])[:, None]
                _p14 = np.array([0])[:, None]
                _p86 = np.array([0])[:, None]
            else:
                _bias = np.array([np.median(masked_residual)])[:, None]
                _p14 = np.array([np.percentile(masked_residual, 16)])[:, None]
                _p86 = np.array([np.percentile(masked_residual, 84)])[:, None]
                    
            if len(bias) == 0:
                bias = _bias
                p14 = _p14
                p86 = _p86
            else:
                bias = np.concatenate((bias, _bias), axis=1)
                p14 = np.concatenate((p14, _p14), axis=1)
                p86 = np.concatenate((p86, _p86), axis=1)
    else:
        # Compute the median residual (bias)
        bias = np.median(residuals)
    
        # Compute the 14th and 86th percentiles
        p14 = np.percentile(residuals, 14)
        p86 = np.percentile(residuals, 86)

    return bias, p14, p86


BASE_DIR = r'/depot/cms/top/jprodger/Bumblebee/src/Paper_Delphes_Stuff/pretraining/output/'

ex_file = uproot.open(BASE_DIR + 'predictions.root')
tree = ex_file['Bumblebee']
branches = tree.arrays()

pred_t_mass = ak.to_numpy(branches['pred_t_mass'])
pred_t_pt = ak.to_numpy(branches['pred_t_pt'])
pred_t_eta = ak.to_numpy(branches['pred_t_eta'])
pred_t_phi = ak.to_numpy(branches['pred_t_phi'])
pred_tbar_mass = ak.to_numpy(branches['pred_tbar_mass'])
pred_tbar_pt = ak.to_numpy(branches['pred_tbar_pt'])
pred_tbar_eta = ak.to_numpy(branches['pred_tbar_eta'])
pred_tbar_phi = ak.to_numpy(branches['pred_tbar_phi'])

gen_t_mass = ak.to_numpy(branches['gen_t_mass'])
gen_t_pt = ak.to_numpy(branches['gen_t_pt'])
gen_t_eta = ak.to_numpy(branches['gen_t_eta'])
gen_t_phi = ak.to_numpy(branches['gen_t_phi'])
gen_tbar_mass = ak.to_numpy(branches['gen_tbar_mass'])
gen_tbar_pt = ak.to_numpy(branches['gen_tbar_pt'])
gen_tbar_eta = ak.to_numpy(branches['gen_tbar_eta'])
gen_tbar_phi = ak.to_numpy(branches['gen_tbar_phi'])

gen_top = vector.array({"pt": gen_t_pt, "phi": gen_t_phi, "eta": gen_t_eta, "mass": gen_t_mass})
gen_topbar = vector.array({"pt": gen_tbar_pt, "phi": gen_tbar_phi, "eta": gen_tbar_eta, "mass": gen_tbar_mass})
pred_top = vector.array({"pt": pred_t_pt, "phi": pred_t_phi, "eta": pred_t_eta, "mass": pred_t_mass})
pred_topbar = vector.array({"pt": pred_tbar_pt, "phi": pred_tbar_phi, "eta": pred_tbar_eta, "mass": pred_tbar_mass})

gen_tt_mass = (gen_top+gen_topbar).mass
dnn_tt_mass = (pred_top+pred_topbar).mass
bin_split = 500.
bins_low = np.linspace(200., bin_split, num=60)
bins_high = np.linspace(bin_split, 1200., num=60)

dnn_bias, dnn_p14, dnn_p86 = compute_bias_and_resolution(dnn_tt_mass - gen_tt_mass, gen_tt_mass, bins_low)

# BIAS PLOTS

fig = plt.figure(figsize=(10, 8))
plt.scatter((bins_low[1:] + bins_low[:-1]) * 0.5, dnn_bias[0], label='Bumblebee')
plt.ylabel('bias')
plt.xlim(275, 500)
plt.xlabel(r'$m(\mathrm{t\bar{t}})$ [GeV]')
plt.savefig(BASE_DIR + 'bias_plot.png')
plt.close(fig)

# RESOLUTION PLOTS

fig = plt.figure(figsize=(10, 8))
plt.scatter((bins_low[1:] + bins_low[:-1]) * 0.5, (dnn_p86[0] - dnn_p14[0]), label='Bumblebee')
plt.ylabel(r'Resolution ($\sigma_{14-86}$)')
plt.xlim(275, 500)
plt.xlabel(r'$m(\mathrm{t\bar{t}})$ [GeV]')
plt.savefig(BASE_DIR + 'res_plot.png')
plt.close(fig)




