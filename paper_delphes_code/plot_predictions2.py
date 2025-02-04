import sys
import numpy as np
import uproot
import awkward as ak
import torch
#import Plotting
#import Resolution
from tqdm import tqdm
import vector
import torch.nn as nn
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# *** very important parameters *** edit at your own risk lol
get_data_from_file = False
post_as_pngs = False # alternative is pdf
N_BINS = 40
BASE_DIR = r'/depot/cms/top/jprodger/Bumblebee/src/Paper_Delphes_Stuff/pretraining/output/'
SAVE_DIR = BASE_DIR
#############################################################

if get_data_from_file:
    
    file = uproot.open(SAVE_DIR + 'predictions.root')
    tree = file["Bumblebee"]
    branches = tree.arrays()
    
    gen_l_pt = ak.to_numpy(branches['gen_l_pt'])
    gen_lbar_pt = ak.to_numpy(branches['gen_lbar_pt'])
    gen_b_pt = ak.to_numpy(branches['gen_b_pt'])
    gen_bbar_pt = ak.to_numpy(branches['gen_bbar_pt'])
    gen_n_pt = ak.to_numpy(branches['gen_n_pt'])
    gen_nbar_pt = ak.to_numpy(branches['gen_nbar_pt'])
    gen_l_eta = ak.to_numpy(branches['gen_l_eta'])
    gen_lbar_eta = ak.to_numpy(branches['gen_lbar_eta'])
    gen_b_eta = ak.to_numpy(branches['gen_b_eta'])
    gen_bbar_eta = ak.to_numpy(branches['gen_bbar_eta'])
    gen_n_eta = ak.to_numpy(branches['gen_n_eta'])
    gen_nbar_eta = ak.to_numpy(branches['gen_nbar_eta'])
    gen_l_phi = ak.to_numpy(branches['gen_l_phi'])
    gen_lbar_phi = ak.to_numpy(branches['gen_lbar_phi'])
    gen_b_phi = ak.to_numpy(branches['gen_b_phi'])
    gen_bbar_phi = ak.to_numpy(branches['gen_bbar_phi'])
    gen_n_phi = ak.to_numpy(branches['gen_n_phi'])
    gen_nbar_phi = ak.to_numpy(branches['gen_nbar_phi'])
    gen_l_mass = ak.to_numpy(branches['gen_l_mass'])
    gen_lbar_mass = ak.to_numpy(branches['gen_lbar_mass'])
    gen_b_mass = ak.to_numpy(branches['gen_b_mass'])
    gen_bbar_mass = ak.to_numpy(branches['gen_bbar_mass'])
    gen_n_mass = ak.to_numpy(branches['gen_n_mass'])
    gen_nbar_mass = ak.to_numpy(branches['gen_nbar_mass'])
    gen_met_pt = ak.to_numpy(branches['gen_met_pt'])
    gen_met_phi = ak.to_numpy(branches['gen_met_phi'])
    gen_met_eta = ak.to_numpy(branches['gen_met_eta'])
    gen_met_mass = ak.to_numpy(branches['gen_met_mass'])
    
    pred_l_pt = ak.to_numpy(branches['pred_l_pt'])
    pred_lbar_pt = ak.to_numpy(branches['pred_lbar_pt'])
    pred_b_pt = ak.to_numpy(branches['pred_b_pt'])
    pred_bbar_pt = ak.to_numpy(branches['pred_bbar_pt'])
    pred_n_pt = ak.to_numpy(branches['pred_n_pt'])
    pred_nbar_pt = ak.to_numpy(branches['pred_nbar_pt'])
    pred_l_eta = ak.to_numpy(branches['pred_l_eta'])
    pred_lbar_eta = ak.to_numpy(branches['pred_lbar_eta'])
    pred_b_eta = ak.to_numpy(branches['pred_b_eta'])
    pred_bbar_eta = ak.to_numpy(branches['pred_bbar_eta'])
    pred_n_eta = ak.to_numpy(branches['pred_n_eta'])
    pred_nbar_eta = ak.to_numpy(branches['pred_nbar_eta'])
    pred_l_phi = ak.to_numpy(branches['pred_l_phi'])
    pred_lbar_phi = ak.to_numpy(branches['pred_lbar_phi'])
    pred_b_phi = ak.to_numpy(branches['pred_b_phi'])
    pred_bbar_phi = ak.to_numpy(branches['pred_bbar_phi'])
    pred_n_phi = ak.to_numpy(branches['pred_n_phi'])
    pred_nbar_phi = ak.to_numpy(branches['pred_nbar_phi'])
    pred_l_mass = ak.to_numpy(branches['pred_l_mass'])
    pred_lbar_mass = ak.to_numpy(branches['pred_lbar_mass'])
    pred_b_mass = ak.to_numpy(branches['pred_b_mass'])
    pred_bbar_mass = ak.to_numpy(branches['pred_bbar_mass'])
    pred_n_mass = ak.to_numpy(branches['pred_n_mass'])
    pred_nbar_mass = ak.to_numpy(branches['pred_nbar_mass'])
    pred_met_pt = ak.to_numpy(branches['pred_met_pt'])
    pred_met_phi = ak.to_numpy(branches['pred_met_phi'])
    pred_met_eta = ak.to_numpy(branches['pred_met_eta'])
    pred_met_mass = ak.to_numpy(branches['pred_met_mass'])
    
    pred_t_mass = ak.to_numpy(branches['pred_t_mass'])
    pred_t_pt = ak.to_numpy(branches['pred_t_pt'])
    pred_t_eta = ak.to_numpy(branches['pred_t_eta'])
    pred_t_phi = ak.to_numpy(branches['pred_t_phi'])
    pred_t_px = ak.to_numpy(branches['pred_t_px'])
    pred_t_py = ak.to_numpy(branches['pred_t_py'])
    pred_t_pz = ak.to_numpy(branches['pred_t_pz'])
    pred_tbar_mass = ak.to_numpy(branches['pred_tbar_mass'])
    pred_tbar_pt = ak.to_numpy(branches['pred_tbar_pt'])
    pred_tbar_eta = ak.to_numpy(branches['pred_tbar_eta'])
    pred_tbar_phi = ak.to_numpy(branches['pred_tbar_phi'])
    pred_tbar_px = ak.to_numpy(branches['pred_tbar_px'])
    pred_tbar_py = ak.to_numpy(branches['pred_tbar_py'])
    pred_tbar_pz = ak.to_numpy(branches['pred_tbar_pz'])
    
    
    gen_t_mass = ak.to_numpy(branches['gen_t_mass'])
    gen_t_pt = ak.to_numpy(branches['gen_t_pt'])
    gen_t_eta = ak.to_numpy(branches['gen_t_eta'])
    gen_t_phi = ak.to_numpy(branches['gen_t_phi'])
    gen_t_px = ak.to_numpy(branches['gen_t_px'])
    gen_t_py = ak.to_numpy(branches['gen_t_py'])
    gen_t_pz = ak.to_numpy(branches['gen_t_pz'])
    gen_tbar_mass = ak.to_numpy(branches['gen_tbar_mass'])
    gen_tbar_pt = ak.to_numpy(branches['gen_tbar_pt'])
    gen_tbar_eta = ak.to_numpy(branches['gen_tbar_eta'])
    gen_tbar_phi = ak.to_numpy(branches['gen_tbar_phi'])
    gen_tbar_px = ak.to_numpy(branches['gen_tbar_px'])
    gen_tbar_py = ak.to_numpy(branches['gen_tbar_py'])
    gen_tbar_pz = ak.to_numpy(branches['gen_tbar_pz'])
    
    pred_Wp_mass = ak.to_numpy(branches['pred_Wp_mass'])
    pred_Wp_pt = ak.to_numpy(branches['pred_Wp_pt'])
    pred_Wp_eta = ak.to_numpy(branches['pred_Wp_eta'])
    pred_Wp_phi = ak.to_numpy(branches['pred_Wp_phi'])
    pred_Wp_px = ak.to_numpy(branches['pred_Wp_px'])
    pred_Wp_py = ak.to_numpy(branches['pred_Wp_py'])
    pred_Wp_pz = ak.to_numpy(branches['pred_Wp_pz'])
    pred_Wm_mass = ak.to_numpy(branches['pred_Wm_mass'])
    pred_Wm_pt = ak.to_numpy(branches['pred_Wm_pt'])
    pred_Wm_eta = ak.to_numpy(branches['pred_Wm_eta'])
    pred_Wm_phi = ak.to_numpy(branches['pred_Wm_phi'])
    pred_Wm_px = ak.to_numpy(branches['pred_Wm_px'])
    pred_Wm_py = ak.to_numpy(branches['pred_Wm_py'])
    pred_Wm_pz = ak.to_numpy(branches['pred_Wm_pz'])
    
    
    gen_Wp_mass = ak.to_numpy(branches['gen_Wp_mass'])
    gen_Wp_pt = ak.to_numpy(branches['gen_Wp_pt'])
    gen_Wp_eta = ak.to_numpy(branches['gen_Wp_eta'])
    gen_Wp_phi = ak.to_numpy(branches['gen_Wp_phi'])
    gen_Wp_px = ak.to_numpy(branches['gen_Wp_px'])
    gen_Wp_py = ak.to_numpy(branches['gen_Wp_py'])
    gen_Wp_pz = ak.to_numpy(branches['gen_Wp_pz'])
    gen_Wm_mass = ak.to_numpy(branches['gen_Wm_mass'])
    gen_Wm_pt = ak.to_numpy(branches['gen_Wm_pt'])
    gen_Wm_eta = ak.to_numpy(branches['gen_Wm_eta'])
    gen_Wm_phi = ak.to_numpy(branches['gen_Wm_phi'])
    gen_Wm_px = ak.to_numpy(branches['gen_Wm_px'])
    gen_Wm_py = ak.to_numpy(branches['gen_Wm_py'])
    gen_Wm_pz = ak.to_numpy(branches['gen_Wm_pz'])
    
    reco_l_pt = ak.to_numpy(branches['reco_l_pt'])
    reco_lbar_pt = ak.to_numpy(branches['reco_lbar_pt'])
    reco_b_pt = ak.to_numpy(branches['reco_b_pt'])
    reco_bbar_pt = ak.to_numpy(branches['reco_bbar_pt'])
    reco_met_pt = ak.to_numpy(branches['reco_met_pt'])
    reco_l_phi = ak.to_numpy(branches['reco_l_phi'])
    reco_lbar_phi = ak.to_numpy(branches['reco_lbar_phi'])
    reco_b_phi = ak.to_numpy(branches['reco_b_phi'])
    reco_bbar_phi = ak.to_numpy(branches['reco_bbar_phi'])
    reco_met_phi = ak.to_numpy(branches['reco_met_phi'])
    reco_l_eta = ak.to_numpy(branches['reco_l_eta'])
    reco_lbar_eta = ak.to_numpy(branches['reco_lbar_eta'])
    reco_b_eta = ak.to_numpy(branches['reco_b_eta'])
    reco_bbar_eta = ak.to_numpy(branches['reco_bbar_eta'])
    reco_met_eta = ak.to_numpy(branches['reco_met_eta'])
    reco_l_mass = ak.to_numpy(branches['reco_l_mass'])
    reco_lbar_mass = ak.to_numpy(branches['reco_lbar_mass'])
    reco_b_mass = ak.to_numpy(branches['reco_b_mass'])
    reco_bbar_mass = ak.to_numpy(branches['reco_bbar_mass'])
    reco_met_mass = ak.to_numpy(branches['reco_met_mass'])
    reco_t_pt = ak.to_numpy(branches['reco_t_pt'])
    reco_t_eta = ak.to_numpy(branches['reco_t_eta'])
    reco_t_phi = ak.to_numpy(branches['reco_t_phi'])
    reco_t_mass = ak.to_numpy(branches['reco_t_mass'])
    reco_t_px = ak.to_numpy(branches['reco_t_px'])
    reco_t_py = ak.to_numpy(branches['reco_t_py'])
    reco_t_pz = ak.to_numpy(branches['reco_t_pz'])
    reco_tbar_pt = ak.to_numpy(branches['reco_tbar_pt'])
    reco_tbar_eta = ak.to_numpy(branches['reco_tbar_eta'])
    reco_tbar_phi = ak.to_numpy(branches['reco_tbar_phi'])
    reco_tbar_mass = ak.to_numpy(branches['reco_tbar_mass'])
    reco_tbar_px = ak.to_numpy(branches['reco_tbar_px'])
    reco_tbar_py = ak.to_numpy(branches['reco_tbar_py'])
    reco_tbar_pz = ak.to_numpy(branches['reco_tbar_pz'])
    

else:
    gen_means = torch.load(BASE_DIR + 'gen_means.pt')
    reco_means = torch.load(BASE_DIR + 'reco_means.pt')
    reco_stdevs = torch.load(BASE_DIR + 'reco_stdevs.pt')
    gen_stdevs = torch.load(BASE_DIR + 'gen_stdevs.pt')
    
    truths = torch.load(BASE_DIR + 'truths.pt')
    predictions = torch.load(BASE_DIR + 'predictions.pt')


    gen_l_pt = []
    gen_lbar_pt = []
    gen_b_pt = []
    gen_bbar_pt = []
    gen_n_pt = []
    gen_nbar_pt = []
    gen_l_eta = []
    gen_lbar_eta = []
    gen_b_eta = []
    gen_bbar_eta = []
    gen_n_eta = []
    gen_nbar_eta = []
    gen_l_phi = []
    gen_lbar_phi = []
    gen_b_phi = []
    gen_bbar_phi = []
    gen_n_phi = []
    gen_nbar_phi = []
    gen_l_mass = []
    gen_lbar_mass = []
    gen_b_mass = []
    gen_bbar_mass = []
    gen_n_mass = []
    gen_nbar_mass = []
    gen_met_pt = []
    gen_met_phi = []
    gen_met_eta = []
    gen_met_mass = []
    
    pred_l_pt = []
    pred_lbar_pt = []
    pred_b_pt = []
    pred_bbar_pt = []
    pred_n_pt = []
    pred_nbar_pt = []
    pred_l_eta = []
    pred_lbar_eta = []
    pred_b_eta = []
    pred_bbar_eta = []
    pred_n_eta = []
    pred_nbar_eta = []
    pred_l_phi = []
    pred_lbar_phi = []
    pred_b_phi = []
    pred_bbar_phi = []
    pred_n_phi = []
    pred_nbar_phi = []
    pred_l_mass = []
    pred_lbar_mass = []
    pred_b_mass = []
    pred_bbar_mass = []
    pred_n_mass = []
    pred_nbar_mass = []
    pred_met_pt = []
    pred_met_phi = []
    pred_met_mass = []
    pred_met_eta = []
    
    pred_t_mass = []
    pred_t_pt = []
    pred_t_eta = []
    pred_t_phi = []
    pred_t_px = []
    pred_t_py = []
    pred_t_pz = []
    pred_tbar_mass = []
    pred_tbar_pt = []
    pred_tbar_eta = []
    pred_tbar_phi = []
    pred_tbar_px = []
    pred_tbar_py = []
    pred_tbar_pz = []

    pred_ttbar_mass = []
    pred_ttbar_pt = []
    pred_ttbar_eta = []
    pred_ttbar_phi = []
    pred_ttbar_px = []
    pred_ttbar_py = []
    pred_ttbar_pz = []
    
    
    gen_t_mass = []
    gen_t_pt = []
    gen_t_eta = []
    gen_t_phi = []
    gen_t_px = []
    gen_t_py = []
    gen_t_pz = []
    gen_tbar_mass = []
    gen_tbar_pt = []
    gen_tbar_eta = []
    gen_tbar_phi = []
    gen_tbar_px = []
    gen_tbar_py = []
    gen_tbar_pz = []

    gen_ttbar_mass = []
    gen_ttbar_pt = []
    gen_ttbar_eta = []
    gen_ttbar_phi = []
    gen_ttbar_px = []
    gen_ttbar_py = []
    gen_ttbar_pz = []
    
    pred_Wp_mass = []
    pred_Wp_pt = []
    pred_Wp_eta = []
    pred_Wp_phi = []
    pred_Wp_px = []
    pred_Wp_py = []
    pred_Wp_pz = []
    pred_Wm_mass = []
    pred_Wm_pt = []
    pred_Wm_eta = []
    pred_Wm_phi = []
    pred_Wm_px = []
    pred_Wm_py = []
    pred_Wm_pz = []
    
    
    gen_Wp_mass = []
    gen_Wp_pt = []
    gen_Wp_eta = []
    gen_Wp_phi = []
    gen_Wp_px = []
    gen_Wp_py = []
    gen_Wp_pz = []
    gen_Wm_mass = []
    gen_Wm_pt = []
    gen_Wm_eta = []
    gen_Wm_phi = []
    gen_Wm_px = []
    gen_Wm_py = []
    gen_Wm_pz = []
    
    
    reco_l_pt = []
    reco_lbar_pt = []
    reco_met_pt = []
    reco_l_eta = []
    reco_lbar_eta = []
    reco_met_eta = []
    reco_l_phi = []
    reco_lbar_phi = []
    reco_met_phi = []
    reco_l_mass = []
    reco_lbar_mass = []
    reco_met_mass = []
    
    e_mass = 0.000510999
    muon_mass = 0.1056584
    
    count = 0
    
    for i in tqdm(range(len(predictions))):
        B, _, _ = predictions[i].shape
        for j in range(B):
            # GEN COMPONENTS
            
            gen_l_pt.append(np.exp((truths[i][j, -6, 0] * gen_stdevs[0, 0]) + gen_means[0, 0]))
            gen_lbar_pt.append(np.exp((truths[i][j, -5, 0] * gen_stdevs[1, 0]) + gen_means[1, 0]))
            gen_b_pt.append(np.exp((truths[i][j, -4, 0] * gen_stdevs[2, 0]) + gen_means[2, 0]))
            gen_bbar_pt.append(np.exp((truths[i][j, -3, 0] * gen_stdevs[3, 0]) + gen_means[3, 0]))
            gen_n_pt.append(np.exp((truths[i][j, -2, 0] * gen_stdevs[4, 0]) + gen_means[4, 0]))
            gen_nbar_pt.append(np.exp((truths[i][j, -1, 0] * gen_stdevs[5, 0]) + gen_means[5, 0]))

            gen_l_eta.append((truths[i][j, -6, 1] * gen_stdevs[0, 1]) + gen_means[0, 1])
            gen_lbar_eta.append((truths[i][j, -5, 1] * gen_stdevs[1, 1]) + gen_means[1, 1])
            gen_b_eta.append((truths[i][j, -4, 1] * gen_stdevs[2, 1]) + gen_means[2, 1])
            gen_bbar_eta.append((truths[i][j, -3, 1] * gen_stdevs[3, 1]) + gen_means[3, 1])
            gen_n_eta.append((truths[i][j, -2, 1] * gen_stdevs[4, 1]) + gen_means[4, 1])
            gen_nbar_eta.append((truths[i][j, -1, 1] * gen_stdevs[5, 1]) + gen_means[5, 1])

            gen_l_phi.append((truths[i][j, -6, 2] * gen_stdevs[0, 2]) + gen_means[0, 2])
            gen_lbar_phi.append((truths[i][j, -5, 2] * gen_stdevs[1, 2]) + gen_means[1, 2])
            gen_b_phi.append((truths[i][j, -4, 2] * gen_stdevs[2, 2]) + gen_means[2, 2])
            gen_bbar_phi.append((truths[i][j, -3, 2] * gen_stdevs[3, 2]) + gen_means[3, 2])
            gen_n_phi.append((truths[i][j, -2, 2] * gen_stdevs[4, 2]) + gen_means[4, 2])
            gen_nbar_phi.append((truths[i][j, -1, 2] * gen_stdevs[5, 2]) + gen_means[5, 2])

            gen_l_mass.append((truths[i][j, -6, 3] * gen_stdevs[0, 3]) + gen_means[0, 3])
            gen_lbar_mass.append((truths[i][j, -5, 3] * gen_stdevs[1, 3]) + gen_means[1, 3])
            gen_b_mass.append((truths[i][j, -4, 3] * gen_stdevs[2, 3]) + gen_means[2, 3])
            gen_bbar_mass.append((truths[i][j, -3, 3] * gen_stdevs[3, 3]) + gen_means[3, 3])
            gen_n_mass.append((truths[i][j, -2, 3] * gen_stdevs[4, 3]) + gen_means[4, 3])
            gen_nbar_mass.append((truths[i][j, -1, 3] * gen_stdevs[5, 3]) + gen_means[5, 3])


            # RECO COMPONENTS
            
            reco_l_pt.append(np.exp((truths[i][j, 0, 0] * reco_stdevs[0, 0]) + reco_means[0, 0]))
            reco_lbar_pt.append(np.exp((truths[i][j, 1, 0] * gen_stdevs[1, 0]) + reco_means[1, 0]))
            reco_met_pt.append(np.exp((truths[i][j, 2, 0] * reco_stdevs[2, 0]) + reco_means[2, 0]))
            
            temp_reco_l_eta = (truths[i][j, 0, 1] * reco_stdevs[0, 1]) + reco_means[0, 1]
            temp_reco_lbar_eta = (truths[i][j, 1, 1] * gen_stdevs[1, 1]) + reco_means[1, 1]

            reco_l_eta.append(temp_reco_l_eta)
            reco_lbar_eta.append(temp_reco_lbar_eta)
            reco_met_eta.append((truths[i][j, 2, 1] * reco_stdevs[2, 1]) + reco_means[2, 1])
            
            temp_reco_l_phi = (truths[i][j, 0, 2] * reco_stdevs[0, 2]) + reco_means[0, 2]
            temp_reco_lbar_phi = (truths[i][j, 1, 2] * gen_stdevs[1, 2]) + reco_means[1, 2]
            
            reco_l_phi.append(temp_reco_l_phi)
            reco_lbar_phi.append(temp_reco_lbar_phi)
            reco_met_phi.append((truths[i][j, 2, 2] * reco_stdevs[2, 2]) + reco_means[2, 2])

            reco_l_mass.append((truths[i][j, 0, 3] * reco_stdevs[0, 3]) + reco_means[0, 3])
            reco_lbar_mass.append((truths[i][j, 1, 3] * gen_stdevs[1, 3]) + reco_means[1, 3])
            reco_met_mass.append((truths[i][j, 2, 3] * reco_stdevs[2, 3]) + reco_means[2, 3])
            
            
            # PRED COMPONENTS
            
            pred_l_pt.append(np.exp((predictions[i][j, -6, 0] * gen_stdevs[0, 0]) + gen_means[0, 0]))
            pred_lbar_pt.append(np.exp((predictions[i][j, -5, 0] * gen_stdevs[1, 0]) + gen_means[1, 0]))
            pred_b_pt.append(np.exp((predictions[i][j, -4, 0] * gen_stdevs[2, 0]) + gen_means[2, 0]))
            pred_bbar_pt.append(np.exp((predictions[i][j, -3, 0] * gen_stdevs[3, 0]) + gen_means[3, 0]))
            pred_n_pt.append(np.exp((predictions[i][j, -2, 0] * gen_stdevs[4, 0]) + gen_means[4, 0]))
            pred_nbar_pt.append(np.exp((predictions[i][j, -1, 0] * gen_stdevs[5, 0]) + gen_means[5, 0]))

            pred_l_eta.append(temp_reco_l_eta)
            pred_lbar_eta.append(temp_reco_lbar_eta)
            pred_b_eta.append((predictions[i][j, -4, 1] * gen_stdevs[2, 1]) + gen_means[2, 1])
            pred_bbar_eta.append((predictions[i][j, -3, 1] * gen_stdevs[3, 1]) + gen_means[3, 1])
            pred_n_eta.append((predictions[i][j, -2, 1] * gen_stdevs[4, 1]) + gen_means[4, 1])
            pred_nbar_eta.append((predictions[i][j, -1, 1] * gen_stdevs[5, 1]) + gen_means[5, 1])

            pred_l_phi.append(temp_reco_l_phi)
            pred_lbar_phi.append(temp_reco_lbar_phi)
            pred_b_phi.append((predictions[i][j, -4, 2] * gen_stdevs[2, 2]) + gen_means[2, 2])
            pred_bbar_phi.append((predictions[i][j, -3, 2] * gen_stdevs[3, 2]) + gen_means[3, 2])
            pred_n_phi.append((predictions[i][j, -2, 2] * gen_stdevs[4, 2]) + gen_means[4, 2])
            pred_nbar_phi.append((predictions[i][j, -1, 2] * gen_stdevs[5, 2]) + gen_means[5, 2])
            
            temp_pred_l_mass = (predictions[i][j, -6, 3] * gen_stdevs[0, 3]) + gen_means[0, 3]
            temp_pred_lbar_mass = (predictions[i][j, -5, 3] * gen_stdevs[1, 3]) + gen_means[1, 3]
            
            if temp_pred_l_mass > muon_mass:
                temp_pred_l_mass = muon_mass
            elif temp_pred_l_mass < e_mass:
                temp_pred_l_mass = e_mass
            if temp_pred_lbar_mass > muon_mass:
                temp_pred_lbar_mass = muon_mass
            elif temp_pred_lbar_mass < e_mass:
                temp_pred_lbar_mass = e_mass
            
            pred_l_mass.append(temp_pred_l_mass)
            pred_lbar_mass.append(temp_pred_lbar_mass)
            pred_b_mass.append((predictions[i][j, -4, 3] * gen_stdevs[2, 3]) + gen_means[2, 3])
            pred_bbar_mass.append((predictions[i][j, -3, 3] * gen_stdevs[3, 3]) + gen_means[3, 3])
            pred_n_mass.append((predictions[i][j, -2, 3] * gen_stdevs[4, 3]) + gen_means[4, 3])
            pred_nbar_mass.append((predictions[i][j, -1, 3] * gen_stdevs[5, 3]) + gen_means[5, 3])

            # t

            t_temp_pred = vector.obj(pt = pred_b_pt[count], phi = pred_b_phi[count], eta = pred_b_eta[count], mass = pred_b_mass[count]) + \
                vector.obj(pt = pred_n_pt[count], phi = pred_n_phi[count], eta = pred_n_eta[count], mass = pred_n_mass[count]) + \
                vector.obj(pt = pred_lbar_pt[count], phi = pred_lbar_phi[count], eta = pred_lbar_eta[count], mass = pred_lbar_mass[count])
    
            t_temp_gen = vector.obj(pt=gen_b_pt[count], phi=gen_b_phi[count], eta=gen_b_eta[count], mass=gen_b_mass[count]) + \
                  vector.obj(pt=gen_n_pt[count], phi=gen_n_phi[count], eta=gen_n_eta[count], mass=gen_n_mass[count]) + \
                  vector.obj(pt=gen_lbar_pt[count], phi=gen_lbar_phi[count], eta=gen_lbar_eta[count], mass=gen_lbar_mass[count])
    
            # tbar
            tbar_temp_pred = vector.obj(pt=pred_bbar_pt[count], phi=pred_bbar_phi[count], eta=pred_bbar_eta[count], mass=pred_bbar_mass[count]) + \
                      vector.obj(pt=pred_nbar_pt[count], phi=pred_nbar_phi[count], eta=pred_nbar_eta[count], mass=pred_nbar_mass[count]) + \
                      vector.obj(pt=pred_l_pt[count], phi=pred_l_phi[count], eta=pred_l_eta[count], mass=pred_l_mass[count])
    
            tbar_temp_gen = vector.obj(pt=gen_bbar_pt[count], phi=gen_bbar_phi[count], eta=gen_bbar_eta[count], mass=gen_bbar_mass[count]) + \
                     vector.obj(pt=gen_nbar_pt[count], phi=gen_nbar_phi[count], eta=gen_nbar_eta[count], mass=gen_nbar_mass[count]) + \
                     vector.obj(pt=gen_l_pt[count], phi=gen_l_phi[count], eta=gen_l_eta[count], mass=gen_l_mass[count])

            # ttbar combined
            ttbar_temp_pred = t_temp_pred + tbar_temp_pred
            ttbar_temp_gen = t_temp_gen + tbar_temp_gen
            
            # W+
            Wp_temp_pred = vector.obj(pt = pred_n_pt[count], phi = pred_n_phi[count], eta = pred_n_eta[count], mass = pred_n_mass[count]) + \
                vector.obj(pt = pred_lbar_pt[count], phi = pred_lbar_phi[count], eta = pred_lbar_eta[count], mass = pred_lbar_mass[count])
    
            Wp_temp_gen = vector.obj(pt=gen_n_pt[count], phi=gen_n_phi[count], eta=gen_n_eta[count], mass=gen_n_mass[count]) + \
                  vector.obj(pt=gen_lbar_pt[count], phi=gen_lbar_phi[count], eta=gen_lbar_eta[count], mass=gen_lbar_mass[count])
                  
            # W-
            Wm_temp_pred = vector.obj(pt=pred_nbar_pt[count], phi=pred_nbar_phi[count], eta=pred_nbar_eta[count], mass=pred_nbar_mass[count]) + \
                      vector.obj(pt=pred_l_pt[count], phi=pred_l_phi[count], eta=pred_l_eta[count], mass=pred_l_mass[count])
    
            Wm_temp_gen = vector.obj(pt=gen_nbar_pt[count], phi=gen_nbar_phi[count], eta=gen_nbar_eta[count], mass=gen_nbar_mass[count]) + \
                     vector.obj(pt=gen_l_pt[count], phi=gen_l_phi[count], eta=gen_l_eta[count], mass=gen_l_mass[count])
                     
            # accounting met
            met_temp_pred = vector.obj(pt=pred_nbar_pt[count], phi=pred_nbar_phi[count], eta=pred_nbar_eta[count], mass=pred_nbar_mass[count]) + \
                    vector.obj(pt=pred_n_pt[count], phi=pred_n_phi[count], eta=pred_n_eta[count], mass=pred_n_mass[count])
            
            met_temp_gen = vector.obj(pt=gen_nbar_pt[count], phi=gen_nbar_phi[count], eta=gen_nbar_eta[count], mass=gen_nbar_mass[count]) + \
                    vector.obj(pt=gen_n_pt[count], phi=gen_n_phi[count], eta=gen_n_eta[count], mass=gen_n_mass[count])
            
            
            # t/tbar data
            pred_t_mass.append(t_temp_pred.mass)
            pred_t_pt.append(t_temp_pred.pt)
            pred_t_eta.append(t_temp_pred.eta)
            pred_t_phi.append(t_temp_pred.phi)
            pred_t_px.append(t_temp_pred.px)
            pred_t_py.append(t_temp_pred.py)
            pred_t_pz.append(t_temp_pred.pz)
    
            pred_tbar_mass.append(tbar_temp_pred.mass)
            pred_tbar_pt.append(tbar_temp_pred.pt)
            pred_tbar_eta.append(tbar_temp_pred.eta)
            pred_tbar_phi.append(tbar_temp_pred.phi)
            pred_tbar_px.append(tbar_temp_pred.px)
            pred_tbar_py.append(tbar_temp_pred.py)
            pred_tbar_pz.append(tbar_temp_pred.pz)
    
            gen_t_mass.append(t_temp_gen.mass)
            gen_t_pt.append(t_temp_gen.pt)
            gen_t_eta.append(t_temp_gen.eta)
            gen_t_phi.append(t_temp_gen.phi)
            gen_t_px.append(t_temp_gen.px)
            gen_t_py.append(t_temp_gen.py)
            gen_t_pz.append(t_temp_gen.pz)
    
            gen_tbar_mass.append(tbar_temp_gen.mass)
            gen_tbar_pt.append(tbar_temp_gen.pt)
            gen_tbar_eta.append(tbar_temp_gen.eta)
            gen_tbar_phi.append(tbar_temp_gen.phi)
            gen_tbar_px.append(tbar_temp_gen.px)
            gen_tbar_py.append(tbar_temp_gen.py)
            gen_tbar_pz.append(tbar_temp_gen.pz)

            # ttbar

            pred_ttbar_mass.append(ttbar_temp_pred.mass)
            pred_ttbar_pt.append(ttbar_temp_pred.pt)
            pred_ttbar_eta.append(ttbar_temp_pred.eta)
            pred_ttbar_phi.append(ttbar_temp_pred.phi)
            pred_ttbar_px.append(ttbar_temp_pred.px)
            pred_ttbar_py.append(ttbar_temp_pred.py)
            pred_ttbar_pz.append(ttbar_temp_pred.pz)

            gen_ttbar_mass.append(ttbar_temp_gen.mass)
            gen_ttbar_pt.append(ttbar_temp_gen.pt)
            gen_ttbar_eta.append(ttbar_temp_gen.eta)
            gen_ttbar_phi.append(ttbar_temp_gen.phi)
            gen_ttbar_px.append(ttbar_temp_gen.px)
            gen_ttbar_py.append(ttbar_temp_gen.py)
            gen_ttbar_pz.append(ttbar_temp_gen.pz)
            
            
            # W+- data
            pred_Wp_mass.append(Wp_temp_pred.mass)
            pred_Wp_pt.append(Wp_temp_pred.pt)
            pred_Wp_eta.append(Wp_temp_pred.eta)
            pred_Wp_phi.append(Wp_temp_pred.phi)
            pred_Wp_px.append(Wp_temp_pred.px)
            pred_Wp_py.append(Wp_temp_pred.py)
            pred_Wp_pz.append(Wp_temp_pred.pz)
    
            pred_Wm_mass.append(Wm_temp_pred.mass)
            pred_Wm_pt.append(Wm_temp_pred.pt)
            pred_Wm_eta.append(Wm_temp_pred.eta)
            pred_Wm_phi.append(Wm_temp_pred.phi)
            pred_Wm_px.append(Wm_temp_pred.px)
            pred_Wm_py.append(Wm_temp_pred.py)
            pred_Wm_pz.append(Wm_temp_pred.pz)
    
            gen_Wp_mass.append(Wp_temp_gen.mass)
            gen_Wp_pt.append(Wp_temp_gen.pt)
            gen_Wp_eta.append(Wp_temp_gen.eta)
            gen_Wp_phi.append(Wp_temp_gen.phi)
            gen_Wp_px.append(Wp_temp_gen.px)
            gen_Wp_py.append(Wp_temp_gen.py)
            gen_Wp_pz.append(Wp_temp_gen.pz)
    
            gen_Wm_mass.append(Wm_temp_gen.mass)
            gen_Wm_pt.append(Wm_temp_gen.pt)
            gen_Wm_eta.append(Wm_temp_gen.eta)
            gen_Wm_phi.append(Wm_temp_gen.phi)
            gen_Wm_px.append(Wm_temp_gen.px)
            gen_Wm_py.append(Wm_temp_gen.py)
            gen_Wm_pz.append(Wm_temp_gen.pz)
            
            # met statistics
            gen_met_pt.append(met_temp_gen.pt)
            gen_met_phi.append(met_temp_gen.phi)
            gen_met_eta.append(met_temp_gen.eta)
            gen_met_mass.append(met_temp_gen.mass)
            pred_met_pt.append(met_temp_pred.pt)
            pred_met_phi.append(met_temp_pred.phi)
            pred_met_eta.append(met_temp_pred.eta)
            pred_met_mass.append(met_temp_pred.mass)
            
            # at end
            count += 1

reconstructed_events_file = uproot.recreate(SAVE_DIR + 'predictions.root')
reconstructed_events_file["Bumblebee"] = {
            "gen_l_pt": gen_l_pt,
            "gen_lbar_pt": gen_lbar_pt,
            "gen_b_pt": gen_b_pt,
            "gen_bbar_pt": gen_bbar_pt,
            "gen_n_pt": gen_n_pt,
            "gen_nbar_pt": gen_nbar_pt,
            "gen_l_eta": gen_l_eta,
            "gen_lbar_eta": gen_lbar_eta,
            "gen_b_eta": gen_b_eta,
            "gen_bbar_eta": gen_bbar_eta,
            "gen_n_eta": gen_n_eta,
            "gen_nbar_eta": gen_nbar_eta,
            "gen_l_phi": gen_l_phi,
            "gen_lbar_phi": gen_lbar_phi,
            "gen_b_phi": gen_b_phi,
            "gen_bbar_phi": gen_bbar_phi,
            "gen_n_phi": gen_n_phi,
            "gen_nbar_phi": gen_nbar_phi,
            "gen_l_mass": gen_l_mass,
            "gen_lbar_mass": gen_lbar_mass,
            "gen_b_mass": gen_b_mass,
            "gen_bbar_mass": gen_bbar_mass,
            "gen_n_mass": gen_n_mass,
            "gen_nbar_mass": gen_nbar_mass,
            "pred_l_pt": pred_l_pt,
            "pred_lbar_pt": pred_lbar_pt,
            "pred_b_pt": pred_b_pt,
            "pred_bbar_pt": pred_bbar_pt,
            "pred_n_pt": pred_n_pt,
            "pred_nbar_pt": pred_nbar_pt,
            "pred_l_eta": pred_l_eta,
            "pred_lbar_eta": pred_lbar_eta,
            "pred_b_eta": pred_b_eta,
            "pred_bbar_eta": pred_bbar_eta,
            "pred_n_eta": pred_n_eta,
            "pred_nbar_eta": pred_nbar_eta,
            "pred_l_phi": pred_l_phi,
            "pred_lbar_phi": pred_lbar_phi,
            "pred_b_phi": pred_b_phi,
            "pred_bbar_phi": pred_bbar_phi,
            "pred_n_phi": pred_n_phi,
            "pred_nbar_phi": pred_nbar_phi,
            "pred_l_mass": pred_l_mass,
            "pred_lbar_mass": pred_lbar_mass,
            "pred_b_mass": pred_b_mass,
            "pred_bbar_mass": pred_bbar_mass,
            "pred_n_mass": pred_n_mass,
            "pred_nbar_mass": pred_nbar_mass,
            "pred_t_mass": pred_t_mass,
            "pred_t_pt": pred_t_pt,
            "pred_t_eta": pred_t_eta,
            "pred_t_phi": pred_t_phi,
            "pred_t_px": pred_t_px,
            "pred_t_py": pred_t_py,
            "pred_t_pz": pred_t_pz,
            "pred_tbar_mass": pred_tbar_mass,
            "pred_tbar_pt": pred_tbar_pt,
            "pred_tbar_eta": pred_tbar_eta,
            "pred_tbar_phi": pred_tbar_phi,
            "pred_tbar_px": pred_tbar_px,
            "pred_tbar_py": pred_tbar_py,
            "pred_tbar_pz": pred_tbar_pz,
            "pred_ttbar_mass": pred_ttbar_mass,
            "pred_ttbar_pt": pred_ttbar_pt,
            "pred_ttbar_eta": pred_ttbar_eta,
            "pred_ttbar_phi": pred_ttbar_phi,
            "pred_ttbar_px": pred_ttbar_px,
            "pred_ttbar_py": pred_ttbar_py,
            "pred_ttbar_pz": pred_ttbar_pz,
            "gen_t_mass": gen_t_mass,
            "gen_t_pt": gen_t_pt,
            "gen_t_eta": gen_t_eta,
            "gen_t_phi": gen_t_phi,
            "gen_t_px": gen_t_px,
            "gen_t_py": gen_t_py,
            "gen_t_pz": gen_t_pz,
            "gen_tbar_mass": gen_tbar_mass,
            "gen_tbar_pt": gen_tbar_pt,
            "gen_tbar_eta": gen_tbar_eta,
            "gen_tbar_phi": gen_tbar_phi,
            "gen_tbar_px": gen_tbar_px,
            "gen_tbar_py": gen_tbar_py,
            "gen_tbar_pz": gen_tbar_pz,
            "gen_ttbar_mass": gen_ttbar_mass,
            "gen_ttbar_pt": gen_ttbar_pt,
            "gen_ttbar_eta": gen_ttbar_eta,
            "gen_ttbar_phi": gen_ttbar_phi,
            "gen_ttbar_px": gen_ttbar_px,
            "gen_ttbar_py": gen_ttbar_py,
            "gen_ttbar_pz": gen_ttbar_pz,
            "pred_Wp_mass": pred_Wp_mass,
            "pred_Wp_pt": pred_Wp_pt,
            "pred_Wp_eta": pred_Wp_eta,
            "pred_Wp_phi": pred_Wp_phi,
            "pred_Wp_px": pred_Wp_px,
            "pred_Wp_py": pred_Wp_py,
            "pred_Wp_pz": pred_Wp_pz,
            "pred_Wm_mass": pred_Wm_mass,
            "pred_Wm_pt": pred_Wm_pt,
            "pred_Wm_eta": pred_Wm_eta,
            "pred_Wm_phi": pred_Wm_phi,
            "pred_Wm_px": pred_Wm_px,
            "pred_Wm_py": pred_Wm_py,
            "pred_Wm_pz": pred_Wm_pz,
            "gen_Wp_mass": gen_Wp_mass,
            "gen_Wp_pt": gen_Wp_pt,
            "gen_Wp_eta": gen_Wp_eta,
            "gen_Wp_phi": gen_Wp_phi,
            "gen_Wp_px": gen_Wp_px,
            "gen_Wp_py": gen_Wp_py,
            "gen_Wp_pz": gen_Wp_pz,
            "gen_Wm_mass": gen_Wm_mass,
            "gen_Wm_pt": gen_Wm_pt,
            "gen_Wm_eta": gen_Wm_eta,
            "gen_Wm_phi": gen_Wm_phi,
            "gen_Wm_px": gen_Wm_px,
            "gen_Wm_py": gen_Wm_py,
            "gen_Wm_pz": gen_Wm_pz,
            "reco_l_pt": reco_l_pt,
            "reco_lbar_pt": reco_lbar_pt,
            "reco_met_pt": reco_met_pt,
            "reco_l_eta": reco_l_eta,
            "reco_lbar_eta": reco_lbar_eta,
            "reco_met_eta": reco_met_eta,
            "reco_l_phi": reco_l_phi,
            "reco_lbar_phi": reco_lbar_phi,
            "reco_met_phi": reco_met_phi,
            "reco_l_mass": reco_l_mass,
            "reco_lbar_mass": reco_lbar_mass,
            "reco_met_mass": reco_met_mass,
            "gen_met_pt": gen_met_pt,
            "gen_met_phi": gen_met_phi,
            "gen_met_eta": gen_met_eta,
            "gen_met_mass": gen_met_mass,
            "pred_met_pt": pred_met_pt,
            "pred_met_phi": pred_met_phi,
            "pred_met_eta": pred_met_eta,
            "pred_met_mass": pred_met_mass
        }

if post_as_pngs:
    # mass
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_mass, bins = N_BINS, label='gen l mass', histtype='step')
    _ = plt.hist(pred_l_mass, bins = bins, label='pred l mass', histtype='step')
    _ = plt.hist(reco_l_mass, bins = bins, label='reco l mass', histtype='step')
    plt.title('l mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_mass, bins = N_BINS, label='gen lbar mass', histtype='step')
    _ = plt.hist(pred_lbar_mass, bins = bins, label='pred lbar mass', histtype='step')
    _ = plt.hist(reco_lbar_mass, bins = bins, label='reco lbar mass', histtype='step')
    plt.title('lbar mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbarmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_mass, bins = N_BINS, label='gen b mass', histtype='step')
    _ = plt.hist(pred_b_mass, bins = bins, label='pred b mass', histtype='step')
    plt.title('b mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_mass, bins = N_BINS, label='gen bbar mass', histtype='step')
    _ = plt.hist(pred_bbar_mass, bins = bins, label='pred bbar mass', histtype='step')
    plt.title('bbar mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbarmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_t_mass, bins = N_BINS, label='gen t mass', histtype='step')
    _ = plt.hist(pred_t_mass, bins = bins, label='pred t mass', histtype='step')
    plt.title('t mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_tbar_mass, bins = N_BINS, label='gen tbar mass', histtype='step')
    _ = plt.hist(pred_tbar_mass, bins = bins, label='pred tbar mass', histtype='step')
    plt.title('tbar mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tbarmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wp_mass, bins = N_BINS, label='gen W+ mass', histtype='step')
    _ = plt.hist(pred_Wp_mass, bins = bins, label='pred W+ mass', histtype='step')
    plt.title('W+ mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wpmasscomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wm_mass, bins = N_BINS, label='gen W- mass', histtype='step')
    _ = plt.hist(pred_Wm_mass, bins = bins, label='pred W- mass', histtype='step')
    plt.title('W- mass')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wmmasscomp.png')
    plt.close()
    
    # eta
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_eta, bins = N_BINS, label='gen l eta', histtype='step')
    _ = plt.hist(pred_l_eta, bins = bins, label='pred l eta', histtype='step')
    _ = plt.hist(reco_l_eta, bins = bins, label='reco l eta', histtype='step')
    plt.title('l eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'letacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_eta, bins = N_BINS, label='gen lbar eta', histtype='step')
    _ = plt.hist(pred_lbar_eta, bins = bins, label='pred lbar eta', histtype='step')
    _ = plt.hist(reco_lbar_eta, bins = bins, label='reco lbar eta', histtype='step')
    plt.title('lbar eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbaretacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_eta, bins = N_BINS, label='gen b eta', histtype='step')
    _ = plt.hist(pred_b_eta, bins = bins, label='pred b eta', histtype='step')
    plt.title('b eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'betacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_eta, bins = N_BINS, label='gen bbar eta', histtype='step')
    _ = plt.hist(pred_bbar_eta, bins = bins, label='pred bbar eta', histtype='step')
    plt.title('bbar eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbaretacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_eta, bins = N_BINS, label='gen n eta', histtype='step')
    _ = plt.hist(pred_n_eta, bins = bins, label='pred n eta', histtype='step')
    plt.title('n eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'netacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_nbar_eta, bins = N_BINS, label='gen nbar eta', histtype='step')
    _ = plt.hist(pred_nbar_eta, bins = bins, label='pred nbar eta', histtype='step')
    plt.title('nbar eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'nbaretacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_t_eta, bins = N_BINS, label='gen t eta', histtype='step')
    _ = plt.hist(pred_t_eta, bins = bins, label='pred t eta', histtype='step')
    plt.title('t eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tetacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_tbar_eta, bins = N_BINS, label='gen tbar eta', histtype='step')
    _ = plt.hist(pred_tbar_eta, bins = bins, label='pred tbar eta', histtype='step')
    plt.title('tbar eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tbaretacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wp_eta, bins = N_BINS, label='gen W+ eta', histtype='step')
    _ = plt.hist(pred_Wp_eta, bins = bins, label='pred W+ eta', histtype='step')
    plt.title('W+ eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wpetacomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wm_eta, bins = N_BINS, label='gen W- eta', histtype='step')
    _ = plt.hist(pred_Wm_eta, bins = bins, label='pred W- eta', histtype='step')
    plt.title('W- eta')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wmetacomp.png')
    plt.close()
    
    # phi
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_phi, bins = N_BINS, label='gen l phi', histtype='step')
    _ = plt.hist(pred_l_phi, bins = bins, label='pred l phi', histtype='step')
    _ = plt.hist(reco_l_phi, bins = bins, label='reco l phi', histtype='step')
    plt.title('l phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_phi, bins = N_BINS, label='gen lbar phi', histtype='step')
    _ = plt.hist(pred_lbar_phi, bins = bins, label='pred lbar phi', histtype='step')
    _ = plt.hist(reco_lbar_phi, bins = bins, label='reco lbar phi', histtype='step')
    plt.title('lbar phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbarphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_phi, bins = N_BINS, label='gen b phi', histtype='step')
    _ = plt.hist(pred_b_phi, bins = bins, label='pred b phi', histtype='step')
    plt.title('b phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_phi, bins = N_BINS, label='gen bbar phi', histtype='step')
    _ = plt.hist(pred_bbar_phi, bins = bins, label='pred bbar phi', histtype='step')
    plt.title('bbar phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbarphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_phi, bins = N_BINS, label='gen n phi', histtype='step')
    _ = plt.hist(pred_n_phi, bins = bins, label='pred n phi', histtype='step')
    plt.title('n phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'nphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_nbar_phi, bins = N_BINS, label='gen nbar phi', histtype='step')
    _ = plt.hist(pred_nbar_phi, bins = bins, label='pred nbar phi', histtype='step')
    plt.title('nbar phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'nbarphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_t_phi, bins = N_BINS, label='gen t phi', histtype='step')
    _ = plt.hist(pred_t_phi, bins = bins, label='pred t phi', histtype='step')
    plt.title('t phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_tbar_phi, bins = N_BINS, label='gen tbar phi', histtype='step')
    _ = plt.hist(pred_tbar_phi, bins = bins, label='pred tbar phi', histtype='step')
    plt.title('tbar phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tbarphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wp_phi, bins = N_BINS, label='gen W+ phi', histtype='step')
    _ = plt.hist(pred_Wp_phi, bins = bins, label='pred W+ phi', histtype='step')
    plt.title('W+ phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wpphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wm_phi, bins = N_BINS, label='gen W- phi', histtype='step')
    _ = plt.hist(pred_Wm_phi, bins = bins, label='pred W- phi', histtype='step')
    plt.title('W- phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wmphicomp.png')
    plt.close()
    
    # pt
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_l_pt, bins = N_BINS, label='gen l pt', histtype='step')
    _ = plt.hist(pred_l_pt, bins = bins, label='pred l pt', histtype='step')
    _ = plt.hist(reco_l_pt, bins = bins, label='reco l pt', histtype='step')
    plt.title('l pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_lbar_pt, bins = N_BINS, label='gen lbar pt', histtype='step')
    _ = plt.hist(pred_lbar_pt, bins = bins, label='pred lbar pt', histtype='step')
    _ = plt.hist(reco_lbar_pt, bins = bins, label='reco lbar pt', histtype='step')
    plt.title('lbar pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'lbarptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_b_pt, bins = N_BINS, label='gen b pt', histtype='step')
    _ = plt.hist(pred_b_pt, bins = bins, label='pred b pt', histtype='step')
    plt.title('b pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_pt, bins = N_BINS, label='gen bbar pt', histtype='step')
    _ = plt.hist(pred_bbar_pt, bins = bins, label='pred bbar pt', histtype='step')
    plt.title('bbar pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'bbarptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_bbar_pt, bins = N_BINS, label='gen n pt', histtype='step')
    _ = plt.hist(pred_n_pt, bins = bins, label='pred n pt', histtype='step')
    plt.title('n pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'nptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_nbar_pt, bins = N_BINS, label='gen nbar pt', histtype='step')
    _ = plt.hist(pred_nbar_pt, bins = bins, label='pred nbar pt', histtype='step')
    plt.title('nbar pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'nbarptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_t_pt, bins = N_BINS, label='gen t pt', histtype='step')
    _ = plt.hist(pred_t_pt, bins = bins, label='pred t pt', histtype='step')
    plt.title('t pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_tbar_pt, bins = N_BINS, label='gen tbar pt', histtype='step')
    _ = plt.hist(pred_tbar_pt, bins = bins, label='pred tbar pt', histtype='step')
    plt.title('tbar pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tbarptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wp_pt, bins = N_BINS, label='gen W+ pt', histtype='step')
    _ = plt.hist(pred_Wp_pt, bins = bins, label='pred W+ pt', histtype='step')
    plt.title('W+ pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wpptcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wm_pt, bins = N_BINS, label='gen W- pt', histtype='step')
    _ = plt.hist(pred_Wm_pt, bins = bins, label='pred W- pt', histtype='step')
    plt.title('W- pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wmptcomp.png')
    plt.close()
    
    # px, py, pz
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_t_px, bins = N_BINS, label='gen t px', histtype='step')
    _ = plt.hist(pred_t_px, bins = bins, label='pred t px', histtype='step')
    plt.title('t px')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tpxcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_tbar_px, bins = N_BINS, label='gen tbar px', histtype='step')
    _ = plt.hist(pred_tbar_px, bins = bins, label='pred tbar px', histtype='step')
    plt.title('tbar px')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tbarpxcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wp_px, bins = N_BINS, label='gen W+ px', histtype='step')
    _ = plt.hist(pred_Wp_px, bins = bins, label='pred W+ px', histtype='step')
    plt.title('gen vs pred W+ px')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wppxcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wm_px, bins = N_BINS, label='gen W- px', histtype='step')
    _ = plt.hist(pred_Wm_px, bins = bins, label='pred W- px', histtype='step')
    plt.title('gen vs pred W- px')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wmpxcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_t_py, bins = N_BINS, label='gen t py', histtype='step')
    _ = plt.hist(pred_t_py, bins = bins, label='pred t py', histtype='step')
    plt.title('t py')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tpycomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_tbar_py, bins = N_BINS, label='gen tbar py', histtype='step')
    _ = plt.hist(pred_tbar_py, bins = bins, label='pred tbar py', histtype='step')
    plt.title('tbar py')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tbarpycomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wp_py, bins = N_BINS, label='gen W+ py', histtype='step')
    _ = plt.hist(pred_Wp_py, bins = bins, label='pred W+ py', histtype='step')
    plt.title('gen vs pred W+ py')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wppycomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wm_py, bins = N_BINS, label='gen W- py', histtype='step')
    _ = plt.hist(pred_Wm_py, bins = bins, label='pred W- py', histtype='step')
    plt.title('gen vs pred W- py')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wmpycomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_t_pz, bins = N_BINS, label='gen t pz', histtype='step')
    _ = plt.hist(pred_t_pz, bins = bins, label='pred t pz', histtype='step')
    plt.title('t pz')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tpzcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_tbar_pz, bins = N_BINS, label='gen tbar pz', histtype='step')
    _ = plt.hist(pred_tbar_pz, bins = bins, label='pred tbar pz', histtype='step')
    plt.title('tbar pz')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'tbarpzcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wp_pz, bins = N_BINS, label='gen W+ pz', histtype='step')
    _ = plt.hist(pred_Wp_pz, bins = bins, label='pred W+ pz', histtype='step')
    plt.title('gen vs pred W+ pz')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wppzcomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_Wm_pz, bins = N_BINS, label='gen W- pz', histtype='step')
    _ = plt.hist(pred_Wm_pz, bins = bins, label='pred W- pz', histtype='step')
    plt.title('gen vs pred W- pz')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'wmpzcomp.png')
    plt.close()
    
    # met
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_met_phi, bins = N_BINS, label='gen met phi', histtype='step')
    _ = plt.hist(pred_met_phi, bins = bins, label='pred met phi', histtype='step')
    _ = plt.hist(reco_met_phi, bins = bins, label='reco met phi', histtype='step')
    plt.title('met phi')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'metphicomp.png')
    plt.close()
    
    fig = plt.figure(figsize = (10,8))
    _, bins, _ = plt.hist(gen_met_pt, bins = N_BINS, label='gen met pt', histtype='step')
    _ = plt.hist(pred_met_pt, bins = bins, label='pred met pt', histtype='step')
    _ = plt.hist(reco_met_pt, bins = bins, label='reco met pt', histtype='step')
    plt.title('met pt')
    plt.yscale('log')
    plt.legend()
    plt.savefig(SAVE_DIR + 'metptcomp.png')
    plt.close()
    
    
    # 2d plots
    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(gen_t_px, pred_t_px, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top px plot')
    plt.xlabel(r'gen top px [GeV]')
    plt.ylabel(r'pred top px [GeV]')
    plt.savefig(SAVE_DIR + "top_px_res2d.png")
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(gen_t_py, pred_t_py, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top py plot')
    plt.xlabel(r'gen top py [GeV]')
    plt.ylabel(r'pred top py [GeV]')
    plt.savefig(SAVE_DIR + "top_py_res2d.png")
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(gen_t_pz, pred_t_pz, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top pz plot')
    plt.xlabel(r'gen top pz [GeV]')
    plt.ylabel(r'pred top pz [GeV]')
    plt.savefig(SAVE_DIR + "top_pz_res2d.png")
    plt.close()
    
    
    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(gen_t_mass, pred_t_mass, bins=100, range=[[50, 250],[0, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top mass plot')
    plt.xlabel(r'gen top mass [GeV]')
    plt.ylabel(r'pred top mass [GeV]')
    plt.savefig(SAVE_DIR + "top_mass_res2d.png")
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(gen_t_phi, pred_t_phi, bins=100, range=[[-3.15, 3.15],[-3.15, 3.15]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top phi plot')
    plt.xlabel(r'gen top phi')
    plt.ylabel(r'pred top phi')
    plt.savefig(SAVE_DIR + "top_phi_res2d.png")
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(gen_t_eta, pred_t_eta, bins=100, range=[[-10, 10],[-10, 10]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top eta plot')
    plt.xlabel(r'gen top eta')
    plt.ylabel(r'pred top eta')
    plt.savefig(SAVE_DIR + "top_eta_res2d.png")
    plt.close()
    
    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(gen_t_pt, pred_t_pt, bins=100, range=[[0, 1600],[0, 1600]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top pT plot')
    plt.xlabel(r'gen top pT [GeV]')
    plt.ylabel(r'pred top pT [GeV]')
    plt.savefig(SAVE_DIR + "top_pt_res2d.png")
    plt.close()
    

else:
    
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True
    
    # mass
    fig1 = plt.figure()
    _, bins, _ = plt.hist(gen_l_mass, bins = N_BINS, label='gen l mass', histtype='step')
    _ = plt.hist(pred_l_mass, bins = bins, label='pred l mass', histtype='step')
    _ = plt.hist(reco_l_mass, bins = bins, label='reco l mass', histtype='step')
    plt.title('l mass')
    plt.yscale('log')
    plt.legend()
    
    fig2 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_mass, bins = N_BINS, label='gen lbar mass', histtype='step')
    _ = plt.hist(pred_lbar_mass, bins = bins, label='pred lbar mass', histtype='step')
    _ = plt.hist(reco_lbar_mass, bins = bins, label='reco lbar mass', histtype='step')
    plt.title('lbar mass')
    plt.yscale('log')
    plt.legend()
    
    fig3 = plt.figure()
    _, bins, _ = plt.hist(gen_b_mass, bins = N_BINS, label='gen b mass', histtype='step')
    _ = plt.hist(pred_b_mass, bins = bins, label='pred b mass', histtype='step')
    plt.title('b mass')
    plt.yscale('log')
    plt.legend()
    
    fig4 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_mass, bins = N_BINS, label='gen bbar mass', histtype='step')
    _ = plt.hist(pred_bbar_mass, bins = bins, label='pred bbar mass', histtype='step')
    plt.title('bbar mass')
    plt.yscale('log')
    plt.legend()
    
    fig5 = plt.figure()
    _, bins, _ = plt.hist(gen_t_mass, bins = N_BINS, label='gen t mass', histtype='step')
    _ = plt.hist(pred_t_mass, bins = bins, label='pred t mass', histtype='step')
    plt.title('t mass')
    plt.yscale('log')
    plt.legend()
    
    fig6 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_mass, bins = N_BINS, label='gen tbar mass', histtype='step')
    _ = plt.hist(pred_tbar_mass, bins = bins, label='pred tbar mass', histtype='step')
    plt.title('tbar mass')
    plt.yscale('log')
    plt.legend()
    
    # eta
    fig7 = plt.figure()
    _, bins, _ = plt.hist(gen_l_eta, bins = N_BINS, label='gen l eta', histtype='step')
    _ = plt.hist(pred_l_eta, bins = bins, label='pred l eta', histtype='step')
    _ = plt.hist(reco_l_eta, bins = bins, label='reco l eta', histtype='step')
    plt.title('l eta')
    plt.yscale('log')
    plt.legend()
    
    fig8 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_eta, bins = N_BINS, label='gen lbar eta', histtype='step')
    _ = plt.hist(pred_lbar_eta, bins = bins, label='pred lbar eta', histtype='step')
    _ = plt.hist(reco_lbar_eta, bins = bins, label='reco lbar eta', histtype='step')
    plt.title('lbar eta')
    plt.yscale('log')
    plt.legend()
    
    fig9 = plt.figure()
    _, bins, _ = plt.hist(gen_b_eta, bins = N_BINS, label='gen b eta', histtype='step')
    _ = plt.hist(pred_b_eta, bins = bins, label='pred b eta', histtype='step')
    plt.title('b eta')
    plt.yscale('log')
    plt.legend()
    
    fig10 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_eta, bins = N_BINS, label='gen bbar eta', histtype='step')
    _ = plt.hist(pred_bbar_eta, bins = bins, label='pred bbar eta', histtype='step')
    plt.title('bbar eta')
    plt.yscale('log')
    plt.legend()
    
    fig11 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_eta, bins = N_BINS, label='gen n eta', histtype='step')
    _ = plt.hist(pred_n_eta, bins = bins, label='pred n eta', histtype='step')
    plt.title('gen vs pred n eta')
    plt.yscale('log')
    plt.legend()
    
    fig12 = plt.figure()
    _, bins, _ = plt.hist(gen_nbar_eta, bins = N_BINS, label='gen nbar eta', histtype='step')
    _ = plt.hist(pred_nbar_eta, bins = bins, label='pred nbar eta', histtype='step')
    plt.title('gen vs pred nbar eta')
    plt.yscale('log')
    plt.legend()
    
    fig13 = plt.figure()
    _, bins, _ = plt.hist(gen_t_eta, bins = N_BINS, label='gen t eta', histtype='step')
    _ = plt.hist(pred_t_eta, bins = bins, label='pred t eta', histtype='step')
    plt.title('t eta')
    plt.yscale('log')
    plt.legend()
    
    fig14 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_eta, bins = N_BINS, label='gen tbar eta', histtype='step')
    _ = plt.hist(pred_tbar_eta, bins = bins, label='pred tbar eta', histtype='step')
    plt.title('tbar eta')
    plt.yscale('log')
    plt.legend()
    
    # phi
    fig15 = plt.figure()
    _, bins, _ = plt.hist(gen_l_phi, bins = N_BINS, label='gen l phi', histtype='step')
    _ = plt.hist(pred_l_phi, bins = bins, label='pred l phi', histtype='step')
    _ = plt.hist(reco_l_phi, bins = bins, label='reco l phi', histtype='step')
    plt.title('l phi')
    plt.yscale('log')
    plt.legend()
    
    fig16 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_phi, bins = N_BINS, label='gen lbar phi', histtype='step')
    _ = plt.hist(pred_lbar_phi, bins = bins, label='pred lbar phi', histtype='step')
    plt.title('lbar phi')
    plt.yscale('log')
    plt.legend()
    
    fig17 = plt.figure()
    _, bins, _ = plt.hist(gen_b_phi, bins = N_BINS, label='gen b phi', histtype='step')
    _ = plt.hist(pred_b_phi, bins = bins, label='pred b phi', histtype='step')
    plt.title('b phi')
    plt.yscale('log')
    plt.legend()
    
    fig18 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_phi, bins = N_BINS, label='gen bbar phi', histtype='step')
    _ = plt.hist(pred_bbar_phi, bins = bins, label='pred bbar phi', histtype='step')
    plt.title('bbar phi')
    plt.yscale('log')
    plt.legend()
    
    fig19 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_phi, bins = N_BINS, label='gen n phi', histtype='step')
    _ = plt.hist(pred_n_phi, bins = bins, label='pred n phi', histtype='step')
    plt.title('gen vs pred n phi')
    plt.yscale('log')
    plt.legend()
    
    fig20 = plt.figure()
    _, bins, _ = plt.hist(gen_nbar_phi, bins = N_BINS, label='gen nbar phi', histtype='step')
    _ = plt.hist(pred_nbar_phi, bins = bins, label='pred nbar phi', histtype='step')
    plt.title('gen vs pred nbar phi')
    plt.yscale('log')
    plt.legend()
    
    fig21 = plt.figure()
    _, bins, _ = plt.hist(gen_t_phi, bins = N_BINS, label='gen t phi', histtype='step')
    _ = plt.hist(pred_t_phi, bins = bins, label='pred t phi', histtype='step')
    plt.title('t phi')
    plt.yscale('log')
    plt.legend()
    
    fig22 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_phi, bins = N_BINS, label='gen tbar phi', histtype='step')
    _ = plt.hist(pred_tbar_phi, bins = bins, label='pred tbar phi', histtype='step')
    plt.title('tbar phi')
    plt.yscale('log')
    plt.legend()
    
    # pt
    fig23 = plt.figure()
    _, bins, _ = plt.hist(gen_l_pt, bins = N_BINS, label='gen l pt', histtype='step')
    _ = plt.hist(pred_l_pt, bins = bins, label='pred l pt', histtype='step')
    _ = plt.hist(reco_l_pt, bins = bins, label='reco l pt', histtype='step')
    plt.title('l pt')
    plt.yscale('log')
    plt.legend()
    
    fig24 = plt.figure()
    _, bins, _ = plt.hist(gen_lbar_pt, bins = N_BINS, label='gen lbar pt', histtype='step')
    _ = plt.hist(pred_lbar_pt, bins = bins, label='pred lbar pt', histtype='step')
    _ = plt.hist(reco_lbar_pt, bins = bins, label='reco lbar pt', histtype='step')
    plt.title('lbar pt')
    plt.yscale('log')
    plt.legend()
    
    fig25 = plt.figure()
    _, bins, _ = plt.hist(gen_b_pt, bins = N_BINS, label='gen b pt', histtype='step')
    _ = plt.hist(pred_b_pt, bins = bins, label='pred b pt', histtype='step')
    plt.title('b pt')
    plt.yscale('log')
    plt.legend()
    
    fig26 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_pt, bins = N_BINS, label='gen bbar pt', histtype='step')
    _ = plt.hist(pred_bbar_pt, bins = bins, label='pred bbar pt', histtype='step')
    plt.title('bbar pt')
    plt.yscale('log')
    plt.legend()
    
    fig27 = plt.figure()
    _, bins, _ = plt.hist(gen_bbar_pt, bins = N_BINS, label='gen n pt', histtype='step')
    _ = plt.hist(pred_n_pt, bins = bins, label='pred n pt', histtype='step')
    plt.title('gen vs pred n pt')
    plt.yscale('log')
    plt.legend()
    
    fig28 = plt.figure()
    _, bins, _ = plt.hist(gen_nbar_pt, bins = N_BINS, label='gen nbar pt', histtype='step')
    _ = plt.hist(pred_nbar_pt, bins = bins, label='pred nbar pt', histtype='step')
    plt.title('gen vs pred nbar pt')
    plt.yscale('log')
    plt.legend()
    
    fig29 = plt.figure()
    _, bins, _ = plt.hist(gen_t_pt, bins = N_BINS, label='gen t pt', histtype='step')
    _ = plt.hist(pred_t_pt, bins = bins, label='pred t pt', histtype='step')
    plt.title('t pt')
    plt.yscale('log')
    plt.legend()
    
    fig30 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_pt, bins = N_BINS, label='gen tbar pt', histtype='step')
    _ = plt.hist(pred_tbar_pt, bins = bins, label='pred tbar pt', histtype='step')
    plt.title('tbar pt')
    plt.yscale('log')
    plt.legend()
    
    # px, py, pz
    fig31 = plt.figure()
    _, bins, _ = plt.hist(gen_t_px, bins = N_BINS, label='gen t px', histtype='step')
    _ = plt.hist(pred_t_px, bins = bins, label='pred t px', histtype='step')
    plt.title('t px')
    plt.yscale('log')
    plt.legend()
    
    fig32 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_px, bins = N_BINS, label='gen tbar px', histtype='step')
    _ = plt.hist(pred_tbar_px, bins = bins, label='pred tbar px', histtype='step')
    plt.title('tbar px')
    plt.yscale('log')
    plt.legend()
    
    fig33 = plt.figure()
    _, bins, _ = plt.hist(gen_t_py, bins = N_BINS, label='gen t py', histtype='step')
    _ = plt.hist(pred_t_py, bins = bins, label='pred t py', histtype='step')
    plt.title('t py')
    plt.yscale('log')
    plt.legend()
    
    fig34 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_py, bins = N_BINS, label='gen tbar py', histtype='step')
    _ = plt.hist(pred_tbar_py, bins = bins, label='pred tbar py', histtype='step')
    plt.title('tbar py')
    plt.yscale('log')
    plt.legend()
    
    fig35 = plt.figure()
    _, bins, _ = plt.hist(gen_t_pz, bins = N_BINS, label='gen t pz', histtype='step')
    _ = plt.hist(pred_t_pz, bins = bins, label='pred t pz', histtype='step')
    plt.title('t pz')
    plt.yscale('log')
    plt.legend()
    
    fig36 = plt.figure()
    _, bins, _ = plt.hist(gen_tbar_pz, bins = N_BINS, label='gen tbar pz', histtype='step')
    _ = plt.hist(pred_tbar_pz, bins = bins, label='pred tbar pz', histtype='step')
    plt.title('tbar pz')
    plt.yscale('log')
    plt.legend()
    
    
    # 2d plots
    fig37 = plt.figure()
    a = plt.hist2d(gen_t_px, pred_t_px, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top px plot')
    plt.xlabel(r'gen top px [GeV]')
    plt.ylabel(r'pred top px [GeV]')
    
    fig38 = plt.figure()
    a = plt.hist2d(gen_t_py, pred_t_py, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top py plot')
    plt.xlabel(r'gen top py [GeV]')
    plt.ylabel(r'pred top py [GeV]')
    
    fig39 = plt.figure()
    a = plt.hist2d(gen_t_pz, pred_t_pz, bins=100, range=[[-400, 400],[-400, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top pz plot')
    plt.xlabel(r'gen top pz [GeV]')
    plt.ylabel(r'pred top pz [GeV]')
    
    
    fig40 = plt.figure()
    a = plt.hist2d(gen_t_mass, pred_t_mass, bins=100, range=[[0, 400],[0, 400]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top mass plot')
    plt.xlabel(r'gen top mass [GeV]')
    plt.ylabel(r'pred top mass [GeV]')
    
    fig41 = plt.figure()
    a = plt.hist2d(gen_t_phi, pred_t_phi, bins=100, range=[[-3.15, 3.15],[-3.15, 3.15]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top phi plot')
    plt.xlabel(r'gen top phi')
    plt.ylabel(r'pred top phi')
    
    fig42 = plt.figure()
    a = plt.hist2d(gen_t_eta, pred_t_eta, bins=100, range=[[-10, 10],[-10, 10]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top eta plot')
    plt.xlabel(r'gen top eta')
    plt.ylabel(r'pred top eta')
    
    fig43 = plt.figure()
    a = plt.hist2d(gen_t_pt, pred_t_pt, bins=100, range=[[0, 1200],[0, 1200]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D top pT plot')
    plt.xlabel(r'gen top pT [GeV]')
    plt.ylabel(r'pred top pT [GeV]')
    
    # W stuff
    fig44 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_mass, bins = N_BINS, label='gen W+ mass', histtype='step')
    _ = plt.hist(pred_Wp_mass, bins = bins, label='pred W+ mass', histtype='step')
    plt.title('gen vs pred W+ mass')
    plt.yscale('log')
    plt.legend()
    
    fig45 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_mass, bins = N_BINS, label='gen W- mass', histtype='step')
    _ = plt.hist(pred_Wm_mass, bins = bins, label='pred W- mass', histtype='step')
    plt.title('gen vs pred W- mass')
    plt.yscale('log')
    plt.legend()
    
    fig46 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_eta, bins = N_BINS, label='gen W+ eta', histtype='step')
    _ = plt.hist(pred_Wp_eta, bins = bins, label='pred W+ eta', histtype='step')
    plt.title('gen vs pred W+ eta')
    plt.yscale('log')
    plt.legend()
    
    fig47 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_eta, bins = N_BINS, label='gen W- eta', histtype='step')
    _ = plt.hist(pred_Wm_eta, bins = bins, label='pred W- eta', histtype='step')
    plt.title('gen vs pred W- eta')
    plt.yscale('log')
    plt.legend()
    
    fig48 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_phi, bins = N_BINS, label='gen W+ phi', histtype='step')
    _ = plt.hist(pred_Wp_phi, bins = bins, label='pred W+ phi', histtype='step')
    plt.title('gen vs pred W+ phi')
    plt.yscale('log')
    plt.legend()
    
    fig49 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_phi, bins = N_BINS, label='gen W- phi', histtype='step')
    _ = plt.hist(pred_Wm_phi, bins = bins, label='pred W- phi', histtype='step')
    plt.title('gen vs pred W- phi')
    plt.yscale('log')
    plt.legend()
    
    fig50 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_pt, bins = N_BINS, label='gen W+ pt', histtype='step')
    _ = plt.hist(pred_Wp_pt, bins = bins, label='pred W+ pt', histtype='step')
    plt.title('gen vs pred W+ pt')
    plt.yscale('log')
    plt.legend()
    
    fig51 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_pt, bins = N_BINS, label='gen W- pt', histtype='step')
    _ = plt.hist(pred_Wm_pt, bins = bins, label='pred W- pt', histtype='step')
    plt.title('gen vs pred W- pt')
    plt.yscale('log')
    plt.legend()
    
    fig52 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_px, bins = N_BINS, label='gen W+ px', histtype='step')
    _ = plt.hist(pred_Wp_px, bins = bins, label='pred W+ px', histtype='step')
    plt.title('gen vs pred W+ px')
    plt.yscale('log')
    plt.legend()
    
    fig53 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_px, bins = N_BINS, label='gen W- px', histtype='step')
    _ = plt.hist(pred_Wm_px, bins = bins, label='pred W- px', histtype='step')
    plt.title('gen vs pred W- px')
    plt.yscale('log')
    plt.legend()
    
    fig54 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_py, bins = N_BINS, label='gen W+ py', histtype='step')
    _ = plt.hist(pred_Wp_py, bins = bins, label='pred W+ py', histtype='step')
    plt.title('gen vs pred W+ py')
    plt.yscale('log')
    plt.legend()
    
    fig55 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_py, bins = N_BINS, label='gen W- py', histtype='step')
    _ = plt.hist(pred_Wm_py, bins = bins, label='pred W- py', histtype='step')
    plt.title('gen vs pred W- py')
    plt.yscale('log')
    plt.legend()
    
    fig56 = plt.figure()
    _, bins, _ = plt.hist(gen_Wp_pz, bins = N_BINS, label='gen W+ pz', histtype='step')
    _ = plt.hist(pred_Wp_pz, bins = bins, label='pred W+ pz', histtype='step')
    plt.title('gen vs pred W+ pz')
    plt.yscale('log')
    plt.legend()
    
    fig57 = plt.figure()
    _, bins, _ = plt.hist(gen_Wm_pz, bins = N_BINS, label='gen W- pz', histtype='step')
    _ = plt.hist(pred_Wm_pz, bins = bins, label='pred W- pz', histtype='step')
    plt.title('gen vs pred W- pz')
    plt.yscale('log')
    plt.legend()
    
    # met
    fig58 = plt.figure()
    _, bins, _ = plt.hist(gen_met_phi, bins = N_BINS, label='gen nnbar phi', histtype='step')
    _ = plt.hist(pred_met_phi, bins = bins, label='pred nnbar phi', histtype='step')
    _ = plt.hist(reco_met_phi, bins = bins, label='reco met phi', histtype='step')
    plt.title('met phi')
    plt.yscale('log')
    plt.legend()
    
    fig59 = plt.figure()
    _, bins, _ = plt.hist(gen_met_pt, bins = N_BINS, label='gen nnbar pt', histtype='step')
    _ = plt.hist(pred_met_pt, bins = bins, label='pred nnbar pt', histtype='step')
    _ = plt.hist(reco_met_pt, bins = bins, label='reco met pt', histtype='step')
    plt.title('met pt')
    plt.yscale('log')
    plt.legend()

    # ttbar

    fig60 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_pt, bins = N_BINS, label='gen ttbar pt', histtype='step')
    _ = plt.hist(pred_ttbar_pt, bins = bins, label='pred ttbar pt', histtype='step')
    plt.title('ttbar pt')
    plt.yscale('log')
    plt.legend()

    fig61 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_eta, bins = N_BINS, label='gen ttbar eta', histtype='step')
    _ = plt.hist(pred_ttbar_eta, bins = bins, label='pred ttbar eta', histtype='step')
    plt.title('ttbar eta')
    plt.yscale('log')
    plt.legend()

    fig62 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_phi, bins = N_BINS, label='gen ttbar phi', histtype='step')
    _ = plt.hist(pred_ttbar_phi, bins = bins, label='pred ttbar phi', histtype='step')
    plt.title('ttbar phi')
    plt.yscale('log')
    plt.legend()

    fig63 = plt.figure()
    _, bins, _ = plt.hist(gen_ttbar_mass, bins = N_BINS, label='gen ttbar mass', histtype='step')
    _ = plt.hist(pred_ttbar_mass, bins = bins, label='pred ttbar mass', histtype='step')
    plt.title('ttbar mass')
    plt.yscale('log')
    plt.legend()

    fig64 = plt.figure()
    a = plt.hist2d(gen_ttbar_mass, pred_ttbar_mass, bins=100, range=[[300, 500],[300, 500]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar mass plot')
    plt.xlabel(r'gen ttbar mass [GeV]')
    plt.ylabel(r'pred ttbar mass [GeV]')
    
    fig65 = plt.figure()
    a = plt.hist2d(gen_ttbar_phi, pred_ttbar_phi, bins=100, range=[[-3.15, 3.15],[-3.15, 3.15]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar phi plot')
    plt.xlabel(r'gen ttbar phi')
    plt.ylabel(r'pred ttbar phi')
    
    fig66 = plt.figure()
    a = plt.hist2d(gen_ttbar_eta, pred_ttbar_eta, bins=100, range=[[-10, 10],[-10, 10]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar eta plot')
    plt.xlabel(r'gen ttbar eta')
    plt.ylabel(r'pred ttbar eta')
    
    fig67 = plt.figure()
    a = plt.hist2d(gen_ttbar_pt, pred_ttbar_pt, bins=100, range=[[0, 1000],[0, 1000]], 
                  norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('2D ttbar pT plot')
    plt.xlabel(r'gen ttbar pT [GeV]')
    plt.ylabel(r'pred ttbar pT [GeV]')
    
    
    # make pdf
    
    p = PdfPages(SAVE_DIR + 'predictions.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    
    for fig in figs:
        fig.savefig(p, format = 'pdf')
    p.close()
