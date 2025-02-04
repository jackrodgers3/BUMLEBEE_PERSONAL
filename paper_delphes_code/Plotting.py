import os
#import re
import numpy as np
#import awkward as ak
#import uproot
from matplotlib import pyplot as plt
import hist
from hist import Hist
import mplhep as hep

physics_var_labels = {'ll_cHel': r"$\cos \varphi$",
                      'b1k': r"$\cos \theta_1^k$",
                      'b2k': r"$\cos \theta_2^k$",
                      'b1n': r"$\cos \theta_1^n$",
                      'b2n': r"$\cos \theta_2^n$",
                      'b1r': r"$\cos \theta_1^r$",
                      'b2r': r"$\cos \theta_2^r$",
                      'c_kk': r"$\cos \theta_1^k \: \cos \theta_2^k$",
                      'c_rr': r"$\cos \theta_1^r \: \cos \theta_2^r$",
                      'c_nn': r"$\cos \theta_1^n \: \cos \theta_2^n$",
                      'c_rk': r"$\cos \theta_1^r \: \cos \theta_2^k$",
                      'c_kr': r"$\cos \theta_1^k \: \cos \theta_2^r$",
                      'c_nr': r"$\cos \theta_1^n \: \cos \theta_2^r$",
                      'c_rn': r"$\cos \theta_1^r \: \cos \theta_2^n$",
                      'c_nk': r"$\cos \theta_1^n \: \cos \theta_2^k$",
                      'c_kn': r"$\cos \theta_1^k \: \cos \theta_2^n$",
                      'llbar_delta_eta': r"$| \Delta \eta_{\ell \bar{\ell}} |$",
                      'llbar_delta_phi': r"$| \Delta \phi_{\ell \bar{\ell}} |$",
                      'ttbar_mass': r"$m_{t \bar{t}}$ (GeV)",
                      'gen_ttbar_mass': r"$m_{t \bar{t}}$ (GeV)",
                      'top_pt': r"$p_T^t$ (GeV)",
                      'top_phi': r"$\phi^t$",
                      'top_eta': r"$\eta^t$",
                      'top_mass': r"$m^t$ (GeV)",
                      'tbar_pt': r"$p_T^{\bar{t}}$ (GeV)",
                      'tbar_phi': r"$\phi^{\bar{t}}$",
                      'tbar_eta': r"$\eta^{\bar{t}}$",
                      'tbar_mass': r"$m^{\bar{t}}$ (GeV)",
                      'nu_pt': r"$p_T^{\nu}$ (GeV)",
                      'nu_eta': r"$\eta^{\nu}$",
                      'nu_phi': r"$\phi^{\nu}$",
                      'nu_mass': r"$m^{\nu}$ (GeV)",
                      'nubar_pt': r"$p_T^{\bar{\nu}}$ (GeV)",
                      'nubar_eta': r"$\eta^{\bar{\nu}}$",
                      'nubar_phi': r"$\phi^{\bar{\nu}}$",
                      'nubar_mass': r"$m^{\bar{\nu}}$ (GeV)",
                      'nunubar_delta_eta': r"$| \Delta \eta_{\nu \bar{\nu}} |$",
                      'nunubar_delta_phi': r"$| \Delta \phi_{\nu \bar{\nu}} |$",
                      'ttbar_delta_eta': r"$| \Delta \eta_{t \bar{t}} |$",
                      'ttbar_delta_phi': r"$| \Delta \phi_{t \bar{t}} |$",
                      'nunubar_delta_eta_signed': r"$\Delta \eta_{\nu \bar{\nu}}$",
                      'nunubar_delta_phi_signed': r"$\Delta \phi_{\nu \bar{\nu}}$",
                      'ttbar_delta_eta_signed': r"$\Delta \eta_{t \bar{t}}$",
                      'ttbar_delta_phi_signed': r"$\Delta \phi_{t \bar{t}}$",
                      'ttbar_delta_r': r"$\Delta R_{t \bar{t}}$",
                      'l_pt': r"$p_T^{\ell}$ (GeV)",
                      'lbar_pt': r"$p_T^{\bar{\ell}}$ (GeV)",
                      'b_pt': r"$p_T^{b}$ (GeV)",
                      'bbar_pt': r"$p_T^{\bar{b}}$ (GeV)",
                      }

phys_labels = list(physics_var_labels.keys())
for label in phys_labels:
    physics_var_labels['gen_'+label] = r"True " + physics_var_labels[label]
del phys_labels

physics_var_labels['top_gen_delta_eta'] = r"$| \Delta \eta(t^{\text{reco}}, t^{\text{gen}}) |$"
physics_var_labels['top_gen_delta_phi'] = r"$| \Delta \phi(t^{\text{reco}}, t^{\text{gen}}) |$"
physics_var_labels['top_gen_delta_r'] = r"$\Delta R(t^{\text{reco}}, t^{\text{gen}})$"
physics_var_labels['tbar_gen_delta_eta'] = r"$| \Delta \eta(\bar{t}^{\text{reco}}, \bar{t}^{\text{gen}}) |$"
physics_var_labels['tbar_gen_delta_phi'] = r"$| \Delta \phi(\bar{t}^{\text{reco}}, \bar{t}^{\text{gen}}) |$"
physics_var_labels['tbar_gen_delta_r'] = r"$\Delta R(\bar{t}^{\text{reco}}, \bar{t}^{\text{gen}})$"

#def ReplaceCloseToZeros(array, threshold=1e-5):
#    return np.where(np.abs(array) < threshold, threshold, array)

def FindRatios(reco, gen):
    '''

    '''
    has_errors = (reco.get('yerr', None) is not None) and (gen.get('yerr', None) is not None)

    x_ratios = []
    ratios = []
    ratio_errors = []
    for idx in range(reco['x'].shape[0]):
        index_candidate = np.argwhere(gen['x'] == reco['x'][idx])
        if (len(index_candidate) >= 1):
            gen_idx = index_candidate[0]

            current_ratio = reco['y'][idx] / gen['y'][gen_idx]

            x_ratios.append(reco['x'][idx])
            ratios.append(current_ratio)
            
            if has_errors:
                # Add in relative errors in quadrature
                current_ratio_err = np.sqrt((np.square(reco['yerr'][idx] / reco['y'][idx]) + np.square(gen['yerr'][gen_idx])).astype(float))
                ratio_errors.append(current_ratio_err)
    
    if (not has_errors):
        return np.array(x_ratios), np.array(ratios)
    
    return np.array(x_ratios), np.array(ratios), np.array(ratio_errors)


def PlotResolutionAsFuncNew(resolutions, xlabel, ylabel, save_folder, ratio_key=None, ratio_label=None, quantity_label='Resolution(fwhm)', cmstext="Work in Progress", lumitext="2017UL", name_addition=""):
    '''
    Updated version of `PlotResolutionAsFunc` that takes a full set of x-values (bin centers) and y-values (and optionally errors)
    for each entry (reconstruction method) in the plot.
    NOTE: CURRENTLY, THIS WILL BREAK IF RATIO PLOT IS USED BUT NOT ALL SETS OF X-VALUES ARE THE SAME

    Arguments:
        resolutions (dict): A nested dictionary whose keys are strings to be used in the plot's legend,
                            and whose values have the structure:
                            {'x': <NumPy array>, 'y': <NumPy array>, 'yerr': <Numpy array (optional)>}
        xlabel (str): x-axis label to use in plot
        ylabel (str): y-axis label to use in plot
        save_folder (str): path to folder to save plot
        ratio_key (str, default=None): if not None, include a ratio plot, with the ratios of each entry in resolutions
                                       to the one specified by this key. 
        ratio_label (str, default=None): y-axis label for the ratio plot
    '''
    if (not os.path.exists(save_folder)):
        os.mkdir(save_folder)
    
    plt.style.use(hep.style.CMS)

    x_min = 0
    x_max = 1.0
    min_value = 0
    max_value = 1.0
    #max_value = np.amax(resolutions[list(resolutions.keys())[0]])

    if (ratio_key is not None):
        fig, ax = plt.subplots(2, 1, dpi=100, height_ratios=[3,1])
        ax_main = ax[0]
        ax_ratio = ax[1]

        min_ratio = 0.8
        max_ratio = 1.2

        # check the dimensions of input arrays to determine whether errorbars are necessary
        #if ((len(resolutions[list(resolutions.keys())[0]].shape) == 1) or (resolutions[list(resolutions.keys())[0]].shape[1] == 1)):
        #max_value = np.amax(resolutions[list(resolutions.keys())[0]])

        for res in resolutions.keys():
            x_min = min(x_min, np.amin(resolutions[res]['x']))
            x_max = max(x_max, np.amin(resolutions[res]['x']))

            if (resolutions[res].get('yerr', None) is None):
                ax_main.scatter(resolutions[res]['x'], resolutions[res]['y'], label=res)

                #ratio = resolutions[res]['y'] / resolutions[ratio_key]['y']
                ratio_x, ratio = FindRatios(resolutions[res], resolutions[ratio_key])
                ax_ratio.scatter(ratio_x, ratio, label=res)

                min_value = min(min_value, np.amin(resolutions[res]['y']))
                max_value = max(max_value, np.amax(resolutions[res]['y']))
                min_ratio = min(min_ratio, np.amin(ratio))
                max_ratio = max(max_ratio, np.amax(ratio))
            else:
                ax_main.errorbar(resolutions[res]['x'], resolutions[res]['y'], yerr=resolutions[res]['yerr'], label=res, fmt='o')

                #ratio = resolutions[res]['y'] / resolutions[ratio_key][y]
                # Add relative errors in quadrature to get error on ratio
                #ratio_err = np.sqrt((np.square(resolutions[res]['yerr'] / resolutions[res]['y']) + np.square(resolutions[ratio_key]['yerr'] / resolutions[ratio_key]['y'])).astype(float))
                ratio_x, ratio, ratio_err = FindRatios(resolutions[res], resolutions[ratio_key])
                ax_ratio.errorbar(ratio_x, ratio, yerr=ratio_err, label=res, fmt='o')

                min_value = min(min_value, np.amin(resolutions[res]['y']))
                max_value = max(max_value, np.amax(resolutions[res]['y']))
                min_ratio = min(min_ratio, np.amin(ratio))
                max_ratio = max(max_ratio, np.amax(ratio))

        ax_ratio.set_ylim(bottom=min_ratio, top=max_ratio)
        ax_ratio.set_ylabel(ratio_label)

    else:
        fig, ax_main = plt.subplots(dpi=100)
        ax_ratio = ax_main # for use in labeling

        for res in resolutions.keys():
            x_min = min(x_min, np.amin(resolutions[res]['x']))
            x_max = max(x_max, np.amin(resolutions[res]['x']))

            if (resolutions[res].get('yerr', None) is None):
                ax_main.scatter(resolutions[res]['x'], resolutions[res]['y'], label=res)

                min_value = min(min_value, np.amin(resolutions[res]['y']))
                max_value = max(max_value, np.amax(resolutions[res]['y']))

            else:
                ax_main.errorbar(resolutions[res]['x'], resolutions[res]['y'], yerr=resolutions[res]['yerr'], label=res, fmt='o')

                min_value = min(min_value, np.amin(resolutions[res]['y']))
                max_value = max(max_value, np.amax(resolutions[res]['y']))
    
    ax_ratio.set_xlim(left=x_min, right=x_max)
    ax_main.set_xlim(left=x_min, right=x_max)
    ax_main.set_ylim(bottom=min_value, top=max_value)


    ax_ratio.set_xlabel(physics_var_labels.get(xlabel, xlabel))
    ax_main.set_ylabel(physics_var_labels.get(ylabel, ylabel) + " " + quantity_label)
    ax_main.legend()

    #hep.cms.text(cmstext, ax=ax_main)
    #hep.cms.lumitext(lumitext, ax=ax_main)
    
    plt.savefig(save_folder+"/" + ylabel + "_" + quantity_label + "_as_function_of_" + xlabel + name_addition + ".pdf", dpi='figure')
    plt.close()

def PlotResolutionAsFunc(resolutions, x, xlabel, ylabel, save_folder, ratio_key=None, ratio_label=None, cmstext="Work in Progress", lumitext="2017UL", name_addition=""):
    '''
    Plots resolution of a variable as a function of another variable (can be same or different),
    as a scatter plot.
    Previously `plot_residuals_by_method`.
    NOTE: IT IS ASSUMED THAT ALL SETS OF Y-VALUES USE THE SAME SET OF X-VALUES.

    Arguments:
        resolutions (dict): A dictionary whose keys are strings to be used in the plot's legend,
                            and whose values are NumPy arrays containing the resolutions (y-values to plot),
                            along with (optionally) errors (format is [[value1, error1], [value2, error2], ...])
        x (NumPy array): the x-values to plot (usually midpoints of bins used to compute resolutions)
        xlabel (str): x-axis label to use in plot
        ylabel (str): y-axis label to use in plot
        save_folder (str): path to folder to save plot
        ratio_key (str, default=None): if not None, include a ratio plot, with the ratios of each entry in resolutions
                                       to the one specified by this key. 
        ratio_label (str, default=None): y-axis label for the ratio plot
    '''
    if (not os.path.exists(save_folder)):
        os.mkdir(save_folder)
    
    plt.style.use(hep.style.CMS)

    min_value = 0
    max_value = 1.0
    #max_value = np.amax(resolutions[list(resolutions.keys())[0]])

    if (ratio_key is not None):
        fig, ax = plt.subplots(2, 1, dpi=100, height_ratios=[3,1])
        ax_main = ax[0]
        ax_ratio = ax[1]

        min_ratio = 0.8
        max_ratio = 1.2

        # check the dimensions of input arrays to determine whether errorbars are necessary
        if ((len(resolutions[list(resolutions.keys())[0]].shape) == 1) or (resolutions[list(resolutions.keys())[0]].shape[1] == 1)):
            max_value = np.amax(resolutions[list(resolutions.keys())[0]])

            for res in resolutions.keys():
                ax_main.scatter(x, resolutions[res], label=res)

                ratio = resolutions[res] / resolutions[ratio_key]
                ax_ratio.scatter(x, ratio, label=res)

                min_value = min(min_value, np.amin(resolutions[res]))
                max_value = max(max_value, np.amax(resolutions[res]))
                min_ratio = min(min_ratio, np.amin(ratio))
                max_ratio = max(max_ratio, np.amax(ratio))

        else:
            max_value = np.amax(resolutions[list(resolutions.keys())[0]][:,0])

            for res in resolutions.keys():
                ax_main.errorbar(x, resolutions[res][:,0], yerr=resolutions[res][:,1], label=res, fmt='o')

                ratio = resolutions[res][:,0] / resolutions[ratio_key][:,0]
                # Add relative errors in quadrature to get error on ratio
                ratio_err = np.sqrt((np.square(resolutions[res][:,1] / resolutions[res][:,0]) + np.square(resolutions[ratio_key][:,1] / resolutions[ratio_key][:,0])).astype(float))
                ax_ratio.errorbar(x, ratio, yerr=ratio_err, label=res, fmt='o')

                min_value = min(min_value, np.amin(resolutions[res][:,0]))
                max_value = max(max_value, np.amax(resolutions[res][:,0]))
                min_ratio = min(min_ratio, np.amin(ratio))
                max_ratio = max(max_ratio, np.amax(ratio))

        ax_ratio.set_ylim(bottom=max(0.1,min_ratio), top=min(max_ratio,1.5)) # min_ratio, max_ratio
        ax_ratio.set_ylabel(ratio_label)

    else:
        fig, ax_main = plt.subplots(dpi=100)
        ax_ratio = ax_main # for use in labeling

        # check the dimensions of input arrays to determine whether errorbars are necessary
        if ((len(resolutions[list(resolutions.keys())[0]].shape) == 1) or (resolutions[list(resolutions.keys())[0]].shape[1] == 1)):
            max_value = np.amax(resolutions[list(resolutions.keys())[0]])

            for res in resolutions.keys():
                ax_main.scatter(x, resolutions[res], label=res)

                min_value = min(min_value, np.amin(resolutions[res]))
                max_value = max(max_value, np.amax(resolutions[res]))
        else:
            max_value = np.amax(resolutions[list(resolutions.keys())[0]][:,0])

            for res in resolutions.keys():
                ax_main.errorbar(x, resolutions[res][:,0], yerr=resolutions[res][:,1], label=res, fmt='o')

                min_value = min(min_value, np.amin(resolutions[res][:,0]))
                max_value = max(max_value, np.amax(resolutions[res][:,0]))
    
    ax_ratio.set_xlim(left=np.amin(x), right=np.amax(x))
    ax_main.set_xlim(left=np.amin(x), right=np.amax(x))
    ax_main.set_ylim(bottom=min_value, top=min(max_value,500.0)) # min_value, max_value


    ax_ratio.set_xlabel(physics_var_labels.get(xlabel, xlabel))
    ax_main.set_ylabel(physics_var_labels.get(ylabel, ylabel))
    ax_main.legend(["Bumblebee", r"$m_{lb}$",])

    #hep.cms.text(cmstext, ax=ax_main)
    #hep.cms.lumitext(lumitext, ax=ax_main)
    
    plt.savefig(save_folder + ylabel + "_as_function_of_" + xlabel + name_addition + ".pdf", dpi='figure')
    plt.close()

def PlotSingleDistribution(processes, xlabel, ylabel, save_folder, num_bins=20, min_value=None, max_value=None, density=True, yscale='linear', cmstext="Work in Progress", lumitext="2017UL"):
    '''
    Plots a 1D histogram.

    Arguments:
        processes (dict): A dictionary whose keys are the labels for the plot legend,
                          and whose values are NumPy arrays of the data in the histogram.
        xlabel (str): the x-axis label
        ylabel (str): the y-axis label
        save_folder (str): folder to save the plot
        num_bins (int): number of evenly-spaced bins to use
    '''
    if (not os.path.exists(save_folder)):
        os.mkdir(save_folder)
    
    plt.style.use(hep.style.CMS)

    if (min_value is None):
        min_value = np.amin([np.amin(processes[key]) for key in processes.keys()])
    if (max_value is None):
        max_value = np.amax([np.amax(processes[key]) for key in processes.keys()])
    
    fig, ax = plt.subplots(dpi=100)

    for key in processes.keys():
        hist_process = Hist(hist.axis.Regular(num_bins, min_value, max_value, name=key, underflow=False, overflow=False))

        hist_process.fill(processes[key])

        hep.histplot(hist_process, ax=ax, density=density, label=key)
    
    ax.set_yscale(yscale)

    ax.set_xlim(left=min_value, right=max_value)
    if (yscale == 'linear'):
        ax.set_ylim(bottom=0)
    elif (yscale == 'log'):
        ax.set_ylim(bottom=0.1)

    ax.set_xlabel(physics_var_labels.get(xlabel, xlabel))
    ax.set_ylabel(physics_var_labels.get(ylabel, ylabel))

    if (len(processes.keys()) > 1):
        ax.legend()

    #hep.cms.text(cmstext, ax=ax)
    #hep.cms.lumitext(lumitext, ax=ax)

    plt.savefig(save_folder+"/"+xlabel+"_distribution.png", dpi='figure')
    plt.close()

