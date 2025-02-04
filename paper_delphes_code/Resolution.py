import os
import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
import hist
from hist import Hist
import lmfit
from lmfit.models import GaussianModel


def GaussianFit(data, constraint_factor=3.0):
    '''
    # Note: this is a binned gaussian fit
    # number of bins is determined by the Rice's rule, that at least 3 bins, and make 2 * N**(1/3) bins (I think at least it needs 5)
    # feed the data into a histogram with the number of bins determined before
    # find the middle point of each bin
    # find the bin has the most event (most frequent bin)
    # find the std of the data (roughly the sigma)
    # only consider the bins in the fit if:
    #     the middle point of a bin >=(<=) the most frequent bin (peak?) -(+) const * rough_sigma, so only fit the data points with in the sigma... 
    #     Note: 1) the problem with this is that you cannot promise that the peak is the true peak...
    #           2) the other problem is if the bins are masked, then you will left with less than 5 bins to fit
    #           3) for low statistic bins and non-gaussian distribution bins, we should not trust them anyway, should not plot them in the plot
    
    # uses lmfit.models.GaussianModel
    # if there are more than 3 bins remained for fitting, fit
    # else, use masked bins as well in the fitting
    '''
    # a rough binning to find the peak
    num_bins = max( int( np.ceil( 2 * np.power(data.shape[0], (1/3) ) ) ) , 5 ) #Rice's rule for binning, minimum of 3 bins
    freqs, bin_edges = np.histogram(data, bins=num_bins)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    print(f"number of data: {data.shape[0]}")
    print(f"number of bins: {num_bins}")
    # Trim fit inputs to region within `constraint_factor` standard deviations of peak
    peak_rough = bin_midpoints[np.argmax(freqs)] # find the most frequent bin
    rms_stdev = np.std(data) 

    x_min = peak_rough - 3 * rms_stdev
    x_max = peak_rough + 3 * rms_stdev
    data_filter = (data < x_max) & (data > x_min)
    data = data[data_filter]

    # left with data for fitting, re-bin it, require at least 5 bins that each bin has at least 10 events
    # remove the bins that has less than 10 events
    num_bins = max( int( np.ceil( 2 * np.power(data.shape[0], (1/3) ) ) ) , 5 )    
    freqs, bin_edges = np.histogram(data, bins=num_bins)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    freqs_after_filter = freqs[freqs>9.0]
    bin_midpoints_after_filter = bin_midpoints[freqs>9.0]

    gmodel = GaussianModel()
    print(f"number of data after filter involved in fitting: {data.shape[0]}")
    print(f"number of bins after filter involved in fitting: {num_bins}")
    # Fit using trimmed inputs if there are enough, otherwise use untrimmed
    if (len(freqs_after_filter) < 5):
        print(f"WARNING: bin does not have enough data")
        freqs_after_filter = np.zeros(5)
        bin_midpoints_after_filter = np.zeros(5)
    
    initial_params = gmodel.guess(freqs_after_filter, x=bin_midpoints_after_filter)
    result = gmodel.fit(freqs_after_filter, initial_params, x=bin_midpoints_after_filter)

    return result

def FittedResolution(data, plot_folder=None, plot_title="Binned_Gaus_Fit"):
    '''
    # Save each bin's fit to a plot if `plot_folder` is not none
    # Note, I think it is good to plot also the data points that are not considered in the fit
    '''
    fit_result = GaussianFit(data, constraint_factor=1.0)

    if (plot_folder is not None):
        fig, ax = plt.subplots()
        fit_result.plot_fit(ax=ax)
        ax.set_title(plot_title + ", r2=" + str(fit_result.rsquared))
        ax.set_xlabel("residual (GeV)")
        ax.set_ylabel("events")
        plt.savefig(plot_folder + "/" + plot_title + ".png", dpi='figure')
        plt.close()

    resolution = fit_result.params['fwhm'].value
    resolution_error = fit_result.params['fwhm'].stderr

    bias = fit_result.params['center'].value
    bias_error = fit_result.params['center'].stderr

    return resolution, resolution_error, bias, bias_error

def SplitInBins(data, x, bins=20):
    '''
    # if bins is an integer, it will be an even split with respect to the min/max of the x
    # else, bins is bin_edges, it takes a list or 1d numpy array
    # return a dictionary that key is the bin_midpoint and value is the data points in the bin
    '''
    min_value = np.amin(x)
    max_value = np.amax(x)

    if isinstance(bins, int):
        bin_width = (max_value - min_value) / bins
        bins = np.linspace(min_value-(bin_width/200), max_value+(bin_width/200), bins)
    elif hasattr(bins, '__iter__'):
        if (not all([isinstance(item, (int, float)) for item in bins])):
            raise ValueError("Invalid bins information passed.")
    else:
        raise ValueError("Invalid bins information passed.")
    
    bin_midpoints = (bins[:-1] + bins[1:]) / 2

    binned_data = {}
    for index, midpoint in enumerate(bin_midpoints):
        bin_mask = ((x >= bins[index]) & (x < bins[index+1]))
        binned_data[midpoint] = data[bin_mask]
    
    return binned_data

def ComputeResolutions(y, x, bins=20, plots_folder=None):
    '''
    x is the gen, and it is the one that is binned and fits are made in each bin
    y can be the residual = reco - gen, or can be the percentage: (reco-gen) /gen
    '''
    if (not ((plots_folder is None) or os.path.exists(plots_folder))):
        os.mkdir(plots_folder)

    binned_residuals = SplitInBins(y, x, bins=bins)

    bin_list = []
    resolutions = []
    resolution_errors = []
    biases = []
    bias_errors = []
    for bin_name in binned_residuals.keys():
        print(f"processing bin centered at {bin_name}...") 
        res, res_err, bias, bias_err = FittedResolution(binned_residuals[bin_name], plots_folder, plot_title="Bin_centered_at_"+str(bin_name)) # maybe instead plot bin edge
        bin_list.append(bin_name)
        resolutions.append(res)
        resolution_errors.append(res_err)
        biases.append(bias)
        bias_errors.append(bias_err)
    
    return np.array(bin_list), np.array(resolutions), np.array(resolution_errors), np.array(biases), np.array(bias_errors)