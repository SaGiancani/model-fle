
import data_cleaning as clean
import fitting as fit
import numpy as np
import os
import pandas as pd
import stat_metrics as stat
import utils
import visualization_fit as visualize

if __name__=="__main__":

    # Load behavioral data from CSV
    data = pd.read_csv(utils.DATA_FILENAME, delimiter=';', header=None).values
    num_subjects = data.shape[0]

    # Define the x-axis (pi/4:pi/4:2*pi)
    xax = utils.X_RANGE

    # Sort properly the condition, according to the X_RANGE
    data = clean.sort_condition(data)

    # Flatten the data to 9x32 for fitting
    data = data.reshape(num_subjects, -1)  # 9x32 matrix
    data_tmp = np.zeros((data.shape[0]+1, data.shape[1]))
    data_tmp[:num_subjects, :] = data   
    # At the bottom of the original data matrix, added the mean across the subjects
    data_tmp[-1, :] = np.nanmean(data, axis = 0) 
    data = data_tmp 

    # Prepare X_data (same values for all subjects, repeated)
    xax = np.tile(xax, 4)
    x_data = np.ones((data.shape[0], len(xax)))*xax

    # Fit the data, either sub by sub and all together
    betas, fit_quality, prediction_subs              = fit.fit_data_sub_B_sub(data, x_data)
    beta_global, fit_quality_global, prediction_glob = fit.fit_all_together(data[:-1, :], x_data[:-1, :])  

    # Data storing as csv
    utils.fit_result_outcome(betas, fit_quality, beta_global, fit_quality_global, filename = 'model_fitting_results.csv')

    # Break fitting sub by sub
    mses_err_sub, rsqr_err_sub, chisq_err_sub = stat.break_fit(data, prediction_subs, utils.N_ITERS)
    # Break fitting global
    mses_err_glob, rsqr_err_glob, chisq_err_glob = stat.break_fit(data[:-1, :].ravel(), prediction_glob.ravel(), utils.N_ITERS*10)

    # Visualization results
    visualize.visualize_fit_n_raw_sub(data, x_data, betas, utils.SUB_NAMES, 'fit_allSubs')
    visualize.visualize_fit_n_raw_allTogether(data[:-1, :], x_data[:-1, :], beta_global, filename = 'fit_allTogether')

    print(f'Betas for average across subjects {betas[-1]}')
    print(f'Betas for all together fit {beta_global}')

    visualize.visualize_betas(betas, filename = 'betas_subs', betas_to_plot = ['Fovea', 'HM', 'VM', 'Eccentricity'])


    visualize.visualize_hists(mses_err_sub, 
                            rsqr_err_sub, 
                            chisq_err_sub, 
                            fit_quality, 
                            filename = 'random_permutation_distributions_subbysub')

    visualize.visualize_hists(np.array([mses_err_glob]), 
                            np.array([rsqr_err_glob]), 
                            np.array([chisq_err_glob]), 
                            fit_quality_global, 
                            sub_names = ['All Together'],                          
                            filename = 'random_permutation_distributions_allTogether')