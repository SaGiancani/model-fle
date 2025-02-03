import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import chisquare
from scipy.stats import ttest_1samp
import utils

def calculate_fit_quality(y_true, y_pred):
    # Define function to calculate fitting metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))    
    r2   = r2_score(y_true, y_pred)
    return rmse, r2


def break_fit(ys_raw, ys_pred, n_inters):
    """
    Break fitting with permutation strategy.
    
    Parameters:
    - ys_raw: 3D NumPy array of raw y values (subjects x curves x data points)
    - xax: 1D NumPy array of x-axis values
    - n_inters: Number of iterations for permutation/randomization
    - random_flag: Boolean flag (only `False` is supported here)
    
    Returns:
    - ys_subs, y_pred_subs, mses_err_subs, rsqr_err_subs, chisq_err_subs, mses_err, rsq_err, chisq_err
    """

    if len(ys_raw.shape)>1:
    # Initialize output variables
        num_subjects, _ = ys_raw.shape
        mses_err  = np.zeros((num_subjects, n_inters))
        rsqr_err  = np.zeros((num_subjects, n_inters))

        # Loop over subjects
        for sb in range(num_subjects):
            print('\n')
            print(f"Processing subject {sb+1}")
            for n_it in range(n_inters):
                # Permutation strategy
                random_ys  = np.random.permutation(ys_raw[sb, :])
                mses_err[sb, n_it], rsqr_err[sb, n_it] = calculate_fit_quality(random_ys, ys_pred[sb, :])

                if n_it % 50 == 0:
                    tmp_rmse = mses_err[sb, n_it]
                    tmp_rsqr = rsqr_err[sb, n_it]
                    print(f"Iteration {n_it + 1}")
                    print(f"RMSE: {tmp_rmse:.4f}, \
                          R²: {tmp_rsqr:.4f}")    
    else:
        # Initialize output variables
        mses_err        = np.zeros(n_inters)
        rsqr_err        = np.zeros(n_inters)
        print(ys_raw.shape, ys_pred.shape)

        for n_it in range(n_inters):
            # Permutation strategy
            random_ys  = np.random.permutation(ys_raw)
            mses_err[n_it], rsqr_err[n_it] = calculate_fit_quality(random_ys, ys_pred)

            if n_it % 50 == 0:
                    tmp_rmse = mses_err[n_it]
                    tmp_rsqr = rsqr_err[n_it]
                    print(f"Iteration {n_it + 1}")
                    print(f"RMSE: {tmp_rmse:.4f}, \
                          R²: {tmp_rsqr:.4f}")    

    return mses_err, rsqr_err


def get_pvalues(glob_measures, sub_measures, fit_quality_global, fit_quality):
    
    mses_err_glob  = glob_measures[0] 
    rsqr_err_glob  = glob_measures[1]

    p_rsqrd_allTog = np.average(rsqr_err_glob  >= fit_quality_global[0][1])
    p_rmse_allTog  = np.average(mses_err_glob  <= fit_quality_global[0][0])

    # Perform a t-test (alternative approach)
    _, p_ttest_rsqrd_allTog = ttest_1samp(rsqr_err_glob, fit_quality_global[0][1])
    _, p_ttest_rmse_allTog   = ttest_1samp(mses_err_glob, fit_quality_global[0][0])

    print(f'All subs together: p-value for RMSE: {p_rmse_allTog:.4f} p-value for R^2: {p_rsqrd_allTog:.4f}')
    print(f'All subs t-test  : p-value for RMSE: {p_ttest_rmse_allTog:.4f} p-value for R^2: {p_ttest_rsqrd_allTog:.4f} \n')
    
    mses_err_sub  = sub_measures[0] 
    rsqr_err_sub  = sub_measures[1]
    
    ps_single_sub   = []
    ps_single_sub_t = []
    
    for i in range(len(fit_quality)):
        p_rsqrd = np.average(rsqr_err_sub[i, :]  >= fit_quality[i][1])
        p_rmse  = np.average(mses_err_sub[i, :]  <= fit_quality[i][0])    

        # Perform a t-test (alternative approach)
        _, p_ttest_rsqrd = ttest_1samp(rsqr_err_sub[i, :], fit_quality[i][1])
        _, p_ttest_rmse   = ttest_1samp(mses_err_sub[i, :], fit_quality[i][0])

        ps_single_sub.append((p_rmse, p_rsqrd))
        ps_single_sub_t.append((p_ttest_rmse, p_ttest_rsqrd))
        print(f'{utils.SUB_NAMES[i]} - pvalue for RMSE: {p_rmse:.4f} R^2: {p_rsqrd:.4f}')
        print(f'   - t-test for RMSE: {p_ttest_rmse:.4f} R^2: {p_ttest_rsqrd:.4f}')

    return (p_rmse_allTog, p_rsqrd_allTog), (p_ttest_rmse_allTog, p_ttest_rsqrd_allTog), ps_single_sub, ps_single_sub_t           