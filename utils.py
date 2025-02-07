import numpy as np
import os
import pandas as pd

BETA_INIT       = np.array([1, 1, 1, 1, 1])  
DATA_FILENAME   = 'FLE7_2020.csv'
N_CONDITIONS    = 8
N_ITERS         = 1500
N_MODELS        = 4
SUB_COLORS      = ["red", "blue", "green", "gray", "purple", "teal", "gold", "brown", "cyan", 'k']
SUB_ENUMERATION = [1, 2, 4, 5, 6, 8, 9, 11, 12]
SUB_NAMES       = [f'S{SUB_ENUMERATION[i]}' if i<len(SUB_ENUMERATION) else 'avg' for i in range(len(SUB_ENUMERATION)+1)]
X_RANGE         =  np.arange(np.pi/4, 2*np.pi + np.pi/4, np.pi/4)


def fit_result_outcome(betas, fit_quality, beta_global, fit_quality_global, filename = 'model_fitting_results.csv'):
    # Data preparation for data storing as csv
    data_for_df = np.array([np.array([i[0], i[1], betas[n, 0], betas[n, 1], betas[n, 2], betas[n, 3], betas[n, 4]]) for n, i in enumerate(fit_quality)])
    data_tmp = np.zeros((data_for_df.shape[0]+1, data_for_df.shape[1]))
    data_tmp[:-1, :] = data_for_df
    data_tmp[-1, :] = [fit_quality_global[0][0], fit_quality_global[0][1], beta_global[0], beta_global[1], beta_global[2], beta_global[3], beta_global[4]]
    data_for_df = data_tmp

    fit_quality_df = pd.DataFrame(data_for_df, columns=["RMSE", "R_squared", "Offset", "Fovea", "HM", "VM", "Eccentricity"], index=SUB_NAMES+ ['AllTogether'])
    fit_quality_df.to_csv(os.path.join('data', filename))
    return

def append_row_to_csv(new_row, new_index=None, filename='model_fitting_results.csv'):
    # Load the existing CSV file
    file_path = os.path.join('data', filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)  # Load with index
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Append the new row with a custom index (if provided)
    if new_index is not None:
        df.loc[new_index] = new_row
    else:
        df.loc[len(df)] = new_row  # Use default integer index

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path)
    return

def stat_outcome(ps_single_sub_randomfit, ps_t_single_sub_randomfit, allTog_list, allTog_t_list, fit_quality_global, fit_quality, filename = 'fit_quality_results'):

    p_rmse_allTog_fit  = allTog_list[0]
    p_rsqrd_allTog_fit = allTog_list[1]
    p_ttest_rmse_allTog_fit  = allTog_t_list[0]
    p_ttest_rsqrd_allTog_fit = allTog_t_list[1]

    # Create a list to store all results
    results = []

    # Add results for the "all subs together" case
    results.append({
        'Case': 'All Subs Together',
        'Metric': 'R^2',
        'Value': fit_quality_global[0][1],  # Actual R^2 value
        'P-value (Randomization)': p_rsqrd_allTog_fit,
        'P-value (t-test)': p_ttest_rsqrd_allTog_fit
    })

    results.append({
        'Case': 'All Subs Together',
        'Metric': 'RMSE',
        'Value': fit_quality_global[0][0],  # Actual RMSE value
        'P-value (Randomization)': p_rmse_allTog_fit,
        'P-value (t-test)': p_ttest_rmse_allTog_fit
    })

    # Add results for each individual subject
    for i in range(len(fit_quality)):
        p_rmse, p_rsqrd = ps_single_sub_randomfit[i]
        p_ttest_rmse, p_ttest_rsqrd = ps_t_single_sub_randomfit[i]

        results.append({
            'Case': SUB_NAMES[i],
            'Metric': 'R^2',
            'Value': fit_quality[i][1],  # Actual R^2 value for this subject
            'P-value (Randomization)': p_rsqrd,
            'P-value (t-test)': p_ttest_rsqrd
        })
        results.append({
            'Case': SUB_NAMES[i],
            'Metric': 'RMSE',
            'Value': fit_quality[i][0],  # Actual RMSE value for this subject
            'P-value (Randomization)': p_rmse,
            'P-value (t-test)': p_ttest_rmse
        })

    # Convert the list of results to a DataFrame
    df_results = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    df_results.to_csv(os.path.join('data', f'{filename}.csv'), index=False)

    print(f"Results saved to {filename}.csv")
    return

def prediction_outcome(x_preds, y_preds, deltas, sub_names):
    # Get number of subjects and datapoints
    num_points = x_preds.shape[1]    # 400 datapoints per subject

    # Repeat subject names for all datapoints
    subject_column = np.repeat(sub_names, num_points)

    # Flatten matrices to create one long column per variable
    x_column = x_preds.flatten()
    y_column = y_preds.flatten()
    delta_column = deltas.flatten()

    # Create a DataFrame
    df = pd.DataFrame({'Subject': subject_column,
                       'X': x_column,
                       'Y': y_column,
                       'Delta': delta_column})

    # Save to CSV
    df.to_csv(os.path.join('data', 'predictions_with_ci.csv'), index=False)

    print("CSV file saved successfully!")
    return