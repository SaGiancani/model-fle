import numpy as np
from scipy.optimize import least_squares
import stat_metrics as stat
import topographic_model as topo
import utils


# Combined model function to fit
def combined_model(beta, x, model_indices):
    models = [topo.mdl1, topo.mdl2, topo.mdl3, topo.mdl4]
    predictions = np.zeros_like(x)
    for i, mdl_idx in enumerate(model_indices):
        predictions[i] = models[mdl_idx](beta, x[i])
    return predictions

# Cost function to minimize across all subjects
def residuals(beta, X, Y, model_indices):
    total_residuals = []
    for i in range(X.shape[0]):  # Loop over subjects (9 subjects)
        pred = combined_model(beta, X[i, :], model_indices)
        total_residuals.extend(pred - Y[i, :])
    return np.array(total_residuals)

# Cost function for a single subject
def residuals_single(beta, x, y, model_indices):
    return combined_model(beta, x, model_indices) - y

def fit_data_sub_B_sub(data, x_data, flag_print = True):
    # Instance storing variables
    fit_quality = list()
    betas = np.zeros((data.shape[0], len(utils.BETA_INIT)))
    prediction = np.zeros(data.shape)
 
    # Model indices to assign 8 conditions to 4 models (repeating pattern across subjects)
    model_indices = np.repeat(np.arange(utils.N_MODELS), utils.N_CONDITIONS)

    for i in range(data.shape[0]):

        result = least_squares(residuals_single, utils.BETA_INIT, args=(x_data[i, :], data[i, :], model_indices))
        betas[i, :] = result.x
        
        # Predicted values for the subject
        prediction[i, :] = combined_model(result.x, x_data[i, :], model_indices)

        # Calculate fit quality metrics
        rmse, r2 = stat.calculate_fit_quality(data[i, :], prediction[i, :])
        fit_quality.append((rmse, r2))

        if flag_print:
            print(f"Processing subject {i+1}")
            print(f"Fitted beta for subject {i+1}: {result.x}")
            print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}\n")    
    return betas, fit_quality, prediction

def fit_all_together(data, x_data, flag_print = True):
    # Check on data shape
    if len(data.shape) != 2:
        print('data shape has to be 2d (n_subs, n_conds).')
        return

    n_subs = data.shape[0]
    fit_quality = list()

    # Model indices to assign 8 conditions to 4 models (repeating pattern across subjects)
    model_indices = np.repeat(np.arange(utils.N_MODELS), utils.N_CONDITIONS)  
    model_indices_ = np.array([model_indices]*n_subs).reshape(-1)

    ### 2. Global fitting across all subjects ###
    result_global = least_squares(residuals_single, utils.BETA_INIT, args=(x_data.ravel(), data.ravel(), model_indices_))
    beta_global = result_global.x

    # Fitted data for quality measure
    prediction = np.zeros_like(data)
    for i in range(n_subs):
        prediction[i, :] = combined_model(beta_global, x_data[i, :], model_indices)
    # Flatten the true and predicted values for evaluation
    y_true = data.flatten()
    y_pred = prediction.flatten()

    # Calculate fit quality metrics
    rmse_glob, r2_glob = stat.calculate_fit_quality(y_true, y_pred)
    fit_quality.append((rmse_glob, r2_glob))
    
    if flag_print:
        print("Fitted global beta parameters:", beta_global)
        print(f"RMSE: {rmse_glob:.4f}, R²: {r2_glob:.4f}\n")

    return beta_global, fit_quality, prediction

def fit_data(data, x_data):
    betas, fit_quality              = fit_data_sub_B_sub(data, x_data)
    beta_global, fit_quality_global = fit_all_together(data, x_data)
    return fit_quality, betas, fit_quality_global, beta_global