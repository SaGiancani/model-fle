import numpy as np
from scipy.optimize import least_squares
import scipy.stats as stats
import stat_metrics as metric
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
        rmse, r2 = metric.calculate_fit_quality(data[i, :], prediction[i, :])
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
    rmse_glob, r2_glob = metric.calculate_fit_quality(y_true, y_pred)
    fit_quality.append((rmse_glob, r2_glob))
    
    if flag_print:
        print("Fitted global beta parameters:", beta_global)
        print(f"RMSE: {rmse_glob:.4f}, R²: {r2_glob:.4f}\n")

    return beta_global, fit_quality, prediction

def fit_data(data, x_data):
    betas, fit_quality              = fit_data_sub_B_sub(data, x_data)
    beta_global, fit_quality_global = fit_all_together(data, x_data)
    return fit_quality, betas, fit_quality_global, beta_global

def get_jacobian(x_data, beta, model_indices, epsilon = 1e-6 ):
    n = len(x_data)
    p = len(beta)
    # Compute new Jacobian at x_pred (numerically)
    J_pred = np.zeros((n, p))
    for j in range(p):
        beta_step = beta.copy()
        beta_step[j] += epsilon
        J_pred[:, j] = (combined_model(beta_step, x_data, model_indices) - 
                        combined_model(beta, x_data, model_indices)) / epsilon
    return J_pred

def get_covariance_mat(J, resid):
    n = J.shape[0]
    p = J.shape[1]    

    # Normalize the Jacobian columns
    J_normalized = J
    
    # Compute residual variance (MSE)
    sigma2 = np.sum(resid**2) / (n - p)
    
    # Estimate Covariance matrix of betas
    J_T_J = J_normalized.T @ J_normalized
    
    # Regularization using singular value thresholding
    U, S, Vt = np.linalg.svd(J_T_J)
    S_inv = np.diag([1/s if s > 1e-6 else 0 for s in S])  # Threshold small singular values
    J_T_J_inv = Vt.T @ S_inv @ U.T

    CovB = sigma2 * J_T_J_inv
    return CovB

def ci(beta_fit, x_pred, CovB, model_indices, alpha=0.05, epsilon=1e-6):
    """
    Compute confidence intervals for predictions at x_pred using the Jacobian.
    """
    J_pred = get_jacobian(x_pred, beta_fit, model_indices, epsilon = epsilon)
    n, p = J_pred.shape
    
    # Compute variance of predicted values
    var_y_pred = np.diag(J_pred @ CovB @ J_pred.T)
    
    # Compute confidence interval half-width
    t_value = stats.t.ppf(1 - alpha / 2, df=n - p)
    delta = t_value * np.sqrt(var_y_pred)
    
    # Clip negative or extreme values (sanity check)
    delta = np.clip(delta, 0, np.max(delta) / 2)

    print(f"Confidence interval half-width (delta):\n{delta[:5]}")  # Print first 5 values
    
    # Compute predicted values
    y_pred = combined_model(beta_fit, x_pred, model_indices)

    return y_pred, delta

def get_confidence_interval(data, x_data, betas, beta_global, prediction_subs, alpha = .05, eps = 1e-5):
    # Compute residuals
    resids = data - prediction_subs

    alpha_    = alpha #95% confidence interval
    epsilon_  = eps
    model_ids = np.repeat(np.arange(utils.N_MODELS), utils.N_CONDITIONS)

    # Compute jacobian for each of subjects + average
    # Compute covariance matrix
    J    = np.zeros((len(x_data), x_data.shape[1], betas.shape[1]))
    CovB = np.zeros((len(x_data), betas.shape[1], betas.shape[1]))
    for i in range(len(x_data)):
        J[i, :, :]    = get_jacobian(x_data[i, :], betas[i, :], model_ids, epsilon = epsilon_)
        CovB[i, :, :] = get_covariance_mat(J[i, :, :], resids[i, :])

    # Define x_pred at original x_data, at a finer grid
    xax_pred       = np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100)
    x_pred_dense   = np.tile(xax_pred, utils.N_MODELS)
    model_id_dense = np.repeat(np.arange(utils.N_MODELS), len(xax_pred))

    y_pred_dense = np.zeros((x_data.shape[0], x_pred_dense.shape[0]))
    deltas       = np.zeros((x_data.shape[0], x_pred_dense.shape[0]))
    # Compute CI at both sets of x_pred
    for i in range(len(x_data)):
        y_pred_dense[i, :], deltas[i, :] = ci(betas[i, :], x_pred_dense, 
                                              CovB[i, :], model_id_dense, 
                                              alpha=alpha_, epsilon=epsilon_)
        
    # Global delta extraction
    ravl_resids = resids[:-1, :].ravel()
    ravl_x      = x_data[:-1, :].ravel()

    model_indices_ = np.repeat(np.arange(utils.N_MODELS), utils.N_CONDITIONS)  
    model_indices_ = np.array([model_indices_]*len(x_data[:-1, :])).reshape(-1)

    J_glbl    = get_jacobian(ravl_x, beta_global, model_indices_, epsilon = epsilon_)
    CovB_glbl = get_covariance_mat(J_glbl, ravl_resids)

    ravl_dense_x    = np.ones((data[:-1, :].shape[0], len(x_pred_dense)))*x_pred_dense
    model_id_dense_ = np.array([model_id_dense]*len(x_data[:-1, :])).reshape(-1)

    y_pred_dense_glbl, deltas_glbl = ci(beta_global, 
                                        ravl_dense_x.reshape(-1), 
                                        CovB_glbl, 
                                        model_id_dense_, 
                                        alpha=alpha_, epsilon=epsilon_)
    
    # Data preparation for output
    d_glb = deltas_glbl.reshape((len(x_data[:-1, :]), len(x_pred_dense)))[0, :]
    y_glb = y_pred_dense_glbl.reshape((len(x_data[:-1, :]), len(x_pred_dense)))[0, :]

    y_preds_ = np.zeros((y_pred_dense.shape[0]+1, y_pred_dense.shape[1]))
    deltas_  = np.zeros((deltas.shape[0]+1, deltas.shape[1]))

    y_preds_[-1, :]  = y_glb
    deltas_[-1, :]   = d_glb
    y_preds_[:-1, :] = y_pred_dense
    deltas_[:-1, :]  = deltas
    x_preds_         = np.ones((len(deltas_), len(x_pred_dense)))*x_pred_dense

    return x_preds_, y_preds_, deltas_