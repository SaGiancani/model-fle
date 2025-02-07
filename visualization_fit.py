from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import topographic_model as topo
import utils

def visualize_fit_n_raw_sub(Y_data_tmp, X_data, beta_subjects, sub_names, filename = None):

    # In case of subplots with as many rows as the subjects are
    num_subjects = Y_data_tmp.shape[0]
    fig, axs = plt.subplots(num_subjects, 4, figsize=(20, 30), sharey='row', sharex='col', dpi=150)
    fig.patch.set_facecolor("white")       
    for i in range(num_subjects):
        max_val = np.nanmax(Y_data_tmp[i, :]) 
        min_val = np.nanmin(Y_data_tmp[i, :])

        axs[i, 0].tick_params(axis='y', labelsize=15)  # Set X tick fontsize

        # Column 1
        axs[i, 0].scatter(X_data[0, :8], Y_data_tmp[i, :8], alpha=0.6, color=utils.SUB_COLORS[i])
        axs[i, 0].plot(
            np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
            topo.mdl1(beta_subjects[i, :], np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
            color=utils.SUB_COLORS[i]
        )
        axs[i, 0].set_ylim([min_val, max_val])
        axs[i, 0].set_ylabel('FLE - ms', fontsize = 18)
        if i == num_subjects-1:
            axs[i, 0].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
            axs[i, 0].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
            axs[i, 0].tick_params(axis='x', labelsize=15)  # Set Y tick fontsize            
            axs[i, 0].set_xlabel('Orientation', fontsize = 18)
        axs[i, 0].text(np.pi/3, max_val-(.2*max_val), f'{sub_names[i]}', fontsize = 18)        

        # Column 2
        axs[i, 1].scatter(X_data[0, 8:16], Y_data_tmp[i, 8:16], alpha=0.6, color=utils.SUB_COLORS[i])
        axs[i, 1].plot(
            np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
            topo.mdl2(beta_subjects[i, :], np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
            color=utils.SUB_COLORS[i]
        )
        axs[i, 1].set_ylim([min_val, max_val])
        if i == num_subjects-1:
            axs[i, 1].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
            axs[i, 1].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
            axs[i, 1].tick_params(axis='x', labelsize=15)  # Set Y tick fontsize            
            axs[i, 1].set_xlabel('Orientation', fontsize = 18)
            
        # Column 3
        axs[i, 2].scatter(X_data[0, 16:24], Y_data_tmp[i, 16:24], alpha=0.6, color=utils.SUB_COLORS[i])
        axs[i, 2].plot(
            np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
            topo.mdl3(beta_subjects[i, :], np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
            color=utils.SUB_COLORS[i]
        )
        axs[i, 2].set_ylim([min_val, max_val])
        if i == num_subjects-1:
            axs[i, 2].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
            axs[i, 2].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
            axs[i, 2].tick_params(axis='x', labelsize=15)  # Set Y tick fontsize            
            axs[i, 2].set_xlabel('Orientation', fontsize = 18)
            
        # Column 4
        axs[i, 3].scatter(X_data[0, 24:], Y_data_tmp[i, 24:], alpha=0.6, color=utils.SUB_COLORS[i])
        axs[i, 3].plot(
            np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
            topo.mdl4(beta_subjects[i, :], np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
            color=utils.SUB_COLORS[i]
        )
        axs[i, 3].set_ylim([min_val, max_val])
        if i == num_subjects-1:
            axs[i, 3].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
            axs[i, 3].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
            axs[i, 3].tick_params(axis='x', labelsize=15)  # Set Y tick fontsize    
            axs[i, 3].set_xlabel('Orientation', fontsize = 18)
        


    # Adjust layout for better spacing
    plt.tight_layout()
    if filename is not None:
        plt.savefig(os.path.join('figures', f'{filename}.png'))
        plt.savefig(os.path.join('figures', f'{filename}.pdf'), format = 'pdf', dpi =500)

    plt.show()
    return



def visualize_fit_n_raw_allTogether(Y_data, X_data, beta, filename = None):

    # In case of subplots with as many rows as the subjects are
    num_subjects = Y_data.shape[0]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey='row', sharex='col', dpi=150)
    fig.patch.set_facecolor("white")       

    max_val = np.nanmax(Y_data.ravel()) 
    min_val = np.nanmin(Y_data.ravel())

    # Column 1
    for i in range(num_subjects):
        axs[0].scatter(X_data[i, :8], Y_data[i, :8], alpha=0.6, color=utils.SUB_COLORS[i])
    axs[0].plot(
        np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
        topo.mdl1(beta, np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
        color='crimson', lw = 3)
    
    axs[0].set_ylim([min_val, max_val])
    axs[0].set_ylabel('FLE - ms', fontsize = 18)
    axs[0].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
    axs[0].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
    axs[0].tick_params(axis='both', labelsize=15)  # Set Y tick fontsize            
    axs[0].set_xlabel('Orientation', fontsize = 18)    
    axs[0].text(np.pi/3, max_val-(.2*max_val), f'All together', fontsize = 18)        

    # Column 2
    for i in range(num_subjects):
        axs[1].scatter(X_data[i, 8:16], Y_data[i, 8:16], alpha=0.6, color=utils.SUB_COLORS[i])
    axs[1].plot(
        np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
        topo.mdl2(beta, np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
        color='crimson', lw = 3)
    axs[1].set_ylim([min_val, max_val])
    axs[1].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
    axs[1].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
    axs[1].tick_params(axis='x', labelsize=15)  # Set Y tick fontsize            
    axs[1].set_xlabel('Orientation', fontsize = 18)    

    # Column 3
    for i in range(num_subjects):
        axs[2].scatter(X_data[i, 16:24], Y_data[i, 16:24], alpha=0.6, color=utils.SUB_COLORS[i])
    axs[2].plot(
        np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
        topo.mdl3(beta, np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
        color='crimson', lw = 3)
    axs[2].set_ylim([min_val, max_val])
    axs[2].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
    axs[2].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
    axs[2].tick_params(axis='x', labelsize=15)  # Set Y tick fontsize            
    axs[2].set_xlabel('Orientation', fontsize = 18)    

    # Column 4
    for i in range(num_subjects):
        axs[3].scatter(X_data[i, 24:], Y_data[i, 24:], alpha=0.6, color=utils.SUB_COLORS[i])
    axs[3].plot(
        np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100),
        topo.mdl4(beta, np.linspace(utils.X_RANGE[0], utils.X_RANGE[-1], 100), a=2, d=2),
        color='crimson', lw = 3)
    axs[3].set_ylim([min_val, max_val])
    axs[3].set_xticks([2 * np.pi, np.pi/4, np.pi/2, np.pi*(3/4), np.pi, (5/4)*np.pi, 3 * np.pi / 2, (7/4)*np.pi])
    axs[3].set_xticklabels(['←', '↙', '↓', '↘', '→', '↗', '↑', '↖'])
    axs[3].tick_params(axis='x', labelsize=15)  # Set Y tick fontsize            
    axs[3].set_xlabel('Orientation', fontsize = 18)    
            

    # Adjust layout for better spacing
    plt.tight_layout()
    if filename is not None:
        plt.savefig(os.path.join('figures', f'{filename}.png'))
        plt.savefig(os.path.join('figures', f'{filename}.pdf'), format = 'pdf', dpi =500)

    plt.show()
    return

def visualize_betas(beta_subjects, filename = None, betas_to_plot = ['Offset', 'Fovea', 'HM', 'VM', 'Eccentricity'], not_significant_subs = None):

    # Data dictionary
    data = {'Offset': beta_subjects[:, 0],
            'Fovea': beta_subjects[:, 1],
            'HM': beta_subjects[:, 2],
            'VM': beta_subjects[:, 3],
            'Eccentricity': beta_subjects[:, 4]}
    
    data = {k: v for k, v in data.items()  if k in betas_to_plot}

    # Get all unique pairs of Beta columns
    beta_columns = list(data.keys())
    beta_pairs = list(combinations(beta_columns, 2))  # Generate all combinations of two
    
    # Plot all combinations of beta pairs
    plt.figure(figsize=(14, 17), facecolor='white')
    # subs = [1, 2, 4, 5, 6, 8, 9, 11, 12]
    subs = utils.SUB_ENUMERATION + ['avg']
    for idx, (beta1, beta2) in enumerate(beta_pairs):
    #     print(beta1, beta2)
        plt.subplot(4, 3, idx+1)  # Adjust rows and columns for better visualization
        if not_significant_subs is None:
            plt.scatter(data[beta1], data[beta2], marker='o', c='k', alpha=0.7)
        else:
            for nots, (bs1, bs2) in enumerate(zip(data[beta1], data[beta2])):
                if subs[nots] in not_significant_subs:
                    plt.scatter(bs1, bs2, marker='o', c='grey', alpha=0.7)
                else:
                    plt.scatter(bs1, bs2, marker='o', c='k', alpha=0.7)
                    
        for n in range(len(subs)):
            plt.text(data[beta1][n]+.1, data[beta2][n]+.1, f'{utils.SUB_NAMES[n]}', fontsize = 11, color = 'k')
        plt.plot([-4, 4], [-4, 4], 'k--', alpha = .4)
            
        plt.xlabel(beta1, fontsize=12)

        plt.ylabel(beta2, fontsize=12)

        plt.hlines(0, -4, 4, ls = '--', color = 'crimson', lw = 3)
        plt.vlines(0, -4, 4, ls = '--', color = 'crimson', lw = 3)
        plt.title(f"{beta1} vs {beta2}", fontsize=14)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(os.path.join('figures', f'{filename}.png'))
        plt.savefig(os.path.join('figures', f'{filename}.pdf'), format = 'pdf', dpi =500)

    plt.show()
    return

def visualize_hists(rmses, rsqrs, fit_quality, sub_names = utils.SUB_NAMES, filename = None):
    num_subjects = rmses.shape[0]
    if num_subjects > 1:
        rows_dim = int(round(num_subjects*1.5))
        cols_dim = num_subjects*3
    else:
        rows_dim = 15
        cols_dim = 5   

    fig, axs = plt.subplots(num_subjects, 2, figsize=(rows_dim, cols_dim), sharey='row', dpi=150)
    fig.patch.set_facecolor("white")       

    for n, rms in enumerate(rmses): 
        # Create the histogram
        if num_subjects > 1:
            asse = [axs[n, 0], axs[n, 1]]
        else:
            asse = [axs[0], axs[1]]

        hist_values_rms, bin_edges_rms   = np.histogram(rms.ravel(), bins=300)
        x_for_sub_text = bin_edges_rms[0]

        hist_values_rsqr, bin_edges_rsqr = np.histogram(rsqrs[n, :].ravel(), bins=300)


        y_for_fit_quality_plotting = np.nanmax([hist_values_rms, hist_values_rsqr])+2
        asse[0].bar(bin_edges_rms[:-1], hist_values_rms, width=np.diff(bin_edges_rms), align='edge', edgecolor='black')    
        asse[0].axvline(fit_quality[n][0], color='r', linewidth=2)
        asse[0].text(fit_quality[n][0], y_for_fit_quality_plotting, f'{fit_quality[n][0]:.4f}', color = 'r')
        asse[0].text(x_for_sub_text, y_for_fit_quality_plotting-4, f'{sub_names[n]}', fontsize = 18)        

        # Add titles and labels
        asse[0].set_title("$RMSE$", fontsize=16)
        asse[0].set_xlabel("Value", fontsize=14)
        asse[0].set_ylabel("Frequency", fontsize=14)
        asse[0].tick_params(axis='both', labelsize=14)  # Set Y tick fontsize            


        asse[1].bar(bin_edges_rsqr[:-1], hist_values_rsqr, width=np.diff(bin_edges_rsqr), align='edge', edgecolor='black')    
        
        asse[1].axvline(fit_quality[n][1], color='r', linewidth=2)
        asse[1].text(fit_quality[n][1], y_for_fit_quality_plotting, f'{fit_quality[n][1]:.4f}', color = 'r')
        # Add titles and labels
        asse[1].set_title("$R^{2}$ ", fontsize=16)
        asse[1].set_xlabel("Value", fontsize=14)
        asse[1].tick_params(axis='both', labelsize=15)  # Set Y tick fontsize            

        # Display the plot
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

    if filename is not None:
        plt.savefig(os.path.join('figures', f'{filename}.png'))
        plt.savefig(os.path.join('figures', f'{filename}.pdf'), format = 'pdf', dpi =500)

    plt.show()
    return