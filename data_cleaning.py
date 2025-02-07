import numpy as np
import utils

def sort_condition(data):
    # Preprocess the data
    num_subjects = data.shape[0]

    y_data = np.zeros((num_subjects, utils.N_MODELS, utils.N_CONDITIONS))  # 9 subjects, 4 models, 8 conditions

    for subj in range(num_subjects):
        loi = data[subj, :]
        ys = loi.reshape(utils.N_MODELS, utils.N_CONDITIONS)  # Reshape to (4, 8)
        ys = -ys[:, [3, 4, 5, 6, 7, 0, 1, 2]] * 10  # Rescale in ms. Rawdata at FLE72020 in tens of ms.
        ys[[1, 2]] = ys[[2, 1]] # Invert the two mid-positions for visualization purposes
        y_data[subj, :, :] = ys

    return y_data