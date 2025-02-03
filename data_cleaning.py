import numpy as np
import utils

def sort_condition(data):
    # Preprocess the data
    num_subjects = data.shape[0]

    y_data = np.zeros((num_subjects, utils.N_MODELS, utils.N_CONDITIONS))  # 9 subjects, 4 models, 8 conditions

    for subj in range(num_subjects):
        loi = data[subj, :]
        ys = loi.reshape(utils.N_MODELS, utils.N_CONDITIONS)  # Reshape to (4, 8)
        ys = -ys[:, [4, 5, 6, 7, 0, 1, 2, 3]] * 10  # Rescale in ms. Rawdata at FLE72020 in tens of ms.
        y_data[subj, :, :] = ys

    return y_data