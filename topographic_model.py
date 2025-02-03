import numpy as np

# Define the four topographic models
def mdl1(beta, x, a=2, d=2):
    return beta[0] + beta[4] * (a / 2) * (beta[1] * (x * 0 + a) + beta[2] * np.abs(a * np.sin(x)) + beta[3] * np.abs(a * np.cos(x)))

def mdl2(beta, x, a=2, d=2):
    return beta[0] + beta[4] * np.sqrt((np.cos(x + np.pi) * (a / 2)) ** 2 + (d - np.sin(x + np.pi) * (a / 2)) ** 2) * (
        beta[1] * (np.sqrt((a * np.cos(x + np.pi)) ** 2 + (d - a * np.sin(x + np.pi)) ** 2) - d) +
        beta[2] * a * np.sin(x) + beta[3] * np.abs(a * np.cos(x)))

def mdl3(beta, x, a=2, d=2):
    return beta[0] + beta[4] * np.sqrt((d + np.cos(x + np.pi) * (a / 2)) ** 2 + (np.sin(x + np.pi) * (a / 2)) ** 2) * (
        beta[1] * (np.sqrt((d + a * np.cos(x + np.pi)) ** 2 + (a * np.sin(x + np.pi)) ** 2) - d) +
        beta[2] * np.abs(a * np.sin(x)) - beta[3] * a * np.cos(x))

def mdl4(beta, x, a=2, d=2):
    return beta[0] + beta[4] * np.sqrt((d + np.cos(x + np.pi) * (a / 2)) ** 2 + (d - np.sin(x + np.pi) * (a / 2)) ** 2) * (
        beta[1] * (np.sqrt((d + a * np.cos(x + np.pi)) ** 2 + (d - a * np.sin(x + np.pi)) ** 2) - np.sqrt(2) * d) +
        beta[2] * a * np.sin(x) - beta[3] * a * np.cos(x))
