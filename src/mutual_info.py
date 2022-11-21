import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import sklearn


def organize_mi_scores(scores, cols):
    scores = pd.Series(scores, name="Mutual Info Scores", index=cols)

    return scores.sort_values(ascending=False)


def plot_mi_scores(scores, cols):
    scores = pd.Series(scores, name="Mutual Info Scores", index=cols)
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.figure(dpi=100, figsize=(8, 5))
