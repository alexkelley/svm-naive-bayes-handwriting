import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.test_kernel_pca_small_nb import test_kernel_pca_small_nb


def plot_kpca():
    x = [i[0] for i in test_kernel_pca_small_nb]
    y = [i[1] for i in test_kernel_pca_small_nb]

    fig, ax = plt.subplots(1,1)
    ax = sns.lineplot(
        x=x,
        y=y,
        color="coral",
        label='Testing Accuracy'
    )
    ax.set_title('Testing Accuracy by Number of Components\nUsed in Kernel PCA',
                 fontsize=18,
                 color="b",
                 alpha=0.5)
    ax.set_xlabel('Number of Components',
                  size = 12,
                  color="b",
                  alpha=0.5)
    ax.set_ylabel('Testing Accuracy',
                  size = 12,
                  color="b",
                  alpha=0.5)
    plt.show()


plot_kpca()
