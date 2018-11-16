import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.test_kernel_pca_small_nb import test_kernel_pca_small_nb
from data.test_linear_pca_small_nb import test_linear_pca_small_nb

from data.test_linear_pca_small_svm import test_linear_pca_small_svm
from data.test_kernel_pca_small_svm import test_kernel_pca_small_svm

def plot_pcs_svm():
    x1 = [i[0] for i in test_linear_pca_small_svm]
    y1 = [i[1] for i in test_linear_pca_small_svm]

    x2 = [i[0] for i in test_kernel_pca_small_svm]
    y2 = [i[1] for i in test_kernel_pca_small_svm]

    fig, ax = plt.subplots(1,1)

    ax = sns.lineplot(
        x=x1,
        y=y1,
        color="green",
        linewidth=6,
        label='LPCA'
    )

    ax = sns.lineplot(
        x=x2,
        y=y2,
        color="coral",
        linewidth=2,
        label='KPCA'
    )

    ax.set_title('Testing Accuracy by Number of Components\nUsed in Linear & Kernel PCA - SVM',
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


def plot_pcs_nb():
    x1  = [i[0] for i in test_linear_pca_small_nb]
    y1 = [i[1] for i in test_linear_pca_small_nb]

    x2 = [i[0] for i in test_kernel_pca_small_nb]
    y2 = [i[1] for i in test_kernel_pca_small_nb]

    fig, ax = plt.subplots(1,1)

    ax = sns.lineplot(
        x=x1,
        y=y1,
        color="green",
        linewidth=6,
        label='LPCA'
    )

    ax = sns.lineplot(
        x=x2,
        y=y2,
        color="coral",
        linewidth=2,
        label='KPCA'
    )

    ax.set_title('Testing Accuracy by Number of Components\nUsed in Linear & Kernel PCA - Naive Bayes',
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


#plot_pcs_nb()
plot_pcs_svm()
