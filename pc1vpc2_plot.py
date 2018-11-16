import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pc1_v_pc2(df, X_columns, y_column):
    X_data = df[X_columns]
    pcs = fit_linear_PCA(X_data, 2)
    pcs[y_column] = df[y_column]

    classes = list(set(df[y_column]))
    plot_data = pd.DataFrame(columns=('label', 'pc1', 'pc2'))

    for i in range(len(classes)):
        df_reduced = pcs.loc[pcs[y_column] == i]
        plot_data.loc[int(i)] = [int(i),
                                 df_reduced[0].mean(),
                                 df_reduced[1].mean()]

    fig, ax = plt.subplots(1,1)

    ax = sns.scatterplot(
        x='pc1',
        y='pc2',
        data=plot_data,
        hue='label',
        palette=sns.color_palette(n_colors=10)
    )

    ax.set_title('Average Value by Label PC1 versus PC2',
                 fontsize=18,
                 color="b",
                 alpha=0.5)
    ax.set_xlabel('PC1',
                  size = 12,
                  color="b",
                  alpha=0.5)
    ax.set_ylabel('PC2',
                  size = 12,
                  color="b",
                  alpha=0.5)
    plt.show()
