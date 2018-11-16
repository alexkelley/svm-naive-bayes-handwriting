import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from project_functions import *

def digit_plot(indices_list):
    plt.figure(figsize=(12,4))
    for i, img_data in enumerate(indices_list):
        n = len(indices_list)
        plt.subplot(1, n, i + 1)
        image = np.array(X_test.iloc[img_data,:])
        label = labels_test[i]
        if label == -1:
            label = 6
        else:
            label = 5
        pixels = image.reshape(16,16)
        plt.imshow(pixels, cmap=plt.cm.gray)
        plt.title('Actual Digit: {}'.format(label))
    plt.show()

#### Load Data ####
df = load_small()
X_columns = list(df)[:-1]
y_column = 'label'

n_components = 5
pca = PCA(n_components=n_components)
data = pca.fit_transform(df[X_columns])
df2 = pd.DataFrame(data)
approximation = pca.inverse_transform(data)


nrows = 10
ncols = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(2,8))

for row in range(nrows):
    label = np.array(df.loc[row])[-1]
    image1 = np.array(df.loc[row])[:-1]
    image2 = approximation[row].reshape(8,8)
    axes[row, 0].imshow(image1.reshape(8,8), cmap=plt.cm.gray)
    axes[row, 0].set_yticklabels([])
    axes[row, 0].set_ylabel(label)
    axes[row, 0].set_xticklabels([])
    axes[row, 1].imshow(image2, cmap=plt.cm.gray)
    axes[row, 1].set_yticklabels([])
    axes[row, 1].set_xticklabels([])

fig.subplots_adjust(hspace=0.2, wspace=0.2)
plt.show()
