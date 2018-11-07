import numpy as np
import pandas as pd

import scipy.stats as stats

from data.full_small_nb import full_small_nb
from data.alternate_features_small_nb import alternate_features_small_nb
from data.linear_pca_small_nb import linear_pca_small_nb
from data.kernel_pca_small_nb import kernel_pca_small_nb
from data.full_small_svm import full_small_svm
from data.alternate_features_small_svm import alternate_features_small_svm
from data.linear_pca_small_svm import linear_pca_small_svm
from data.kernel_pca_small_svm import kernel_pca_small_svm

nb_means = [full_small_nb, alternate_features_small_nb, linear_pca_small_nb, kernel_pca_small_nb]

svm_means = [full_small_svm, alternate_features_small_svm, linear_pca_small_svm, kernel_pca_small_svm]


trial_names = ['No feature reduction', 'Select alternate features', 'Linear PCA', 'Kernel PCA']

pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def calculate_t_tests(means):
    t_results = ''
    for i in pairs:
        trial1 = means[i[0]]
        name1 = trial_names[i[0]]
        trial2 = means[i[1]]
        name2 = trial_names[i[1]]
        t_test = stats.ttest_rel(trial1, trial2)

        if t_test[1] < 0.05:
            hypothesis = 'Reject the null hypothesis that two trials have equal means so accuracy results are significantly different'
        else:
            hypothesis = "Can't reject null hypothesis so the accuracy scores are not significantly different"


        t_results += '{}\t{}\t{:.10f}\t{}\n'.format(name1, name2, t_test[1], hypothesis)

    return t_results


#print(calculate_t_tests(nb_means))
print(calculate_t_tests(svm_means))
