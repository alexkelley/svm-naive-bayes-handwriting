import pprint
from project_functions import *

emnist = load_emnist()
print(display_df_stats(emnist))

X_columns = list(emnist)[1:]
y_column = list(emnist)[0]

print(count_class_variables(emnist, y_column))

#### Raw Data ####

# build 30 training/testing samples using raw data
samples = {}
for i in range(30):
    X_train, y_train, X_test, y_test = bootstrap_split(
        emnist, i, X_columns, y_column)
    samples[i] = [X_train, y_train, X_test, y_test]

# run Naïve Bayes classifier
nb_scores = nb_trial(samples)

title = 'Naïve Bayes classifier on EMNIST using Raw Data'
print(trials_report(nb_scores, 0.95, title))



#### LPCA ####

# X = emnist[X_columns]
# # print(evaluate_principal_components(X, 50))

# # reduce X to new DateFrame with desired number of PCs
# emnist_lpca = fit_linear_PCA(X, 32)

# # reattach labels to reduced DataFrame
# emnist_lpca['label'] = emnist[y_column]
# print(display_df_stats(emnist_lpca))

# # build 30 training/testing samples using Linear PCA data
# X_columns_lpca = list(emnist_lpca)[:-1]
# samples = {}
# for i in range(30):
#     X_train, y_train, X_test, y_test = bootstrap_split(
#         emnist_lpca, i, X_columns_lpca, 'label')
#     samples[i] = [X_train, y_train, X_test, y_test]

# # run Naïve Bayes classifier
# nb_scores = nb_trial(samples)

# title = 'Naïve Bayes classifier on EMNIST using Linear PCA'
# print(trials_report(nb_scores, 0.95, title))
