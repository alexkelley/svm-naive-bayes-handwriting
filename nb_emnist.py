import pprint
from project_functions import *

#### Load Data ####
emnist = load_emnist()
X_columns = list(emnist)[1:]
y_column = list(emnist)[0]

# print(display_df_stats(emnist))
# print(count_class_variables(emnist, y_column))


def all_features_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a full data set.
    '''
    samples = build_nb_dataset(df, X_columns, y_column)

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on EMNIST using All Features'

    print(trials_report(nb_scores, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'full_emnist_nb'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(nb_scores['accuracy_test'])))

    return summary_statistics(nb_scores['accuracy_test'], 0.95)


def alternate_features_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on 50% of the features sampled from alternating
    columns in the data sets.
    '''
    X_subset = select_feature_sample(df, X_columns)

    # build 30 training/testing samples using reduced data
    samples = build_nb_dataset(df, X_subset, y_column)

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on EMNIST using {} random features'.format(len(X_subset))
    print(trials_report(nb_scores, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'alternate_features_emnist_nb'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(nb_scores['accuracy_test'])))

    return summary_statistics(nb_scores['accuracy_test'], 0.95)


def linear_pca_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a reduced set of feature using Linear PCA.
    '''
    X_data = df[X_columns]

    # evaluate variance of each PC and export to tab delimited test file
    pc_variance = evaluate_linear_pcs(X_data, len(X_columns))

    filename = 'data/pc_variance_emnist.txt'

    with open(filename, 'w') as f:
        f.write(pc_variance)

    # reduce X to new DateFrame with desired number of PCs
    mnist_lpca = fit_linear_PCA(X_data, 32)

    # reattach labels to reduced DataFrame
    mnist_lpca[y_column] = df[y_column]
    print(display_df_stats(mnist_lpca))

    mnist_lpca = mnist_lpca.dropna(axis='rows')
    print(display_df_stats(mnist_lpca))

    # build 30 training/testing samples using Linear PCA data
    X_columns_lpca = list(mnist_lpca)[:-1]
    samples = build_nb_dataset(mnist_lpca, X_columns_lpca, y_column)

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on EMNIST using 13 PCs from Linear PCA'
    print(trials_report(nb_scores, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'linear_pca_emnist_nb'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(nb_scores['accuracy_test'])))

    return summary_statistics(nb_scores['accuracy_test'], 0.95)


def kernel_pca_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a reduced set of feature using Kernel PCA.
    '''
    X_data = df[X_columns]

    # reduce X to new DateFrame with 32 PCs
    mnist_kpca = fit_kernel_PCA(X_data, 32)

    # reattach labels to reduced DataFrame
    mnist_kpca['label'] = df[y_column]

    # build 30 training/testing samples using Linear PCA data
    X_columns_kpca = list(mnist_kpca)[:-1]
    samples = build_nb_dataset(mnist_kpca, X_columns_kpca, y_column)

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on EMNIST using {} PCs from Kernel PCA'.format(len(X_columns_kpca))
    print(trials_report(nb_scores, 0.95, title))

     # save testing accuracy data to local file
    list_name = 'kernel_pca_emnist_nb'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(nb_scores['accuracy_test'])))

    return summary_statistics(nb_scores['accuracy_test'], 0.95)


################
## Run Trials ##
################

all_features_nb = all_features_nb(emnist, X_columns, y_column)
# alternate_nb = alternate_features_nb(emnist, X_columns, y_column)
# lpca_nb = linear_pca_nb(emnist, X_columns, y_column)
# kpca_nb = kernel_pca_nb(emnist, X_columns, y_column)
