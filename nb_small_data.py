import pprint
from project_functions import *

#### Load Data ####
df_small = load_small()
X_columns = list(df_small)[:-1]
y_column = 'label'
#print(df_small.head())
#print(display_df_stats(df_small))
#print(count_class_variables(df_small, y_column))


def all_features_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a full data set.
    '''
    samples = build_nb_dataset(df, X_columns, y_column)

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on Small Dataset using All Features'

    print(title)

    average_confusion_matrices(
        nb_scores['confusion_matrix'],
        'nb_full_confusion')

    # save testing accuracy data to local file
    list_name = 'full_small_nb'
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

    title = 'Naïve Bayes classifier on Small Dataset using {} random features'.format(len(X_subset))

    print(title)

    average_confusion_matrices(
        nb_scores['confusion_matrix'],
        'nb_alternate_confusion')

    # save testing accuracy data to local file
    list_name = 'alternate_features_small_nb'
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

    filename = 'data/pc_variance_small.txt'

    with open(filename, 'w') as f:
        f.write(pc_variance)

    # reduce X to new DateFrame with desired number of PCs
    n_components = 13
    mnist_lpca = fit_linear_PCA(X_data, n_components)

    # reattach labels to reduced DataFrame
    mnist_lpca[y_column] = df[y_column]

    # build 30 training/testing samples using Linear PCA data
    X_columns_lpca = list(mnist_lpca)[:-1]
    samples = build_nb_dataset(mnist_lpca, X_columns_lpca, y_column)

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on Small Dataset using {} PCs from Linear PCA'.format(n_components)
    print(title)

    average_confusion_matrices(
        nb_scores['confusion_matrix'],
        'nb_lpca_confusion')

    # save testing accuracy data to local file
    list_name = 'linear_pca_small_nb'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(nb_scores['accuracy_test'])))

    return summary_statistics(nb_scores['accuracy_test'], 0.95)


def kernel_pca_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a reduced set of feature using Kernel PCA.
    '''
    X_data = df[X_columns]

    # reduce X to new DateFrame with 35 PCs
    n_components = 35
    mnist_kpca = fit_kernel_PCA(X_data, n_components)

    # reattach labels to reduced DataFrame
    mnist_kpca['label'] = df[y_column]

    # build 30 training/testing samples using Linear PCA data
    X_columns_kpca = list(mnist_kpca)[:-1]
    samples = build_nb_dataset(mnist_kpca, X_columns_kpca, y_column)

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on Small Dataset using {} PCs from Kernel PCA'.format(n_components)
    print(title)

    average_confusion_matrices(
        nb_scores['confusion_matrix'],
        'nb_kpca_confusion')

     # save testing accuracy data to local file
    list_name = 'kernel_pca_small_nb'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(nb_scores['accuracy_test'])))

    return summary_statistics(nb_scores['accuracy_test'], 0.95)


def compare_lpca(df, X_columns, y_column):
    X_data = df[X_columns]
    results = []
    for i in range(1, 65):
        # reduce X to new DateFrame with varying PCs
        mnist_lpca = fit_linear_PCA(X_data, i)

        # reattach labels to reduced DataFrame
        mnist_lpca['label'] = df[y_column]

        # build 30 training/testing samples using Linear PCA data
        X_columns_lpca = list(mnist_lpca)[:-1]
        samples = build_nb_dataset(mnist_lpca, X_columns_lpca, y_column)

        # run Naïve Bayes classifier
        nb_scores = nb_trial(samples)

        title = 'Naïve Bayes classifier on Small Dataset using {} PCs from Linear PCA'.format(i)
        print(lpca_trials_report(nb_scores, 0.95, title))

        accuracy = summary_statistics(
            nb_scores['accuracy_test'],
            0.95)['mean']
        results.append((i, accuracy))

    # save testing accuracy data to local file
    list_name = 'test_linear_pca_small_nb'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, results))

    return True


def compare_kpca(df, X_columns, y_column):
    X_data = df[X_columns]
    results = []
    for i in range(1, 65):
        # reduce X to new DateFrame with varying PCs
        mnist_kpca = fit_kernel_PCA(X_data, i)

        # reattach labels to reduced DataFrame
        mnist_kpca['label'] = df[y_column]

        # build 30 training/testing samples using Linear PCA data
        X_columns_kpca = list(mnist_kpca)[:-1]
        samples = build_nb_dataset(mnist_kpca, X_columns_kpca, y_column)

        # run Naïve Bayes classifier
        nb_scores = nb_trial(samples)

        title = 'Naïve Bayes classifier on Small Dataset using {} PCs from Kernel PCA'.format(i)
        print(kpca_trials_report(nb_scores, 0.95, title))

        accuracy = summary_statistics(nb_scores['accuracy_test'], 0.95)['mean']
        results.append((i, accuracy))

    # save testing accuracy data to local file
    list_name = 'test_kernel_pca_small_nb2'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, results))

    return True


################
## Run Trials ##
################

# all_features_nb = all_features_nb(df_small, X_columns, y_column)
# print(all_features_nb)
# alternate_nb = alternate_features_nb(df_small, X_columns, y_column)
# print(alternate_nb)
# lpca_nb = linear_pca_nb(df_small, X_columns, y_column)
# print(lpca_nb)
# kpca_nb = kernel_pca_nb(df_small, X_columns, y_column)
# print(kpca_nb)

# compare_lpca(df_small, X_columns, y_column)
compare_kpca(df_small, X_columns, y_column)

## Visualize the KPCA accuracy scores by number of components
#plot_pc1_v_pc2(df_small, X_columns, y_column)
