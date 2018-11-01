import pprint
from project_functions import *

#### Load Data ####
mnist = load_mnist()
X_columns = list(mnist)[:-1]
y_column = 'label'
# print(display_df_stats(mnist))
# print(count_class_variables(mnist, y_column))


def raw_data_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a full data set.
    '''
    samples = {}
    for i in range(30):
        X_train, y_train, X_test, y_test = bootstrap_split(
            df, i, X_columns, y_column)
        samples[i] = [X_train, y_train, X_test, y_test]

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on MNIST using Raw Data'

    print(trials_report(nb_scores, 0.95, title))

    return summary_statistics(nb_scores['accurracy_test'], 0.95)


def linear_pca_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a reduced set of feature using Linear PCA.
    '''
    X = df[X_columns]
    print(evaluate_linear_pcs(X, 25))

    # reduce X to new DateFrame with desired number of PCs
    mnist_lpca = fit_linear_PCA(X, 13)

    # reattach labels to reduced DataFrame
    mnist_lpca[y_column] = df[y_column]
    print(display_df_stats(mnist_lpca))

    # build 30 training/testing samples using Linear PCA data
    X_columns_lpca = list(mnist_lpca)[:-1]
    samples = {}
    for i in range(30):
        X_train, y_train, X_test, y_test = bootstrap_split(
            mnist_lpca, i, X_columns_lpca, y_column)
        samples[i] = [X_train, y_train, X_test, y_test]

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on MNIST using 13 PCs from Linear PCA'
    print(trials_report(nb_scores, 0.95, title))

    return summary_statistics(nb_scores['accurracy_test'], 0.95)


def kernel_pca_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a reduced set of feature using Kernel PCA.
    '''
    X = df[X_columns]

    # reduce X to new DateFrame with desired number of PCs
    mnist_kpca = fit_kernel_PCA(X, 13)

    # reattach labels to reduced DataFrame
    mnist_kpca['label'] = df[y_column]
    print(display_df_stats(mnist_kpca))

    # build 30 training/testing samples using Linear PCA data
    X_columns_kpca = list(mnist_kpca)[:-1]
    samples = {}
    for i in range(30):
        X_train, y_train, X_test, y_test = bootstrap_split(
            mnist_kpca, i, X_columns_kpca, y_column)
        samples[i] = [X_train, y_train, X_test, y_test]

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on MNIST using {} PCs from Kernel PCA'.format(len(X_columns_kpca))
    print(trials_report(nb_scores, 0.95, title))

    return summary_statistics(nb_scores['accurracy_test'], 0.95)


def random_features_nb(df, X_columns, y_column):
    '''
    Run Naïve Bayes on a random sample of 50% of the
    features from the middle portion of the data set.
    '''
    X_subset = select_feature_sample(df, X_columns)

    # build 30 training/testing samples using reduced data
    samples = {}
    for i in range(30):
        X_train, y_train, X_test, y_test = bootstrap_split(
            df, i, X_subset, y_column)
        samples[i] = [X_train, y_train, X_test, y_test]

    # run Naïve Bayes classifier
    nb_scores = nb_trial(samples)

    title = 'Naïve Bayes classifier on MNIST using {} random features'.format(len(X_subset))
    print(trials_report(nb_scores, 0.95, title))

    return summary_statistics(nb_scores['accurracy_test'], 0.95)


################
## Run Trials ##
################

raw_nb = raw_data_nb(mnist, X_columns, y_column)
pprint.pprint(raw_nb)

# lpca_nb = linear_pca_nb(mnist, X_columns, y_column)
# pprint.pprint(lpca_nb)

# kpca_nb = kernel_pca_nb(mnist, X_columns, y_column)
# pprint.pprint(kpca_nb)

# random_nb = random_features_nb(mnist, X_columns, y_column)
# pprint.pprint(random_nb)
