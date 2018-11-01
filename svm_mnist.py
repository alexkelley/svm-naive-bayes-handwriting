import pprint
from project_functions import *

#### Load Data ####
mnist = load_mnist()
X_columns = list(mnist)[:-1]
y_column = 'label'
# print(display_df_stats(mnist))
# print(count_class_variables(mnist, y_column))


def raw_data_svm(df, X_columns, y_column):
    '''
    Run SVM on a full data set.
    '''
    # binarize_classes
    class_set = set(df[y_column])
    binarized_data = {}
    for i in class_set:
        target_class = float(i)
        df_binarized = binarize_classes(df, y_column, target_class)
        binarized_data[i] = df_binarized

    # create 30 bootstrap samples for each binarized class
    svm_data = {}
    for label, data in binarized_data.items():
        samples = {}
        for i in range(30):
            X_train, y_train, X_test, y_test = bootstrap_split(
                data, i, X_columns, y_column)
            samples[i] = [X_train, y_train, X_test, y_test]
        svm_data[label] = samples

    # run SVM on each sample for each class
    label_results = {}
    for label, trial_samples in svm_data.items():
        svm_scores = svm_trial(trial_samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    for label, results in label_results.items():
        title = 'SVM on MNIST using all Features: {} class.'.format(label)
        #print(trials_report(results, 0.95, title))
        # print(title)
        # print(results['accuracy_test'][0])
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)

    # compute overall mean accuracy
    title = 'SVM on MNIST using all Features'
    print(svm_report(overall_data, 0.95, title))

    return True


def linear_pca_svm(df, X_columns, y_column):
    X = df[X_columns]
    print(evaluate_linear_pcs(X, 25))

    # reduce X to new DateFrame with desired number of PCs
    mnist_lpca = fit_linear_PCA(X, 13)

    # reattach labels to reduced DataFrame
    mnist_lpca[y_column] = df[y_column]

    # binarize_classes
    class_set = set(df[y_column])
    binned_data = {}
    for i in class_set:
        target_class = float(i)
        df_binned = binarize_classes(mnist_lpca, y_column, target_class)
        binned_data[i] = df_binned

    # create 30 bootstrap samples for each binarized class
    X_columns_lpca = list(mnist_lpca)[:-1]
    svm_data = {}
    for label, data in binned_data.items():
        samples = {}
        for i in range(30):
            X_train, y_train, X_test, y_test = bootstrap_split(
                data, i, X_columns_lpca, y_column)
            samples[i] = [X_train, y_train, X_test, y_test]
        svm_data[label] = samples

    # run SVM on each sample for each class
    label_results = {}
    for label, samples in svm_data.items():
        svm_scores = svm_trial(samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    for label, results in label_results.items():
        title = 'SVM on MNIST using all Features: {} class.'.format(label)
        #print(trials_report(svm_scores, 0.95, title))
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)

    # compute overall mean accuracy
    title = 'SVM on MNIST using Linear PCA'
    print(svm_report(overall_data, 0.95, title))

    return True


def random_svm(df, X_columns, y_column):
    '''
    Run SVM on reduced feature set selected randomly.
    '''
    X_subset = select_feature_sample(df, X_columns)

    # binarize_classes
    class_set = set(df[y_column])
    binarized_data = {}
    for i in class_set:
        target_class = float(i)
        df_binarized = binarize_classes(df, y_column, target_class)
        binarized_data[i] = df_binarized

    # create 30 bootstrap samples for each binarized class
    svm_data = {}
    for label, data in binarized_data.items():
        samples = {}
        for i in range(30):
            X_train, y_train, X_test, y_test = bootstrap_split(
                data, i, X_subset, y_column)
            samples[i] = [X_train, y_train, X_test, y_test]
        svm_data[label] = samples

    # run SVM on each sample for each class
    label_results = {}
    for label, trial_samples in svm_data.items():
        svm_scores = svm_trial(trial_samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    for label, results in label_results.items():
        title = 'SVM on MNIST using all Features: {} class.'.format(label)
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)

    # compute overall mean accuracy
    title = 'SVM on MNIST using a random subset of Features'
    print(svm_report(overall_data, 0.95, title))

    return True


def kernel_pca_svm(df, X_columns, y_column):
    X = df[X_columns]

    # reduce X to new DateFrame with PC count = 1/2 original features
    mnist_kpca = fit_kernel_PCA(X, int(len(X)/2))

    # reattach labels to reduced DataFrame
    mnist_kpca['label'] = df[y_column]

    # binarize_classes
    class_set = set(df[y_column])
    binned_data = {}
    for i in class_set:
        target_class = float(i)
        df_binned = binarize_classes(mnist_kpca, y_column, target_class)
        binned_data[i] = df_binned

    # create 30 bootstrap samples for each binarized class
    X_columns_kpca = list(mnist_kpca)[:-1]
    svm_data = {}
    for label, data in binned_data.items():
        samples = {}
        for i in range(30):
            X_train, y_train, X_test, y_test = bootstrap_split(
                data, i, X_columns_kpca, y_column)
            samples[i] = [X_train, y_train, X_test, y_test]
        svm_data[label] = samples

    # run SVM on each sample for each class
    label_results = {}
    for label, samples in svm_data.items():
        svm_scores = svm_trial(samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    for label, results in label_results.items():
        title = 'SVM on MNIST using Kernel PCA: {} class.'.format(label)
        #print(trials_report(svm_scores, 0.95, title))
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)

    # compute overall mean accuracy
    title = 'SVM on MNIST using Kernel PCA'
    print(svm_report(overall_data, 0.95, title))

    return True

################
## Run Trials ##
################
# raw_data_svm(mnist, X_columns, y_column)
# linear_pca_svm(mnist, X_columns, y_column)
# random_svm(mnist, X_columns, y_column)
kernel_pca_svm(mnist, X_columns, y_column)
