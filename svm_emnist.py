import pprint
from project_functions import *

#### Load Data ####
emnist = load_emnist()
X_columns = list(emnist)[1:]
y_column = list(emnist)[0]
# print(display_df_stats(emnist))
# print(count_class_variables(emnist, y_column))


def all_features_svm(df, X_columns, y_column):
    '''
    Run SVM on a full data set.
    '''
    svm_data = build_svm_dataset(df, X_columns, y_column)

    # run SVM on each sample for each class
    label_results = {}
    for label, trial_samples in svm_data.items():
        svm_scores = svm_trial(trial_samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        title = 'SVM on EMNIST using all Features: {} class.'.format(label)
        #print(trials_report(results, 0.95, title))
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on EMNIST using all Features'
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'full_emnist_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(export_list)))

    return True


def alternate_features_svm(df, X_columns, y_column):
    '''
    Run SVM on reduced feature set selected randomly.
    '''
    X_subset = select_feature_sample(df, X_columns)

    svm_data = build_svm_dataset(df, X_subset, y_column)

    # run SVM on each sample for each class
    label_results = {}
    for label, trial_samples in svm_data.items():
        svm_scores = svm_trial(trial_samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        title = 'SVM on EMNIST using all Features: {} class.'.format(label)
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on EMNIST using Alternate Features'
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'alternate_features_emnist_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(export_list)))

    return True


def linear_pca_svm(df, X_columns, y_column):
    X_data = df[X_columns]

    # reduce X to new DateFrame with desired number of PCs
    emnist_lpca = fit_linear_PCA(X_data, 13)

    # reattach labels to reduced DataFrame
    emnist_lpca[y_column] = df[y_column]
    X_columns_lpca = list(emnist_lpca)[:-1]

    svm_data = build_svm_dataset(emnist_lpca, X_columns_lpca, y_column)

    # run SVM on each sample for each class
    label_results = {}
    for label, samples in svm_data.items():
        svm_scores = svm_trial(samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        title = 'SVM on EMNIST using all Features: {} class.'.format(label)
        #print(trials_report(svm_scores, 0.95, title))
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on EMNIST using Linear PCA'
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'linear_pca_emnist_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(export_list)))

    return True


def kernel_pca_svm(df, X_columns, y_column):
    X_data = df[X_columns]

    # reduce X to new DateFrame with 13 PCs
    emnist_kpca = fit_kernel_PCA(X_data, 13)

    # reattach labels to reduced DataFrame
    emnist_kpca['label'] = df[y_column]
    X_columns_kpca = list(emnist_kpca)[:-1]

    svm_data = build_svm_dataset(emnist_kpca, X_columns_kpca, y_column)

    # run SVM on each sample for each class
    label_results = {}
    for label, samples in svm_data.items():
        svm_scores = svm_trial(samples)
        label_results[label] = svm_scores

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        title = 'SVM on EMNIST using Kernel PCA: {} class.'.format(label)
        #print(trials_report(svm_scores, 0.95, title))
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on EMNIST using Kernel PCA'
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'kernel_pca_emnist_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(export_list)))

    return True

################
## Run Trials ##
################
all_features_svm(emnist, X_columns, y_column)
# alternate_features_svm(emnist, X_columns, y_column)
# linear_pca_svm(emnist, X_columns, y_column)
# kernel_pca_svm(emnist, X_columns, y_column)
