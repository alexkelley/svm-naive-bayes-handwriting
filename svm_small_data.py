import time
import pprint
from project_functions import *

#### Load Data ####
df_small = load_small()
X_columns = list(df_small)[:-1]
y_column = 'label'
# print(display_df_stats(mnist))
# print(count_class_variables(mnist, y_column))


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

    summarize_labels(label_results, 'svm_all_results')

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        title = 'SVM on Small Dataset using all Features: {} class.'.format(label)
        #print(trials_report(results, 0.95, title))
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on Small Dataset using all Features'
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'full_small_svm'
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

    summarize_labels(label_results, 'svm_alternate_results')

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        title = 'SVM on Small Dataset using all Features: {} class.'.format(label)
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on Small Dataset using Alternate Features'
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'alternate_features_small_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(export_list)))

    return True


def linear_pca_svm(df, X_columns, y_column):
    X_data = df[X_columns]

    # reduce X to new DateFrame with desired number of PCs
    n_components = 6
    mnist_lpca = fit_linear_PCA(X_data, n_components)

    # reattach labels to reduced DataFrame
    mnist_lpca[y_column] = df[y_column]
    X_columns_lpca = list(mnist_lpca)[:-1]

    svm_data = build_svm_dataset(mnist_lpca, X_columns_lpca, y_column)

    # run SVM on each sample for each class
    label_results = {}
    for label, samples in svm_data.items():
        svm_scores = svm_trial(samples)
        label_results[label] = svm_scores

    summarize_labels(label_results, 'svm_lpca_results')

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        title = 'SVM on Small Dataset using all Features: {} class.'.format(label)
        #print(trials_report(svm_scores, 0.95, title))
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on Small Dataset using {} PCs with Linear PCA'.format(n_components)
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'linear_pca_small_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(export_list)))

    return True


def kernel_pca_svm(df, X_columns, y_column):
    X_data = df[X_columns]

    # reduce X to new DateFrame with selected PCs
    n_components = 5
    df_kpca = fit_kernel_PCA(X_data, n_components)

    # reattach labels to reduced DataFrame
    df_kpca['label'] = df[y_column]
    X_columns_kpca = list(df_kpca)[:-1]

    svm_data = build_svm_dataset(df_kpca, X_columns_kpca, y_column)

    # run SVM on each sample for each class
    label_results = {}
    for label, samples in svm_data.items():
        svm_scores = svm_trial(samples)
        label_results[label] = svm_scores

    summarize_labels(label_results, 'svm_kpca_results')

    # report mean accuracy & CI for each class
    overall_data = {}
    export_list = []
    for label, results in label_results.items():
        overall_data[label] = summary_statistics(results['accuracy_test'], 0.95)
        export_list.extend(results['accuracy_test'])

    # compute overall mean accuracy
    title = 'SVM on Small Dataset using {} PCs from Kernel PCA'.format(n_components)
    #print(title)
    print(svm_report(overall_data, 0.95, title))

    # save testing accuracy data to local file
    list_name = 'kernel_pca_small_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, pprint.pformat(export_list)))

    return True


def compare_lpca(df, X_columns, y_column):
    X_data = df[X_columns]
    results = []
    for i in range(1, 3): #65
        # reduce X to new DateFrame with varying PCs
        df_lpca = fit_linear_PCA(X_data, i)

        # reattach labels to reduced DataFrame
        df_lpca['label'] = df[y_column]

        # build 30 training/testing samples using Linear PCA data
        X_columns_lpca = list(df_lpca)[:-1]
        svm_data = build_svm_dataset(df_lpca, X_columns_lpca, y_column)

        # run SVM on each sample for each class
        label_results = {}
        export_list = []
        for label, samples in svm_data.items():
            svm_scores = svm_trial(samples)
            label_results[label] = svm_scores
            export_list.extend(svm_scores['accuracy_test'])

        #print(export_list)
        # report mean accuracy & CI for each class
        # overall_data = {}
        # for label, r in label_results.items():
        #     overall_data[label] = summary_statistics(r['accuracy_test'], 0.95)

        title = 'SVM classifier using {} PCs from Linear PCA'.format(i)
        print(title)
        #print(kpca_trials_report(export_list, 0.95, title))

        accuracy = summary_statistics(export_list, 0.95)['mean']
        results.append((i, accuracy))

    # save testing accuracy data to local file
    list_name = 'test_linear_pca_small_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, results))

    return True


def compare_kpca(df, X_columns, y_column):
    X_data = df[X_columns]
    results = []
    for i in range(1, 65): #65
        # reduce X to new DateFrame with varying PCs
        df_kpca = fit_kernel_PCA(X_data, i)

        # reattach labels to reduced DataFrame
        df_kpca['label'] = df[y_column]

        # build 30 training/testing samples using Linear PCA data
        X_columns_kpca = list(df_kpca)[:-1]
        svm_data = build_svm_dataset(df_kpca, X_columns_kpca, y_column)

        # run SVM on each sample for each class
        label_results = {}
        export_list = []
        for label, samples in svm_data.items():
            svm_scores = svm_trial(samples)
            label_results[label] = svm_scores
            export_list.extend(svm_scores['accuracy_test'])

        print(export_list)
        # report mean accuracy & CI for each class
        # overall_data = {}
        # for label, r in label_results.items():
        #     overall_data[label] = summary_statistics(r['accuracy_test'], 0.95)

        title = 'SVM classifier using {} PCs from Kernel PCA'.format(i)
        print(title)
        #print(kpca_trials_report(export_list, 0.95, title))

        accuracy = summary_statistics(export_list, 0.95)['mean']
        results.append((i, accuracy))

    # save testing accuracy data to local file
    list_name = 'test_kernel_pca_small_svm'
    filename = 'data/{}.py'.format(list_name)

    with open(filename, 'w') as f:
        f.write('{0} = {1}'.format(list_name, results))

    return True


################
## Run Trials ##
################
if __name__ == '__main__':
    start_time = time.time()

    # all_features_svm(df_small, X_columns, y_column)
    # alternate_features_svm(df_small, X_columns, y_column)
    linear_pca_svm(df_small, X_columns, y_column)
    #kernel_pca_svm(df_small, X_columns, y_column)

    #compare_lpca(df_small, X_columns, y_column)
    #compare_kpca(df_small, X_columns, y_column)

    end_time = time.time()
    print('\nElapsed time: {:.2f} seconds\n'.format(end_time - start_time))
