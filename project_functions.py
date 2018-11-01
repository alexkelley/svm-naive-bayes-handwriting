import random
from operator import itemgetter
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import scipy.stats as stats


def load_mnist():
    d = load_digits()
    df = pd.DataFrame(data = d.data)
    df['label'] = d.target
    df[['label']] = df[['label']].astype('float64')
    return df


def load_emnist():
    df = pd.read_csv('emnist-digits-test.csv')
    return df


def display_df_stats(df):
    '''
    Takes a DataFrame object and returns a string of information regarding it.
    '''
    n = len(df)
    summary = 'Number of rows: {}\n'.format(n)
    summary += 'Number of columns: {}\n'.format(len(list(df)))
    summary += 'Number of missing values: {}\n'.format(
        df.isnull().values.ravel().sum())

    return summary


def count_class_variables(df, y):
    '''
    Computes the distribution of class variables.

    Parameters:
    - df: DataFrame object
    - y: string, target class column name

    Returns: a string detailing the class variable distribution
    '''
    y_cases = list(set(df[y]))
    observations = len(df)
    summary = 'Count of observations: {}\n'.format(observations)
    summary += 'Observations by Subset of Cases:\n'

    case_summary = []
    for i in y_cases:
        df_temp = df.ix[df[y] == i]
        case_summary.append([i, len(df_temp), len(df_temp)/float(observations)])

    for j in sorted(case_summary, reverse=False):
        summary += '\t{:>15}: {:>5}\t{:.2f}%\n'.format(
            j[0],
            j[1],
            j[2]*100)

    return summary


def evaluate_linear_pcs(X_data, n_components):
    results = '{}\t{}\t{}\n'.format('Component', 'Variance', 'Cumulative Variance')
    pca = PCA(n_components=n_components)
    pca.fit(X_data)
    variance = pca.explained_variance_ratio_

    cumsum = 0.0
    for count, i in enumerate(variance):
        cumsum += i
        results += '{:<15}\t{:<15.3f}\t{:<15.2f}\n'.format(count+1, i, cumsum)

    return results


def fit_linear_PCA(X_data, n_components):
    transform_data = PCA(n_components=n_components).fit_transform(X_data)
    df = pd.DataFrame(transform_data)
    return df


def fit_kernel_PCA(X_data, n_components):
    '''
   Documentation at: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
    '''
    transform_data = KernelPCA(
        n_components=n_components,
        kernel='rbf',
        gamma = 0.0002,
        fit_inverse_transform=True
    ).fit_transform(X_data)
    df = pd.DataFrame(transform_data)
    return df


def select_feature_sample(df, X_columns):
    '''
    Select alternate features as a new sub-sample of features to include in the model
    '''
    subset = []
    for i in X_columns[0::2]:
        subset.append(i)
    return subset


def bootstrap_split(df, random_state, X, y):
    '''
    Split a DataFrame into training and testing sets using the Bootstrap with
    replacement method.

    Parameters:
    - df: DataFrame object
    - random_state: int to set randomization start point for repeatability
    - X: list of column names for predictor attributes
    - y: string of column name for target attribute

    Returns a 4-tuple of data objects for use in a classifier algorithm
    '''
    boot_train = resample(df, n_samples=len(df), random_state=random_state)
    boot_indices = list(set(boot_train.index.tolist()))
    boot_test = df.loc[~df.index.isin(boot_indices)]
    X_train = boot_train[X]
    y_train = boot_train[y]
    X_test = boot_test[X]
    y_test = boot_test[y]

    return X_train, y_train, X_test, y_test


def nb_trial(samples):
    '''
    Run an instance of a Naive Bayes classifier for a list of samples

    Parameters:
    - sample: dictionary with {trial_number: data_list}
              data_list = [X_train, y_train, X_test, y_test]

    Returns a dictionary of labeled training and testing accuracy scores.
    '''
    acc_train = []
    acc_test = []

    for trial, data in samples.items():
        X_train, y_train, X_test, y_test = data
        nbclf = GaussianNB()
        fitnb = nbclf.fit(X_train, y_train)

        y_pred = nbclf.predict(X_test)

        acc_train.append(nbclf.score(X_train, y_train))
        acc_test.append(nbclf.score(X_test, y_test))
        #fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())

    scores = {
        'accuracy_train': acc_train,
        'accuracy_test': acc_test
        #'sensitivity_test': tpr,
        #'specificity_test': 1 - fpr
    }

    return scores


def summary_statistics(data, confidence):
    '''
    Calculates summary statistics for a list of values

    Parameters:
    - data: list of float values
    - confidence: float of confidence level for significance test

    Returns: dictionary of labeled summary statistics
    '''
    a = 1.0 * np.array(data)
    n = len(a)
    mean, std = np.mean(a), np.std(a)
    if n > 1:
        se = stats.sem(a)
        ci = se * stats.t.ppf((1 + confidence) / 2., n-1)
    else:
        ci = 0.0

    summary = {
        'length': n,
        'mean': mean,
        'error': 1 - mean,
        'standard_deviation': std,
        'variance': std**2,
        'confidence_bound': ci,
        'confidence_level': confidence
    }

    return summary


def trials_report(scores, confidence, title):
    '''
    Returns a string summarizing the accuracy results from a series of trial runs
    of a classifier algorithm.
    '''
    summary = '{}\n'.format(title)

    for label, data in scores.items():

        stats = summary_statistics(data, confidence)
        subtitle = '{} Set Accuracy for {} Trials'.format(label.title(), stats['length'])
        summary += '{1}\n{0}\n{1}\n'.format(subtitle, len(subtitle) * '-')
        summary += 'Mean: {:.2f}%\n'.format(stats['mean'] * 100)
        summary += 'Standard Deviation: {:.2f}%\n'.format(sqrt(stats['variance']) * 100)
        summary += 'With {:.0f}% confidence the mean lies within [{:.2f}%, {:.2f}%]\n\n'.format(
            confidence * 100,
            (stats['mean'] - stats['confidence_bound']) * 100,
            (stats['mean'] + stats['confidence_bound']) * 100)

    return summary


def binarize_classes(df, y_column, target_class):
    '''
    Create a new DataFrame that converts multiclass data into a binary class
    structure.

    Takes a
    - DataFrame object
    - string, name of y_column
    - float, target_class for the positive class
    '''
    # split into two DataFrames: target_class and all others
    n_classes = len(set(df[y_column]))
    target_df = df.loc[df[y_column] == target_class]
    n_rows = len(target_df)
    negative_df = df.loc[df[y_column] != target_class]

    # change y_column to 1 for target_class 1.0
    target_df[y_column] = 1.0

    # stratify sample negative class data set to size of target_class data set
    new_negative_df = pd.DataFrame(columns=list(negative_df))
    strat_n_rows = int(n_rows / (n_classes - 1))
    strat_values = set(negative_df[y_column])

    for i in strat_values:
        df_strat = negative_df.ix[negative_df[y_column] == i]
        df_strat = df_strat.sample(frac=1.0)
        new_negative_df = new_negative_df.append(df_strat.iloc[:strat_n_rows])

    # convert negative class y_column to -1
    new_negative_df[y_column] = 0

    # rejoin split DataFrames and return
    final_df = target_df.append(new_negative_df)

    return final_df


def build_nb_dataset(df, X_columns, y_column):
    samples = {}
    for i in range(30):
        X_train, y_train, X_test, y_test = bootstrap_split(
            df, i, X_columns, y_column)
        samples[i] = [X_train, y_train, X_test, y_test]

    return samples


def build_svm_dataset(df, X_columns, y_column):
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

    return svm_data


def svm_trial(samples):
    '''
    Run an instance of an SVM classifier for a list of samples

    Parameters:
    - sample: dictionary with {trial_number: data_list}
              data_list = [X_train, y_train, X_test, y_test]

    Returns a dictionary of labeled training and testing accuracy scores.
    '''
    acc_train = []
    acc_test = []

    for trial, data in samples.items():
        X_train, y_train, X_test, y_test = data
        clf = svm.SVC(kernel='rbf', C=3, gamma=0.01)
        fit = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc_train.append(clf.score(X_train, y_train))
        acc_test.append(clf.score(X_test, y_test))
        fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())

    scores = {
        'accuracy_train': acc_train,
        'accuracy_test': acc_test,
        'sensitivity_test': tpr,
        'specificity_test': 1 - fpr
    }

    return scores



def svm_report(overall_data, confidence, title):
    '''
    Returns a string summarizing the accuracy results from a series of trial runs
    of a classifier algorithm.
    '''
    label_means = []
    for i in overall_data.values():
        label_means.append(i['mean'])

    svm_raw = summary_statistics(label_means, 0.95)
    summary = '{}\n'.format(title)
    subtitle = 'Testing Set Accuracy for 10 Labels (0-9)'
    summary += '{1}\n{0}\n{1}\n'.format(subtitle, len(subtitle) * '-')
    summary += 'Mean: {:.2f}%\n'.format(svm_raw['mean'] * 100)
    summary += 'Standard Deviation: {:.2f}%\n'.format(sqrt(svm_raw['variance']) * 100)
    summary += 'With 95% confidence the mean lies within [{:.2f}%, {:.2f}%]\n\n'.format(
        (svm_raw['mean'] - svm_raw['confidence_bound']) * 100,
        (svm_raw['mean'] + svm_raw['confidence_bound']) * 100
    )
    return summary
