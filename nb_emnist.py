import pprint
from project_functions import *

#### Load Data ####
emnist = load_emnist()
X_columns = list(emnist)[1:]
y_column = list(emnist)[0]
print(display_df_stats(emnist))
print(count_class_variables(emnist, y_column))


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


################
## Run Trials ##
################

all_features_nb = all_features_nb(emnist, X_columns, y_column)
