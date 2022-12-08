import pandas as pd
import numpy as np
import sys

from math import ceil
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor


def accuracy_assesment(imputed_df_s: list, original_df, columns, numeric_columns=[], vector_columns=[]) -> list:
    """
    This method compute the accuracy of the imputed dataframes with respect to the original dataframe.
    :param imputed_df_s: list of imputed pandas dataframes.
    :param original_df: pandas dataframe taken as reference when computing the accuracy.
    :param columns: columns to check.
    :param numeric_columns: list of column names that have a numeric value.
    :param vector_columns: list of vector variables.
    :return: list of accuracies.
    """
    percentages = [f'{p}%' for p in range(50, 100, 10)]
    accuracies = []

    tot_size = original_df.shape[0] * original_df.shape[1]

    for i_df in imputed_df_s:
        distance_error = 0
        for c in columns:

            # defining distance function based on the type of variable.
            if c in numeric_columns:
                maximum_distance = original_df[c].max() - original_df[c].min()

                def distance_function(x, y): return (
                    np.abs(x - y) / maximum_distance)
            elif c in vector_columns:
                
                def distance_function(x, y):
                    x = np.array(x) / np.linalg.norm(x)
                    y = np.array(y) / np.linalg.norm(y)
                    assert np.linalg.norm(x) == 1
                    assert np.linalg.norm(y) == 1
                    distance = (1 - np.dot(x, y)) / 2
                    assert distance <= 1
                    return distance
            else:
                def distance_function(x, y): return 1 if x != y else 0

            # retriving values for a specific column
            imputed_column = pd.Series(i_df[c]).values
            original_column = pd.Series(original_df[c]).values

            # compare columns
            for i, o in zip(imputed_column, original_column):
                distance_error += distance_function(i, o)

        accuracy = (tot_size - distance_error) / tot_size
        accuracies.append(accuracy)

    return dict(zip(percentages, accuracies))


def iterative_imputation_DT(df_in, target, numerical_columns=[], min_split=8, n_iter=10):
    """
    This method uses Decision trees to iteratively impute NaN values.
    :param df_in: pandas dataframe to impute.
    :param target: target column. This column should not be considered during imputation.
    :param numerical_columns: columns that must be imputed with regression algorithms. Default empty list.
    :param neighbours: number of neighbours for KNN.
    :param n_iter: number of times the dataset must be scanned.
    :return: a new pandas dataframe imputed.
    """

    df = df_in.copy()
    missing_columns, missing_indexes, full_indexes, train_columns = prepare_iterative_imputation(df, target)

    # start with basic imputation
    simple_imputer = SimpleImputer(
        missing_values=np.NaN, strategy='most_frequent')
    df = simple_imputer.fit_transform(df)
    df = pd.DataFrame(df, columns=df_in.columns)
    progressbar = ProgressBar(n_iter)
    for _ in range(n_iter):
        progressbar.next()
        for c, l, f, m in zip(missing_columns, train_columns, full_indexes, missing_indexes):
            # Prepare data for the imputatoin: fit the model with features belonging to
            # l, which are all the labels but the one that must be imputed.
            # Samples selected for fitting are the ones havin index f, thus the ones
            # having a value in column c in the original dataset.
            
            if len(m) == 0:
                continue
            train_y = df[c].iloc[f]
            X = df[l].copy()
            X = pd.get_dummies(X)
            train_X = X.iloc[f]
            imputed_X = X.iloc[m]

            if (c not in numerical_columns):
                knn = DecisionTreeClassifier(min_samples_split=8)
                knn.fit(train_X, train_y)
                imputed_y = knn.predict(imputed_X)
            else:
                regr = DecisionTreeRegressor(min_samples_split=8)
                regr.fit(train_X, train_y)
                imputed_y = regr.predict(imputed_X)

            df[c].iloc[m] = imputed_y
    progressbar.reset()
    return df

def iterative_imputation_KNN(df_in, target, numerical_columns=[], neighbours=3, n_iter=10):
    """
    This method uses KNN to iteratively impute NaN values.
    :param df_in: pandas dataframe to impute.
    :param target: target column. This column should not be considered during imputation.
    :param numerical_columns: columns that must be imputed with regression algorithms. Default empty list.
    :param neighbours: number of neighbours for KNN.
    :param n_iter: number of times the dataset must be scanned.
    :return: a new pandas dataframe imputed.
    """
    df = df_in.copy()
    missing_columns, missing_indexes, full_indexes, train_columns = prepare_iterative_imputation(df, target)

    # start with basic imputation
    simple_imputer = SimpleImputer(
        missing_values=np.NaN, strategy='most_frequent')
    df = simple_imputer.fit_transform(df)
    df = pd.DataFrame(df, columns=df_in.columns)
    progressbar = ProgressBar(n_iter)
    for _ in range(n_iter):
        progressbar.next()
        for c, l, f, m in zip(missing_columns, train_columns, full_indexes, missing_indexes):
            # Prepare data for the imputatoin: fit the model with features belonging to
            # l, which are all the labels but the one that must be imputed.
            # Samples selected for fitting are the ones havin index f, thus the ones
            # having a value in column c in the original dataset.
            
            if len(m) == 0:
                continue
            train_y = df[c].iloc[f]
            X = df[l].copy()
            X = pd.get_dummies(X)
            train_X = X.iloc[f]
            imputed_X = X.iloc[m]

            if (c not in numerical_columns):
                knn = KNeighborsClassifier(n_neighbors=neighbours)
                knn.fit(train_X, train_y)
                imputed_y = knn.predict(imputed_X)
            else:
                regr = KNeighborsRegressor(n_neighbors=neighbours)
                regr.fit(train_X, train_y)
                imputed_y = regr.predict(imputed_X)

            df[c].iloc[m] = imputed_y
    progressbar.reset()
    return df


def prepare_iterative_imputation(df_in, target):
    """
    Prepare data for the iterative imputation.
    :param df_in: pandas dataframe for which data must be retrived.
    :param target: column representing the target variable for the ML task.
    :return: 
        - missing_columns: list of columns that must be imputed.
        - missing indexes: list of lists. List at index i contains the indexes for column i which there is a missing value.
        - full indexes: list of complementaries of missing indexes.
        - train_columns: list of lists. List at index i contains the column labels needed to impute column i.
    """
    df = df_in.copy()
    missing_columns = list(df.columns.values)
    missing_columns.remove(target)

    # for each column, get indexes related to missing values.
    missing_indexes = []
    full_indexes = []
    for c in missing_columns:
        missing_indexes.append(list(np.where(df[c].isnull())[0]))
        full_indexes.append(list(np.where(df[c].notnull())[0]))

    # create train columns
    train_columns = []
    for c in missing_columns:
        train_column = missing_columns.copy()
        train_column.remove(c)
        train_columns.append(train_column)
    
    return missing_columns, missing_indexes, full_indexes, train_columns



class ProgressBar:

    def __init__(self, end, width=15, step_size=1) -> None:
        """
        This class implements a dynamic progress bar.
        :param end: number of iteration of the process to represent.
        :param width: width of the diplayed progress bar. Default is 15.
        :param step_size: size of the steps at each iteration. By default it is set to 1.
        """
        self.step = 0
        self.end = end
        self.width = width
        self.step_size = step_size

    def reset(self):
        """
        reset the learner to the initial state.
        :return: None
        """
        self.__init__(self.end, self.width, self.step_size)

    def next(self):
        """
        print updated progress bar.
        :return: None
        """
        self.step += self.step_size
        percentage = self.step / self.end * 100
        n_completed = ceil(percentage / 100 * self.width)
        completed = "=" * n_completed
        to_complete = " " * (self.width - n_completed)
        sys.stdout.write("\rloading: [{}{}] {:0.1f}%".format(
            completed, to_complete, percentage))
        if self.step == self.end:
            print()


def pipeline_ML(df, target_name, seed, validation_function):
    """
    This function implements a full machine learning pipeline. It implements training, validation and evaluation of the model.
    :param df: pandas dataframe object containing all the necessary data for the pipeline. 
    :param target_name: name of the column related to the target of the classification ML task.
    :param seed: random seed for reproducibility.
    :param validation_function: function used to make model selection. The validation function signature should be: n_splits, X_train, y_train, seed.
    The validation function should perform a k-fold validation with n_splits for a learning algorithm defined in that function.
    :return: the best model, test accuracy, report containing f1-score, precision, recall for each target class a pandas dataframe containing the confusion matrix and
    weighted f1-score. 
    """
    # distinguish columns related to covariates and for targets.
    covariates_columns = list(df.columns.values)
    covariates_columns.remove(target_name)

    # splitting the dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        df[covariates_columns], df[target_name], test_size=0.3, random_state=seed, stratify=df[target_name])

    # model_selection
    best_model = validation_function(10, X_train, y_train, seed)
    test_accuracy, report, confusion_matrix_df, weighted_f1_score= evaluate_model(
        best_model, X_test, y_test)
    # evaluate best model on test data.
    return best_model, test_accuracy, report, confusion_matrix_df, weighted_f1_score


def model_selection_decision_tree(n_splits, X_train, y_train, seed):
    """
    Model selection function for a decision tree.
    :param n_splits: number of splits in the k-fold procedure.
    :param X_train: covariates used for training.
    :param y_train: target variables for training.
    :param seed: for reproducibility.
    :return : best decision tree found.
    """
    # hyperparameters to validate
    grid_values = range(2, 20)
    accuracies = []
    std_accuracies = []

    # model validation
    for v in grid_values:
        decision_tree = DecisionTreeClassifier(
            criterion='gini', min_samples_split=v)
        scores = cross_val_score(
            decision_tree, X_train, y_train, cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed))
        accuracies.append(np.mean(scores))
        std_accuracies.append(np.std(scores))

    # since the results present noise, we select the model with the highest pessimist validation accuracy.
    lower_bounds = np.array(accuracies) - np.array(std_accuracies)
    # model selection
    # hyperparameter with value x corresponds to model at index x + 2
    best_model = np.argmax(lower_bounds) + 2
    best_decision_tree = DecisionTreeClassifier(
        criterion='gini', min_samples_split=best_model)
    best_decision_tree = best_decision_tree.fit(X_train, y_train)
    return best_decision_tree


def model_selection_logistic_regression(n_splits, X_train, y_train, seed):
    # hyperparameters to validate
    C_values = [1e-2, .1, .2, .5, .8, 1, 2, 3, 5, 10, 20]
    accuracies = []
    std_accuracies = []

    # model validation
    for c in C_values:
        lr = LogisticRegression(random_state=seed, C=c, multi_class='multinomial', max_iter=200)
        scores = cross_val_score(
            lr, X_train, y_train, cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed))
        accuracies.append(np.mean(scores))
        std_accuracies.append(np.std(scores))

    # since the results present noise, we select the model with the highest pessimist validation accuracy.
    lower_bounds = np.array(accuracies) - np.array(std_accuracies)

    # model selection
    best_model_index = np.argmax(lower_bounds)
    best_c = C_values[best_model_index]
    best_lr = LogisticRegression(random_state=seed, C=best_c, multi_class='multinomial')
    best_lr = best_lr.fit(X_train, y_train)
    return best_lr



def model_selection_SVM(n_splits, X_train, y_train, seed):
    """
    Model selection function for a SVM with round basis function kernel.
    :param n_splits: number of splits in the k-fold procedure.
    :param X_train: covariates used for training.
    :param y_train: target variables for training.
    :param seed: for reproducibility.
    :return : best SVM found.
    """
    # hyperparameters to validate
    C_values = [1e-2, .1, .2, .5, .8, 1, 2, 3, 5, 10]
    accuracies = []
    std_accuracies = []

    # model validation
    for c in C_values:
        svm_mc = svm.SVC(
            kernel='rbf',
            max_iter=1000,
            random_state=seed,
            C=c
        )
        scores = cross_val_score(
            svm_mc, X_train, y_train, cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed))
        accuracies.append(np.mean(scores))
        std_accuracies.append(np.std(scores))

    # since the results present noise, we select the model with the highest pessimist validation accuracy.
    lower_bounds = np.array(accuracies) - np.array(std_accuracies)

    # model selection
    best_model_index = np.argmax(lower_bounds)
    best_c = C_values[best_model_index]
    best_svm = svm.SVC(
        kernel='rbf',
        max_iter=1000,
        random_state=seed,
        C=best_c
    )
    best_svm = best_svm.fit(X_train, y_train)
    return best_svm


def evaluate_model(model, X_test, y_test):
    """
    This function takes a model in input and evaluate it with overall accuracy, and for each class F1-score, recall and precision and weighted f1 score.
    :param model: trained model to evaluate.
    :param X_test: covariates for testing.
    :param y_test: labels for testing.
    :return : test accuracy, pandas dataframe containing f1-score, precision and recall metrics, pandas dataframe containing confusion matrix and weighted f1 score.
    """
    # compute accuracy
    test_accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    # compute f1-score, precision and recall for each target category
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)
    report = pd.DataFrame(report)
    report = report.drop(
        labels=['accuracy', 'macro avg', 'weighted avg'], axis=1)

    # compute heatmap for confusion matrix.
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, normalize="true"))

    # compute weighted macro f1-score
    weighted_f1_score = f1_score(y_test, y_pred, average='weighted')

    return test_accuracy, report, confusion_matrix_df, weighted_f1_score
