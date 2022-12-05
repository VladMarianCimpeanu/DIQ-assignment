import pandas as pd
import numpy as np
import sys

from math import ceil
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix



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
    
    accuracies = []
    
    tot_size = original_df.shape[0] * original_df.shape[1] 
    
    for i_df in imputed_df_s:
        distance_error = 0
        for c in columns:
            
            # defining distance function based on the type of variable.
            if c in numeric_columns:
                maximum_distance = original_df[c].max() - original_df[c].min()
                distance_function = lambda x, y: (np.abs(x - y) / maximum_distance)
            elif c in vector_columns:
                # here we assume vectors are normalized.
                distance_function = lambda x, y: np.abs(1 - (np.dot(np.array(x), np.array(y))))
            else:
                distance_function = lambda x, y: 1 if x != y else 0
            
            # retriving values for a specific column
            imputed_column = pd.Series(i_df[c]).values
            original_column = pd.Series(original_df[c]).values
            
            # compare columns
            for i, o in zip(imputed_column, original_column):
                distance_error += distance_function(i, o)
                
       
        accuracy = (tot_size - distance_error) / tot_size
        accuracies.append(accuracy)

    return accuracies


def iterative_imputation_KNN(df_in, target, neighbours=3, n_iter=10):
    """
    This method uses KNN to iteratively impute Nan values.
    :param df_in: pandas dataframe to impute.
    :param target: target column. This column should not be considered during imputation.
    :param neighbours: number of neighbours for KNN.
    :param n_iter: number of times the dataset must be scanned.
    :return: a new pandas dataframe imputed.
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

    # start with basic imputation
    simple_imputer = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
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
            train_y = df[c].iloc[f]
            X = df[l].copy()
            X = pd.get_dummies(X)
            train_X = X.iloc[f]
            imputed_X = X.iloc[m]
            
            knn = KNeighborsClassifier(n_neighbors=neighbours)
            knn.fit(train_X, train_y)
            
            imputed_y = knn.predict(imputed_X)
            df[c].iloc[m] = imputed_y
    progressbar.reset()
    return df


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
        sys.stdout.write("\rloading: [{}{}] {:0.1f}%".format(completed, to_complete, percentage))
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
    :return: the best model, test accuracy, report containing f1-score, precision, recall for each target class and a pandas dataframe containing the confusion matrix.
    """
    # distinguish columns related to covariates and for targets.
    covariates_columns = list(df.columns.values)
    covariates_columns.remove(target_name)

    # splitting the dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        df[covariates_columns], df[target_name], test_size=0.3, random_state=seed, stratify=df[target_name])
    
    # model_selection
    best_model = validation_function(10, X_train, y_train, seed)
    test_accuracy, report, confusion_matrix_df = evaluate_model(best_model, X_test, y_test)
    # evaluate best model on test data.
    return best_model, test_accuracy, report, confusion_matrix_df
    
    
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

    # model validation
    for v in grid_values:
        decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=v)
        scores = cross_val_score(
            decision_tree, X_train, y_train, cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed))
        accuracies.append(np.mean(scores))
    
    # model selection
    best_model = np.argmax(accuracies) + 2 # hyperparameter with value x corresponds to model at index x - 2 
    best_decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split= best_model)
    best_decision_tree = best_decision_tree.fit(X_train, y_train)
    return best_decision_tree
    
    
def evaluate_model(model, X_test, y_test):
    """
    This function takes a model in input and evaluate it with overall accuracy, and for each class F1-score, recall and precision.
    :param model: trained model to evaluate.
    :param X_test: covariates for testing.
    :param y_test: labels for testing.
    :return : test accuracy, pandas dataframe containing f1-score, precision and recall metrics, pandas dataframe containing confusion matrix.
    """
    # compute accuracy
    test_accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    # compute f1-score, precision and recall for each target category
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report)
    report = report.drop(labels=['accuracy', 'macro avg', 'weighted avg'], axis=1)
    
    # compute heatmap for confusion matrix.
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize="true"))
    return test_accuracy, report, confusion_matrix_df