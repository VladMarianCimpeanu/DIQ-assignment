import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.neighbors import KNeighborsClassifier


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

                def distance_function(x, y): return (
                    np.abs(x - y) / maximum_distance)
            elif c in vector_columns:
                # here we assume vectors are normalized.
                def distance_function(x, y): return np.abs(
                    1 - (np.dot(np.array(x), np.array(y))))
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

    return accuracies


def iterative_imputation_KNN(df_in, target, neighbours=3, n_iter=10):
    """
    This method uses KNN to iteratively impute NaN values.
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
    simple_imputer = SimpleImputer(
        missing_values=np.NaN, strategy='most_frequent')
    df = simple_imputer.fit_transform(df)
    df = pd.DataFrame(df, columns=df_in.columns)

    for i in range(n_iter):
        # print(f'iteration: {i}')
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

    return df
