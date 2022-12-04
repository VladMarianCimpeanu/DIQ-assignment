import pandas as pd
import numpy as np

def accuracy_assesment(imputed_df_s: list, original_df, numeric_columns=[]) -> list:
    """
    This method compute the accuracy of the imputed dataframes with respect to the original dataframe.
    :param imputed_df_s: list of imputed pandas dataframes.
    :param original_df: pandas dataframe taken as reference when computing the accuracy.
    :param numeric_columns: list of column names that have a numeric value.
    :return: list of accuracies.
    """

    columns = original_df.columns
    
    accuracies = []
    
    tot_size = original_df.shape[0] * original_df.shape[1] 
    
    for i_df in imputed_df_s:
        distance_error = 0
        for c in columns:
            
            # defining distance function based on the type of variable.
            if c in numeric_columns:
                maximum_distance = original_df[c].max() - original_df[c].min()
                distance_function = lambda x, y: (np.abs(x - y) / maximum_distance) 
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