import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from scipy.sparse import csc_matrix
import pandas as pd
from typing import List


import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Custom Dataset class for loading paired data (features and labels) for PyTorch models.

    Args:
        x (Tensor or ndarray): The input features, typically of shape (num_samples, num_features).
        y (Tensor or ndarray): The corresponding labels, typically of shape (num_samples,).

    """
    def __init__(self, x, y):
        """
        Initializes the dataset with features and labels.

        Parameters:
            x (Tensor or ndarray): The input features of the dataset.
            y (Tensor or ndarray): The labels corresponding to each sample in the dataset.
        """
        self.x = x  # Store the input features
        self.y = y  # Store the labels

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at a specified index.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input features and corresponding label at the given index.
                   (features, label).
        """
        features = self.x[index]  # Get the input features for the sample
        label = self.y[index]     # Get the label for the sample
        return features, label

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.x)  # Return the number of samples in the dataset (assumed equal to len(y))



def read_rds(rds_file):
    """
    read rds file as DataFrame

    """
    # Activate the automatic pandas conversion (useful if you convert to DataFrame)
    pandas2ri.activate()

    # Load the RDS file in R
    r_matrix = r['readRDS'](rds_file)  # Replace with your file path

    # Extract row and column names using R functions
    row_names = list(r['rownames'](r_matrix))
    col_names = list(r['colnames'](r_matrix))

    # If the object is a sparse matrix, we can convert it to a Scipy sparse matrix
    if r_matrix.rclass[0] == "dgCMatrix":
        # Extract the sparse matrix components
        i = r_matrix.slots['i']
        p = r_matrix.slots['p']
        x = r_matrix.slots['x']
        shape = tuple(r_matrix.slots['Dim'])

        # Convert to Scipy csc_matrix
        sparse_matrix_py = csc_matrix((x, i, p), shape=shape)


        dense_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix_py,index=row_names, columns=col_names).T    #cell as row, gene as col
        return dense_df

    else:
        print("The RDS file does not contain a dgCMatrix object.")



def split_data_by_cmo_label(data):
    """
    Extracts CMO labels from the index, splits the data into training and test sets based on the presence 
    of CMO labels, and returns the summary of CMO class distributions.

    Args:
        data (pd.DataFrame): The input data frame with an index containing CMO identifiers.

    Returns:
        tuple: A tuple containing:
            - train_data (pd.DataFrame): Data frame with rows having valid CMO labels.
            - test_data (pd.DataFrame): Data frame with rows lacking CMO labels.
            - class_summary (pd.Series): Summary of CMO class distribution.
    """
    
    # Extract the prefix (e.g., 'CMO301') from each index entry
    data['label'] = data.index.str.extract(r'^(CMO\d+)', expand=False).fillna('No label')
    
    # Count the occurrences of each unique "CMO" class
    class_summary = data['label'].value_counts()
    
    # Display the summary
    print("Summary of CMO classes:")
    print(class_summary)
    
    # Split the data into training and test sets
    train_data = data[data["label"] != "No label"]
    test_data = data[data["label"] == "No label"]
    
    return train_data, test_data, class_summary




def select_top_features_by_non_zero_count(data, top_n=6000):
    """
    Selects the top N features (columns) from the dataset based on the number of non-zero entries 
    in each column. The function counts non-zero elements in each column and selects the top N columns 
    with the highest counts.

    Args:
        data (pd.DataFrame): The input data frame with features as columns.
        top_n (int): The number of top features to select based on non-zero counts. Default is 2000.

    Returns:
        pd.DataFrame: A new DataFrame containing only the top N features (columns).
    """
    
    # Count non-zero elements in each column
    non_zero_counts = (data != 0).sum(axis=0)
    
    # Select the top N columns with the highest non-zero counts
    top_n_cols = non_zero_counts.nlargest(top_n).index
    
    # Filter the original dataframe to keep only the top N columns
    data_top_n = data[top_n_cols]
    
    # Display summary of the selected columns
    print(f"Selected top {top_n} columns based on non-zero counts:\n{data_top_n.head()}")  # Show a preview of the selected data
    
    return data_top_n, top_n_cols





def accuracy_score(y_test: List[int], y_pred: List[int]) -> float:
    """
    Calculate the accuracy of predictions.

    This function compares the predicted labels with the true labels and calculates 
    the proportion of correct predictions (accuracy).

    Parameters:
    y_test (List[int]): The true labels for the test dataset.
    y_pred (List[int]): The predicted labels.

    Returns:
    float: The accuracy score, a value between 0 and 1, representing the proportion 
           of correct predictions out of the total predictions.
    """
    # Initialize counters for true and false predictions
    correct_predictions = 0
    incorrect_predictions = 0

    # Loop through each pair of true and predicted labels
    for true, pred in zip(y_test, y_pred):
        if true == pred:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
    
    # Return the accuracy as the ratio of correct predictions to total predictions
    total_predictions = correct_predictions + incorrect_predictions
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    