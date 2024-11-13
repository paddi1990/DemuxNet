import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from scipy.sparse import csc_matrix
import pandas as pd
from typing import List



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
    