from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

def helloworld() -> str:
    print("Hello world")



def ols_fit(df: pd.DataFrame, 
            dep_col: str,
            indep_col: str = None) -> pd.DataFrame:
    '''Function ingests a dataframe, returns the OLS line of best fit as an additional column in the dataframe

    Args:
    df (pd.DataFrame): The DataFrame ingested  
    dep_col (str): The name of the column in the DataFrame holding the dependent variable data  
    indep_col (str) (Optional): The name of the column in the DataFrame holding the independent variable data

    Returns:
    DataFrame with the additional calculated column.
    '''

    if indep_col is not None:  
        # Check whether the column exists in the dataframe
        if indep_col not in df.columns:
            raise ValueError(f"Column '{indep_col}' does not exist in the DataFrame.")
    # Check if the column is numeric
        if not np.issubdtype(df[indep_col].dtype, np.number):
            raise TypeError(f"The column '{indep_col}' is not numeric. It must be numeric for OLS fit.")
        
        # If it's numeric, use it as the independent variable (x)
        x = df[indep_col].to_numpy()  # Convert the specified x column to a numpy array
    else:
        # If no indep_col is provided, use a default range of numbers as x
        x = np.arange(len(df))  # Default x as 0, 1, 2, ..., n-1

    y = df[dep_col].to_numpy()
    

    beta = np.cov(y, x)[0][-1] / np.var(x)
    alpha = np.mean(y) - beta*np.mean(x)

    df["OLS_Fit"] = alpha + x*beta

    return df

## Simple Moving Average
def sma(df: pd.DataFrame, col_name: str, length: int) -> pd.DataFrame:
    '''Function ingests a dataframe, returns simple moving average values as an additional column in the dataframe
    Args:
    df (pd.DataFrame): The DataFrame ingested  
    col_name (str): The name of the column in the DataFrame holding the data to calculate over
    length (int): The length of the moving average to calculate
    Returns:
    DataFrame with the additional calculated column.
    '''
    if col_name not in df.columns:  # Check whether the column exists in the dataframe
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
        
    if not np.issubdtype(df[col_name].dtype, np.number):  # Check if the column is numeric
        raise TypeError(f"The column '{col_name}' is not numeric. It must be numeric for SMA calculation.")
        
    df[f"sma_{length}"] = df[col_name].rolling(length).sum() / length   
    return df


## Bollinger Bands
def bollinger_bands(df: pd.DataFrame, col_name: str,
                    length: int = 20, stdev: int = 2) -> pd.DataFrame:
    '''Function ingests a dataframe, returns upper and lower bollinger bands with the SMA line
    Args:
    df (pd.DataFrame): The DataFrame ingested  
    col_name (str): The name of the column in the DataFrame holding the data to calculate over
    length (int): The length of the moving average to calculate
    stdev (int): The number of standard deviations used in the creation of the bands
    Returns:
    DataFrame with the additional calculated column.
    '''
    if col_name not in df.columns:  # Check whether the column exists in the dataframe
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
        
    if not np.issubdtype(df[col_name].dtype, np.number):  # Check if the column is numeric
        raise TypeError(f"The column '{col_name}' is not numeric. It must be numeric for SMA calculation.")
        
    df[f"sma_{length}"] = df[col_name].rolling(length).sum() / length 
    df[f"bb_{length}_{stdev}_min"] = df[f"sma_{length}"] - test_data[col_name].rolling(length).std() * stdev
    df[f"bb_{length}_{stdev}_max"] = df[f"sma_{length}"] + test_data[col_name].rolling(length).std() * stdev
     
    return df


## Bollinger Bands Percentage
def bollinger_bands_perc(df: pd.DataFrame, col_name: str,
                         length: int = 20, stdev: int = 2) -> pd.DataFrame:
    '''Function ingests a dataframe, returns bollinger band percentage values in a separate column in the dataframe
    Args:
    df (pd.DataFrame): The DataFrame ingested  
    col_name (str): The name of the column in the DataFrame holding the data to calculate over
    length (int): The length of the moving average to calculate
    stdev (int): The number of standard deviations used in the creation of the bands
    Returns:
    DataFrame with the additional calculated column. 
    '''
    if col_name not in df.columns:  # Check whether the column exists in the dataframe
        raise ValueError(f"Column '{col_name}' does not exist in the DataFrame.")
        
    if not np.issubdtype(df[col_name].dtype, np.number):  # Check if the column is numeric
        raise TypeError(f"The column '{col_name}' is not numeric. It must be numeric for SMA calculation.")
    
    # Upper and Lower bands
    bb_lower = (df[col_name].rolling(length).sum() / length) - test_data[col_name].rolling(length).std() * stdev
    bb_upper = (df[col_name].rolling(length).sum() / length) + test_data[col_name].rolling(length).std() * stdev

    # %b = (Current Price - Lower Band) / (Upper Band - Lower Band)
    df[f"bb_{length}_{stdev}_perc"] = (df[col_name] - bb_lower) / (bb_upper - bb_lower)
     
    return df




## Momentum?


## RSI


# Stochastic Indicator?













