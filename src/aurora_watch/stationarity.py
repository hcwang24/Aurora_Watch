import pandas as pd
import matplotlib.pyplot as plt

def plot_stationarity(df: pd.DataFrame, column: str, wd_size: int = 14) -> None:
    """
    Plots the original time series, its rolling mean, and rolling standard deviation.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing the time series data.
    column : str
        The column name of the time series to be analyzed.
    wd_size : int, optional
        The window size for calculating rolling statistics. Default is 14.
    """
    
    # Create subplots for the original series, rolling mean, and rolling std
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Plot original time series
    df[column].plot(ax=axes[0])
    axes[0].set_title(f'{column}')

    # Plot rolling mean
    df[column].rolling(wd_size, min_periods=5).mean().plot(ax=axes[1])
    axes[1].set_title('Rolling Mean')

    # Plot rolling std
    df[column].rolling(wd_size, min_periods=5).std().plot(ax=axes[2])
    axes[2].set_title('Rolling Standard Deviation')

    plt.tight_layout()
    plt.show()

from statsmodels.tsa.stattools import adfuller

def check_stationarity(df: pd.DataFrame, original_column: str, detrended_column:str) -> pd.DataFrame:
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing the time series data.
    original_column : str
        The column name of the original time series data.
    detrended_column : str
        The column name of the detrended time series data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing ADF statistics and p-values for the original and detrended series.
    """
    
    # Perform ADF test for original series
    result_original = adfuller(df[original_column])

    # Perform ADF test for detrended residuals
    result_detrended = adfuller(df[detrended_column])

    # Create a DataFrame with ADF statistic and p-values
    adf_results = pd.DataFrame({
        'Test': ['Original', 'Residual Detrended'],
        'ADF Statistic': [f'{result_original[0]:.2e}', f'{result_detrended[0]:.2e}'],
        'p-value': [f'{result_original[1]:.2e}', f'{result_detrended[1]:.2e}']
    })

    return adf_results
