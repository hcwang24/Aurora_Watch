import pandas as pd
from datetime import datetime

def preprocess_kp_index(df, start_date="2000-01-01"):
    """
    Preprocesses the 'Kp_Index' time series data by filtering based on a start date and adding 
    additional columns representing seasonal patterns (27-day Sun rotation, month of the year, 
    and 11-year solar cycle).

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing a 'Datetime' column (datetime format) and a 'Kp_Index' column.
    start_date : str, optional
        The starting date for filtering the data. Default is '2000-01-01'.

    Returns:
    -------
    pd.DataFrame
        A filtered and augmented DataFrame with columns for month, day in the 27-day Sun rotation,
        and 11-year solar cycle group.
    """
    # Convert start_date and Datetime column to datetime format
    start_date = pd.to_datetime(start_date)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Filter the data based on start_date
    filtered_df = df[df['Datetime'] >= start_date].copy().reset_index(drop=True)
    
    # Add the month column (1-12)
    filtered_df['month_of_year'] = filtered_df['Datetime'].dt.month
    
    # Add the day within the 27-day Sun rotation cycle (1-27)
    filtered_df['Day_Sun_rotation'] = ((filtered_df['Datetime'] - pd.to_datetime("2000-01-01")).dt.days % 27) + 1
    
    # Add the 11-year solar cycle year group (1-11)
    filtered_df['11_Year_Group'] = ((filtered_df['Datetime'].dt.year - 1755) % 11) + 1
    
    return filtered_df
