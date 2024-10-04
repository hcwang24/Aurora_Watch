import pandas as pd
import numpy as np
from datetime import datetime


def load_and_clean_kpindex(start_time="1932-01-01", end_time=datetime.today().strftime("%Y-%m-%d"), index="Kp"):
    """
    Downloads and cleans the Kp index data for the given date range.

    Args:
    - start_time: str, the start date for data retrieval in 'YYYY-MM-DD' format
    - end_time: str, the end date for data retrieval in 'YYYY-MM-DD' format

    Returns:
    - kpindex_df: DataFrame, cleaned daily Kp Index data
    """
    from getKpindex import getKpindex

    # Download data
    kpindex_data = getKpindex(starttime=start_time, endtime=end_time, index=index)

    # Since kpindex_data is a transposed 2D list or array, transpose it back to the correct format.
    kpindex_data_transposed = list(zip(*kpindex_data))

    # Convert to a DataFrame
    kpindex_df = pd.DataFrame(
        kpindex_data_transposed, columns=["Datetime", "Kp_Index", "Status"]
    )

    # Convert the 'Datetime' column to datetime format
    kpindex_df["Datetime"] = pd.to_datetime(kpindex_df["Datetime"])

    # Basic cleaning - handle missing values
    kpindex_df = kpindex_df.dropna()

    # Create a new column for the date only
    kpindex_df["Date"] = kpindex_df["Datetime"].dt.date

    # Group by the 'Date' column and calculate the daily max and std of Kp index
    daily_max_df = (
        kpindex_df.groupby("Date").agg(Daily_Kp_max=("Kp_Index", "max")).reset_index()
    )

    daily_max_df = pd.DataFrame(daily_max_df)

    daily_max_df = daily_max_df.rename(
        columns={"Date": "Datetime", "Daily_Kp_max": "Kp_Index"}
    )

    return daily_max_df
