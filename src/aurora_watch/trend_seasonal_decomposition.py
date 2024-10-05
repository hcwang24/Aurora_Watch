import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

def trend_seasonal_decomposition(df: pd.DataFrame, start_date: str = '2000-01-01') -> pd.DataFrame:
    """
    Decomposes the 'Kp_Index' time series data into trend and seasonal components for different periods 
    (27-Day Sun Rotation, Month of Year, and 11-Year Solar Cycle).

    Parameters:
    -----------
    df : pd.DataFrame
        A DataFrame containing a 'Datetime' column and a 'Kp_Index' column with time series data.
    start_date : str, optional
        The starting date from which to filter the data. Default is '2000-01-01'.

    Returns:
    --------
    pd.DataFrame
        The filtered and decomposed DataFrame with additional columns for residuals and seasonal components.
    """
    # Filter the data for entries after the start_date
    filtered_df = df[df['Datetime'] >= start_date].reset_index(drop=True)

    # Step 1: Decompose the 27-Day Sun Rotation Trend and Seasonality
    stl_27 = STL(filtered_df['Kp_Index'], period=27, seasonal=27).fit()
    filtered_df['Residual_27'] = filtered_df['Kp_Index'] - stl_27.trend
    filtered_df['Residual_detrended_27'] = filtered_df['Residual_27'] - stl_27.seasonal
    filtered_df['Trend_27'] = stl_27.trend
    filtered_df['Seasonal_27'] = stl_27.seasonal

    # Step 2: Decompose the Month of Year Seasonality (approximately 365.25 days per year)
    days_per_year = int(365.25)
    days_per_month = int(days_per_year / 12) + 1  # Make this 31 as the seasonal parameter requires an odd number
    stl_monthly = STL(filtered_df['Residual_detrended_27'], period=days_per_year, seasonal=days_per_month).fit()
    filtered_df['Residual_detrended_27_MoY'] = filtered_df['Residual_detrended_27'] - stl_monthly.seasonal
    filtered_df['Seasonal_MonthofYear'] = stl_monthly.seasonal

    # Step 3: Decompose the 11-Year Solar Cycle Seasonality
    solar_cycle_period = int(11 * days_per_year)  # ~4015 days
    stl_11year = STL(filtered_df['Residual_detrended_27_MoY'], period=solar_cycle_period, seasonal=days_per_year).fit()
    filtered_df['Detrended_Deseasonalized_Kp_Index'] = filtered_df['Residual_detrended_27_MoY'] - stl_11year.seasonal
    filtered_df['Seasonal_SolarCycle'] = stl_11year.seasonal

    return filtered_df

def plot_decomposition(filtered_df: pd.DataFrame, lags: int = 100) -> plt.Figure:
    """
    Plots the ACF and Residuals for each decomposition step.

    Parameters:
    -----------
    filtered_df : pd.DataFrame
        A DataFrame containing the decomposed Kp_Index data.
    lags : int, optional
        The number of lags to display in the ACF plots. Default is 100.

    Returns:
    --------
    plt.Figure
        The matplotlib Figure object with the ACF and residual plots.
    """
    # Plot ACF and Residuals side by side
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    
    # Original Data
    plot_acf(filtered_df['Kp_Index'], ax=axes[0, 0], lags=lags)
    axes[0, 0].set_title('Original Kp Index (ACF)')
    filtered_df['Kp_Index'].plot.line(ylabel="Kp Index", title="Original Kp Index", ax=axes[0, 1])

    # Step 1: ACF and Residual after removing 27-Day Sun Rotation Seasonality
    plot_acf(filtered_df['Residual_27'], ax=axes[1, 0], lags=lags)
    axes[1, 0].set_title('ACF after Removing 27-Day Trend')
    filtered_df['Residual_27'].plot.line(ylabel="Kp Index", title="Residual", ax=axes[1, 1])

    plot_acf(filtered_df['Residual_detrended_27'], ax=axes[2, 0], lags=lags)
    axes[2, 0].set_title('ACF of Residuals after Removing Trend and 27-Day Seasonality')
    filtered_df['Residual_detrended_27'].plot.line(ylabel="Kp Index", title="Residual", ax=axes[2, 1])

    # Step 2: ACF and Residual after removing 27-Day + Month of Year Seasonality
    plot_acf(filtered_df['Residual_detrended_27_MoY'], ax=axes[3, 0], lags=lags)
    axes[3, 0].set_title('ACF after Removing Trend + 27-Day + Month of Year Seasonality')
    filtered_df['Residual_detrended_27_MoY'].plot.line(ylabel="Kp Index", title="Residual", ax=axes[3, 1])

    # Step 3: ACF and Residual after removing 27-Day, Month of Year, and 11-Year Solar Cycle Seasonality
    plot_acf(filtered_df['Detrended_Deseasonalized_Kp_Index'], ax=axes[4, 0], lags=lags)
    axes[4, 0].set_title('ACF after Removing Trend + 27-Day + Month of Year + 11-Year Seasonality')
    filtered_df['Detrended_Deseasonalized_Kp_Index'].plot.line(ylabel="Kp Index", title="Residual", ax=axes[4, 1])

    return fig
