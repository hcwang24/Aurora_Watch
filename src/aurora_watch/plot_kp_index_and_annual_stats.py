import pandas as pd
import matplotlib.pyplot as plt

def plot_kp_index_and_annual_stats(kpindex_data, start_date=None, end_date=None, index_column="Datetime", annual_min_max=False, y_lim=None):
    """
    Function to plot Kp index data, either for a specified date range or with annual statistics (min, max, mean).
    
    Parameters:
    kpindex_data (pd.DataFrame): DataFrame containing 'Datetime' and 'Kp_Index' columns.
    start_date (str, optional): The start date (inclusive) in the format 'YYYY-MM-DD'. Defaults to None for all data.
    end_date (str, optional): The end date (exclusive) in the format 'YYYY-MM-DD'. Defaults to None for all data.
    index_column (str, optional): Column name for the datetime information. Defaults to 'Datetime'.
    annual_min_max (bool, optional): Whether to display annual statistics (min, max, mean). Defaults to False for plotting raw Kp Index data.
    
    Returns:
    fig (matplotlib.figure.Figure): The figure object of the plot.
    """

    # Ensure the datetime column is in the correct format
    kpindex_data[index_column] = pd.to_datetime(kpindex_data[index_column])
    
    # Filter the data based on the given date range
    if start_date and end_date:
        filtered_data = kpindex_data[(kpindex_data[index_column] >= start_date) & 
                                     (kpindex_data[index_column] <= end_date)]
    else:
        filtered_data = kpindex_data

    # Check if data exists after filtering
    if filtered_data.empty:
        print("No data available for the specified date range.")
        return None
    
    # Create the figure object
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not annual_min_max:
        # Identify the peak Kp index value and corresponding time
        peak_kp_value = filtered_data['Kp_Index'].max()
        peak_kp_time = filtered_data[filtered_data['Kp_Index'] == peak_kp_value][index_column].values[0]

        # Plot the Kp Index for the given date range
        ax.plot(filtered_data[index_column], filtered_data['Kp_Index'], label="Kp Index", color="blue")
        
        # Highlight the peak point
        ax.scatter(peak_kp_time, peak_kp_value, color='blue', zorder=5, 
                    label=f'Peak Kp Index ({pd.to_datetime(peak_kp_time).strftime("%Y-%m-%d")} UTC)')
        
        # Annotate the peak Kp Index value
        ax.text(peak_kp_time, peak_kp_value + 0.2, f"Peak: {peak_kp_value:.1f}", color='blue')

        # Set title for the plot
        title = f'Kp Index from {start_date} to {end_date}' if start_date and end_date else 'Kp Index (All Data)'

    else:
        # Compute annual statistics: max, min, and mean Kp index values
        filtered_data['Year'] = filtered_data[index_column].dt.year
        agg_filtered_data = filtered_data.groupby('Year').agg(
            Kp_max=('Kp_Index', 'max'),
            Kp_min=('Kp_Index', 'min'),
            Kp_mean=('Kp_Index', 'mean')
        ).reset_index()

        # Plot mean Kp Index over the years
        ax.plot(agg_filtered_data['Year'], agg_filtered_data['Kp_mean'], label="Kp Mean", color='blue')
        
        # Fill the area between the min and max Kp Index values
        ax.fill_between(agg_filtered_data['Year'], agg_filtered_data['Kp_min'], 
                         agg_filtered_data['Kp_max'], color='lightgray', alpha=1, label="Min-Max Range")
        
        # Set title for the annual statistics plot
        title = "Annual Peak, Mean, and Min Kp Index"

    # Common plot settings
    ax.set_xlabel('Date' if not annual_min_max else 'Year')
    ax.set_ylabel('Kp Index')
    ax.set_title(title)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    
    # Close the figure to prevent it from displaying immediately
    plt.close(fig)

    # Return the figure object instead of showing the plot
    return fig
