import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from warnings import simplefilter
import seaborn as sns
from matplotlib.ticker import MaxNLocator


# A function for plotting the Kp index over a specified date range. 
def plot_kp_index_by_date_range(kpindex_data, start_date=None, end_date=None):
    """
    Function to filter Kp index data by a specific date range, plot the data, and highlight the highest point.
    
    Parameters:
    kpindex_data (pd.DataFrame): The full Kp index dataset containing 'Datetime' and 'Kp_Index'.
    start_date (str, optional): The start date (inclusive) in the format 'YYYY-MM-DD'. Defaults to None, showing all data.
    end_date (str, optional): The end date (exclusive) in the format 'YYYY-MM-DD'. Defaults to None, showing all data.
    
    Returns:
    None
    """

    # Ensure 'Datetime' is in datetime format
    kpindex_data['Datetime'] = pd.to_datetime(kpindex_data['Datetime'])
    
    # Filter the data for the given date range
    if start_date and end_date:
        filtered_data = kpindex_data[(kpindex_data['Datetime'] >= start_date) & 
                                     (kpindex_data['Datetime'] < end_date)]
    else:
        filtered_data = kpindex_data

    # Check if data exists in the date range
    if filtered_data.empty:
        print("No data available for the specified date range.")
        return
    
    # Find the highest Kp index value and its corresponding time
    peak_kp_value = filtered_data['Kp_Index'].max()
    peak_kp_time = filtered_data[filtered_data['Kp_Index'] == peak_kp_value]['Datetime'].values[0]
    
    # Plot the filtered data
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['Datetime'], filtered_data['Kp_Index'], label="Kp Index", color = "blue")

    # Highlight the peak
    plt.scatter(peak_kp_time, peak_kp_value, color='blue', zorder=5, label=f'Peak Kp Index ({pd.to_datetime(peak_kp_time).strftime("%Y-%m-%d")} UTC)')

    # Annotate the peak value
    plt.text(peak_kp_time, peak_kp_value + 0.2, f"Peak: {peak_kp_value:.1f}", color='blue')

    # Labels and title
    if start_date and end_date:
        title = f'Kp Index from {start_date} to {end_date}'
    else:
        title = 'Kp Index (for All Data)'
    
    plt.xlabel('Date')
    plt.ylabel('Kp Index')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_prediction_intervals(
    y,
    p,
    col="mean",
    coverage="95%",
    valid=None,
    xlabel=None,
    ylabel=None,
    width=700,
    height=400,
):
    """
    y = series
    p = prediction dataframe from statsmodels .summary_frame() method
    """
    if xlabel is None:
        xlabel = y.index.name

    if ylabel is None:
        ylabel = y.name

    if "pi_lower" not in p.columns:
        pilabel = "mean_ci"
    else:
        pilabel = "pi"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name="Observed"))
    if valid is not None:
        fig.add_trace(
            go.Scatter(
                x=valid.index,
                y=valid,
                mode="lines",
                line=dict(color="#00CC96"),
                name="Validation",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=p.index,
            y=p[col],
            mode="lines",
            line=dict(color="salmon"),
            name=f"{col.title()} predicted",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p.index,
            y=p[f"{pilabel}_lower"],
            mode="lines",
            line=dict(width=0.5, color="salmon"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p.index,
            y=p[f"{pilabel}_upper"],
            mode="lines",
            line=dict(width=0.5, color="salmon"),
            fillcolor="rgba(250, 128, 114, 0.2)",
            fill="tonexty",
            name=f"{coverage} confidence interval",
        )
    )
    fig.update_xaxes(title=xlabel, title_standoff=0)
    fig.update_yaxes(title=ylabel, title_standoff=0)
    fig.update_layout(
        width=width,
        height=height,
        title_x=0.5,
        title_y=0.93,
        margin=dict(t=60, b=10),
    )
    return fig

def plot_windows(y, train_windows, test_windows, title=""):
    """Visualize training and test windows"""

    simplefilter("ignore", category=UserWarning)

    def get_y(length, split):
        # Create a constant vector based on the split for y-axis."""
        return np.ones(length) * split

    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    train_color, test_color = sns.color_palette("colorblind")[:2]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(
            np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
        )
        ax.plot(
            train,
            get_y(len(train), i),
            marker="o",
            c=train_color,
            label="Train",
        )
        ax.plot(
            test,
            get_y(len_test, i),
            marker="o",
            c=test_color,
            label="Validation",
        )
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
    )
    # remove duplicate labels/handles
    handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
    ax.legend(handles, labels);
    
def get_windows(y, cv):
    """Generate windows"""
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows

