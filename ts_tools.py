import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf


def sim_arma(ar_coefs=None, ma_coefs=None, scale=1, n=100, random_state=42):
    """
    Simulate of an ARMA process.

    Parameters:
        ar_coefs (list, optional): Coefficients for the AR part of the process. Default is None.
        ma_coefs (list, optional): Coefficients for the MA part of the process. Default is None.
        scale (float, optional): Standard deviation of the random noise. Default is 1.
        n (int, optional): Number of time steps to simulate. Default is 100.
        random_state (int, optional): used to set seed for reproducibility
    """    
    np.random.seed(random_state)
    # Default coefficients for white noise if no AR or MA coefficients are provided
    if ar_coefs is None:
        ar_coefs = [1]  # AR(0)
    if ma_coefs is None:
        ma_coefs = [1]  # MA(0)

    # Create the ARMA process using statsmodels package
    arma_process = ArmaProcess(ar_coefs, ma_coefs)    
    ts = arma_process.generate_sample(nsample=n, scale=scale)    
    
    return ts

def plot_ts(*args, title="", xlabel="Time Step", ylabel="Value", save_path=None):
    """
    Plot a time series.

    Parameters:
        args (tuple): Can be (y,) or (x, y).
                      y (np.array): Time series to plot.
                      x (np.array, optional): Used to index plotted y. Default is None.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis. Default is "Time Step".
        ylabel (str, optional): Label for the y-axis. Default is "Value".
        save_path (str, optional): File path with name and extension to save the plot. Default is None.
    """
    if len(args) == 1:
        y = args[0]
        x = None
    elif len(args) == 2:
        x, y = args
    else:
        raise ValueError("Invalid arguments: pass either (y,) or (x, y)")

    # Plot the data
    plt.figure(figsize=(10, 5))
    if x is not None:
        plt.plot(x, y, linestyle='-', label=title)
    else:
        plt.plot(y, linestyle='-', label=title)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()



def plt_acf(ts, max_k=54, title=None, vlines=None, save_path=None):
    """
    Wrapper for ACF plot using bars instead of the default statsmodels' dots and whiskers.

    Parameters:
        ts (np.array or list): The time series to analyze.
        max_k (int, optional): Maximum number of lags to display. Default is 54.
        title (str, optional): Title of the plot. Default is None.
        vlines (list, optional): List of lag values where vertical lines should be drawn. Default is None.
        save_path (str, optional): Path to save the plot, including file name and extension. Default is None.

    Returns:
        None
    """
    
    # Calculate ACF values
    acf_values = acf(ts, nlags=max_k, fft=True)
    lags = np.arange(len(acf_values))

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot ACF as bars
    default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    ax.bar(lags, acf_values, width=0.6, color=default_blue, alpha=0.7, label='ACF')
    
    # Add horizontal lines for confidence intervals
    conf_int = 1.96 / np.sqrt(len(ts))
    ax.axhline(y=conf_int, linestyle='--', color='gray', linewidth=1, label="95% CI")
    ax.axhline(y=-conf_int, linestyle='--', color='gray', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Set the title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Add vertical lines if specified
    if vlines:
        for lag in vlines:
            ax.axvline(x=lag, color=default_blue, linestyle='--', linewidth=1)
            ax.text(
                x=lag, y=ax.get_ylim()[1] * 0.9, 
                s=f'Lag {lag}', color=default_blue, rotation=90,
                va='top', ha='center', fontsize=10, fontweight='bold'
            )

    # Customize grid and axis
    ax.grid(alpha=0.3)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend()

    # Save the plot if a path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()


def plt_pacf(ts, max_k=54, title=None, vlines=None, save_path=None):
    """
    Wrapper for PACF plot using bars instead of the default statsmodels' dots and whiskers.

    Parameters:
        ts (np.array or list): The time series to analyze.
        max_k (int, optional): Maximum number of lags to display. Default is 54.
        title (str, optional): Title of the plot. Default is None.
        vlines (list, optional): List of lag values where vertical lines should be drawn. Default is None.
        save_path (str, optional): Path to save the plot, including file name and extension. Default is None.

    Returns:
        None
    """
    
    # Calculate PACF values
    pacf_values = pacf(ts, nlags=max_k)
    lags = np.arange(len(pacf_values))

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot PACF as bars
    default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    ax.bar(lags, pacf_values, width=0.6, color=default_blue, alpha=0.7, label='PACF')
    
    # Add horizontal lines for confidence intervals
    conf_int = 1.96 / np.sqrt(len(ts))
    ax.axhline(y=conf_int, linestyle='--', color='gray', linewidth=1, label="95% CI")
    ax.axhline(y=-conf_int, linestyle='--', color='gray', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Set the title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Add vertical lines if specified
    if vlines:
        for lag in vlines:
            ax.axvline(x=lag, color=default_blue, linestyle='--', linewidth=1)
            ax.text(
                x=lag, y=ax.get_ylim()[1] * 0.9, 
                s=f'Lag {lag}', color=default_blue, rotation=90,
                va='top', ha='center', fontsize=10, fontweight='bold'
            )

    # Customize grid and axis
    ax.grid(alpha=0.3)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("Partial Autocorrelation", fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend()

    # Save the plot if a path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()







    