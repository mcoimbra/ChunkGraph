from dataclasses import dataclass, field
import os
from typing import Callable, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

# Update Matplotlib's global configuration
plt.rcParams.update({
    'figure.figsize': (10, 6),        # Default figure size
    'axes.titlesize': 'large',       # Title font size
    'axes.labelsize': 'medium',      # Axis label font size
    'xtick.labelsize': 'small',      # X-axis tick label size
    'ytick.labelsize': 'small',      # Y-axis tick label size
    'legend.fontsize': 'small',      # Legend font size
    'axes.grid': True,               # Enable grid by default
    'grid.alpha': 0.7,               # Grid transparency
    'grid.color': 'gray',            # Grid color
})

@dataclass
class PlotConfig:
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    bins: Optional[int] = None  # For histograms
    edgecolor: Optional[str] = field(default_factory=lambda: plt.rcParams.get('grid.color', 'black'))
    label: Optional[str] = None  # Legend label
    do_grid: bool = field(default_factory=lambda: plt.rcParams.get('axes.grid', True))
    figsize: Tuple[int, int] = field(default_factory=lambda: plt.rcParams.get('figure.figsize', (10, 6)))
    do_legend: bool = True  # Whether to display legend
    save_extensions: List[str] = field(default_factory=lambda: ['png', 'pdf'])
    dpi: int = 300
    function_type: Optional[str] = "plot"  # The type of plotting function ("plot", "hist", "bar")
    alpha: float = field(default_factory=lambda: plt.rcParams.get('grid.alpha', 1.0))
    width: Optional[float] = None  # Width parameter for bar plots
    plot_basename: str = "default_name"
    output_dir: str = os.getcwd()
    xlim: Optional[Tuple[float, float]] = field(default_factory=lambda: plt.rcParams.get('axes.xlim', (0, 1)))
    ylim: Optional[Tuple[float, float]] = field(default_factory=lambda: plt.rcParams.get('axes.ylim', (0, 1)))
    yaxis_major_locator_function: Optional[Callable] = field(default_factory=lambda: MultipleLocator(1))
    align: Optional[str] = "center"
    range: Tuple[int, int] = None
    xscale: Optional[str] = field(default_factory=lambda: plt.rcParams.get('xscale', 'linear'))

def apply_plot_config(
    x_values: List[float],
    y_values: Optional[List[float]],
    config: PlotConfig
) -> None:
    """
    Applies the plot configuration, calls the appropriate Matplotlib plotting function,
    and saves the plot in specified formats.

    Args:
        x_values (List[float]): The x-axis values for the plot.
        y_values (Optional[List[float]]): The y-axis values for the plot (not required for histograms).
        config (PlotConfig): Configuration for the plot.
    """
    # Create figure
    plt.figure(figsize=config.figsize)

    # Call the appropriate Matplotlib function
    if config.function_type == "plot":
        plt.plot(x_values, y_values, label=config.label, alpha=config.alpha)
    elif config.function_type == "bar":
        plt.bar(
            x=x_values,
            height=y_values,
            width=config.width if config.width else 0.8,  # Default width for bar plots
            edgecolor=config.edgecolor,
            color="blue",  # Default color; can be parameterized further
            alpha=config.alpha,
            label=config.label,
            align=config.align
        )
    elif config.function_type == "hist":
        bins = config.bins
        if isinstance(bins, (list, np.ndarray)):
            bins = np.array(bins)  # Ensure it’s a NumPy array
        elif bins is None or not isinstance(bins, int):
            bins = 10  # Default number of bins if invalid or None
        if config.range == None:
            plt.hist(
                x=x_values,
                #bins=config.bins if config.bins else 10,  # Default number of bins
                #bins=config.bins if isinstance(config.bins, (int, list, np.ndarray)) and config.bins else 10,  # Default number of bins
                bins=bins,

                edgecolor=config.edgecolor,
                alpha=config.alpha,
                label=config.label,
            )
        else:
            plt.hist(
                x=x_values,
                #bins=config.bins if config.bins else 10,  # Default number of bins
                #bins=config.bins if isinstance(config.bins, (int, list, np.ndarray)) and config.bins else 10,  # 
                bins=bins,
                edgecolor=config.edgecolor,
                alpha=config.alpha,
                label=config.label,
                range=config.range,
            )
    else:
        raise ValueError(f"Unsupported function_type: {config.function_type}")

    # Apply titles and labels
    if config.title:
        plt.title(config.title)
    if config.xlabel:
        plt.xlabel(config.xlabel)
    if config.ylabel:
        plt.ylabel(config.ylabel)
    if config.xscale:
        plt.xscale(config.xscale)

    # Configure grid
    if config.do_grid:
        plt.grid()

    # Set axis limits
    if config.xlim:
        plt.xlim(config.xlim)
    if config.ylim:
        plt.ylim(config.ylim)

    # Set Y-axis major locator
    if config.yaxis_major_locator_function:
        plt.gca().yaxis.set_major_locator(config.yaxis_major_locator_function)

    # Add legend if enabled
    if config.do_legend:
        plt.legend()

    # Save the plot in specified formats
    for current_fig_ext in config.save_extensions:
        fig_output_path: str = os.path.join(config.output_dir, f"{config.plot_basename}.{current_fig_ext}")
        if current_fig_ext != "pdf":
            plt.savefig(fig_output_path, dpi=config.dpi)
        else:
            plt.savefig(fig_output_path)

    # Close the plot
    plt.close()
