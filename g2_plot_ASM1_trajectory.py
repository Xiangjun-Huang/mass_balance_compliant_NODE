### Plot EASM1 component concentration trajectory, 12 components ###
### RMSE and R2 calculation are always compared to the first y ###
from typing import List, Optional, Union
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def plot_ASM1_trajectory(
    t: torch.Tensor,
    *ydata: torch.Tensor,
    y_types: List[str] = None,
    y_linestyles: List[str] = None,
    y_colors: List[str] = None,
    is_dy: bool = False,
    show_RMSE: bool = False,
    show_R2: bool = False,
    n_rows: int = 3,
    n_cols: int = 4,
    fig_size: tuple = (10, 6),
    t_unit_rate: float = 24,
    t_show_unit: str = "hour",
    t_ticks: List[float] = [0, 2, 4, 6],
    t_labels: List[str] = ["0", "2", "4", "6"],
    y_names: List[str] = None,
    y_units: List[str] = None,
    title: Optional[str] = None
):

    # Input validation
    assert len(ydata) > 0, "Must provide at least one y dataset"
    n_ydata = len(ydata)
    n_comp = ydata[0].shape[-1]
    assert n_rows * n_cols >= n_comp, "n_rows * n_cols must >= component count"

    # Auto-complete labels and styles
    y_types = _complete_types(y_types, n_ydata)
    y_linestyles = _complete_linestyles(y_linestyles, n_ydata)
    y_colors = _complete_colors(y_colors, n_ydata)

    # Default component names and units
    y_names = y_names or ['$S_S$', '$X_S$', '$X_{BH}$', '$X_{BA}$', '$X_P$', '$S_O$',
                        '$S_{NO}$', '$S_{NH}$', '$S_{ND}$', '$X_{ND}$', '$S_{ALK}$', '$S_{N_2}$']
    y_units = y_units or ['mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l',
                        '$mg O_2/l$', 'mg N/l', 'mg N/l', 'mg N/l', 'mg N/l', 'eq ALK/l', 'mg N/l']

    if is_dy:
        y_names = [f'{s[0]}d{s[1:]}' for s in y_names]
        y_units = [f'{u}/d' for u in y_units]

        # calculate overall RMSE and R2
    loss_mse = torch.nn.MSELoss(reduction='mean')   
    overall_RMSE_R2 = ""
    if show_RMSE:
        if n_ydata<2:
            show_RMSE = False
        else:
            for y_label, y_color, y in zip(y_types[1:], y_colors[1:], ydata[1:]):
                RMSE = torch.sqrt(loss_mse(ydata[0], y)).item()  # only compared to first y
                overall_RMSE_R2 += f"Overall $RMSE$ ({y_label} to {y_types[0]}) = {RMSE: .2f} | "

    if show_R2:
        if n_ydata<2:
            show_R2 = False
        else:
            TSS = torch.sum((ydata[0] - torch.mean(ydata[0]))**2).item()
            if overall_RMSE_R2 != "":
                overall_RMSE_R2 += f"\n"
            for y_label, y_color, y in zip(y_types[1:], y_colors[1:], ydata[1:]):
                R2 = 1 - (torch.sum((ydata[0] - y)**2).item() / TSS )
                overall_RMSE_R2 += f"Overall $R^2$ ({y_label} to {y_types[0]}) = {R2: .1%} | "

    # Create figure
    plt.style.use('seaborn-v0_8-paper')
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, constrained_layout=True)
    fig = plt.figure(figsize=fig_size)
    if title:
        fig.suptitle(title, y=1.02)

    for i in range(n_comp):
        ax = fig.add_subplot(n_rows, n_cols, i + 1) 

        # Plot each dataset
        for y_linestyle, y_color, y_type, y in zip(y_linestyles, y_colors, y_types, ydata):
            ax.plot(t * t_unit_rate, y[:, i], ls=y_linestyle, color=y_color, label=y_type, lw=1)

        # Configure axes
        ax.margins(x=0,y=0)
        ax.set_xlabel(t_show_unit, fontsize=7)
        ax.set_ylabel(y_units[i], fontsize=7) 
        ax.set_xlim(t[0]*t_unit_rate, t[-1]*t_unit_rate)
        ax.set_xticks(ticks=t_ticks)
        ax.set_xticklabels(labels=t_labels)
        ax.xaxis.set_label_coords(1, -0.15)
        formatter = ticker.ScalarFormatter(useOffset=False, useMathText=False) #useOffset=False to prevent offset notation
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(which='both', linewidth=0.2)
        ax.tick_params(which="both", direction="in")
        if i == 0:
            ax.legend()

        # Add metrics to subtitle
        y_name_i = y_names[i]
        if show_RMSE:
            for y_color, y in zip(y_colors[1:], ydata[1:]):
                y_name_i += f"|{torch.sqrt(loss_mse(ydata[0][:,i], y[:,i])):.0f}"
        if show_R2:
            TSS = torch.sum((ydata[0][:,i] - torch.mean(ydata[0][:,i]))**2) # Total Sum of Squares of first y
            for y_color, y in zip(y_colors[1:], ydata[1:]):
                y_name_i += f"|{1 - (torch.sum((ydata[0][:,i] - y[:,i])**2) / TSS):.0%}"          

        ax.set_title(y_name_i, pad=4)

    if show_RMSE or show_R2:
        bottom = 0.07
        fig.text(0.5, 0.02, overall_RMSE_R2, fontsize=8, ha="center")
    else:
        bottom = 0.03

    fig.tight_layout(rect=[0.03, bottom, 0.97, 0.97], pad=0.5, w_pad=0.5, h_pad=0.5)

    plt.show()
    return 

def _complete_types(types, n_needed):
    """Auto-complete labels with y_0, y_1..."""
    if not types:
        return [f'y_{i}' for i in range(n_needed)]
    return types + [f'y_{len(types)+i}' for i in range(n_needed - len(types))]

def _complete_linestyles(styles, n_needed):
    """Complete line styles with default cycle"""
    default = [':', '--', '-.']
    if not styles:
        styles = default
    return [styles[i % len(styles)] for i in range(n_needed)]

def _complete_colors(colors, n_needed):
    """Complete colors with matplotlib default cycle"""
    default = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if not colors:
        colors = default
    return [colors[i % len(colors)] for i in range(n_needed)]