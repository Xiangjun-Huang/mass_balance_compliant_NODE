from typing import List, Tuple, Optional, Sequence
import matplotlib.pyplot as plt

def plot_lr_loss(
    lr_loss_pairs: Sequence[Tuple[List[float], List[float]]], 
    loss_labels: Optional[List[str]] = None,
    fig_size: Tuple[float, float] = (6, 4),
    label_size: int = 9,
    line_width: float = 1,
    line_styles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title_txt: str = "",
    legend_loc: str = "center",
    loss_yscale: str = "log",
    lr_yscale: str = "log",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    explicitely enforce (lr, loss) input in pair in tuple
    
    example:
    lr_loss_pairs = [
        ([0.1, 0.09, 0.08], [3.2, 2.9, 2.5]),  # first tuple (lr, loss)
        ([0.05, 0.04, 0.03], [4.1, 3.8, 3.6])   # second
    ]
    """
    # input validation
    if not all(len(pair) == 2 for pair in lr_loss_pairs):
        raise ValueError("every input must be (lr, loss) in pair")
    
    for lr, loss in lr_loss_pairs:
        if len(lr) != len(loss):
            raise ValueError(f"The length of LR does not match loss: LR has {len(lr)} point, loss has {len(loss)} point")

    n_groups = len(lr_loss_pairs)
    
    # default values
    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    line_styles = line_styles or ['-', '--', '-.', ':']
    loss_labels = loss_labels or [f'Model {i+1}' for i in range(n_groups)]
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax1 = plt.subplots(figsize=fig_size)
    ax2 = ax1.twinx()
    all_lines = []
    
    for idx, (lr_data, loss_data) in enumerate(lr_loss_pairs):
        color = colors[idx % len(colors)]
        
        # plot loss at left y axis
        loss_line = ax1.plot(
            loss_data,
            color=color,
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=line_width,
            label=f'{loss_labels[idx]} Loss'
        )[0]
        
        # plot lr at right y axis
        lr_line = ax2.plot(
            lr_data,
            color=color,
            linestyle=':',  # LR dot line
            alpha=0.5,
            label=f'{loss_labels[idx]} LR'
        )[0]
        
        all_lines.extend([loss_line, lr_line])
    
    # axis settings
    ax1.set_yscale(loss_yscale)
    ax2.set_yscale(lr_yscale)
    ax2.set_ylabel('Learning rate')

    ax1.set_xlabel('Iterations')
    ax1.tick_params(which="both", direction="in",labelsize=label_size)
    ax1.margins(x=0)
    ax1.set_ylabel('Loss', )

    ax1.grid(which='both', linewidth=0.1)
    ax1.legend(all_lines, [l.get_label() for l in all_lines], loc=legend_loc, fontsize=label_size, framealpha=0.5)
    plt.title(title_txt)
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)