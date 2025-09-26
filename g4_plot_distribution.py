import matplotlib.pyplot as plt
import torch
import seaborn as sns  
def plot_distribution(y,
    y_name = ['$S_S$', '$X_S$', '$X_{BH}$', '$X_{BA}$', '$X_P$', '$S_O$', '$S_{NO}$', '$S_{NH}$', '$S_{ND}$', '$X_{ND}$', '$S_{ALK}$', '$S_{N_2}$'],
    y_unit = ['mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l', '$mg O_2/l$', 'mg N/l', 'mg N/l', 'mg N/l', 'mg N/l', 'eq ALK/l', 'mg N/l']):
    
    n_comp = y.shape[-1]
    n_rows = 2
    n_cols = 6

    fig = plt.figure(figsize=(16, 5))  
    for i in range(n_comp):
        ax = fig.add_subplot(n_rows, n_cols, i+1)

        mean = torch.mean(y[:, i].float()).item()
        std = torch.std(y[:, i].float()).item()

        sns.histplot(y[:,i].numpy(), kde=True, ax=ax, alpha=0.7, bins=50)  # frequency，bins=50
        ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.3f}')
        ax.axvline(mean + std, color='green', linestyle='dashed', linewidth=1, label=f'Std: {std:.4f}')
        ax.axvline(mean - std, color='green', linestyle='dashed', linewidth=1)

        ax.set_title(y_name[i])
        ax.set_xlabel(y_unit[i])
        # ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.show()
    return None