from cvxpy import std
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns  

y_name = ['$S_S$', '$X_S$', '$X_{BH}$', '$X_{BA}$', '$X_P$', '$S_O$', '$S_{NO}$', '$S_{NH}$', '$S_{ND}$', '$X_{ND}$', '$S_{ALK}$', '$S_{N_2}$'],
y_unit = ['mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l', '$mg O_2/l$', 'mg N/l', 'mg N/l', 'mg N/l', 'mg N/l', 'eq ALK/l', 'mg N/l']

dy_labels = ['COD', 'Nitrogen', 'Charge']
dy_units = ['mg COD/L/d', 'mg N/L/d', 'eq ALK/L/d']

mass_lim = [(-2000, 2000),(-200,200),(-0.015, 0.015)]

file_path = rf"result\noise_100_runs_constrained_unconstrained_rsum_RMSE_R2_timing_v2.xlsx"
df = pd.read_excel(file_path, skiprows=3)

# The Excel contains following columns：
# run: run number (1-100)
# ncm_COD, ncs_COD: mean and sd of noise constrained COD
# nnm_COD, nns_COD: mean and sd of noise unconstrained COD
# ncm_nitrogen, ncs_nitrogen: mean and sd of noise constrained nitrogen
# nnm_nitrogen, nns_nitrogen: mean and sd of noise unconstrained nitrogen
# ncm_charge, ncs_charge: mean and sd of noise constrained charge
# nnm_charge, nns_charge: mean and sd of noise unconstrained charge
# time: running time

# check data structure
# print("data structure：")
# print(df.head())

# define the name
pairs = [
    {"name": "COD", 
     "var1_mean": "ncm_COD", "var1_std": "ncs_COD", 
     "var2_mean": "nnm_COD", "var2_std": "nns_COD"},
    {"name": "Nitrogen", 
     "var1_mean": "ncm_nitrogen", "var1_std": "ncs_nitrogen", 
     "var2_mean": "nnm_nitrogen", "var2_std": "nns_nitrogen"},
    {"name": "Charge", 
     "var1_mean": "ncm_charge", "var1_std": "ncs_charge", 
     "var2_mean": "nnm_charge", "var2_std": "nns_charge"}
]

fig, axes = plt.subplots(3, 1, figsize=(4, 6), sharex=True)  

for i, pair in enumerate(pairs):
    ax = axes[i]
    
    var1_mean = df[pair["var1_mean"]]
    var1_std = df[pair["var1_std"]]
    var2_mean = df[pair["var2_mean"]]
    var2_std = df[pair["var2_std"]]
    
    global_mean1 = var1_mean.mean()
    global_std1 = var1_mean.std()
    global_mean2 = var2_mean.mean()
    global_std2 = var2_mean.std()
    print(f"mean1:\n {global_mean1}")
    print(f"std1:\n {global_std1}")
    print(f"mean2:\n {global_mean2} | {global_mean2/global_mean1:.1f}")
    print(f"std2:\n {global_std2} | {global_std2/global_std1:.1f}")
    
    line1, = ax.plot(df["run"], var1_mean, label=f"{pair['name']} Var1 Mean", color="blue", marker="o", markersize=1)
    shadow1 = ax.fill_between(df["run"], var1_mean - var1_std, var1_mean + var1_std,  color="blue", alpha=0.2)
    
    line2, = ax.plot(df["run"], var2_mean, label=f"{pair['name']} Var2 Mean", color="red", marker="s", markersize=1)
    shadow2 = ax.fill_between(df["run"], var2_mean - var2_std, var2_mean + var2_std, color="red", alpha=0.2)
    
    # # add global mean line
    # ax.axhline(global_mean1, color="blue", linestyle="--")
    # ax.axhline(global_mean2, color="orange", linestyle="--")
    
    # # add global sd line
    # ax.axhspan(global_mean1 - global_std1, global_mean1 + global_std1, color="blue", alpha=0.1)
    # ax.axhspan(global_mean2 - global_std2, global_mean2 + global_std2, color="orange", alpha=0.1)
    
    ax.set_title(rf'{dy_labels[i]} $r_{{sum}}$ over 100 runs - Mean and SD', fontsize=8)
    # ax.set_title(rf'$A_{{{dy_labels[i]}}}⋅dy$ - {dy_labels[i]} Sequence - Mean and std', fontsize=8) 
    ax.set_ylabel(f'{dy_units[i]}', fontsize=7)  
    ax.tick_params(axis='both', direction='in', labelsize=6) 
    ax.set_xlim(1,100)
    ax.set_xticks(ticks=[1, 20, 40, 60, 80, 100])
    ax.set_xticklabels(labels=["1", "20", "40", "60", "80", "100"])
    ax.set_ylim(mass_lim[i])
    ax.grid(True)

# set shared x axis labels
axes[-1].set_xlabel("Run Number", fontsize=7) 

# global legend
fig.legend([line1, line2, shadow1,shadow2], 
    ["Constrained Mean", "Unconstrained Mean", "± Constrained SD", "± Unconstrained SD"], 
    loc="lower center", bbox_to_anchor=(0.52, 0.0), ncol=2, fontsize=6 )

plt.subplots_adjust(left=0.2, bottom=0.11, right=0.92, top=0.92, wspace=0.25,hspace=0.25)
# plt.tight_layout()

plt.show()

RMSE_lim = [(0,6),(0,600)]

fig, axes = plt.subplots(2, 1, figsize=(4, 4), sharex=True) 

# define names
variables = [
    {"name": "component states", 
     "rmse_constrained": "ncrmse_y", "r2_constrained": "ncr2_y",
     "rmse_nonconstrained": "nnrmse_y", "r2_nonconstrained": "nnr2_y"},
    {"name": "component derivatives", 
     "rmse_constrained": "ncrmse_dy", "r2_constrained": "ncr2_dy",
     "rmse_nonconstrained": "nnrmse_dy", "r2_nonconstrained": "nnr2_dy"}
]

for i, var in enumerate(variables):
    ax1 = axes[i]  # RMSE
    ax2 = ax1.twinx()  # R2

    run = df["run"]
    rmse_constrained = df[var["rmse_constrained"]]
    r2_constrained = df[var["r2_constrained"]] * 100
    rmse_nonconstrained = df[var["rmse_nonconstrained"]]
    r2_nonconstrained = df[var["r2_nonconstrained"]] * 100

    mean_rmse_constrained = rmse_constrained.mean()
    std_rmse_constrained = rmse_constrained.std()
    mean_rmse_nonconstrained = rmse_nonconstrained.mean()
    std_rmse_nonconstrained = rmse_nonconstrained.std()
    print(f"mean_rmse_constrained: {mean_rmse_constrained}")
    print(f"std_rmse_constrained: {std_rmse_constrained}")
    print(f"mean_rmse_nonconstrained: {mean_rmse_nonconstrained}")
    print(f"std_rmse_nonconstrained: {std_rmse_nonconstrained}")

    mean_r2_constrained = r2_constrained.mean()
    std_r2_constrained = r2_constrained.std()
    mean_r2_nonconstrained = r2_nonconstrained.mean()
    std_r2_nonconstrained = r2_nonconstrained.std()
    print(f"mean_r2_constrained: {mean_r2_constrained}")
    print(f"std_r2_constrained: {std_r2_constrained}")
    print(f"mean_r2_nonconstrained: {mean_r2_nonconstrained}")
    print(f"std_r2_nonconstrained: {std_r2_nonconstrained}")

    # plot RMSE curve
    line1, = ax1.plot(run, rmse_constrained, label="Constrained RMSE", color="blue", linestyle="-", marker="o", markersize=0.5)
    line2, = ax1.plot(run, rmse_nonconstrained, label="Unconstrained RMSE", color="red", linestyle="--", marker="s", markersize=0.5)
    ax1.set_ylabel("RMSE", fontsize=7, color="black")
    ax1.tick_params(axis='both',direction='in', labelsize=6)
    ax1.set_ylim(RMSE_lim[i])
    ax1.set_xticks(ticks=[1, 20, 40, 60, 80, 100])
    ax1.set_xticklabels(labels=["1", "20", "40", "60", "80", "100"])

    # plot R2 curve
    line3, = ax2.plot(run, r2_constrained, label="Constrained R²", color="green", linestyle="-.", linewidth=1)
    line4, = ax2.plot(run, r2_nonconstrained, label="Unconstrained R²", color="purple", linestyle=":", linewidth=1)
    ax2.set_ylabel("R² (%)", fontsize=7, color="black")
    ax2.tick_params(axis='both', direction='in', labelsize=6)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))  
    ax2.set_ylim(0, 120)  

    ax1.set_title(f"RMSE and R² for {var['name']} over 100 runs", fontsize=8)
    ax1.grid(True)

# shared x axis labels
axes[-1].set_xlabel("Run Number", fontsize=7)
axes[-1].set_xlim(1,100)
axes[-1].tick_params(axis='x',direction='in', labelsize=6)

# global legend
fig.legend([line1, line2, line3, line4], 
    ["Constrained $RMSE$", "Unconstrained $RMSE$","Constrained $R^2$", "Unconstrained $R^2$"], 
    loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6)

plt.subplots_adjust(left=0.12, bottom=0.16, right=0.86, top=0.92, wspace=0.25,hspace=0.25)
# plt.tight_layout(rect=[0, 0, 1, 0.95]) 

plt.show()