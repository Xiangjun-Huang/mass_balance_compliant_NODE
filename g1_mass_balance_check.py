""""
mass balance check 
 1    2    3     4     5    6    7     8     9     10    11     12
  
S_S, X_S, X_BH, X_BA, X_P, S_O, S_NO, S_NH, S_ND, X_ND, S_ALK, S_N2
 0    1    2     3     4    5    6     7     8     9     10     11  

"""
import numpy as np
import torch
import matplotlib.pyplot as plt

def mass_balance_check(comp_m, kla, t, *y_dy, 
                       dy_types=['True','Predicted','Preicted 2'], dy_styles=[':','--','-.'], dy_colors=['g','b','r'],
                       element_names=['COD','Nitrogen','Charge'],element_units=['mg COD/L/d', 'mg N/L/d', 'eq ALK/L/d'],
                       is_lim= False, rsum_lims=[(-50,50),(-0.1,0.1),(-0.01, 0.01)], 
                       t_unit_rate=24, t_show_unit='hour', t_ticks=[0,2,4,6], t_labels=['0','2','4','6'],
                       figsize=(4,6)):

    plt.style.use('seaborn-v0_8-paper')

    n_elements = comp_m.shape[1]
    fig, axs = plt.subplots(n_elements, 1, figsize=figsize) 

    n_y = round(len(y_dy)/2)
    result_m = np.zeros((5*n_y, n_elements))
    c_ratio_m = np.zeros((y_dy[0].size(0), n_elements*9))

    for j in range(n_y):
        y_j = y_dy[j*2]
        dy_j = y_dy[j*2+1]
        dy_j = dy_j.clone() # avoid in-place operation
        dy_j[...,5:6] = dy_j[...,5:6] - kla * (8 - y_j[...,5:6])
        mass_dy = dy_j @ comp_m

        # save the data
        mass_dy_min = torch.min(mass_dy, dim=0).values
        mass_dy_max = torch.max(mass_dy, dim=0).values
        mass_dy_mean = torch.mean(mass_dy, dim=0)
        mass_dy_std = torch.std(mass_dy, dim=0)

        mass_dy_abs_sum = torch.sum(torch.abs(mass_dy), dim=0)
        abs_product = torch.abs(dy_j.unsqueeze(2) * comp_m.unsqueeze(0))  # to (m, n, 3)
        a_dy_j_abs_product = abs_product.sum(dim=1)  # (m, 3)
        a_dy_j_abs_sum = torch.sum(torch.abs(a_dy_j_abs_product),dim=0)
        
        c_ratio = mass_dy_abs_sum/a_dy_j_abs_sum

        c_ratio_mat = torch.abs(mass_dy/a_dy_j_abs_product)
        c_ratio_m[:,j*9:(j+1)*9] = torch.hstack((mass_dy, a_dy_j_abs_product, c_ratio_mat)).numpy()

        result_m[j*5:(j+1)*5, :] = torch.vstack((mass_dy_min, mass_dy_max, mass_dy_mean, mass_dy_std, c_ratio)).numpy()
           
        for i in range(n_elements):
            ax = axs[i]
            ax.plot(t*t_unit_rate, mass_dy[:, i], label=dy_types[j], ls=dy_styles[j], color=dy_colors[j], lw=1)             
            ax.set_ylabel(f'{element_units[i]}')
            ax.set_xlabel(t_show_unit, fontsize=7)
            ax.xaxis.set_label_coords(1, -0.15)
            if is_lim:
                ax.set_ylim(rsum_lims[i])
            ax.set_xlim(t[0]*t_unit_rate, t[-1]*t_unit_rate)
            ax.set_xticks(ticks=t_ticks)
            ax.set_xticklabels(labels=t_labels)
            ax.grid(which='both', linewidth=0.2)
            ax.tick_params(which="both", direction="in")
            if i == 0:
                ax.legend()
            ax.set_title(rf'{element_names[i]} $r_{{sum}}$ Balance Check')


    # plt.subplots_adjust(left=0.225, bottom=0.08, right=0.9, top=0.92, wspace=0.5,hspace=0.5)
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97], pad=0.5, w_pad=0.5, h_pad=0.5) 
    plt.show()
 
    np.savetxt(fname=f'result/noise_single_run_metrics.csv', 
        X=result_m, delimiter=',', comments='') 
    np.savetxt(fname=f'result/noise_single_run_conservation_ratio_mat.csv', 
        X=c_ratio_m, delimiter=',', comments='') 

    return None