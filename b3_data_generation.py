import torch
import numpy as np
from scipy.signal import savgol_filter
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint # for adjoint method!
from b1_EASM1_model import EASM1
from b2_default_stoichiometric_kinetic_value import default_stoichimetric_kinetic_value
from g1_mass_balance_check import mass_balance_check

y0 = torch.tensor([59.8, 260.1, 2552.0, 148, 449, 2, 0, 23.0, 1.8, 7.8, 0.007, 0])
# y0 = torch.tensor([59.2, 260, 2550.0, 147, 448.5, 2, 0, 22.0, 1.7, 7.7, 0.006, 0]) # for testing

t0 = 0.0
tn = 0.25   # unit :day 0.25 day = 6 hours
t_points = 1000
t_span  = torch.tensor([t0, tn, t_points]) 

noise_sd = 0.05
window_size = 50
poly_order = 3

y_name = ['S_S', 'X_S', 'X_BH', 'X_BA', 'X_P', 'S_O', 'S_NO', 'S_NH', 'S_ND', 'X_ND', 'S_ALK', 'S_N2']
y_unit = ['mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l', 'mg COD/l', 'mg O2/l', 'mg N/l', 'mg N/l',
    'mg N/l', 'mg N/l', 'eq ALK/l', 'mg N/l']

t_y_name = ['time'] + y_name
t_y_unit = ['day'] + y_unit
t_y_header = ",".join(t_y_name) + "\n" + ",".join(t_y_unit)

t_dy_name = ['time'] + ['d' + i for i in y_name]
t_dy_unit = ['day'] + [i + '/d' for i in y_unit]
t_dy_header = ",".join(t_dy_name) + "\n" + ",".join(t_dy_unit)

# Generating training data from mechanistic ODEs without autograd
t = torch.linspace(t_span[0], t_span[1], int(t_span[2].item()))
stoi_p, kine_p, stoi_m, comp_m, kla = default_stoichimetric_kinetic_value()
with torch.no_grad():
    mechanistic_model = EASM1(stoi_p, kine_p, kla)
    true_y = odeint(mechanistic_model, y0, t, method='dopri5')
    # true_y = odeint(mechanistic_model, y0, t, method='dopri5', rtol=1e-12, atol=1e-12)
    true_dy = mechanistic_model(0, true_y)
print("Data generated.")

# =========== add noise based on std of components ==============
if noise_sd!=0:
    comp_std = torch.std(true_y, dim=0) # components standard deviation
    noise_y = true_y + noise_sd * comp_std * torch.randn_like(true_y)
    smooth_y = torch.tensor(savgol_filter(noise_y,window_size, poly_order,axis=0)) # smooth y with savitzky_golay filter
else:
    noise_y = true_y
    smooth_y = true_y

# mass balance check
mass_balance_check(comp_m.float(), kla, t, true_y, true_dy, dy_types=['true'])

# save data to csv file
save_t_true_y = torch.hstack((t.unsqueeze(-1), true_y)).numpy()
save_path = rf'data/ASM1_component_concentration_trajectory_kla'
np.savetxt(fname=f'{save_path}_{kla}_t_{t_span[0]}_{t_span[1]}_{t_span[2]:.0f}_true_y.csv', 
    X=save_t_true_y, fmt='%.6f', delimiter=',', header=t_y_header, comments='') 
save_t_true_dy = torch.hstack((t.unsqueeze(-1), true_dy)).numpy()
np.savetxt(fname=f'{save_path}_{kla}_t_{t_span[0]}_{t_span[1]}_{t_span[2]:.0f}_true_dy.csv', 
    X=save_t_true_dy, fmt='%.6f', delimiter=',', header=t_dy_header, comments='') 
save_t_noise_y = torch.hstack((t.unsqueeze(-1), noise_y)).numpy()
np.savetxt(fname=f'{save_path}_{kla}_t_{t_span[0]}_{t_span[1]}_{t_span[2]:.0f}_noise_y_{noise_sd}.csv', 
    X=save_t_noise_y, fmt='%.6f', delimiter=',', header=t_y_header, comments='') 
save_t_smooth_y = torch.hstack((t.unsqueeze(-1), smooth_y)).numpy()
np.savetxt(fname=f'{save_path}_{kla}_t_{t_span[0]}_{t_span[1]}_{t_span[2]:.0f}_smooth_y_{noise_sd}.csv', 
    X=save_t_smooth_y, fmt='%.6f', delimiter=',', header=t_y_header, comments='') 