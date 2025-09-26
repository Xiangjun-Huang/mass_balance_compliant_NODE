import numpy as np
import torch
from torchdiffeq import odeint
from c1_constrained_NODE_model import constrained_NODE
from c2_unconstrained_NODE_model import unconstrained_NODE
from c3_NODE_training import NODE_training
from c4_NODE_evaluation import NODE_evaluation
# from g1_mass_balance_check import mass_balance_check
# from g2_ASM1_trajectory import plot_ASM1_trajectory
# from g4_plot_distribution import plot_distribution
import time

# ============== 1. data preparation  ==============
kla = 240
t0 = 0.0
tn = 0.25   # unit :day 0.25 day = 6 hours
t_points = 1000
noise_sd = 0.05

file_path = rf"data\ASM1_component_concentration_trajectory_kla_{kla}_t_{t0}_{tn}_{t_points}_"

# read noise_y data
t_y_data = np.loadtxt(rf"{file_path}noise_y_{noise_sd}.csv", delimiter=',', skiprows=2)
t = torch.tensor(t_y_data[:,0], dtype=torch.float32)
noise_y = torch.tensor(t_y_data[:,1:], dtype=torch.float32)

# read smooth_y data
t_y_data = np.loadtxt(rf"{file_path}smooth_y_{noise_sd}.csv", delimiter=',', skiprows=2)
smooth_y = torch.tensor(t_y_data[:,1:], dtype=torch.float32)

# read true_y data from csv file
t_y_data = np.loadtxt(rf"{file_path}true_y.csv", delimiter=',', skiprows=2)
true_y = torch.tensor(t_y_data[:,1:], dtype=torch.float32)

# read true_dy data
true_dy_data = np.loadtxt(rf"{file_path}true_dy.csv", delimiter=',', skiprows=2)
true_dy = torch.tensor(true_dy_data[:,1:], dtype=torch.float32)

# specify which data to use
y_type = 'noise'
y = noise_y

# define initial condition
y0 = y[0,:]

# define elemental composition matrix as mass balance constraints
comp_matrix = torch.t(torch.tensor([
    [1, 1, 1, 1, 1, -1, -32/7, 0, 0, 0, 0, -12/7],     # COD
    [0, 0, 0.086, 0.086, 0.06, 0, 1, 1, 1, 1, 0, 1],   # nitrogen
    [0, 0, 0, 0, 0, 0, -1/14000, 1/14000, 0, 0, -1, 0] # charge
    ], dtype=torch.float32))

kla = torch.tensor(kla,dtype=torch.float32)

# calculate mean and sd of training data
y_mean = torch.mean(y, dim=0)
y_sd = torch.std(y, dim=0)
y_sd[y_sd == 0.0] = 1e-12  # prevent division by zero

# data normalization
y0_normalized = (y0 - y_mean) / y_sd
y_normalized = (y - y_mean) / y_sd

# composition matrix normalization
y_sd_diag = torch.diag(y_sd)
comp_matrix_normalized = torch.matmul(y_sd_diag, comp_matrix,)

# visulise data
# plot_distribution(y)
# mass_balance_check(comp_matrix, kla, t, true_y, true_dy, dy_types=['true'])

# plot_ASM1_trajectory(t, y, y_types=[y_type])
# plot_ASM1_trajectory(t, true_dy, y_types=['true'], is_dy=True)

# ============== 2. training parameters ==============
n_iters = 5000
batch_time = 16
batch_size = 512

lr_initial = 0.01
lr_factor = 0.2
lr_patience = 200

n_runs = 10 # specify how many runs
result_m = np.zeros((n_runs, 22))
loss_mse = torch.nn.MSELoss(reduction='mean')

# ============== 3. multiple times training ==============
for j in range(n_runs):

    # ============== 4. constrained model training ==============
    constrained_NODE_model = constrained_NODE(comp_matrix=comp_matrix_normalized, kla=kla, y_mean=y_mean, y_sd=y_sd)

    start_time = time.time()
    constrained_NODE_model, lr_list_con, loss_list_con = NODE_training(model=constrained_NODE_model, t=t, y=y_normalized, 
        n_iters=n_iters, batch_time=batch_time, batch_size=batch_size, lr_initial=lr_initial, lr_factor=lr_factor, lr_patience=lr_patience,show_loss=False)
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"{j}: constrained training time: {elapsed_time:.1f} minutes")

    # ============== 5. constrained model evlaution ==============
    pred_y_constrained, pred_dy_constrained=NODE_evaluation(constrained_NODE_model, y0_normalized, t, y_mean, y_sd)

    # save the data
    dy_j = pred_dy_constrained.clone() # can not do in-place operation
    dy_j[...,5:6] = dy_j[...,5:6] - kla * (8 - pred_y_constrained[...,5:6])
    mass_dy_constrained= dy_j @ comp_matrix

    mass_dy_constrained_mean = torch.mean(mass_dy_constrained, dim=0).numpy()
    mass_dy_constrained_std = torch.std(mass_dy_constrained, dim=0).numpy()

    pred_y_constrained_RMSE = torch.sqrt(loss_mse(y, pred_y_constrained)).numpy()
    pred_y_constrained_R2 = 1 - (torch.sum((y - pred_y_constrained)**2) / torch.sum((y - torch.mean(y))**2)).numpy()

    pred_dy_constrained_RMSE = torch.sqrt(loss_mse(true_dy, pred_dy_constrained)).numpy()
    pred_dy_constrained_R2 = 1 - (torch.sum((true_dy - pred_dy_constrained)**2) / torch.sum((true_dy - torch.mean(true_dy))**2)).numpy()

    result_m[j,0:3] = mass_dy_constrained_mean
    result_m[j,3:6] = mass_dy_constrained_std
    result_m[j,6] = pred_y_constrained_RMSE
    result_m[j,7] = pred_dy_constrained_RMSE
    result_m[j,8] = pred_y_constrained_R2
    result_m[j,9] = pred_dy_constrained_R2
    result_m[j,10] = elapsed_time

    # ============== 6. unconstrained model training ==============
    unconstrained_NODE_model = unconstrained_NODE(kla=kla, y_mean=y_mean, y_sd=y_sd)

    start_time = time.time()
    unconstrained_NODE_model, lr_list_un, loss_list_un = NODE_training(model=unconstrained_NODE_model, t=t, y=y_normalized, 
        n_iters=n_iters, batch_time=batch_time, batch_size=batch_size, lr_initial=lr_initial, lr_factor=lr_factor, lr_patience=lr_patience, show_loss=False)
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"{j}: unconstrained training time: {elapsed_time:.1f} minutes")

    # ============== 7. unconstrained model evlaution ==============
    pred_y_unconstrained, pred_dy_unconstrained=NODE_evaluation(unconstrained_NODE_model, y0_normalized, t, y_mean, y_sd)

    # save the data
    dy_j = pred_dy_unconstrained.clone() # can not do in-place operation
    dy_j[...,5:6] = dy_j[...,5:6] - kla * (8 - pred_y_unconstrained[...,5:6])
    mass_dy_unconstrained = dy_j @ comp_matrix

    mass_dy_unconstrained_mean = torch.mean(mass_dy_unconstrained, dim=0).numpy()
    mass_dy_unconstrained_std = torch.std(mass_dy_unconstrained, dim=0).numpy()

    pred_y_unconstrained_RMSE = torch.sqrt(loss_mse(y, pred_y_unconstrained)).numpy()
    pred_y_unconstrained_R2 = 1 - (torch.sum((y - pred_y_unconstrained)**2) / torch.sum((y - torch.mean(y))**2)).numpy()

    pred_dy_unconstrained_RMSE = torch.sqrt(loss_mse(true_dy, pred_dy_unconstrained)).numpy()
    pred_dy_unconstrained_R2 = 1 - (torch.sum((true_dy - pred_dy_unconstrained)**2) / torch.sum((true_dy - torch.mean(true_dy))**2)).numpy()

    result_m[j,11:14] = mass_dy_unconstrained_mean
    result_m[j,14:17] = mass_dy_unconstrained_std
    result_m[j,17] = pred_y_unconstrained_RMSE
    result_m[j,18] = pred_dy_unconstrained_RMSE
    result_m[j,19] = pred_y_unconstrained_R2
    result_m[j,20] = pred_dy_unconstrained_R2
    result_m[j,21] = elapsed_time
    
np.savetxt(fname=f'result/{y_type}_{n_runs}_runs_rsum_RMSE_R2_timing.csv', 
    X=result_m, delimiter=',', comments='') 