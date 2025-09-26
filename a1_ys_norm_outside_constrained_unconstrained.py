import numpy as np
import torch
from c1_constrained_NODE_model import constrained_NODE
from c2_unconstrained_NODE_model import unconstrained_NODE
from c3_NODE_training import NODE_training
from c4_NODE_evaluation import NODE_evaluation
from g1_mass_balance_check import mass_balance_check
from g2_plot_ASM1_trajectory import plot_ASM1_trajectory
from g3_plot_lr_loss import plot_lr_loss
from g4_plot_distribution import plot_distribution

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

# define ASM1 elemental composition matrix (12 x 3)
comp_matrix = torch.t(torch.tensor([
    [1, 1, 1, 1, 1, -1, -32/7, 0, 0, 0, 0, -12/7],     # COD
    [0, 0, 0.086, 0.086, 0.06, 0, 1, 1, 1, 1, 0, 1],   # nitrogen
    [0, 0, 0, 0, 0, 0, -1/14000, 1/14000, 0, 0, -1, 0] # charge
    ], dtype=torch.float32))

kla = torch.tensor(kla, dtype=torch.float32)

# calculate mean and sd of training data
y_mean = torch.mean(y, dim=0)
y_sd = torch.std(y, dim=0)
y_sd[y_sd == 0.0] = 1e-12  # prevent division by zero

# data normalization
y0_normalized = (y0 - y_mean) / y_sd
y_normalized = (y - y_mean) / y_sd

# composition matrix normalization
y_sd_diag = torch.diag(y_sd)
comp_matrix_normalized = torch.matmul(y_sd_diag, comp_matrix)

# visulise data
# plot_distribution(y)
# mass_balance_check(comp_matrix, kla, t, true_y, true_dy, dy_types=['true'])

plot_ASM1_trajectory(t, y, y_types=[y_type], show_RMSE=True)
plot_ASM1_trajectory(t, true_y, noise_y, smooth_y, y_types=['true', 'noise', 'smooth'],show_RMSE=True,show_R2=True)
plot_ASM1_trajectory(t, true_dy, y_types=['true'], is_dy=True)

# ============== 2. training parameters ==============
n_iters = 5000
batch_time = 16
batch_size = 512

lr_initial = 0.01
lr_factor = 0.2
lr_patience = 200

# ============== 3. constrained model training ==============
constrained_NODE_model = constrained_NODE(comp_matrix=comp_matrix_normalized, kla=kla, y_mean=y_mean, y_sd=y_sd)

constrained_NODE_model, lr_list_con, loss_list_con = NODE_training(model=constrained_NODE_model, t=t, y=y_normalized, 
    n_iters=n_iters, batch_time=batch_time, batch_size=batch_size, lr_initial=lr_initial, lr_factor=lr_factor, lr_patience=lr_patience)

plot_lr_loss([(lr_list_con,loss_list_con)],loss_labels=['constrained loss'])
# torch.save(constrained_NODE_model, rf'result/{y_type}_constrained_NODE_model.pth')

# ============== 4. constrained model evlaution ==============
# constrained_NODE_model = torch.load(rf'result/{y_type}_constrained_NODE_model.pth')
pred_y_constrained, pred_dy_constrained=NODE_evaluation(constrained_NODE_model, y0_normalized, t, y_mean, y_sd)

# mass_balance_check(comp_matrix, kla, t, y, true_dy, pred_y_constrained, pred_dy_constrained, dy_types=[y_type, 'constrained'])
# plot_ASM1_trajectory(t, y, pred_y_constrained, y_types=[y_type, 'constrained'], show_RMSE=True, show_R2=True)
# plot_ASM1_trajectory(t, true_dy, pred_dy_constrained, y_types=['true', 'constrained'], is_dy=True, show_RMSE=True, show_R2=True)

# ============== 5. unconstrained model training ==============
unconstrained_NODE_model = unconstrained_NODE(kla=kla, y_mean=y_mean, y_sd=y_sd)

unconstrained_NODE_model, lr_list_un, loss_list_un = NODE_training(model=unconstrained_NODE_model, t=t, y=y_normalized, 
    n_iters=n_iters, batch_time=batch_time, batch_size=batch_size, lr_initial=lr_initial, lr_factor=lr_factor, lr_patience=lr_patience)

# plot_lr_loss([(lr_list_un,loss_list_un)], loss_labels=['unconstrained'])
# torch.save(unconstrained_NODE_model, rf'result/{y_type}_unconstrained_NODE_model.pth')

# ============== 6. unconstrained model evlaution ==============
# unconstrained_NODE_model = torch.load('result/{y_type}_unconstrained_NODE_model.pth') 

pred_y_unconstrained, pred_dy_unconstrained=NODE_evaluation(unconstrained_NODE_model, y0_normalized, t, y_mean, y_sd)

plot_ASM1_trajectory(t, true_dy, pred_dy_constrained, pred_dy_unconstrained, y_types=['true', 'constrained','unconstrained'], is_dy=True, show_RMSE=True, show_R2=True)
plot_ASM1_trajectory(t, y, pred_y_constrained, pred_y_unconstrained, y_types=[y_type, 'constrained','unconstrained'], show_RMSE=True, show_R2=True)
plot_lr_loss([(lr_list_con,loss_list_con), (lr_list_un, loss_list_un)],loss_labels=['constrained','unconstrained'])
mass_balance_check(comp_matrix, kla, t, y, true_dy, pred_y_constrained, pred_dy_constrained, pred_y_unconstrained, pred_dy_unconstrained, dy_types=[y_type, 'constrained','unconstrained'])