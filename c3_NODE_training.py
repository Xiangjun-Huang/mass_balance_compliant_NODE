import numpy as np
import torch
from torchdiffeq import odeint
# ============== batch function ==============
def get_batch(t, y, batch_time=16, batch_size=512):
    data_size = t.size(dim=t.ndim-1)
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True))
    batch_y0 = y[s]
    batch_t = t[:batch_time]
    batch_y = torch.stack([y[s + i] for i in range(batch_time)], dim=0)
    return batch_y0, batch_t, batch_y

def NODE_training(model, t, y, n_iters=50, batch_time=16, batch_size=512, lr_initial = 0.01, lr_factor = 0.2, lr_patience = 200, show_loss=True):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    lr_list = []
    loss_list = []

    for i in range(n_iters):   
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(t, y, batch_time=batch_time, batch_size=batch_size)
        pred_y = odeint(model, batch_y0, batch_t, method='dopri5') 
        loss = torch.mean(torch.abs(pred_y - batch_y))

        loss.backward()   
        optimizer.step()
        scheduler.step(loss)

        lr_list.append(optimizer.param_groups[0]['lr'])
        loss_list.append(loss.item())

        if show_loss:    
            print(f"Iter {i:4d} | lr: {lr_list[i]:.7f} | Loss: {loss_list[i]:.7f}")

    return model, lr_list, loss_list