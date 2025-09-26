import torch
from torchdiffeq import odeint

def NODE_evaluation(model, y0_normalized, t, y_mean, y_sd):
    model.eval()
    with torch.no_grad():
        pred_y_normalized = odeint(model, y0_normalized, t, method='dopri5')
        pred_y = (pred_y_normalized * y_sd) + y_mean
        
        # normalized derivatives
        pred_dy_normalized = model(t, pred_y_normalized)
        
        # original derivatives by inverse normalisation
        pred_dy = pred_dy_normalized * y_sd
    return pred_y, pred_dy