import torch
# ============= neural ODE model without physical constraints of mass balance ===============    
class unconstrained_NODE(torch.nn.Module):
    def __init__(self, kla, y_mean, y_sd):
        """
        Args:
            comp_matrix (torch.Tensor): EASM1 elemental composition matrix 
            for mass balance in COD, nitrogen and charge shape [12, 3]), if input y is normalized, 
            then comp_matrix should be also normalized (multiplied by y_sd_diagonal matrix)
        """
        super(unconstrained_NODE, self).__init__()
        
        # define MLP neural network
        self.net = torch.nn.Sequential(
            torch.nn.Linear(12, 50),
            torch.nn.GELU(),
            torch.nn.Linear(50, 50),
            torch.nn.GELU(),
            torch.nn.Linear(50, 50),
            torch.nn.GELU(),
            torch.nn.Linear(50, 12)
        )
        
        # register composition matrix for physics constraint and calculate its pseudo inverse matrix
        self.register_buffer("kla", kla)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_sd", y_sd)

    def forward(self, t, y):
        # 1. predicte dy
        # y[y<0] = torch.tensor(0.0)
        dy = self.net(y)

        # 2. dy is normalized, so kla*() should be normalized, y[...,5:6] should be normalised, can not directly substract!
        dy = dy.clone()
        dy[..., 5:6] = dy[..., 5:6] + self.kla*(8 - self.y_mean[5:6]) / self.y_sd[5:6] - self.kla*y[...,5:6] 
         
        return dy
