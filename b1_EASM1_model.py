# extended EASM1 mechanistic model with full 12 components （excluding S_I, X_I, X_INORG）
import torch

class EASM1(torch.nn.Module):
    def __init__(self, stoichi_params, kinetic_params, kla):        
        # stoichi_params (torch.Tensor): Stoichiometric parameters.
        # kinetic_params (list): List of kinetic parameters.
        # kla: constant tensor

        super(EASM1, self).__init__()

        # Stoichiometric parameters
        self.Y_A           = stoichi_params[0]
        self.Y_H           = stoichi_params[1]
        self.f_P           = stoichi_params[2]
        self.i_XB          = stoichi_params[3]
        self.i_XP          = stoichi_params[4]
        self.i_COD_N2      = stoichi_params[5]
        self.i_COD_NO3     = stoichi_params[6]
        self.i_NO3_N2      = stoichi_params[7]
        self.i_Charge_SNHx = stoichi_params[8]
        self.i_Charge_SNOx = stoichi_params[9]

        # Kinetic parameters
        self.u_A    = kinetic_params[0]
        self.b_A    = kinetic_params[1]
        self.k_a    = kinetic_params[2]
        self.K_O_A  = kinetic_params[3]
        self.K_NH   = kinetic_params[4]
        self.u_H    = kinetic_params[5]
        self.n_g    = kinetic_params[6]
        self.K_S    = kinetic_params[7]
        self.b_H    = kinetic_params[8]
        self.K_O_H  = kinetic_params[9]
        self.K_NO   = kinetic_params[10]
        self.K_NH_H = kinetic_params[11]
        self.k_h    = kinetic_params[12]
        self.K_X    = kinetic_params[13]
        self.n_h    = kinetic_params[14]
        self.K_ALK  = kinetic_params[15]

        # kla input
        self.kla = kla

    def forward(self, t, y):

        # components
        S_S     = y[..., 0:1]
        X_S     = y[..., 1:2]
        X_BH    = y[..., 2:3]
        X_BA    = y[..., 3:4]
        X_P     = y[..., 4:5]
        S_O     = y[..., 5:6]
        S_NO    = y[..., 6:7]
        S_NH    = y[..., 7:8]
        S_ND    = y[..., 8:9]
        X_ND    = y[..., 9:10]
        S_ALK   = y[..., 10:11]
        S_N2    = y[..., 11:12]

        # reaction rates
        r1 = self.u_H * (S_S / (self.K_S + S_S)) * (S_O / (self.K_O_H + S_O)) * (S_NH / (self.K_NH_H + S_NH)) * (S_ALK / (self.K_ALK + S_ALK)) * X_BH
        r2 = self.u_H * (S_S / (self.K_S + S_S)) * (self.K_O_H / (self.K_O_H + S_O)) * (S_NO / (self.K_NO + S_NO)) * (S_NH / (self.K_NH_H + S_NH)) * self.n_g * X_BH
        r3 = self.u_A * (S_NH / (self.K_NH + S_NH)) * (S_O / (self.K_O_A + S_O)) * (S_ALK / (self.K_ALK + S_ALK)) * X_BA
        r4 = self.b_H * X_BH
        r5 = self.b_A * X_BA
        r6 = self.k_a * S_ND * X_BH
        r7 = self.k_h * ((X_S / X_BH) / (self.K_X + (X_S / X_BH))) * (S_O / (self.K_O_H + S_O) + self.n_h * (self.K_O_H / (self.K_O_H + S_O)) * (S_NO / (self.K_NO + S_NO))) * X_BH
        r8 = r7 * (X_ND / X_S)

        # derivatives
        dS_S   = r7 - (r1 + r2) / self.Y_H
        dX_S   = (1 - self.f_P) * (r4 + r5) - r7
        dX_BH  = r1 + r2 - r4
        dX_BA  = r3 - r5
        dX_P   = self.f_P * (r4 + r5)
        dS_O   = (1 - 1/ self.Y_H) * r1 + (1 + self.i_COD_NO3 / self.Y_A) * r3 + self.kla * (8 - S_O) # saturation concentration of O2 is 8 mg/L
        dS_NO  = r3 / self.Y_A - ((1 - self.Y_H) / (self.i_NO3_N2 * self.Y_H)) * r2
        dS_NH  = r6 - self.i_XB * r1 - self.i_XB * r2 - (self.i_XB + 1 / self.Y_A) * r3
        dS_ND  = r8 - r6
        dX_ND  = (self.i_XB - self.f_P * self.i_XP) * (r4 + r5) - r8
        dS_ALK = self.i_Charge_SNHx * r6 - self.i_XB * self.i_Charge_SNHx * r1 - (((1 - self.Y_H)/(self.i_NO3_N2 * self.Y_H))*self.i_Charge_SNOx + self.i_XB * self.i_Charge_SNHx ) * r2 - ((self.i_XB + 1 / self.Y_A) * self.i_Charge_SNHx - self.i_Charge_SNOx / self.Y_A) * r3
        dS_N2  = r2 * ((1 - self.Y_H) / (self.i_NO3_N2 * self.Y_H))

        return torch.cat([dS_S, dX_S, dX_BH, dX_BA, dX_P, dS_O, dS_NO, dS_NH, dS_ND, dX_ND, dS_ALK, dS_N2],dim=-1)