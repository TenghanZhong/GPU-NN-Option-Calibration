import numpy as np
import torch
import joblib
import pandas as pd
from pathlib import Path
import torch.nn as nn

# -------- 1. 网络结构和工具函数 --------
def get_act(name):
    return dict(linear=nn.Identity(), none=nn.Identity(),
                relu=nn.ReLU(),  elu=nn.ELU(),
                leaky=nn.LeakyReLU(0.01), tanh=nn.Tanh())[name]

class GenericNet(nn.Module):
    def __init__(self, d_in, d_out, hidden, layers, act="elu"):
        super().__init__()
        if isinstance(hidden, int):
            hidden = [hidden] * layers
        net, inp = [], d_in
        for h in hidden:
            net += [nn.Linear(inp, h), get_act(act)]
            inp = h
        net.append(nn.Linear(inp, d_out))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

# -------- 2. 参数路径 --------
PROJECT_DIR = Path(r"C:\Users\26876\Desktop\Math548\Project")
MODEL_W     = PROJECT_DIR / "model_5000_days.pt"
SCALERS_PKL = PROJECT_DIR / "scalers_5000_days.pkl"

# -------- 3. 加载模型和缩放器 --------
saved = joblib.load(SCALERS_PKL)
net = GenericNet(len(saved["in_cols"]), 1, [64, 32, 32], 3, "elu").double()
net.load_state_dict(torch.load(MODEL_W, map_location="cpu"))
net.eval()
scX, scY = saved["scX"], saved["scY"]

# -------- 4. Calibrated 参数 --------
theta_opt = [
    0.020442,
    2.725817,
    0.087589,
    0.906155,
    0.945427,
    0.123439,  # r1 (T = 258)
    0.123986,  # r2 (T = 100)
    0.12495,   # r3 (T = 48)
    0.12889    # r4 (T = 20)
]

# -------- 5. 输入期权的 T 和 k --------
T_value = 20     # 剩余到期时间，例如20天
k_value = 0.2   # 执行价（归一化）

T_LIST = np.array([258, 100, 48, 20], dtype=float)
T_idx = np.argmin(np.abs(T_LIST - T_value))
r_value = theta_opt[5 + T_idx]
print(r_value)

X_input = np.array([
    theta_opt[0],  # a
    theta_opt[1],  # b
    theta_opt[2],  # c
    theta_opt[3],  # d
    theta_opt[4],  # kappa
    r_value,       # r
    T_value,       # T
    k_value        # strike
]).reshape(1, -1)

# -------- 6. 推断价格 --------
X_std = torch.tensor(scX.transform(X_input)).double()
with torch.no_grad():
    y_std = net(X_std).numpy()
y_pred = scY.inverse_transform(y_std.reshape(-1, 1)).ravel()[0]

print(f"📌 Predicted Put Option Price: {y_pred:.6f}")
