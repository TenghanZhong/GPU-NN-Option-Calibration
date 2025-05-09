import torch
import joblib
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# ---- 1. 加载数据 ----
data_file = r"C:\Users\26876\Desktop\Math548\Project\Option_put_price_dataset_mpmath.parquet"
model_file = r"C:\Users\26876\Desktop\Math548\Project\model.pt"
scaler_file = r"C:\Users\26876\Desktop\Math548\Project\scalers.pkl"

# 加载数据
df = pd.read_parquet(data_file)
scalers = joblib.load(scaler_file)

in_cols = scalers["in_cols"]
out_cols = scalers["out_cols"]
scX = scalers["scX"]
scY = scalers["scY"]

# ---- 2. 分割数据（严格按照 t_K 分层）----
if "t" not in df.columns or "K" not in df.columns:
    raise ValueError("数据中缺少 't' 和 'K' 列，无法进行 stratified split！")

strata = df["t"].astype(str) + "_" + df["K"].astype(str)

train_df, test_df = train_test_split(
    df, test_size=0.05, stratify=strata, random_state=2025
)

print(f"📦 Train size: {len(train_df)}, Test size: {len(test_df)}")

# ---- 3. 从测试集中随机取100个点 ----
np.random.seed(42)
sample_df = test_df.sample(n=200, random_state=2025)

X = sample_df[in_cols].values
Y_true = sample_df[out_cols].values

# 标准化（注意使用训练时的标准化器）
X_std = scX.transform(X)

# ---- 4. 定义网络并加载权重 ----
def get_act(name):
    name = name.lower()
    if name in ("linear", "none"): return nn.Identity()
    if name == "relu": return nn.ReLU()
    if name == "elu": return nn.ELU()
    if name == "leaky": return nn.LeakyReLU(0.01)
    if name == "tanh": return nn.Tanh()
    raise ValueError(f"unknown activation {name}")

class GenericNet(nn.Module):
    def __init__(self, dim_in, dim_out, hidden, layers, act="elu", out_act="linear"):
        super().__init__()
        if isinstance(hidden, int):
            hidden = [hidden] * layers
        blocks, in_d = [], dim_in
        for h in hidden:
            blocks += [nn.Linear(in_d, h), get_act(act)]
            in_d = h
        blocks += [nn.Linear(in_d, dim_out), get_act(out_act)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

# 建立和训练时一模一样的网络
net = GenericNet(
    dim_in=len(in_cols),
    dim_out=len(out_cols),
    hidden=[64, 64, 32],  # 🔥 这里一定要和训练时保持一致！
    layers=3,
    act="elu",
    out_act="linear",
).double()

net.load_state_dict(torch.load(model_file))
net.eval()

# ---- 5. 预测 ----
with torch.no_grad():
    preds_std = net(torch.tensor(X_std, dtype=torch.double)).numpy()

# 反标准化
preds = scY.inverse_transform(preds_std)

# ---- 6. 比较 true vs pred ----
plt.figure(figsize=(8,6))
plt.scatter(range(len(Y_true)), Y_true, color="green", label="True", marker="o", alpha=0.7)
plt.scatter(range(len(preds)), preds, color="blue", label="Predicted", marker="x", alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("True vs Predicted Values (100 random samples from Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 7. 计算 RMSE ----
rmse = np.sqrt(((Y_true - preds) ** 2).mean())
print(f"🏁 RMSE on 100 test samples = {rmse:.6f}")
