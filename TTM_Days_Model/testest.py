import os as _os
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _Path(_os.environ.get("GPU_NN_PROJECT_ROOT", str(_REPO_ROOT / "Data_Results"))).expanduser()

import pandas as pd
import numpy as np

# 读取 Excel 文件
df = pd.read_excel(str(_PROJECT_ROOT / "ttm100.xlsx"))

# 修改列
df['spot_price'] = df['spot_price'] / 100
df['mid_price_Put'] /= 100

# 保存为新文件（或覆盖原文件）
df.to_excel(str(_PROJECT_ROOT / "ttm100_modified.xlsx"), index=False)

