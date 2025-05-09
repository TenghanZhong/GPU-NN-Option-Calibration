import pandas as pd
import numpy as np

# 读取 Excel 文件
df = pd.read_excel(r"C:\Users\26876\Desktop\Math548\Project\ttm100.xlsx")

# 修改列
df['spot_price'] = df['spot_price'] / 100
df['mid_price_Put'] /= 100

# 保存为新文件（或覆盖原文件）
df.to_excel(r"C:\Users\26876\Desktop\Math548\Project\ttm100_modified.xlsx", index=False)

