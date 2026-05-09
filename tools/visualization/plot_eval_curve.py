import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据
df_base = pd.read_csv(r'C:\Users\20439\Downloads\UAV-DETR\result_csv\results_r18.csv') # 请确保文件路径正确，比如 C:/Users/.../results_r18.csv
df_ours = pd.read_csv(r'C:\Users\20439\Downloads\UAV-DETR\result_csv\results_redetr_r18.csv')

# 【关键步骤】清除列名中的前后空格
df_base.columns = df_base.columns.str.strip()
df_ours.columns = df_ours.columns.str.strip()

# 2. 打印一下列名，确认现在变成了干净的名字（可选，用于调试）
# print(df_base.columns)

# 3. 绘制 mAP@50 对比图
plt.figure(figsize=(10, 6))

# 现在可以直接使用干净的列名了
plt.plot(df_base['metrics/mAP50(B)'], label='Baseline (R18)', color='blue', linestyle='--')
plt.plot(df_ours['metrics/mAP50(B)'], label='Ours (UAV-DETR)', color='red', linewidth=2)

plt.title('Comparison of Detection Accuracy (mAP@50)')
plt.xlabel('Epochs')
plt.ylabel('mAP@0.5')
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig('comparison_map.png', dpi=300)
plt.show()