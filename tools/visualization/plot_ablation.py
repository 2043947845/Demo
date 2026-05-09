import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def set_font_style():
    """借鉴 YOLO 的全局 Matplotlib 绘图风格设置"""
    # 启用 seaborn 的默认样式，这与 YOLO 风格类似，比较美观
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5})
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
        "figure.titlesize": 18,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"]
    })


def main():
    set_font_style()

    # 1. 定义 CSV 文件路径和对应标签
    # 确保脚本即使在不同目录下运行也能找到正确的绝对路径
    base_dir = r"c:\Users\20439\Downloads\UAV-DETR\result_csv"
    
    csv_files = {
        "Baseline (R18)": os.path.join(base_dir, "results_r18.csv"),
        "w/o FD": os.path.join(base_dir, "results_r18_no_fd.csv"),
        "w/o MFFF": os.path.join(base_dir, "results_r18_no_mfff.csv"),
        "w/o SAC": os.path.join(base_dir, "results_r18_no_sac.csv")
    }

    # 读取数据并将字典转换为 DataFrame 列表，同时清理列名
    dataframes = {}
    for label, path in csv_files.items():
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
            
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip() # 去除列名空格
        dataframes[label] = df

    if not dataframes:
        print("Error: No CSV files loaded.")
        return

    # 2. 绘制 mAP@50 对比图 (Ablation Study)
    plt.figure(figsize=(10, 6), dpi=300)

    # YOLO 常用冷暖色调搭配
    colors = {
        "Baseline (R18)": "#2A82D7", # 经典蓝
        "w/o FD": "#FF8C00",         # 橙色
        "w/o MFFF": "#2E8B57",       # 海绿色
        "w/o SAC": "#DC143C"         # 猩红
    }

    start_epoch = 100 # 截掉前多少轮

    # 遍历每个 DataFrame 并绘制曲线
    for label, df in dataframes.items():
        if 'metrics/mAP50(B)' in df.columns:
            # 基础网络使用虚线以示区分，消融实验用实线，可以自行根据需要调整
            linestyle = '--' if 'Baseline' in label else '-'
            linewidth = 2.5 if 'Baseline' not in label else 2.0
            
            # 使用 epoch 作为 x 轴，如果没有 epoch 列直接用 index
            x_axis = df['epoch'] if 'epoch' in df.columns else df.index + 1
            
            # 过滤出 start_epoch 之后的数据
            mask = x_axis > start_epoch
            plot_x = x_axis[mask]
            plot_y = df.loc[mask, 'metrics/mAP50(B)']
            
            if len(plot_x) == 0:
                print(f"Warning: No data available for {label} after epoch {start_epoch}.")
                continue
                
            color = colors.get(label, "tab:blue")
            plt.plot(plot_x, plot_y, 
                     label=label, 
                     color=color,
                     linestyle=linestyle,
                     linewidth=linewidth)
                     
            # 找到这段区间的最高点
            max_y = plot_y.max()
            max_x = plot_x[plot_y == max_y].iloc[0] # 取第一个达到最高点所在的 epoch
            
            # 画一个圆点标记最高点
            plt.plot(max_x, max_y, marker='o', color=color, markersize=6)
            # 画横向虚线对准坐标轴
            plt.axhline(y=max_y, xmin=0, xmax=(max_x - start_epoch)/(plt.xlim()[1] - start_epoch) if plt.xlim()[1] > start_epoch else 0.5, color=color, linestyle=':', alpha=0.7)
            # 在图中标注出具体的数值，放在点的上方一点
            plt.annotate(f'{max_y:.4f}', 
                         xy=(max_x, max_y), 
                         xytext=(10, 5), # 偏移量调整，可以防止互相遮挡
                         textcoords='offset points', 
                         color=color, 
                         fontsize=10, 
                         fontweight='bold')
        else:
            print(f"Warning: Column 'metrics/mAP50(B)' not found in {label}.")

    # 3. 设置图表属性
    plt.title('Ablation Study: Detection Accuracy (mAP@50)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    
    # 将图例放在最佳位置，添加透明背景阴影，类似于 YOLO 的高级感
    plt.legend(loc="lower right", frameon=True, shadow=True, fancybox=True, edgecolor='black', framealpha=0.9)
    
    # 优化坐标轴范围，使曲线更紧凑，放大精度差距
    max_epoch = max([len(df) for df in dataframes.values()])
    plt.xlim(start_epoch, max_epoch)
    
    # 动态计算 start_epoch 之后的最小值和最大值，给y轴留出一点余量，放大显示
    all_y_values_after_start = []
    for df in dataframes.values():
        if 'metrics/mAP50(B)' in df.columns:
            x_axis = df['epoch'] if 'epoch' in df.columns else df.index + 1
            all_y_values_after_start.append(df.loc[x_axis > start_epoch, 'metrics/mAP50(B)'])
    
    if all_y_values_after_start:
        y_values = pd.concat(all_y_values_after_start)
        y_min, y_max = y_values.min(), y_values.max()
        plt.ylim(max(0, y_min - 0.01), min(1.0, y_max + 0.01)) # 放大到 0.01 的余量，因为只看后期肯定差距更小
    
    plt.tight_layout()

    # 4. 保存并显示图片
    save_path = os.path.join(r"c:\Users\20439\Downloads\UAV-DETR\tools\visualization", "ablation_map50.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"=========================================")
    print(f"Ablation study plot saved to: {save_path}")
    print(f"=========================================")
    plt.show()

if __name__ == "__main__":
    main()
