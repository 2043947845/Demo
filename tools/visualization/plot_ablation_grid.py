import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def set_font_style():
    """借鉴 YOLO 的全局 Matplotlib 绘图风格设置"""
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5})
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "figure.titlesize": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"]
    })

def get_actual_column_name(columns, target):
    """Find the exact column name handling potential spacing/slash variations"""
    for col in columns:
        if col.strip() == target.strip():
            return col
            
    # Fallback loosely matching e.g. 'trainl1_loss' -> 'train/l1_loss'
    target_clean = target.replace('/', '').replace('_', '')
    for col in columns:
        if target_clean in col.replace('/', '').replace('_', ''):
            return col
    return None

def main():
    set_font_style()

    base_dir = r"c:\Users\20439\Downloads\UAV-DETR\result_csv"
    
    # User specified the four models
    csv_files = {
        "Baseline (r18)": os.path.join(base_dir, "results_r18.csv"),
        "w/o FD": os.path.join(base_dir, "results_r18_no_fd.csv"),
        "w/o MFFF": os.path.join(base_dir, "results_r18_no_mfff.csv"),
        "w/o SAC": os.path.join(base_dir, "results_r18_no_sac.csv")
    }

    metrics_list = [
        "train/giou_loss", "train/cls_loss", "train/l1_loss", "metrics/precision(B)", "metrics/recall(B)",
        "val/giou_loss", "val/cls_loss", "val/l1_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    ]

    dataframes = {}
    for label, path in csv_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            dataframes[label] = df
        else:
            print(f"File not found for {label}: {path}")

    if not dataframes:
        print("No CSV files found.")
        return

    colors = {
        "Baseline (r18)": "#2A82D7", # 蓝
        "w/o FD": "#FF8C00",         # 橙
        "w/o MFFF": "#2E8B57",       # 绿
        "w/o SAC": "#DC143C"         # 红
    }

    start_epoch = 100

    # 创建 2行5列 的网格，类似于 results.png
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), dpi=300)
    fig.suptitle('Ablation Study Training Results', fontweight='bold', y=1.02)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        
        has_data = False
        for label, df in dataframes.items():
            actual_col = get_actual_column_name(df.columns, metric)
            
            if actual_col:
                linestyle = '--' if 'Baseline' in label else '-'
                linewidth = 2.0 if 'Baseline' in label else 1.5
                alpha = 0.8 if 'Baseline' in label else 1.0
                
                x_axis = df['epoch'] if 'epoch' in df.columns else df.index + 1
                y_data = df[actual_col]
                
                # Plot removing NaNs and after start_epoch
                mask = y_data.notna() & (x_axis > start_epoch)
                
                ax.plot(x_axis[mask], y_data[mask], 
                        label=label, 
                        color=colors.get(label, "tab:blue"),
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha)
                has_data = True
            else:
                pass # print(f"Column {metric} not found in {label}.")

        if has_data:
            ax.set_title(metric, fontweight='bold')
            ax.set_xlabel('Epoch')
            # 自动调整边界
            ax.autoscale(enable=True, axis='x', tight=True)
            ax.tick_params(axis='both', which='major', labelsize=10)
        else:
            ax.set_title(f"{metric}\n(Not Found)", color='red')
            
    # 全局添加一个图例
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05),
                   frameon=True, shadow=True, fancybox=True, edgecolor='black', framealpha=0.9)

    plt.tight_layout()
    
    # 底部稍微留空以便放下 Legend
    plt.subplots_adjust(bottom=0.08)

    save_path = os.path.join(r"c:\Users\20439\Downloads\UAV-DETR\tools\visualization", "ablation_results_grid.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print("=" * 50)
    print(f"Grid plot successfully generated and saved to: {save_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
