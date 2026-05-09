import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def set_font_style():
    """全局 Matplotlib 绘图风格设置"""
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

def get_actual_column_name(columns, target):
    """查找准确的列名，处理可能存在的空格或格式差异"""
    for col in columns:
        if col.strip() == target.strip():
            return col
            
    target_clean = target.replace('/', '').replace('_', '')
    for col in columns:
        if target_clean in col.replace('/', '').replace('_', ''):
            return col
    return None

def main():
    set_font_style()
    base_dir = r"c:\Users\20439\Downloads\UAV-DETR\result_csv"
    output_dir = r"c:\Users\20439\Downloads\UAV-DETR\tools\visualization\ablation_individual"
    
    # 创建输出子目录以避免文件杂乱
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = {
        "Baseline (r18)": os.path.join(base_dir, "results_r18.csv"),
        "w/o FD": os.path.join(base_dir, "results_r18_no_fd.csv"),
        "w/o MFFF": os.path.join(base_dir, "results_r18_no_mfff.csv"),
        "w/o SAC": os.path.join(base_dir, "results_r18_no_sac.csv")
    }

    metrics_list = [
        "train/giou_loss", "train/cls_loss", "train/l1_loss", 
        "metrics/precision(B)", "metrics/recall(B)",
        "val/giou_loss", "val/cls_loss", "val/l1_loss", 
        "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    ]

    dataframes = {}
    for label, path in csv_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            dataframes[label] = df

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

    print(f"Generating dual-view plots in: {output_dir}")
    print("-" * 50)
    
    # 为每个指标单独生成一张左右对比图
    for metric in metrics_list:
        # 清理文件名中不允许的字符（如斜杠）
        safe_metric_name = metric.replace('/', '_').replace('(', '_').replace(')', '')
        

        has_data = False
        empty_plot = True

        all_y_zoom = []
        
        # 定义全局统一的值
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
        fig.suptitle(f"{metric} Comparison (Ablation Study)", fontweight='bold', y=1.05)
        
        for label, df in dataframes.items():
            actual_col = get_actual_column_name(df.columns, metric)
            if actual_col:
                x_axis = df['epoch'] if 'epoch' in df.columns else df.index + 1
                y_data = df[actual_col]
                mask = y_data.notna()
                
                if mask.any():
                    empty_plot = False
                
                linestyle = '--' if 'Baseline' in label else '-'
                linewidth = 2.5 if 'Baseline' in label else 2.0
                alpha = 0.8 if 'Baseline' in label else 1.0
                color = colors.get(label, "tab:blue")
                
                # 绘制左侧全景图
                ax1.plot(x_axis[mask], y_data[mask], 
                         label=label, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                
                # 绘制右侧截长图 (epoch > start_epoch)
                mask_zoom = mask & (x_axis > start_epoch)
                ax2.plot(x_axis[mask_zoom], y_data[mask_zoom], 
                         label=label, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                
                # 保存放大区间的数值用于后续自动缩放Y轴
                zoom_data = y_data[mask_zoom]
                if len(zoom_data) > 0:
                    all_y_zoom.append(zoom_data)
                
                has_data = True

        if not empty_plot:
            # 配置左图 (0~400)
            ax1.set_title("Full Process (Epoch 0 - End)", fontsize=14)
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel(metric.split('/')[-1])
            ax1.autoscale(enable=True, axis='tight')
            ax1.legend(loc="best", frameon=True, fancybox=True, shadow=True)
            
            # 配置右图 (100~400+ 放大截取)
            ax2.set_title(f"Zoomed Process (Epoch > {start_epoch})", fontsize=14)
            ax2.set_xlabel('Epochs')
            ax2.autoscale(enable=True, axis='tight')
            
            # 动态调整右图的Y轴使得放大更加清晰
            if all_y_zoom:
                merged_y = pd.concat(all_y_zoom)
                y_min, y_max = merged_y.min(), merged_y.max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.05
                ax2.set_ylim(y_min - margin, y_max + margin)

            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"{safe_metric_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved: {safe_metric_name}.png")
        else:
            print(f"Skipped {metric} (No valid data found)")
            
        plt.close(fig) # 防止内存泄漏警告
        
    print("-" * 50)
    print("All individual plots generated.")

if __name__ == "__main__":
    main()
