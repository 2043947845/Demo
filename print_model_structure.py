from ultralytics import YOLO
import os

def export_model_layers_to_md(model_path, model_name, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n# {model_name} 模型结构分析\n")
        f.write(f"- **模型权重路径**: `{model_path}`\n\n")
        try:
            model = YOLO(model_path)
            
            # 1. 直接打印模型整体结构
            f.write(f"## {model_name} 完整网络结构\n")
            f.write("```text\n")
            f.write(str(model.model) + "\n")
            f.write("```\n\n")
            
            # 2. 打印所有包含的子模块名称（即我们在 target_layers 中填写的名称）
            f.write(f"## {model_name} 可供提取特征的图层列表\n")
            f.write("这些是你可以填入 `TARGET_STAGES` 或 `TARGET_LAYERS` 字典中的键（如 'model.14'）。\n\n")
            f.write("| 模块名称 (Layer Name) | 模块类型 (Module Class) |\n")
            f.write("| --- | --- |\n")
            for name, module in model.model.named_modules():
                if name.startswith('model.'):
                    f.write(f"| `{name}` | {module.__class__.__name__} |\n")
                    
        except Exception as e:
            f.write(f"❌ 加载 {model_name} 失败: {e}\n")

if __name__ == "__main__":
    MODEL_RT_DETR = 'runs/train/rtdetr_r18/weights/best.pt'
    MODEL_UAV_DETR = 'runs/train/r18/weights/best.pt'
    
    OUTPUT_MD = "model_structure_analysis.md"
    
    # 清空旧文件（如果存在）
    if os.path.exists(OUTPUT_MD):
        os.remove(OUTPUT_MD)
        
    export_model_layers_to_md(MODEL_RT_DETR, "基线模型: RT-DETR-R18", OUTPUT_MD)
    export_model_layers_to_md(MODEL_UAV_DETR, "改进模型: UAV-DETR-R18", OUTPUT_MD)
    
    print(f"✅ 模型结构已成功导出至: {os.path.abspath(OUTPUT_MD)}")
