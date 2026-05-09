# UAV-DETR 项目文件结构与说明文档

本项目（UAV-DETR）是针对无人机（UAV）图像目标检测的端到端高效目标检测模型。以下是本项目根目录下各个主干脚本和核心文件的详细功能说明，按其所属的模块和用途进行了分类，方便您快速理解和开发。

---

## 🚀 1. 核心模型运行脚本
这一类脚本用于模型的训练、验证和前向预测，是日常使用中最常打交道的文件。这些文件处于项目的 **根目录** 下。

*   **`train.py`**
    *   **作用**：模型训练的入口程序。
    *   **详细**：负责初始化环境并且加载 `RTDETR` 模型（例如 `uavdetr-r50.yaml`），定义训练超参数（如 `batch`、`epochs`、`imgsz`、`workers` 等），启动对 VisDrone 或 UAVVaste 等数据集的训练过程。训练产生的结果全部保存在 `runs/train` 目录下。
*   **`test.py`**
    *   **作用**：模型评估与验证程序。
    *   **详细**：加载训练好的权重文件（`best.pt`），在指定的数据集（如验证集或测试集）上进行计算，输出精度指标（如 AP50 等），并可选择保存推断时的 JSON 预测结果以供进一步计算。
*   **`predict_demo.py`**
    *   **作用**：单图或多图的可视化预测与推理演示。
    *   **详细**：载入已训练的模型权重，对 `image_test` 或其他文件夹下的图片进行推理。提供置信度设定（`conf`）、画框粗细（`line_width`）等参数，预测画好框的图像会保存在 `runs/detect/predict` 目录中。

---

## 📊 2. 数据集格式转换脚本
各类目标检测数据集由于格式五花八门，本项目提供了一整套将其转换为 YOLO / COCO 格式的标准转换脚本。此类代码现已整理分类至 `tools/dataset/` 路径下。

*   **`tools/dataset/visdrone_to_yolo.py`** （原 `visdrone2yolo.py`）
    *   **作用**：将原始 VisDrone 数据集转换为 YOLO 格式。
    *   **详细**：解析原始的 txt 文件，将中心坐标和宽高转换为 YOLO 需要的 `(class, x_center, y_center, width, height)` 归一化格式，提取出主要的 10 类需要检测的目标。
*   **`tools/dataset/visdrone_to_coco.py` & `tools/dataset/yolo_to_coco.py`** （原 `visdrone2coco.py`、`yolo2coco.py`）
    *   **作用**：将 VisDrone 原格式 或 YOLO 格式转换为标准的 COCO JSON 标记格式。
    *   **详细**：这对于使用 COCO 官方 API 工具进行严格客观的指标评估至关重要。
*   **`tools/dataset/uavvaste_to_yolo.py`**
    *   **作用**：将 UAVVaste 数据集从 COCO(JSON) 格式转换为 YOLO 的 txt 格式。
    *   **详细**：同时会自动解析切分集配置并自动为你生成对应的 `uavvaste.yaml`，方便 YOLO/RT-DETR 引擎读取。
*   **`tools/dataset/uavvaste_split.py`** （原 `split_uavvaste.py`）
    *   **作用**：UAVVaste 专用数据集拆分脚本。
    *   **详细**：根据配套提供的 json 分布文件（区分了 Train/Val/Test），从包含全部图片标记的总表 json 中将需要的验证集干净地剥离并单独保存为标准格式。

---

## 🔧 3. 评测指标计算与JSON修复工具
YOLO 体系生成的 JSON 预测文件有时跟原始 COCO 标签在类别 ID 或 Image ID 的对齐上存在偏差，这些脚本用于修复这种偏差并调用 COCO API 进行公平衡量。此类代码现已分类至 `tools/metrics/` 路径下。

*   **`tools/metrics/get_coco_metrics.py`** （原 `get_COCO_metrice.py`）
    *   **作用**：计算 COCO 标准评价指标（AP, AP50, AP75 等）。
    *   **详细**：调用 `pycocotools.cocoeval`，需要输入真实的 JSON 标签和模型输出的预测 JSON。论文中的高精度性能指标一般均由该代码标准计算得出。
*   **`tools/metrics/fix_visdrone_json.py`** （原 `fix_json.py`）
    *   **作用**：修复模型在 VisDrone 数据集上预测输出的 JSON 文件。
    *   **详细**：主要用于解决由于文件命名、YOLO 输出标签从 0 开始而 COCO 可能从 1 开始导致的类别偏移问题（如将 `category_id` +1，转换图片为纯数字 ID 等）。
*   **`tools/metrics/fix_uavvaste_json.py`**
    *   **作用**：同上，但是专门致力于修复 UAVVaste 数据集的预测 JSON。

---

## 🎨 4. 高级图像可视化与对比画图工具
用于论文插图作图、效果对比等分析工具。此类脚本全部整理存放在 `tools/visualization/` 路径下。

*   **`tools/visualization/visualize_heatmap.py`** （原 `visualize_final.py`）
    *   **作用**：特征图（Heatmap）可视化生成器。
    *   **详细**：通过 PyTorch 钩子（Forward Hook）注册从底层取出网络中某一隐藏层的特征，进行均值等聚合操作后绘制其激活的热力图（Heatmap），直观呈现模型在何处投入了注意力。支持批量处理多张图和多个指定层（如 layer1, layer2...）。
*   **`tools/visualization/view_ground_truth.py`** （原 `view_gt.py`）
    *   **作用**：绘制特定图片的真实标签框（Ground Truth）。
    *   **详细**：提取原始标签（txt格式），并使用与模型预测时相同的配色盘在图片上将所有真实目标框完整画出，这通常用来跟模型的预测输出做纯净对比。
*   **`tools/visualization/draw_magnifier_tangent.py`** （原 `pinjie.py`）
    *   **作用**：放大镜/局部切线对比图生成器。
    *   **详细**：专为论文高质量配图编写。能够针对小目标密集区域绘制红圈，以三倍或更高倍率放大画出，并使用数学算法连接外切线构造完美的“放大镜”物理透视效果，最后将 Ground Truth 与 预测图 左右无缝拼接成横向对比图表。
*   **`tools/visualization/plot_eval_curve.py`** （原 `eva.py`）
    *   **作用**：绘制验证精度曲线（如 mAP@50）。
    *   **详细**：解析保存好结果的 CSV 文件（如 `results_r18.csv` 与 `results_r50.csv`），利用 `matplotlib` 生成训练过程中的精度直观对比和折线走势图。

---

## 📁 5. 项目核心目录
*   **`tools/`**：上述所有的各类辅助工具（数据处理、指标计算、制图）分类存放的新目录。
*   **`ultralytics/`**：核心模型库源代码。UAV-DETR 基于著名的 ultralytics 开发框架深度魔改并内建集成，里面的 `nn/`、`models/`、`engine/` 是真正的网络底层所在地。
*   **`dataset/`**：存放上述格式化操作整理后的数据集及 YAML 配置文件的归属地。
*   **`runs/`**：保存每一次训练（`train`）、验证（`val`）及预测（`detect`）产出的记录和最佳权重的输出地点。
*   **`image_test/`**：存放着用于测试模型的样例图片。
