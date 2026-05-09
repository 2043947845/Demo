import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import RTDETR

# ============ 配置 ============
RTDETR_WEIGHTS  = r'runs/train/rtdetr_r18/weights/best.pt'
UAVDETR_WEIGHTS = r'runs/train/r18/weights/best.pt'
IMG_PATH        = r'image_test/0000129_02411_d_0000138.jpg'

# # 每一行: '展示标题': (RT-DETR层名, UAV-DETR层名)
# COMPARE_LAYERS = {
#     'P5_Backbone_out': ('model.7',  'model.7'),
#     'AIFI_Encoder':    ('model.9',  'model.9'),
#     'P3_SmallObj':     ('model.19', 'model.19'),   # ⭐ 核心差异层
#     'P4_output':       ('model.23', 'model.23'),
# }
COMPARE_LAYERS = {
    'MSFF-FE': ('model.19',  'model.20'),
    'FD':    ('model.20',  'model.21'),
    'SAC':       ('model.25', 'model.27'),
}


# ============ Hook 工具 ============
class HookExtractor:
    def __init__(self):
        self.features = {}
        self._hooks = []

    def register(self, model, layer_name, save_key):
        """注册 hook，输出存到 self.features[save_key]"""
        named = dict(model.named_modules())
        if layer_name not in named:
            print(f"[警告] 找不到层: {layer_name}，已跳过")
            return
        def _hook(m, inp, out):
            # out 可能是 tensor 或 tuple，统一取第一个 tensor
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.features[save_key] = out.detach().cpu()
        self._hooks.append(named[layer_name].register_forward_hook(_hook))

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def to_heatmap(feat):
    """[B,C,H,W] 或 [C,H,W] → 归一化灰度热力图"""
def to_heatmap(feat):
    """[B,C,H,W] 或 [C,H,W] → 归一化灰度热力图"""
    if feat.dim() == 4:
        feat = feat[0]          # 取 batch=0
    
    # 使用由强到弱更公平、能保留尖锐激活的 L2 范数（或者也可选最大值）
    hm = feat.norm(dim=0).numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    # 通过 1.5 次方稍微压暗背景噪声，让高亮红点更加聚焦
    hm = np.power(hm, 1.5)
    return hm.astype(np.float32)

def overlay_heatmap(heatmap, img_bgr, alpha=0.55):
    h, w = img_bgr.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)


# ============ 单模型推理 + 特征提取 ============
def extract_features(weights_path, img_path, layer_name_dict):
    """
    layer_name_dict: { save_key: layer_name_str }
    返回: { save_key: feature_tensor }
    """
    model = RTDETR(weights_path)
    model.model.eval()

    extractor = HookExtractor()
    for save_key, layer_name in layer_name_dict.items():
        extractor.register(model.model, layer_name, save_key)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"读取图片失败: {img_path}")

    with torch.no_grad():
        model.predict(img_path, verbose=False)

    extractor.remove_all()
    
    # 打印已捕获的 key，方便调试
    print(f"[{weights_path}] 捕获到的层: {list(extractor.features.keys())}")
    return extractor.features, img_bgr


# ============ 对比可视化 & 保存 ============
def compare_and_save(img_path):
    # 分别构造两个模型要注册的 {save_key: layer_name} 字典
    rt_layer_dict  = {title + '_rt':  v[0] for title, v in COMPARE_LAYERS.items()}
    uav_layer_dict = {title + '_uav': v[1] for title, v in COMPARE_LAYERS.items()}

    rt_feats,  img = extract_features(RTDETR_WEIGHTS,  img_path, rt_layer_dict)
    uav_feats, _   = extract_features(UAVDETR_WEIGHTS, img_path, uav_layer_dict)

    n_rows = len(COMPARE_LAYERS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]   # 保证是二维列表

    for i, (title, (rt_layer, uav_layer)) in enumerate(COMPARE_LAYERS.items()):
        rt_key  = title + '_rt'
        uav_key = title + '_uav'

        # 原图
        axes[i][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title('Original', fontsize=12)
        axes[i][0].axis('off')

        # RT-DETR 热力图
        if rt_key in rt_feats:
            hm_rt = to_heatmap(rt_feats[rt_key])
            axes[i][1].imshow(cv2.cvtColor(overlay_heatmap(hm_rt, img), cv2.COLOR_BGR2RGB))
        else:
            axes[i][1].text(0.5, 0.5, 'Hook未捕获', ha='center', va='center')
        rt_title_str = f'RT-DETR Base ({rt_layer})\nBlurry/Diminished/Offset' if i == 0 else (f'RT-DETR Base ({rt_layer})' if i==1 else f'RT-DETR Base ({rt_layer})')
        if title == 'MSFF-FE': rt_title_str = f'RT-DETR Base ({rt_layer})\nBlurry blocks on small objects'
        if title == 'FD': rt_title_str = f'RT-DETR Base ({rt_layer})\nWeak activations (Diminished)'
        if title == 'SAC': rt_title_str = f'RT-DETR Base ({rt_layer})\nFocus offsets target center'
        
        axes[i][1].set_title(rt_title_str, fontsize=12)
        axes[i][1].axis('off')

        # UAV-DETR 热力图
        if uav_key in uav_feats:
            hm_uav = to_heatmap(uav_feats[uav_key])
            axes[i][2].imshow(cv2.cvtColor(overlay_heatmap(hm_uav, img), cv2.COLOR_BGR2RGB))
        else:
            axes[i][2].text(0.5, 0.5, 'Hook未捕获', ha='center', va='center')
            
        uav_title_str = f'UAV-DETR Ours ({uav_layer})'
        if title == 'MSFF-FE': uav_title_str = f'UAV-DETR Ours ({uav_layer})\nSharp, independent outlines'
        if title == 'FD': uav_title_str = f'UAV-DETR Ours ({uav_layer})\nStrong responses retained'
        if title == 'SAC': uav_title_str = f'UAV-DETR Ours ({uav_layer})\nPerfectly anchored center'
        
        axes[i][2].set_title(uav_title_str, fontsize=12, fontweight='bold', color='darkred')
        axes[i][2].axis('off')

        axes[i][0].set_ylabel(title, fontsize=14, fontweight='bold')

    plt.suptitle('Feature Heatmap: RT-DETR vs UAV-DETR', fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = 'heatmap_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存 → {out_path}")
    plt.show()


if __name__ == '__main__':
    compare_and_save(IMG_PATH)