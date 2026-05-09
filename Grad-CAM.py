import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import RTDETR

# ============ 配置 ============
RTDETR_WEIGHTS  = r'runs/train/rtdetr_r18/weights/best.pt'
UAVDETR_WEIGHTS = r'runs/train/r18/weights/best.pt'
IMG_PATH        = r'image_test/test00.jpg'

# 要对比的目标层（title: (rt层名, uav层名)）
TARGET_LAYERS = {
    'P3_SmallObj': ('model.19', 'model.19'),
    'P4_output':   ('model.22', 'model.23'),
}

IMG_SIZE = 640

# ============ Grad-CAM 核心类 ============
class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        named = dict(self.model.named_modules())
        if self.layer_name not in named:
            raise ValueError(f"找不到层: {self.layer_name}\n"
                             f"可用层: {[k for k in named if 'model.' in k][:20]}")
        target = named[self.layer_name]

        # 前向 hook：保存激活值
        def forward_hook(m, inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.activations = out  # [B, C, H, W]

        # 反向 hook：保存梯度
        def backward_hook(m, grad_in, grad_out):
            g = grad_out[0]
            if isinstance(g, (tuple, list)):
                g = g[0]
            self.gradients = g  # [B, C, H, W]

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))

    def compute(self, img_tensor, score_fn):
        """
        img_tensor: [1, 3, H, W], 已归一化
        score_fn:   接收模型输出，返回用于反向传播的标量
        """
        self.model.zero_grad()
        img_tensor = img_tensor.requires_grad_(True)

        # 前向
        output = self.model(img_tensor)

        # 计算目标标量并反向
        score = score_fn(output)
        score.backward()

        # Grad-CAM 计算
        grads = self.gradients        # [1, C, H, W]
        acts  = self.activations      # [1, C, H, W]

        # 全局平均池化梯度 → 权重
        weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # 加权求和 + ReLU
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()        # [H, W]

        # 归一化
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============ 图像预处理 ============
def preprocess(img_path, size=IMG_SIZE):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"图片不存在: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.
    return tensor.unsqueeze(0), img_bgr  # [1,3,H,W], 原图BGR


# ============ 目标得分函数 ============
def make_score_fn(model_wrapper):
    """
    对 RTDETRDetectionModel 的输出提取可微分得分。
    output 在 eval+no_grad 外调用时是 (pred_logits, pred_boxes) 的 tuple，
    或者是 ultralytics 的 Results 对象。
    这里直接 hook 内部 decoder 的 enc_score_head 输出。
    """
    score_storage = {}

    named = dict(model_wrapper.named_modules())

    # 优先用 enc_score_head：Encoder 对所有位置的分类得分，[B, HW, num_classes]
    # RT-DETR:  model.28.enc_score_head
    # UAV-DETR: model.28.enc_score_head  (层号可能是26或28，按你的结构)
    for key in ['model.28.enc_score_head', 'model.26.enc_score_head']:
        if key in named:
            def _hook(m, inp, out):
                score_storage['enc_score'] = out   # [B, N, C]
            named[key].register_forward_hook(_hook)
            print(f"  得分 hook 注册到: {key}")
            break

    def score_fn(output):
        if 'enc_score' in score_storage:
            # 取所有位置、所有类别的最大得分之和 → 标量
            s = score_storage['enc_score']   # [B, N, C]
            return s.sigmoid().max(dim=-1).values.sum()
        else:
            # fallback：如果 output 是 tensor
            if isinstance(output, torch.Tensor):
                return output.sigmoid().max(dim=-1).values.sum()
            raise RuntimeError("无法提取得分，请检查模型输出格式")

    return score_fn, score_storage


# ============ 叠加热力图到原图 ============
def overlay_cam(cam, img_bgr, alpha=0.5):
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, colored, alpha, 0)


# ============ 单模型 Grad-CAM 提取 ============
def run_gradcam(weights_path, img_tensor, layer_name):
    # 加载模型，必须 train 模式才能反向传播
    model_wrapper = RTDETR(weights_path)
    inner_model = model_wrapper.model
    inner_model.train()   # ← Grad-CAM 必须用 train 模式

    # 注册 Grad-CAM hook
    gcam = GradCAM(inner_model, layer_name)

    # 注册得分 hook
    score_fn, _ = make_score_fn(inner_model)

    # 计算 Grad-CAM
    cam = gcam.compute(img_tensor.clone(), score_fn)

    gcam.remove_hooks()
    return cam


# ============ 主流程 ============
def compare_gradcam(img_path):
    img_tensor, img_bgr = preprocess(img_path)

    n = len(TARGET_LAYERS)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = [axes]

    for i, (title, (rt_layer, uav_layer)) in enumerate(TARGET_LAYERS.items()):
        print(f"\n[{title}] 提取 RT-DETR ({rt_layer})...")
        cam_rt  = run_gradcam(RTDETR_WEIGHTS,  img_tensor, rt_layer)

        print(f"[{title}] 提取 UAV-DETR ({uav_layer})...")
        cam_uav = run_gradcam(UAVDETR_WEIGHTS, img_tensor, uav_layer)

        # 原图
        axes[i][0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title('Original', fontsize=12)
        axes[i][0].axis('off')

        # RT-DETR Grad-CAM
        vis_rt = overlay_cam(cam_rt, img_bgr)
        axes[i][1].imshow(cv2.cvtColor(vis_rt, cv2.COLOR_BGR2RGB))
        axes[i][1].set_title(f'RT-DETR Grad-CAM\n{title} ({rt_layer})', fontsize=10)
        axes[i][1].axis('off')

        # UAV-DETR Grad-CAM
        vis_uav = overlay_cam(cam_uav, img_bgr)
        axes[i][2].imshow(cv2.cvtColor(vis_uav, cv2.COLOR_BGR2RGB))
        axes[i][2].set_title(f'UAV-DETR Grad-CAM\n{title} ({uav_layer})', fontsize=10)
        axes[i][2].axis('off')

    plt.suptitle('Grad-CAM: RT-DETR vs UAV-DETR', fontsize=14, y=1.01)
    plt.tight_layout()
    out = 'gradcam_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n✅ 已保存 → {out}")
    plt.show()


if __name__ == '__main__':
    compare_gradcam(IMG_PATH)