
# 基线模型: RT-DETR-R18 模型结构分析
- **模型权重路径**: `runs/train/rtdetr_r18/weights/best.pt`

## 基线模型: RT-DETR-R18 完整网络结构
```text
RTDETRDetectionModel(
  (model): Sequential(
    (0): ConvNormLayer(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (norm): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): ReLU(inplace=True)
    )
    (1): ConvNormLayer(
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): ReLU(inplace=True)
    )
    (2): ConvNormLayer(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): ReLU(inplace=True)
    )
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (5): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (conv): ConvNormLayer(
              (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (6): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (conv): ConvNormLayer(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (7): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (conv): ConvNormLayer(
              (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (8): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): Identity()
    )
    (9): AIFI(
      (ma): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
      )
      (fc1): Linear(in_features=256, out_features=1024, bias=True)
      (fc2): Linear(in_features=1024, out_features=256, bias=True)
      (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0, inplace=False)
      (dropout1): Dropout(p=0, inplace=False)
      (dropout2): Dropout(p=0, inplace=False)
      (act): GELU(approximate='none')
    )
    (10): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (11): Upsample(scale_factor=2.0, mode='nearest')
    (12): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): Identity()
    )
    (13): Concat()
    (14): RepC3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (15): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (16): Upsample(scale_factor=2.0, mode='nearest')
    (17): Focus(
      (conv): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (18): Concat()
    (19): RepC3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (20): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (21): Concat()
    (22): RepC3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (23): Conv(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (24): Concat()
    (25): RepC3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (26): RTDETRDecoder(
      (input_proj): ModuleList(
        (0-2): 3 x Sequential(
          (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        )
      )
      (decoder): DeformableTransformerDecoder(
        (layers): ModuleList(
          (0-2): 3 x DeformableTransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
            )
            (dropout1): Dropout(p=0.0, inplace=False)
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (cross_attn): MSDeformAttn(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True)
              (attention_weights): Linear(in_features=256, out_features=96, bias=True)
              (value_proj): Linear(in_features=256, out_features=256, bias=True)
              (output_proj): Linear(in_features=256, out_features=256, bias=True)
            )
            (dropout2): Dropout(p=0.0, inplace=False)
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (linear1): Linear(in_features=256, out_features=1024, bias=True)
            (act): ReLU(inplace=True)
            (dropout3): Dropout(p=0.0, inplace=False)
            (linear2): Linear(in_features=1024, out_features=256, bias=True)
            (dropout4): Dropout(p=0.0, inplace=False)
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (denoising_class_embed): Embedding(11, 256)
      (query_pos_head): MLP(
        (layers): ModuleList(
          (0): Linear(in_features=4, out_features=512, bias=True)
          (1): Linear(in_features=512, out_features=256, bias=True)
        )
      )
      (enc_output): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
      (enc_score_head): Linear(in_features=256, out_features=10, bias=True)
      (enc_bbox_head): MLP(
        (layers): ModuleList(
          (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
          (2): Linear(in_features=256, out_features=4, bias=True)
        )
      )
      (dec_score_head): ModuleList(
        (0-2): 3 x Linear(in_features=256, out_features=10, bias=True)
      )
      (dec_bbox_head): ModuleList(
        (0-2): 3 x MLP(
          (layers): ModuleList(
            (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
            (2): Linear(in_features=256, out_features=4, bias=True)
          )
        )
      )
    )
  )
  (criterion): RTDETRDetectionLoss(
    (matcher): HungarianMatcher()
    (fl): FocalLoss()
    (vfl): VarifocalLoss()
  )
)
```

## 基线模型: RT-DETR-R18 可供提取特征的图层列表
这些是你可以填入 `TARGET_STAGES` 或 `TARGET_LAYERS` 字典中的键（如 'model.14'）。

| 模块名称 (Layer Name) | 模块类型 (Module Class) |
| --- | --- |
| `model.0` | ConvNormLayer |
| `model.0.conv` | Conv2d |
| `model.0.norm` | BatchNorm2d |
| `model.0.act` | ReLU |
| `model.1` | ConvNormLayer |
| `model.1.conv` | Conv2d |
| `model.1.norm` | BatchNorm2d |
| `model.1.act` | ReLU |
| `model.2` | ConvNormLayer |
| `model.2.conv` | Conv2d |
| `model.2.norm` | BatchNorm2d |
| `model.2.act` | ReLU |
| `model.3` | MaxPool2d |
| `model.4` | Blocks |
| `model.4.blocks` | ModuleList |
| `model.4.blocks.0` | BasicBlock |
| `model.4.blocks.0.short` | ConvNormLayer |
| `model.4.blocks.0.short.conv` | Conv2d |
| `model.4.blocks.0.short.norm` | BatchNorm2d |
| `model.4.blocks.0.short.act` | Identity |
| `model.4.blocks.0.branch2a` | ConvNormLayer |
| `model.4.blocks.0.branch2a.conv` | Conv2d |
| `model.4.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.4.blocks.0.branch2a.act` | ReLU |
| `model.4.blocks.0.branch2b` | ConvNormLayer |
| `model.4.blocks.0.branch2b.conv` | Conv2d |
| `model.4.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.4.blocks.0.branch2b.act` | Identity |
| `model.4.blocks.0.act` | ReLU |
| `model.4.blocks.1` | BasicBlock |
| `model.4.blocks.1.branch2a` | ConvNormLayer |
| `model.4.blocks.1.branch2a.conv` | Conv2d |
| `model.4.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.4.blocks.1.branch2a.act` | ReLU |
| `model.4.blocks.1.branch2b` | ConvNormLayer |
| `model.4.blocks.1.branch2b.conv` | Conv2d |
| `model.4.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.4.blocks.1.branch2b.act` | Identity |
| `model.4.blocks.1.act` | ReLU |
| `model.5` | Blocks |
| `model.5.blocks` | ModuleList |
| `model.5.blocks.0` | BasicBlock |
| `model.5.blocks.0.short` | Sequential |
| `model.5.blocks.0.short.pool` | AvgPool2d |
| `model.5.blocks.0.short.conv` | ConvNormLayer |
| `model.5.blocks.0.short.conv.conv` | Conv2d |
| `model.5.blocks.0.short.conv.norm` | BatchNorm2d |
| `model.5.blocks.0.short.conv.act` | Identity |
| `model.5.blocks.0.branch2a` | ConvNormLayer |
| `model.5.blocks.0.branch2a.conv` | Conv2d |
| `model.5.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.5.blocks.0.branch2a.act` | ReLU |
| `model.5.blocks.0.branch2b` | ConvNormLayer |
| `model.5.blocks.0.branch2b.conv` | Conv2d |
| `model.5.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.5.blocks.0.branch2b.act` | Identity |
| `model.5.blocks.0.act` | ReLU |
| `model.5.blocks.1` | BasicBlock |
| `model.5.blocks.1.branch2a` | ConvNormLayer |
| `model.5.blocks.1.branch2a.conv` | Conv2d |
| `model.5.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.5.blocks.1.branch2a.act` | ReLU |
| `model.5.blocks.1.branch2b` | ConvNormLayer |
| `model.5.blocks.1.branch2b.conv` | Conv2d |
| `model.5.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.5.blocks.1.branch2b.act` | Identity |
| `model.5.blocks.1.act` | ReLU |
| `model.6` | Blocks |
| `model.6.blocks` | ModuleList |
| `model.6.blocks.0` | BasicBlock |
| `model.6.blocks.0.short` | Sequential |
| `model.6.blocks.0.short.pool` | AvgPool2d |
| `model.6.blocks.0.short.conv` | ConvNormLayer |
| `model.6.blocks.0.short.conv.conv` | Conv2d |
| `model.6.blocks.0.short.conv.norm` | BatchNorm2d |
| `model.6.blocks.0.short.conv.act` | Identity |
| `model.6.blocks.0.branch2a` | ConvNormLayer |
| `model.6.blocks.0.branch2a.conv` | Conv2d |
| `model.6.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.6.blocks.0.branch2a.act` | ReLU |
| `model.6.blocks.0.branch2b` | ConvNormLayer |
| `model.6.blocks.0.branch2b.conv` | Conv2d |
| `model.6.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.6.blocks.0.branch2b.act` | Identity |
| `model.6.blocks.0.act` | ReLU |
| `model.6.blocks.1` | BasicBlock |
| `model.6.blocks.1.branch2a` | ConvNormLayer |
| `model.6.blocks.1.branch2a.conv` | Conv2d |
| `model.6.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.6.blocks.1.branch2a.act` | ReLU |
| `model.6.blocks.1.branch2b` | ConvNormLayer |
| `model.6.blocks.1.branch2b.conv` | Conv2d |
| `model.6.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.6.blocks.1.branch2b.act` | Identity |
| `model.6.blocks.1.act` | ReLU |
| `model.7` | Blocks |
| `model.7.blocks` | ModuleList |
| `model.7.blocks.0` | BasicBlock |
| `model.7.blocks.0.short` | Sequential |
| `model.7.blocks.0.short.pool` | AvgPool2d |
| `model.7.blocks.0.short.conv` | ConvNormLayer |
| `model.7.blocks.0.short.conv.conv` | Conv2d |
| `model.7.blocks.0.short.conv.norm` | BatchNorm2d |
| `model.7.blocks.0.short.conv.act` | Identity |
| `model.7.blocks.0.branch2a` | ConvNormLayer |
| `model.7.blocks.0.branch2a.conv` | Conv2d |
| `model.7.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.7.blocks.0.branch2a.act` | ReLU |
| `model.7.blocks.0.branch2b` | ConvNormLayer |
| `model.7.blocks.0.branch2b.conv` | Conv2d |
| `model.7.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.7.blocks.0.branch2b.act` | Identity |
| `model.7.blocks.0.act` | ReLU |
| `model.7.blocks.1` | BasicBlock |
| `model.7.blocks.1.branch2a` | ConvNormLayer |
| `model.7.blocks.1.branch2a.conv` | Conv2d |
| `model.7.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.7.blocks.1.branch2a.act` | ReLU |
| `model.7.blocks.1.branch2b` | ConvNormLayer |
| `model.7.blocks.1.branch2b.conv` | Conv2d |
| `model.7.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.7.blocks.1.branch2b.act` | Identity |
| `model.7.blocks.1.act` | ReLU |
| `model.8` | Conv |
| `model.8.conv` | Conv2d |
| `model.8.bn` | BatchNorm2d |
| `model.8.act` | Identity |
| `model.9` | AIFI |
| `model.9.ma` | MultiheadAttention |
| `model.9.ma.out_proj` | NonDynamicallyQuantizableLinear |
| `model.9.fc1` | Linear |
| `model.9.fc2` | Linear |
| `model.9.norm1` | LayerNorm |
| `model.9.norm2` | LayerNorm |
| `model.9.dropout` | Dropout |
| `model.9.dropout1` | Dropout |
| `model.9.dropout2` | Dropout |
| `model.9.act` | GELU |
| `model.10` | Conv |
| `model.10.conv` | Conv2d |
| `model.10.bn` | BatchNorm2d |
| `model.10.act` | SiLU |
| `model.11` | Upsample |
| `model.12` | Conv |
| `model.12.conv` | Conv2d |
| `model.12.bn` | BatchNorm2d |
| `model.12.act` | Identity |
| `model.13` | Concat |
| `model.14` | RepC3 |
| `model.14.cv1` | Conv |
| `model.14.cv1.conv` | Conv2d |
| `model.14.cv1.bn` | BatchNorm2d |
| `model.14.cv2` | Conv |
| `model.14.cv2.conv` | Conv2d |
| `model.14.cv2.bn` | BatchNorm2d |
| `model.14.m` | Sequential |
| `model.14.m.0` | RepConv |
| `model.14.m.0.act` | SiLU |
| `model.14.m.0.conv1` | Conv |
| `model.14.m.0.conv1.conv` | Conv2d |
| `model.14.m.0.conv1.bn` | BatchNorm2d |
| `model.14.m.0.conv1.act` | Identity |
| `model.14.m.0.conv2` | Conv |
| `model.14.m.0.conv2.conv` | Conv2d |
| `model.14.m.0.conv2.bn` | BatchNorm2d |
| `model.14.m.0.conv2.act` | Identity |
| `model.14.m.1` | RepConv |
| `model.14.m.1.conv1` | Conv |
| `model.14.m.1.conv1.conv` | Conv2d |
| `model.14.m.1.conv1.bn` | BatchNorm2d |
| `model.14.m.1.conv1.act` | Identity |
| `model.14.m.1.conv2` | Conv |
| `model.14.m.1.conv2.conv` | Conv2d |
| `model.14.m.1.conv2.bn` | BatchNorm2d |
| `model.14.m.1.conv2.act` | Identity |
| `model.14.m.2` | RepConv |
| `model.14.m.2.conv1` | Conv |
| `model.14.m.2.conv1.conv` | Conv2d |
| `model.14.m.2.conv1.bn` | BatchNorm2d |
| `model.14.m.2.conv1.act` | Identity |
| `model.14.m.2.conv2` | Conv |
| `model.14.m.2.conv2.conv` | Conv2d |
| `model.14.m.2.conv2.bn` | BatchNorm2d |
| `model.14.m.2.conv2.act` | Identity |
| `model.14.cv3` | Conv |
| `model.14.cv3.conv` | Conv2d |
| `model.14.cv3.bn` | BatchNorm2d |
| `model.15` | Conv |
| `model.15.conv` | Conv2d |
| `model.15.bn` | BatchNorm2d |
| `model.16` | Upsample |
| `model.17` | Focus |
| `model.17.conv` | Conv |
| `model.17.conv.conv` | Conv2d |
| `model.17.conv.bn` | BatchNorm2d |
| `model.18` | Concat |
| `model.19` | RepC3 |
| `model.19.cv1` | Conv |
| `model.19.cv1.conv` | Conv2d |
| `model.19.cv1.bn` | BatchNorm2d |
| `model.19.cv2` | Conv |
| `model.19.cv2.conv` | Conv2d |
| `model.19.cv2.bn` | BatchNorm2d |
| `model.19.m` | Sequential |
| `model.19.m.0` | RepConv |
| `model.19.m.0.conv1` | Conv |
| `model.19.m.0.conv1.conv` | Conv2d |
| `model.19.m.0.conv1.bn` | BatchNorm2d |
| `model.19.m.0.conv1.act` | Identity |
| `model.19.m.0.conv2` | Conv |
| `model.19.m.0.conv2.conv` | Conv2d |
| `model.19.m.0.conv2.bn` | BatchNorm2d |
| `model.19.m.0.conv2.act` | Identity |
| `model.19.m.1` | RepConv |
| `model.19.m.1.conv1` | Conv |
| `model.19.m.1.conv1.conv` | Conv2d |
| `model.19.m.1.conv1.bn` | BatchNorm2d |
| `model.19.m.1.conv1.act` | Identity |
| `model.19.m.1.conv2` | Conv |
| `model.19.m.1.conv2.conv` | Conv2d |
| `model.19.m.1.conv2.bn` | BatchNorm2d |
| `model.19.m.1.conv2.act` | Identity |
| `model.19.m.2` | RepConv |
| `model.19.m.2.conv1` | Conv |
| `model.19.m.2.conv1.conv` | Conv2d |
| `model.19.m.2.conv1.bn` | BatchNorm2d |
| `model.19.m.2.conv1.act` | Identity |
| `model.19.m.2.conv2` | Conv |
| `model.19.m.2.conv2.conv` | Conv2d |
| `model.19.m.2.conv2.bn` | BatchNorm2d |
| `model.19.m.2.conv2.act` | Identity |
| `model.19.cv3` | Conv |
| `model.19.cv3.conv` | Conv2d |
| `model.19.cv3.bn` | BatchNorm2d |
| `model.20` | Conv |
| `model.20.conv` | Conv2d |
| `model.20.bn` | BatchNorm2d |
| `model.21` | Concat |
| `model.22` | RepC3 |
| `model.22.cv1` | Conv |
| `model.22.cv1.conv` | Conv2d |
| `model.22.cv1.bn` | BatchNorm2d |
| `model.22.cv2` | Conv |
| `model.22.cv2.conv` | Conv2d |
| `model.22.cv2.bn` | BatchNorm2d |
| `model.22.m` | Sequential |
| `model.22.m.0` | RepConv |
| `model.22.m.0.conv1` | Conv |
| `model.22.m.0.conv1.conv` | Conv2d |
| `model.22.m.0.conv1.bn` | BatchNorm2d |
| `model.22.m.0.conv1.act` | Identity |
| `model.22.m.0.conv2` | Conv |
| `model.22.m.0.conv2.conv` | Conv2d |
| `model.22.m.0.conv2.bn` | BatchNorm2d |
| `model.22.m.0.conv2.act` | Identity |
| `model.22.m.1` | RepConv |
| `model.22.m.1.conv1` | Conv |
| `model.22.m.1.conv1.conv` | Conv2d |
| `model.22.m.1.conv1.bn` | BatchNorm2d |
| `model.22.m.1.conv1.act` | Identity |
| `model.22.m.1.conv2` | Conv |
| `model.22.m.1.conv2.conv` | Conv2d |
| `model.22.m.1.conv2.bn` | BatchNorm2d |
| `model.22.m.1.conv2.act` | Identity |
| `model.22.m.2` | RepConv |
| `model.22.m.2.conv1` | Conv |
| `model.22.m.2.conv1.conv` | Conv2d |
| `model.22.m.2.conv1.bn` | BatchNorm2d |
| `model.22.m.2.conv1.act` | Identity |
| `model.22.m.2.conv2` | Conv |
| `model.22.m.2.conv2.conv` | Conv2d |
| `model.22.m.2.conv2.bn` | BatchNorm2d |
| `model.22.m.2.conv2.act` | Identity |
| `model.22.cv3` | Conv |
| `model.22.cv3.conv` | Conv2d |
| `model.22.cv3.bn` | BatchNorm2d |
| `model.23` | Conv |
| `model.23.conv` | Conv2d |
| `model.23.bn` | BatchNorm2d |
| `model.24` | Concat |
| `model.25` | RepC3 |
| `model.25.cv1` | Conv |
| `model.25.cv1.conv` | Conv2d |
| `model.25.cv1.bn` | BatchNorm2d |
| `model.25.cv2` | Conv |
| `model.25.cv2.conv` | Conv2d |
| `model.25.cv2.bn` | BatchNorm2d |
| `model.25.m` | Sequential |
| `model.25.m.0` | RepConv |
| `model.25.m.0.conv1` | Conv |
| `model.25.m.0.conv1.conv` | Conv2d |
| `model.25.m.0.conv1.bn` | BatchNorm2d |
| `model.25.m.0.conv1.act` | Identity |
| `model.25.m.0.conv2` | Conv |
| `model.25.m.0.conv2.conv` | Conv2d |
| `model.25.m.0.conv2.bn` | BatchNorm2d |
| `model.25.m.0.conv2.act` | Identity |
| `model.25.m.1` | RepConv |
| `model.25.m.1.conv1` | Conv |
| `model.25.m.1.conv1.conv` | Conv2d |
| `model.25.m.1.conv1.bn` | BatchNorm2d |
| `model.25.m.1.conv1.act` | Identity |
| `model.25.m.1.conv2` | Conv |
| `model.25.m.1.conv2.conv` | Conv2d |
| `model.25.m.1.conv2.bn` | BatchNorm2d |
| `model.25.m.1.conv2.act` | Identity |
| `model.25.m.2` | RepConv |
| `model.25.m.2.conv1` | Conv |
| `model.25.m.2.conv1.conv` | Conv2d |
| `model.25.m.2.conv1.bn` | BatchNorm2d |
| `model.25.m.2.conv1.act` | Identity |
| `model.25.m.2.conv2` | Conv |
| `model.25.m.2.conv2.conv` | Conv2d |
| `model.25.m.2.conv2.bn` | BatchNorm2d |
| `model.25.m.2.conv2.act` | Identity |
| `model.25.cv3` | Conv |
| `model.25.cv3.conv` | Conv2d |
| `model.25.cv3.bn` | BatchNorm2d |
| `model.26` | RTDETRDecoder |
| `model.26.input_proj` | ModuleList |
| `model.26.input_proj.0` | Sequential |
| `model.26.input_proj.0.0` | Conv2d |
| `model.26.input_proj.0.1` | BatchNorm2d |
| `model.26.input_proj.1` | Sequential |
| `model.26.input_proj.1.0` | Conv2d |
| `model.26.input_proj.1.1` | BatchNorm2d |
| `model.26.input_proj.2` | Sequential |
| `model.26.input_proj.2.0` | Conv2d |
| `model.26.input_proj.2.1` | BatchNorm2d |
| `model.26.decoder` | DeformableTransformerDecoder |
| `model.26.decoder.layers` | ModuleList |
| `model.26.decoder.layers.0` | DeformableTransformerDecoderLayer |
| `model.26.decoder.layers.0.self_attn` | MultiheadAttention |
| `model.26.decoder.layers.0.self_attn.out_proj` | NonDynamicallyQuantizableLinear |
| `model.26.decoder.layers.0.dropout1` | Dropout |
| `model.26.decoder.layers.0.norm1` | LayerNorm |
| `model.26.decoder.layers.0.cross_attn` | MSDeformAttn |
| `model.26.decoder.layers.0.cross_attn.sampling_offsets` | Linear |
| `model.26.decoder.layers.0.cross_attn.attention_weights` | Linear |
| `model.26.decoder.layers.0.cross_attn.value_proj` | Linear |
| `model.26.decoder.layers.0.cross_attn.output_proj` | Linear |
| `model.26.decoder.layers.0.dropout2` | Dropout |
| `model.26.decoder.layers.0.norm2` | LayerNorm |
| `model.26.decoder.layers.0.linear1` | Linear |
| `model.26.decoder.layers.0.act` | ReLU |
| `model.26.decoder.layers.0.dropout3` | Dropout |
| `model.26.decoder.layers.0.linear2` | Linear |
| `model.26.decoder.layers.0.dropout4` | Dropout |
| `model.26.decoder.layers.0.norm3` | LayerNorm |
| `model.26.decoder.layers.1` | DeformableTransformerDecoderLayer |
| `model.26.decoder.layers.1.self_attn` | MultiheadAttention |
| `model.26.decoder.layers.1.self_attn.out_proj` | NonDynamicallyQuantizableLinear |
| `model.26.decoder.layers.1.dropout1` | Dropout |
| `model.26.decoder.layers.1.norm1` | LayerNorm |
| `model.26.decoder.layers.1.cross_attn` | MSDeformAttn |
| `model.26.decoder.layers.1.cross_attn.sampling_offsets` | Linear |
| `model.26.decoder.layers.1.cross_attn.attention_weights` | Linear |
| `model.26.decoder.layers.1.cross_attn.value_proj` | Linear |
| `model.26.decoder.layers.1.cross_attn.output_proj` | Linear |
| `model.26.decoder.layers.1.dropout2` | Dropout |
| `model.26.decoder.layers.1.norm2` | LayerNorm |
| `model.26.decoder.layers.1.linear1` | Linear |
| `model.26.decoder.layers.1.act` | ReLU |
| `model.26.decoder.layers.1.dropout3` | Dropout |
| `model.26.decoder.layers.1.linear2` | Linear |
| `model.26.decoder.layers.1.dropout4` | Dropout |
| `model.26.decoder.layers.1.norm3` | LayerNorm |
| `model.26.decoder.layers.2` | DeformableTransformerDecoderLayer |
| `model.26.decoder.layers.2.self_attn` | MultiheadAttention |
| `model.26.decoder.layers.2.self_attn.out_proj` | NonDynamicallyQuantizableLinear |
| `model.26.decoder.layers.2.dropout1` | Dropout |
| `model.26.decoder.layers.2.norm1` | LayerNorm |
| `model.26.decoder.layers.2.cross_attn` | MSDeformAttn |
| `model.26.decoder.layers.2.cross_attn.sampling_offsets` | Linear |
| `model.26.decoder.layers.2.cross_attn.attention_weights` | Linear |
| `model.26.decoder.layers.2.cross_attn.value_proj` | Linear |
| `model.26.decoder.layers.2.cross_attn.output_proj` | Linear |
| `model.26.decoder.layers.2.dropout2` | Dropout |
| `model.26.decoder.layers.2.norm2` | LayerNorm |
| `model.26.decoder.layers.2.linear1` | Linear |
| `model.26.decoder.layers.2.act` | ReLU |
| `model.26.decoder.layers.2.dropout3` | Dropout |
| `model.26.decoder.layers.2.linear2` | Linear |
| `model.26.decoder.layers.2.dropout4` | Dropout |
| `model.26.decoder.layers.2.norm3` | LayerNorm |
| `model.26.denoising_class_embed` | Embedding |
| `model.26.query_pos_head` | MLP |
| `model.26.query_pos_head.layers` | ModuleList |
| `model.26.query_pos_head.layers.0` | Linear |
| `model.26.query_pos_head.layers.1` | Linear |
| `model.26.enc_output` | Sequential |
| `model.26.enc_output.0` | Linear |
| `model.26.enc_output.1` | LayerNorm |
| `model.26.enc_score_head` | Linear |
| `model.26.enc_bbox_head` | MLP |
| `model.26.enc_bbox_head.layers` | ModuleList |
| `model.26.enc_bbox_head.layers.0` | Linear |
| `model.26.enc_bbox_head.layers.1` | Linear |
| `model.26.enc_bbox_head.layers.2` | Linear |
| `model.26.dec_score_head` | ModuleList |
| `model.26.dec_score_head.0` | Linear |
| `model.26.dec_score_head.1` | Linear |
| `model.26.dec_score_head.2` | Linear |
| `model.26.dec_bbox_head` | ModuleList |
| `model.26.dec_bbox_head.0` | MLP |
| `model.26.dec_bbox_head.0.layers` | ModuleList |
| `model.26.dec_bbox_head.0.layers.0` | Linear |
| `model.26.dec_bbox_head.0.layers.1` | Linear |
| `model.26.dec_bbox_head.0.layers.2` | Linear |
| `model.26.dec_bbox_head.1` | MLP |
| `model.26.dec_bbox_head.1.layers` | ModuleList |
| `model.26.dec_bbox_head.1.layers.0` | Linear |
| `model.26.dec_bbox_head.1.layers.1` | Linear |
| `model.26.dec_bbox_head.1.layers.2` | Linear |
| `model.26.dec_bbox_head.2` | MLP |
| `model.26.dec_bbox_head.2.layers` | ModuleList |
| `model.26.dec_bbox_head.2.layers.0` | Linear |
| `model.26.dec_bbox_head.2.layers.1` | Linear |
| `model.26.dec_bbox_head.2.layers.2` | Linear |

# 改进模型: UAV-DETR-R18 模型结构分析
- **模型权重路径**: `runs/train/r18/weights/best.pt`

## 改进模型: UAV-DETR-R18 完整网络结构
```text
RTDETRDetectionModel(
  (model): Sequential(
    (0): ConvNormLayer(
      (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (norm): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): ReLU(inplace=True)
    )
    (1): ConvNormLayer(
      (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): ReLU(inplace=True)
    )
    (2): ConvNormLayer(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): ReLU(inplace=True)
    )
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (5): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (conv): ConvNormLayer(
              (conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (6): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (conv): ConvNormLayer(
              (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (7): Blocks(
      (blocks): ModuleList(
        (0): BasicBlock(
          (short): Sequential(
            (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
            (conv): ConvNormLayer(
              (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
              (act): Identity()
            )
          )
          (branch2a): ConvNormLayer(
            (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (branch2a): ConvNormLayer(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): ReLU(inplace=True)
          )
          (branch2b): ConvNormLayer(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (norm): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): ReLU(inplace=True)
        )
      )
    )
    (8): Conv(
      (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): Identity()
    )
    (9): AIFI(
      (ma): MultiheadAttention(
        (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
      )
      (fc1): Linear(in_features=256, out_features=1024, bias=True)
      (fc2): Linear(in_features=1024, out_features=256, bias=True)
      (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0, inplace=False)
      (dropout1): Dropout(p=0, inplace=False)
      (dropout2): Dropout(p=0, inplace=False)
      (act): GELU(approximate='none')
    )
    (10): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (11): DySample(
      (offset): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
    )
    (12): Conv(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): Identity()
    )
    (13): Concat()
    (14): RepC3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (15): Conv(
      (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (16): DySample(
      (offset): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
    )
    (17): Focus(
      (conv): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (18): Concat()
    (19): MFFF(
      (cv1): Conv(
        (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(384, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): ImprovedFFTKernel(
        (in_conv): Sequential(
          (0): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1))
          (1): GELU(approximate='none')
        )
        (out_conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1))
        (dw_33): Conv2d(96, 96, kernel_size=(31, 31), stride=(1, 1), padding=(15, 15), groups=96)
        (dw_11): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), groups=96)
        (act): SiLU(inplace=True)
        (conv1x1): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1))
        (conv3x3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96)
        (conv5x5): Conv2d(96, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=96)
        (fac_conv): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1))
        (fac_pool): AdaptiveAvgPool2d(output_size=(1, 1))
        (ffm): FFM(
          (conv): Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96)
          (dwconv1): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1))
          (dwconv2): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1))
        )
        (channel_attention): Sequential(
          (0): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
          (3): Sigmoid()
        )
      )
    )
    (20): RepC3(
      (cv1): Conv(
        (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (21): FrequencyFocusedDownSampling(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (ffm): FFM(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        (dwconv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (dwconv2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv_reduce): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (conv_resize): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (22): Concat()
    (23): RepC3(
      (cv1): Conv(
        (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (24): FrequencyFocusedDownSampling(
      (cv1): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (ffm): FFM(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        (dwconv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (dwconv2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      )
      (conv_reduce): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (conv_resize): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (25): Concat()
    (26): RepC3(
      (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (1): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
        (2): RepConv(
          (act): SiLU(inplace=True)
          (conv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
        )
      )
      (cv3): Conv(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
    (27): SemanticAlignmenCalibration(
      (spatial_conv): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (semantic_conv): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (frequency_enhancer): FFM(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        (dwconv1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (dwconv2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (gating_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (offset_conv): Sequential(
        (0): Conv(
          (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (1): Conv2d(64, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    (28): RTDETRDecoder(
      (input_proj): ModuleList(
        (0-2): 3 x Sequential(
          (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        )
      )
      (decoder): DeformableTransformerDecoder(
        (layers): ModuleList(
          (0-2): 3 x DeformableTransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
            )
            (dropout1): Dropout(p=0.0, inplace=False)
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (cross_attn): MSDeformAttn(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True)
              (attention_weights): Linear(in_features=256, out_features=96, bias=True)
              (value_proj): Linear(in_features=256, out_features=256, bias=True)
              (output_proj): Linear(in_features=256, out_features=256, bias=True)
            )
            (dropout2): Dropout(p=0.0, inplace=False)
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (linear1): Linear(in_features=256, out_features=1024, bias=True)
            (act): ReLU(inplace=True)
            (dropout3): Dropout(p=0.0, inplace=False)
            (linear2): Linear(in_features=1024, out_features=256, bias=True)
            (dropout4): Dropout(p=0.0, inplace=False)
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (denoising_class_embed): Embedding(11, 256)
      (query_pos_head): MLP(
        (layers): ModuleList(
          (0): Linear(in_features=4, out_features=512, bias=True)
          (1): Linear(in_features=512, out_features=256, bias=True)
        )
      )
      (enc_output): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
      (enc_score_head): Linear(in_features=256, out_features=10, bias=True)
      (enc_bbox_head): MLP(
        (layers): ModuleList(
          (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
          (2): Linear(in_features=256, out_features=4, bias=True)
        )
      )
      (dec_score_head): ModuleList(
        (0-2): 3 x Linear(in_features=256, out_features=10, bias=True)
      )
      (dec_bbox_head): ModuleList(
        (0-2): 3 x MLP(
          (layers): ModuleList(
            (0-1): 2 x Linear(in_features=256, out_features=256, bias=True)
            (2): Linear(in_features=256, out_features=4, bias=True)
          )
        )
      )
    )
  )
  (criterion): RTDETRDetectionLoss(
    (matcher): HungarianMatcher()
    (fl): FocalLoss()
    (vfl): VarifocalLoss()
  )
)
```

## 改进模型: UAV-DETR-R18 可供提取特征的图层列表
这些是你可以填入 `TARGET_STAGES` 或 `TARGET_LAYERS` 字典中的键（如 'model.14'）。

| 模块名称 (Layer Name) | 模块类型 (Module Class) |
| --- | --- |
| `model.0` | ConvNormLayer |
| `model.0.conv` | Conv2d |
| `model.0.norm` | BatchNorm2d |
| `model.0.act` | ReLU |
| `model.1` | ConvNormLayer |
| `model.1.conv` | Conv2d |
| `model.1.norm` | BatchNorm2d |
| `model.1.act` | ReLU |
| `model.2` | ConvNormLayer |
| `model.2.conv` | Conv2d |
| `model.2.norm` | BatchNorm2d |
| `model.2.act` | ReLU |
| `model.3` | MaxPool2d |
| `model.4` | Blocks |
| `model.4.blocks` | ModuleList |
| `model.4.blocks.0` | BasicBlock |
| `model.4.blocks.0.short` | ConvNormLayer |
| `model.4.blocks.0.short.conv` | Conv2d |
| `model.4.blocks.0.short.norm` | BatchNorm2d |
| `model.4.blocks.0.short.act` | Identity |
| `model.4.blocks.0.branch2a` | ConvNormLayer |
| `model.4.blocks.0.branch2a.conv` | Conv2d |
| `model.4.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.4.blocks.0.branch2a.act` | ReLU |
| `model.4.blocks.0.branch2b` | ConvNormLayer |
| `model.4.blocks.0.branch2b.conv` | Conv2d |
| `model.4.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.4.blocks.0.branch2b.act` | Identity |
| `model.4.blocks.0.act` | ReLU |
| `model.4.blocks.1` | BasicBlock |
| `model.4.blocks.1.branch2a` | ConvNormLayer |
| `model.4.blocks.1.branch2a.conv` | Conv2d |
| `model.4.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.4.blocks.1.branch2a.act` | ReLU |
| `model.4.blocks.1.branch2b` | ConvNormLayer |
| `model.4.blocks.1.branch2b.conv` | Conv2d |
| `model.4.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.4.blocks.1.branch2b.act` | Identity |
| `model.4.blocks.1.act` | ReLU |
| `model.5` | Blocks |
| `model.5.blocks` | ModuleList |
| `model.5.blocks.0` | BasicBlock |
| `model.5.blocks.0.short` | Sequential |
| `model.5.blocks.0.short.pool` | AvgPool2d |
| `model.5.blocks.0.short.conv` | ConvNormLayer |
| `model.5.blocks.0.short.conv.conv` | Conv2d |
| `model.5.blocks.0.short.conv.norm` | BatchNorm2d |
| `model.5.blocks.0.short.conv.act` | Identity |
| `model.5.blocks.0.branch2a` | ConvNormLayer |
| `model.5.blocks.0.branch2a.conv` | Conv2d |
| `model.5.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.5.blocks.0.branch2a.act` | ReLU |
| `model.5.blocks.0.branch2b` | ConvNormLayer |
| `model.5.blocks.0.branch2b.conv` | Conv2d |
| `model.5.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.5.blocks.0.branch2b.act` | Identity |
| `model.5.blocks.0.act` | ReLU |
| `model.5.blocks.1` | BasicBlock |
| `model.5.blocks.1.branch2a` | ConvNormLayer |
| `model.5.blocks.1.branch2a.conv` | Conv2d |
| `model.5.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.5.blocks.1.branch2a.act` | ReLU |
| `model.5.blocks.1.branch2b` | ConvNormLayer |
| `model.5.blocks.1.branch2b.conv` | Conv2d |
| `model.5.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.5.blocks.1.branch2b.act` | Identity |
| `model.5.blocks.1.act` | ReLU |
| `model.6` | Blocks |
| `model.6.blocks` | ModuleList |
| `model.6.blocks.0` | BasicBlock |
| `model.6.blocks.0.short` | Sequential |
| `model.6.blocks.0.short.pool` | AvgPool2d |
| `model.6.blocks.0.short.conv` | ConvNormLayer |
| `model.6.blocks.0.short.conv.conv` | Conv2d |
| `model.6.blocks.0.short.conv.norm` | BatchNorm2d |
| `model.6.blocks.0.short.conv.act` | Identity |
| `model.6.blocks.0.branch2a` | ConvNormLayer |
| `model.6.blocks.0.branch2a.conv` | Conv2d |
| `model.6.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.6.blocks.0.branch2a.act` | ReLU |
| `model.6.blocks.0.branch2b` | ConvNormLayer |
| `model.6.blocks.0.branch2b.conv` | Conv2d |
| `model.6.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.6.blocks.0.branch2b.act` | Identity |
| `model.6.blocks.0.act` | ReLU |
| `model.6.blocks.1` | BasicBlock |
| `model.6.blocks.1.branch2a` | ConvNormLayer |
| `model.6.blocks.1.branch2a.conv` | Conv2d |
| `model.6.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.6.blocks.1.branch2a.act` | ReLU |
| `model.6.blocks.1.branch2b` | ConvNormLayer |
| `model.6.blocks.1.branch2b.conv` | Conv2d |
| `model.6.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.6.blocks.1.branch2b.act` | Identity |
| `model.6.blocks.1.act` | ReLU |
| `model.7` | Blocks |
| `model.7.blocks` | ModuleList |
| `model.7.blocks.0` | BasicBlock |
| `model.7.blocks.0.short` | Sequential |
| `model.7.blocks.0.short.pool` | AvgPool2d |
| `model.7.blocks.0.short.conv` | ConvNormLayer |
| `model.7.blocks.0.short.conv.conv` | Conv2d |
| `model.7.blocks.0.short.conv.norm` | BatchNorm2d |
| `model.7.blocks.0.short.conv.act` | Identity |
| `model.7.blocks.0.branch2a` | ConvNormLayer |
| `model.7.blocks.0.branch2a.conv` | Conv2d |
| `model.7.blocks.0.branch2a.norm` | BatchNorm2d |
| `model.7.blocks.0.branch2a.act` | ReLU |
| `model.7.blocks.0.branch2b` | ConvNormLayer |
| `model.7.blocks.0.branch2b.conv` | Conv2d |
| `model.7.blocks.0.branch2b.norm` | BatchNorm2d |
| `model.7.blocks.0.branch2b.act` | Identity |
| `model.7.blocks.0.act` | ReLU |
| `model.7.blocks.1` | BasicBlock |
| `model.7.blocks.1.branch2a` | ConvNormLayer |
| `model.7.blocks.1.branch2a.conv` | Conv2d |
| `model.7.blocks.1.branch2a.norm` | BatchNorm2d |
| `model.7.blocks.1.branch2a.act` | ReLU |
| `model.7.blocks.1.branch2b` | ConvNormLayer |
| `model.7.blocks.1.branch2b.conv` | Conv2d |
| `model.7.blocks.1.branch2b.norm` | BatchNorm2d |
| `model.7.blocks.1.branch2b.act` | Identity |
| `model.7.blocks.1.act` | ReLU |
| `model.8` | Conv |
| `model.8.conv` | Conv2d |
| `model.8.bn` | BatchNorm2d |
| `model.8.act` | Identity |
| `model.9` | AIFI |
| `model.9.ma` | MultiheadAttention |
| `model.9.ma.out_proj` | NonDynamicallyQuantizableLinear |
| `model.9.fc1` | Linear |
| `model.9.fc2` | Linear |
| `model.9.norm1` | LayerNorm |
| `model.9.norm2` | LayerNorm |
| `model.9.dropout` | Dropout |
| `model.9.dropout1` | Dropout |
| `model.9.dropout2` | Dropout |
| `model.9.act` | GELU |
| `model.10` | Conv |
| `model.10.conv` | Conv2d |
| `model.10.bn` | BatchNorm2d |
| `model.10.act` | SiLU |
| `model.11` | DySample |
| `model.11.offset` | Conv2d |
| `model.12` | Conv |
| `model.12.conv` | Conv2d |
| `model.12.bn` | BatchNorm2d |
| `model.12.act` | Identity |
| `model.13` | Concat |
| `model.14` | RepC3 |
| `model.14.cv1` | Conv |
| `model.14.cv1.conv` | Conv2d |
| `model.14.cv1.bn` | BatchNorm2d |
| `model.14.cv2` | Conv |
| `model.14.cv2.conv` | Conv2d |
| `model.14.cv2.bn` | BatchNorm2d |
| `model.14.m` | Sequential |
| `model.14.m.0` | RepConv |
| `model.14.m.0.act` | SiLU |
| `model.14.m.0.conv1` | Conv |
| `model.14.m.0.conv1.conv` | Conv2d |
| `model.14.m.0.conv1.bn` | BatchNorm2d |
| `model.14.m.0.conv1.act` | Identity |
| `model.14.m.0.conv2` | Conv |
| `model.14.m.0.conv2.conv` | Conv2d |
| `model.14.m.0.conv2.bn` | BatchNorm2d |
| `model.14.m.0.conv2.act` | Identity |
| `model.14.m.1` | RepConv |
| `model.14.m.1.conv1` | Conv |
| `model.14.m.1.conv1.conv` | Conv2d |
| `model.14.m.1.conv1.bn` | BatchNorm2d |
| `model.14.m.1.conv1.act` | Identity |
| `model.14.m.1.conv2` | Conv |
| `model.14.m.1.conv2.conv` | Conv2d |
| `model.14.m.1.conv2.bn` | BatchNorm2d |
| `model.14.m.1.conv2.act` | Identity |
| `model.14.m.2` | RepConv |
| `model.14.m.2.conv1` | Conv |
| `model.14.m.2.conv1.conv` | Conv2d |
| `model.14.m.2.conv1.bn` | BatchNorm2d |
| `model.14.m.2.conv1.act` | Identity |
| `model.14.m.2.conv2` | Conv |
| `model.14.m.2.conv2.conv` | Conv2d |
| `model.14.m.2.conv2.bn` | BatchNorm2d |
| `model.14.m.2.conv2.act` | Identity |
| `model.14.cv3` | Conv |
| `model.14.cv3.conv` | Conv2d |
| `model.14.cv3.bn` | BatchNorm2d |
| `model.15` | Conv |
| `model.15.conv` | Conv2d |
| `model.15.bn` | BatchNorm2d |
| `model.16` | DySample |
| `model.16.offset` | Conv2d |
| `model.17` | Focus |
| `model.17.conv` | Conv |
| `model.17.conv.conv` | Conv2d |
| `model.17.conv.bn` | BatchNorm2d |
| `model.18` | Concat |
| `model.19` | MFFF |
| `model.19.cv1` | Conv |
| `model.19.cv1.conv` | Conv2d |
| `model.19.cv1.bn` | BatchNorm2d |
| `model.19.cv2` | Conv |
| `model.19.cv2.conv` | Conv2d |
| `model.19.cv2.bn` | BatchNorm2d |
| `model.19.m` | ImprovedFFTKernel |
| `model.19.m.in_conv` | Sequential |
| `model.19.m.in_conv.0` | Conv2d |
| `model.19.m.in_conv.1` | GELU |
| `model.19.m.out_conv` | Conv2d |
| `model.19.m.dw_33` | Conv2d |
| `model.19.m.dw_11` | Conv2d |
| `model.19.m.act` | SiLU |
| `model.19.m.conv1x1` | Conv2d |
| `model.19.m.conv3x3` | Conv2d |
| `model.19.m.conv5x5` | Conv2d |
| `model.19.m.fac_conv` | Conv2d |
| `model.19.m.fac_pool` | AdaptiveAvgPool2d |
| `model.19.m.ffm` | FFM |
| `model.19.m.ffm.conv` | Conv2d |
| `model.19.m.ffm.dwconv1` | Conv2d |
| `model.19.m.ffm.dwconv2` | Conv2d |
| `model.19.m.channel_attention` | Sequential |
| `model.19.m.channel_attention.0` | Conv2d |
| `model.19.m.channel_attention.1` | ReLU |
| `model.19.m.channel_attention.2` | Conv2d |
| `model.19.m.channel_attention.3` | Sigmoid |
| `model.20` | RepC3 |
| `model.20.cv1` | Conv |
| `model.20.cv1.conv` | Conv2d |
| `model.20.cv1.bn` | BatchNorm2d |
| `model.20.cv2` | Conv |
| `model.20.cv2.conv` | Conv2d |
| `model.20.cv2.bn` | BatchNorm2d |
| `model.20.m` | Sequential |
| `model.20.m.0` | RepConv |
| `model.20.m.0.conv1` | Conv |
| `model.20.m.0.conv1.conv` | Conv2d |
| `model.20.m.0.conv1.bn` | BatchNorm2d |
| `model.20.m.0.conv1.act` | Identity |
| `model.20.m.0.conv2` | Conv |
| `model.20.m.0.conv2.conv` | Conv2d |
| `model.20.m.0.conv2.bn` | BatchNorm2d |
| `model.20.m.0.conv2.act` | Identity |
| `model.20.m.1` | RepConv |
| `model.20.m.1.conv1` | Conv |
| `model.20.m.1.conv1.conv` | Conv2d |
| `model.20.m.1.conv1.bn` | BatchNorm2d |
| `model.20.m.1.conv1.act` | Identity |
| `model.20.m.1.conv2` | Conv |
| `model.20.m.1.conv2.conv` | Conv2d |
| `model.20.m.1.conv2.bn` | BatchNorm2d |
| `model.20.m.1.conv2.act` | Identity |
| `model.20.m.2` | RepConv |
| `model.20.m.2.conv1` | Conv |
| `model.20.m.2.conv1.conv` | Conv2d |
| `model.20.m.2.conv1.bn` | BatchNorm2d |
| `model.20.m.2.conv1.act` | Identity |
| `model.20.m.2.conv2` | Conv |
| `model.20.m.2.conv2.conv` | Conv2d |
| `model.20.m.2.conv2.bn` | BatchNorm2d |
| `model.20.m.2.conv2.act` | Identity |
| `model.20.cv3` | Conv |
| `model.20.cv3.conv` | Conv2d |
| `model.20.cv3.bn` | BatchNorm2d |
| `model.21` | FrequencyFocusedDownSampling |
| `model.21.cv1` | Conv |
| `model.21.cv1.conv` | Conv2d |
| `model.21.cv1.bn` | BatchNorm2d |
| `model.21.cv2` | Conv |
| `model.21.cv2.conv` | Conv2d |
| `model.21.cv2.bn` | BatchNorm2d |
| `model.21.ffm` | FFM |
| `model.21.ffm.conv` | Conv2d |
| `model.21.ffm.dwconv1` | Conv2d |
| `model.21.ffm.dwconv2` | Conv2d |
| `model.21.conv_reduce` | Conv |
| `model.21.conv_reduce.conv` | Conv2d |
| `model.21.conv_reduce.bn` | BatchNorm2d |
| `model.21.conv_resize` | Conv |
| `model.21.conv_resize.conv` | Conv2d |
| `model.21.conv_resize.bn` | BatchNorm2d |
| `model.22` | Concat |
| `model.23` | RepC3 |
| `model.23.cv1` | Conv |
| `model.23.cv1.conv` | Conv2d |
| `model.23.cv1.bn` | BatchNorm2d |
| `model.23.cv2` | Conv |
| `model.23.cv2.conv` | Conv2d |
| `model.23.cv2.bn` | BatchNorm2d |
| `model.23.m` | Sequential |
| `model.23.m.0` | RepConv |
| `model.23.m.0.conv1` | Conv |
| `model.23.m.0.conv1.conv` | Conv2d |
| `model.23.m.0.conv1.bn` | BatchNorm2d |
| `model.23.m.0.conv1.act` | Identity |
| `model.23.m.0.conv2` | Conv |
| `model.23.m.0.conv2.conv` | Conv2d |
| `model.23.m.0.conv2.bn` | BatchNorm2d |
| `model.23.m.0.conv2.act` | Identity |
| `model.23.m.1` | RepConv |
| `model.23.m.1.conv1` | Conv |
| `model.23.m.1.conv1.conv` | Conv2d |
| `model.23.m.1.conv1.bn` | BatchNorm2d |
| `model.23.m.1.conv1.act` | Identity |
| `model.23.m.1.conv2` | Conv |
| `model.23.m.1.conv2.conv` | Conv2d |
| `model.23.m.1.conv2.bn` | BatchNorm2d |
| `model.23.m.1.conv2.act` | Identity |
| `model.23.m.2` | RepConv |
| `model.23.m.2.conv1` | Conv |
| `model.23.m.2.conv1.conv` | Conv2d |
| `model.23.m.2.conv1.bn` | BatchNorm2d |
| `model.23.m.2.conv1.act` | Identity |
| `model.23.m.2.conv2` | Conv |
| `model.23.m.2.conv2.conv` | Conv2d |
| `model.23.m.2.conv2.bn` | BatchNorm2d |
| `model.23.m.2.conv2.act` | Identity |
| `model.23.cv3` | Conv |
| `model.23.cv3.conv` | Conv2d |
| `model.23.cv3.bn` | BatchNorm2d |
| `model.24` | FrequencyFocusedDownSampling |
| `model.24.cv1` | Conv |
| `model.24.cv1.conv` | Conv2d |
| `model.24.cv1.bn` | BatchNorm2d |
| `model.24.cv2` | Conv |
| `model.24.cv2.conv` | Conv2d |
| `model.24.cv2.bn` | BatchNorm2d |
| `model.24.ffm` | FFM |
| `model.24.ffm.conv` | Conv2d |
| `model.24.ffm.dwconv1` | Conv2d |
| `model.24.ffm.dwconv2` | Conv2d |
| `model.24.conv_reduce` | Conv |
| `model.24.conv_reduce.conv` | Conv2d |
| `model.24.conv_reduce.bn` | BatchNorm2d |
| `model.24.conv_resize` | Conv |
| `model.24.conv_resize.conv` | Conv2d |
| `model.24.conv_resize.bn` | BatchNorm2d |
| `model.25` | Concat |
| `model.26` | RepC3 |
| `model.26.cv1` | Conv |
| `model.26.cv1.conv` | Conv2d |
| `model.26.cv1.bn` | BatchNorm2d |
| `model.26.cv2` | Conv |
| `model.26.cv2.conv` | Conv2d |
| `model.26.cv2.bn` | BatchNorm2d |
| `model.26.m` | Sequential |
| `model.26.m.0` | RepConv |
| `model.26.m.0.conv1` | Conv |
| `model.26.m.0.conv1.conv` | Conv2d |
| `model.26.m.0.conv1.bn` | BatchNorm2d |
| `model.26.m.0.conv1.act` | Identity |
| `model.26.m.0.conv2` | Conv |
| `model.26.m.0.conv2.conv` | Conv2d |
| `model.26.m.0.conv2.bn` | BatchNorm2d |
| `model.26.m.0.conv2.act` | Identity |
| `model.26.m.1` | RepConv |
| `model.26.m.1.conv1` | Conv |
| `model.26.m.1.conv1.conv` | Conv2d |
| `model.26.m.1.conv1.bn` | BatchNorm2d |
| `model.26.m.1.conv1.act` | Identity |
| `model.26.m.1.conv2` | Conv |
| `model.26.m.1.conv2.conv` | Conv2d |
| `model.26.m.1.conv2.bn` | BatchNorm2d |
| `model.26.m.1.conv2.act` | Identity |
| `model.26.m.2` | RepConv |
| `model.26.m.2.conv1` | Conv |
| `model.26.m.2.conv1.conv` | Conv2d |
| `model.26.m.2.conv1.bn` | BatchNorm2d |
| `model.26.m.2.conv1.act` | Identity |
| `model.26.m.2.conv2` | Conv |
| `model.26.m.2.conv2.conv` | Conv2d |
| `model.26.m.2.conv2.bn` | BatchNorm2d |
| `model.26.m.2.conv2.act` | Identity |
| `model.26.cv3` | Conv |
| `model.26.cv3.conv` | Conv2d |
| `model.26.cv3.bn` | BatchNorm2d |
| `model.27` | SemanticAlignmenCalibration |
| `model.27.spatial_conv` | Conv |
| `model.27.spatial_conv.conv` | Conv2d |
| `model.27.spatial_conv.bn` | BatchNorm2d |
| `model.27.semantic_conv` | Conv |
| `model.27.semantic_conv.conv` | Conv2d |
| `model.27.semantic_conv.bn` | BatchNorm2d |
| `model.27.frequency_enhancer` | FFM |
| `model.27.frequency_enhancer.conv` | Conv2d |
| `model.27.frequency_enhancer.dwconv1` | Conv2d |
| `model.27.frequency_enhancer.dwconv2` | Conv2d |
| `model.27.gating_conv` | Conv2d |
| `model.27.offset_conv` | Sequential |
| `model.27.offset_conv.0` | Conv |
| `model.27.offset_conv.0.conv` | Conv2d |
| `model.27.offset_conv.0.bn` | BatchNorm2d |
| `model.27.offset_conv.1` | Conv2d |
| `model.28` | RTDETRDecoder |
| `model.28.input_proj` | ModuleList |
| `model.28.input_proj.0` | Sequential |
| `model.28.input_proj.0.0` | Conv2d |
| `model.28.input_proj.0.1` | BatchNorm2d |
| `model.28.input_proj.1` | Sequential |
| `model.28.input_proj.1.0` | Conv2d |
| `model.28.input_proj.1.1` | BatchNorm2d |
| `model.28.input_proj.2` | Sequential |
| `model.28.input_proj.2.0` | Conv2d |
| `model.28.input_proj.2.1` | BatchNorm2d |
| `model.28.decoder` | DeformableTransformerDecoder |
| `model.28.decoder.layers` | ModuleList |
| `model.28.decoder.layers.0` | DeformableTransformerDecoderLayer |
| `model.28.decoder.layers.0.self_attn` | MultiheadAttention |
| `model.28.decoder.layers.0.self_attn.out_proj` | NonDynamicallyQuantizableLinear |
| `model.28.decoder.layers.0.dropout1` | Dropout |
| `model.28.decoder.layers.0.norm1` | LayerNorm |
| `model.28.decoder.layers.0.cross_attn` | MSDeformAttn |
| `model.28.decoder.layers.0.cross_attn.sampling_offsets` | Linear |
| `model.28.decoder.layers.0.cross_attn.attention_weights` | Linear |
| `model.28.decoder.layers.0.cross_attn.value_proj` | Linear |
| `model.28.decoder.layers.0.cross_attn.output_proj` | Linear |
| `model.28.decoder.layers.0.dropout2` | Dropout |
| `model.28.decoder.layers.0.norm2` | LayerNorm |
| `model.28.decoder.layers.0.linear1` | Linear |
| `model.28.decoder.layers.0.act` | ReLU |
| `model.28.decoder.layers.0.dropout3` | Dropout |
| `model.28.decoder.layers.0.linear2` | Linear |
| `model.28.decoder.layers.0.dropout4` | Dropout |
| `model.28.decoder.layers.0.norm3` | LayerNorm |
| `model.28.decoder.layers.1` | DeformableTransformerDecoderLayer |
| `model.28.decoder.layers.1.self_attn` | MultiheadAttention |
| `model.28.decoder.layers.1.self_attn.out_proj` | NonDynamicallyQuantizableLinear |
| `model.28.decoder.layers.1.dropout1` | Dropout |
| `model.28.decoder.layers.1.norm1` | LayerNorm |
| `model.28.decoder.layers.1.cross_attn` | MSDeformAttn |
| `model.28.decoder.layers.1.cross_attn.sampling_offsets` | Linear |
| `model.28.decoder.layers.1.cross_attn.attention_weights` | Linear |
| `model.28.decoder.layers.1.cross_attn.value_proj` | Linear |
| `model.28.decoder.layers.1.cross_attn.output_proj` | Linear |
| `model.28.decoder.layers.1.dropout2` | Dropout |
| `model.28.decoder.layers.1.norm2` | LayerNorm |
| `model.28.decoder.layers.1.linear1` | Linear |
| `model.28.decoder.layers.1.act` | ReLU |
| `model.28.decoder.layers.1.dropout3` | Dropout |
| `model.28.decoder.layers.1.linear2` | Linear |
| `model.28.decoder.layers.1.dropout4` | Dropout |
| `model.28.decoder.layers.1.norm3` | LayerNorm |
| `model.28.decoder.layers.2` | DeformableTransformerDecoderLayer |
| `model.28.decoder.layers.2.self_attn` | MultiheadAttention |
| `model.28.decoder.layers.2.self_attn.out_proj` | NonDynamicallyQuantizableLinear |
| `model.28.decoder.layers.2.dropout1` | Dropout |
| `model.28.decoder.layers.2.norm1` | LayerNorm |
| `model.28.decoder.layers.2.cross_attn` | MSDeformAttn |
| `model.28.decoder.layers.2.cross_attn.sampling_offsets` | Linear |
| `model.28.decoder.layers.2.cross_attn.attention_weights` | Linear |
| `model.28.decoder.layers.2.cross_attn.value_proj` | Linear |
| `model.28.decoder.layers.2.cross_attn.output_proj` | Linear |
| `model.28.decoder.layers.2.dropout2` | Dropout |
| `model.28.decoder.layers.2.norm2` | LayerNorm |
| `model.28.decoder.layers.2.linear1` | Linear |
| `model.28.decoder.layers.2.act` | ReLU |
| `model.28.decoder.layers.2.dropout3` | Dropout |
| `model.28.decoder.layers.2.linear2` | Linear |
| `model.28.decoder.layers.2.dropout4` | Dropout |
| `model.28.decoder.layers.2.norm3` | LayerNorm |
| `model.28.denoising_class_embed` | Embedding |
| `model.28.query_pos_head` | MLP |
| `model.28.query_pos_head.layers` | ModuleList |
| `model.28.query_pos_head.layers.0` | Linear |
| `model.28.query_pos_head.layers.1` | Linear |
| `model.28.enc_output` | Sequential |
| `model.28.enc_output.0` | Linear |
| `model.28.enc_output.1` | LayerNorm |
| `model.28.enc_score_head` | Linear |
| `model.28.enc_bbox_head` | MLP |
| `model.28.enc_bbox_head.layers` | ModuleList |
| `model.28.enc_bbox_head.layers.0` | Linear |
| `model.28.enc_bbox_head.layers.1` | Linear |
| `model.28.enc_bbox_head.layers.2` | Linear |
| `model.28.dec_score_head` | ModuleList |
| `model.28.dec_score_head.0` | Linear |
| `model.28.dec_score_head.1` | Linear |
| `model.28.dec_score_head.2` | Linear |
| `model.28.dec_bbox_head` | ModuleList |
| `model.28.dec_bbox_head.0` | MLP |
| `model.28.dec_bbox_head.0.layers` | ModuleList |
| `model.28.dec_bbox_head.0.layers.0` | Linear |
| `model.28.dec_bbox_head.0.layers.1` | Linear |
| `model.28.dec_bbox_head.0.layers.2` | Linear |
| `model.28.dec_bbox_head.1` | MLP |
| `model.28.dec_bbox_head.1.layers` | ModuleList |
| `model.28.dec_bbox_head.1.layers.0` | Linear |
| `model.28.dec_bbox_head.1.layers.1` | Linear |
| `model.28.dec_bbox_head.1.layers.2` | Linear |
| `model.28.dec_bbox_head.2` | MLP |
| `model.28.dec_bbox_head.2.layers` | ModuleList |
| `model.28.dec_bbox_head.2.layers.0` | Linear |
| `model.28.dec_bbox_head.2.layers.1` | Linear |
| `model.28.dec_bbox_head.2.layers.2` | Linear |
