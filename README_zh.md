# ComfyUI DepthAnythingAccel

在 ComfyUI 中以原生分辨率运行 **Depth Anything V2 / Distill-Any-Depth** 深度估计的节点包
——同一套管线,可跑在 **PyTorch / ONNX Runtime / TensorRT** 三种后端上。

英文文档见 [README.md](README.md)。

## 功能简介

- **原生分辨率**：输入不会被缩放成 518×518,只按 14 像素 patch 对齐;深度图最后
  会还原回原始图片尺寸。为避免显存耗尽，PyTorch 与 ONNX 节点会拒绝超过约 2.6 MP
  (FP32)或 4.7 MP(FP16/BF16)的输入，请先缩小图片;TensorRT 则受引擎的
  `Max H` / `Max W` 限制。
- **三种后端,一条流程**：同一模型可跑 PyTorch(参考)/ ONNX Runtime(CUDA 或 CPU)/ TensorRT。
  节点可在图内直接串联:`Load → Export to ONNX → (Fuse) → (Build TensorRT) → Estimate`,
  转换进度输出到控制台;无需命令行、无需刷新。
- **DA2 与 DAD 用同一套节点**：Distill-Any-Depth 与 Depth Anything V2 架构一致,
  两者共用全部节点;encoder(small/base/large/giant)与家族(DA2/DAD)会从
  checkpoint 自动识别。
- **动态输入尺寸**：一个导出的 ONNX 模型可接受任意 14 对齐的 batch/H/W。TensorRT
  引擎接受 `Max H` / `Max W` 范围内的任意 H/W，batch 会逐张处理。
- **自动下载**：受支持的 checkpoint 缺失时,Load 节点会自动下载到
  `ComfyUI/models/depthanything/`。

实测速度(RTX 5070 Ti,FP16,784×1176):

| Encoder | PyTorch | ONNX Runtime(融合) | TensorRT |
| --- | ---: | ---: | ---: |
| Small(`vits`) | 29.1 ms | 20.5 ms | 7.5 ms |
| Base(`vitb`) | 51.0 ms | 37.4 ms | 16.5 ms |
| Large(`vitl`) | 127.3 ms | 93.5 ms | 47.3 ms |

TensorRT 约为 ONNX Runtime 的 2–2.7×,ONNX Runtime 约为 PyTorch 的 1.4×。
各后端输出与 FP32 参考高度一致(相关性 ≈ 1.0,归一化误差 ≈ 1e-3)。

## 安装

### 方式一：ComfyUI Manager（推荐）

在 ComfyUI Manager 中搜索 `DepthAnythingAccel` 安装。

### 方式二：手动安装

```powershell
git clone https://github.com/PineCookie/ComfyUI-DepthAnythingAccel "$env:ComfyUI\custom_nodes\ComfyUI-DepthAnythingAccel"
& "$env:ComfyUI\.venv\Scripts\python.exe" -m pip install -r "$env:ComfyUI\custom_nodes\ComfyUI-DepthAnythingAccel\requirements.txt"
```

### 可选后端

PyTorch 节点在任何能运行 ComfyUI 的机器上都可用，无需额外依赖。ONNX Runtime
与 TensorRT 为可选项，请按你的机器安装到 ComfyUI 所用的 Python 中：

| 机器 | ONNX Runtime | TensorRT |
| --- | --- | --- |
| NVIDIA 显卡 | `onnxruntime-gpu` | `tensorrt-cu12` 或 `tensorrt-cu13`，与 `torch.version.cuda` 一致 |
| Mac、AMD、Intel、纯 CPU | `onnxruntime`（在 CPU 上运行） | 不支持 |

- `onnxruntime` 与 `onnxruntime-gpu` 只能装一个，二者会互相覆盖。若其他节点包已安装
  `onnxruntime`，请先卸载它再安装 `onnxruntime-gpu`。
- `Export to ONNX` 与 `Fuse ONNX Attention` 需要 `onnx`（`onnxslim` 可选）。导出已在
  CUDA 与 CPU 上验证。
- PyTorch 加载器的 FP16/BF16 精度仅在 CUDA/ROCm 上生效，其他设备按 FP32 运行。

## 下载模型

权重不随本仓库分发。你可以手动下载，或在 Load 节点的下拉里选择受支持的 checkpoint——本地没有的会自动下载到 `ComfyUI/models/depthanything/`：

| 模型 | 来源 | 许可 |
| --- | --- | --- |
| DA2 Small | `Kijai/DepthAnythingV2-safetensors`（默认）或 `depth-anything/Depth-Anything-V2-Small`（`.pth`） | Apache-2.0 |
| DA2 Base / Large | `Kijai/DepthAnythingV2-safetensors`（默认）或 `depth-anything/Depth-Anything-V2-Base` / `-Large`（`.pth`） | CC-BY-NC-4.0，**非商业** |
| Distill-Any-Depth Small / Base / Large | `xingyang1/Distill-Any-Depth` | Apache-2.0（模型卡） |

> ⚠️ Kijai 的转换版本标注为 CC-BY-4.0，但它们由官方权重转换而来，Base / Large
> 仍应按非商业许可对待。使用或再分发前请阅读 [THIRD_PARTY.md](THIRD_PARTY.md)。

DA2 Giant 尚未公开发布。你自行放入 `ComfyUI/models/depthanything/` 的 `vitg`
checkpoint 会被自动识别并加载。

## 快速上手

1. 在节点面板找到 `DepthAnythingAccel` 分类。
2. 接入一张图片，连接：

```text
[Load Depth Model (PyTorch)] -> [Estimate Depth (PyTorch)] -> 保存图片
```

`Load` 节点只需选择模型文件(可选调整精度);`encoder` 与 DA2/DAD 来源都会从
checkpoint 自动检测。

## 想要更快？用 ONNX / TensorRT

转换节点**可直接串联**:每个转换节点接收上一个模型、返回可用的模型句柄,
全程无需刷新 ComfyUI。生成的文件也会写入 `ComfyUI/models/depthanything/`,
供 `Load ...` 节点日后直接复用。

**ONNX(约 1.4× 加速)**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Estimate Depth (ONNX)]
```

`Export to ONNX` 返回显式图模型;`Fuse ONNX Attention`(可选,更快)把它改写为
ORT 融合注意力图。若不打算构建 TensorRT 引擎,建议用融合后的句柄:

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Fuse ONNX Attention] -> [Estimate Depth (ONNX)]
```

**TensorRT(约 2–2.7×,最快)**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Build TensorRT Engine] -> [Estimate Depth (TensorRT)]
```

> TensorRT engine 必须从「显式(未融合)」的 ONNX 图构建;它的 `Max H` / `Max W`
> 决定了能处理的最大输入分辨率。
> 引擎与构建它的 TensorRT 版本及 GPU 绑定——升级 TensorRT 或更换显卡后请重建
> (加载器在版本不一致时会给出明确提示)。

## 节点一览

**PyTorch**

- `Load Depth Model (PyTorch)`
- `Estimate Depth (PyTorch)`

**ONNX Runtime**

- `Load Depth Model (ONNX)`
- `Estimate Depth (ONNX)`

**TensorRT**

- `Load Depth Model (TensorRT)`
- `Estimate Depth (TensorRT)`

**转换**

- `Export to ONNX` —— PyTorch 模型 → 显式 ONNX 模型句柄
- `Fuse ONNX Attention` —— 显式 ONNX → 融合 ONNX 模型句柄
- `Build TensorRT Engine` —— 显式 ONNX → TensorRT 引擎句柄

## 许可证

本仓库源代码采用 Apache License 2.0；见 [LICENSE](LICENSE)。模型权重由其各自所有者单独授权；见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 开发者

CLI 工具、测试与开发说明见 [DEVELOPMENT.md](DEVELOPMENT.md)。
