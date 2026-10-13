# ComfyUI DepthAnythingAccel

在 ComfyUI 中以原生分辨率运行 **Depth Anything V2 / Distill-Any-Depth** 深度估计的节点包
——同一套管线，可跑在 **PyTorch / ONNX Runtime / TensorRT** 三种后端上。

英文文档见 [README.md](README.md)。

## 功能简介

- **原生分辨率**：输入不会被缩放成 518×518，只按 14 像素 patch 对齐；深度图最后
  会还原回原始图片尺寸。
- **三种后端，一条流程**：同一模型可跑 PyTorch（参考）/ ONNX Runtime（CUDA 或 CPU）/
  TensorRT。转换在图内完成：`Load → Export to ONNX → (Fuse) → (Build TensorRT) → Estimate`，
  进度输出到控制台；无需命令行、无需刷新。
- **DA2 与 DAD 用同一套节点**：Distill-Any-Depth 与 Depth Anything V2 架构一致，
  两者共用全部节点；encoder（small/base/large/giant）与家族（DA2/DAD）会自动识别。
- **自动下载**：在 Load 节点中选择受支持的 checkpoint，缺失时会自动下载到
  `ComfyUI/models/depthanything/`。

实测速度（RTX 5070 Ti，FP16，784×1176）：

| Encoder | PyTorch | ONNX Runtime（融合） | TensorRT |
| --- | ---: | ---: | ---: |
| Small（`vits`） | 29.1 ms | 20.5 ms | 7.5 ms |
| Base（`vitb`） | 51.0 ms | 37.4 ms | 16.5 ms |
| Large（`vitl`） | 127.3 ms | 93.5 ms | 47.3 ms |

TensorRT 约为融合 ONNX Runtime 的 2–2.7×，融合 ONNX Runtime 约为 PyTorch 的 1.4×。
各后端输出与 FP32 参考高度一致（相关性 ≈ 1.0，归一化误差 ≈ 1e-3）。

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

## 模型

权重不随本仓库分发。在 Load 节点下拉中选择即可自动下载，也可以自行把文件放进
`ComfyUI/models/depthanything/`（支持子文件夹）。

| 模型 | 来源 | 许可 |
| --- | --- | --- |
| DA2 Small | `Kijai/DepthAnythingV2-safetensors`（默认）或 `depth-anything/Depth-Anything-V2-Small`（`.pth`） | Apache-2.0 |
| DA2 Base / Large | `Kijai/DepthAnythingV2-safetensors`（默认）或 `depth-anything/Depth-Anything-V2-Base` / `-Large`（`.pth`） | CC-BY-NC-4.0，**非商业** |
| Distill-Any-Depth Small / Base / Large | `xingyang1/Distill-Any-Depth` | Apache-2.0（模型卡） |

> ⚠️ Kijai 的转换版本标注为 CC-BY-4.0，但它们由官方权重转换而来，Base / Large
> 仍应按非商业许可对待。你导出的 ONNX 模型与引擎沿用其 checkpoint 的许可。
> 详见 [THIRD_PARTY.md](THIRD_PARTY.md)。

DA2 Giant 尚未公开发布；你自行提供的 `vitg` checkpoint 会被自动识别并加载。

## 快速上手

```text
[Load Image] -> [Estimate Depth (PyTorch)] -> [Save Image]
                         ^
        [Load Depth Model (PyTorch)]
```

节点位于 `Depth Anything Accel` 分类。选择模型文件（可选调整精度）即可；encoder 与
DA2/DAD 家族会从 checkpoint 自动识别。

## 更快：ONNX Runtime / TensorRT

转换节点**可直接串联**：每个节点接收上一个模型、返回可用的模型句柄，全程无需刷新
ComfyUI。生成的文件写入 `ComfyUI/models/depthanything/`，供 `Load ...` 节点日后复用。

**ONNX Runtime（约 1.4× 加速）**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Fuse ONNX Attention] -> [Estimate Depth (ONNX)]
```

`Fuse ONNX Attention` 把导出的图改写为 ONNX Runtime 的融合注意力，加速主要来自这一步。
它是可选的；未融合的模型也能运行，只是更慢。

**TensorRT（在此基础上再快约 2–2.7×，仅限 NVIDIA）**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Build TensorRT Engine] -> [Estimate Depth (TensorRT)]
```

> 引擎需从未融合的 ONNX 模型构建。引擎接受 `Max H` / `Max W` 以内的任意输入，并与构建它的
> TensorRT 版本及显卡绑定——升级 TensorRT 或更换显卡后请重建（由本节点包构建的引擎在
> 版本不一致时会在加载时提示）。

### 输入尺寸限制

为避免显存耗尽，PyTorch 与 ONNX 节点会拒绝超过约 2.6 MP（FP32）或 4.7 MP（FP16/BF16）
的输入，请先缩小图片；TensorRT 则受引擎的 `Max H` / `Max W` 限制。一个 batch 在
ONNX Runtime 上一次处理，在 PyTorch 与 TensorRT 上逐张处理。

## 节点一览

| 节点 | 作用 |
| --- | --- |
| `Load Depth Model (PyTorch)` / `Estimate Depth (PyTorch)` | 参考后端 |
| `Load Depth Model (ONNX)` / `Estimate Depth (ONNX)` | ONNX Runtime 后端 |
| `Load Depth Model (TensorRT)` / `Estimate Depth (TensorRT)` | TensorRT 后端 |
| `Export to ONNX` | PyTorch 模型 → ONNX 模型 |
| `Fuse ONNX Attention` | ONNX 模型 → 融合 ONNX 模型 |
| `Build TensorRT Engine` | 未融合 ONNX 模型 → TensorRT 引擎 |

## 许可证

本仓库源代码采用 Apache License 2.0；见 [LICENSE](LICENSE)。模型权重由其各自所有者
单独授权；见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 开发者

CLI 工具、测试与开发说明见 [DEVELOPMENT.md](DEVELOPMENT.md)。
