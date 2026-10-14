# ComfyUI DepthAnythingAccel

在 ComfyUI 中以原生分辨率运行 **Depth Anything V2 / Distill-Any-Depth** 深度估计的节点包
——同一套管线，可跑在 **PyTorch / ONNX Runtime / TensorRT** 三种后端上。

英文文档见 [README.md](README.md)。

## 为什么选择本节点包

- **全分辨率深度**：深度节点通常默认以约 500 px 运行模型再放大结果。本节点包按原图
  分辨率运行，大图的细节得以保留在深度图中。
- **更快**：TensorRT 比 PyTorch 快 2.7–3.9×（见下表），且在工作流内用节点完成转换，
  无需命令行、无需重启。
- **DA2 与 Distill-Any-Depth 共用一套节点**：模型大小与家族会从 checkpoint 自动识别。
- **与其他模型共享显存**：ComfyUI 会像管理自带模型一样加载和释放 PyTorch 模型。
- **自动下载**：在 Load 节点中选择受支持的模型，首次使用时自动下载。

实测速度（RTX 5070 Ti，FP16，784×1176）：

| Encoder | PyTorch | ONNX Runtime（融合） | TensorRT |
| --- | ---: | ---: | ---: |
| Small（`vits`） | 29.1 ms | 20.5 ms | 7.5 ms |
| Base（`vitb`） | 51.0 ms | 37.4 ms | 16.5 ms |
| Large（`vitl`） | 127.3 ms | 93.5 ms | 47.3 ms |

TensorRT 约为融合 ONNX Runtime 的 2–2.7×，融合 ONNX Runtime 约为 PyTorch 的 1.4×。
各后端输出与 FP32 参考高度一致（相关性 ≈ 1.0，归一化误差 ≈ 1e-3）。

## 安装

**ComfyUI Manager（推荐）**：搜索 `DepthAnythingAccel` 并安装。

**手动安装**：

```powershell
git clone https://github.com/PineCookie/ComfyUI-DepthAnythingAccel "$env:ComfyUI\custom_nodes\ComfyUI-DepthAnythingAccel"
& "$env:ComfyUI\.venv\Scripts\python.exe" -m pip install -r "$env:ComfyUI\custom_nodes\ComfyUI-DepthAnythingAccel\requirements.txt"
```

PyTorch 节点只需这些。ONNX Runtime 与 TensorRT 为可选项，见下文「更快」一节。

## 快速上手

打开 ComfyUI 的工作流模板，在本节点包下选择 *Depth estimation - PyTorch*，或自行搭建：

```text
[Load Image] -> [Estimate Depth (PyTorch)] -> [Save Image]
                         ^
        [Load Depth Model (PyTorch)]
```

节点位于 `Depth Anything Accel` 分类。选择模型（可选调整精度）即可。示例使用 DA2
Small，体积小且允许商用；其他模型见「模型」一节。

## 更快：ONNX Runtime / TensorRT

转换节点可在图内直接串联：每个节点接收上一个模型并返回可直接使用的模型。转换后的
文件保存在 `ComfyUI/models/depthanything/`，之后可用 `Load Depth Model (ONNX)` /
`(TensorRT)` 节点直接加载，跳过转换。

**ONNX Runtime（约 1.4× 加速）**

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Fuse ONNX Attention] -> [Estimate Depth (ONNX)]
```

加速主要来自 `Fuse ONNX Attention`。它是可选的；未融合的模型也能运行，只是更慢。

**TensorRT（在此基础上再快约 2–2.7×，仅限 NVIDIA）** —— 模板 *Depth estimation - TensorRT*

```text
[Load Depth Model (PyTorch)] -> [Export to ONNX] -> [Build TensorRT Engine] -> [Estimate Depth (TensorRT)]
```

> 引擎需从未融合的 ONNX 模型构建。引擎接受 `Max H` / `Max W` 以内的任意输入，并与构建它的
> TensorRT 版本及显卡绑定——升级 TensorRT 或更换显卡后请重建（由本节点包构建的引擎在
> 版本不一致时会在加载时提示）。

### 安装可选后端

请按你的机器安装到 ComfyUI 所用的 Python 中：

| 机器 | ONNX Runtime | TensorRT |
| --- | --- | --- |
| NVIDIA 显卡 | `onnxruntime-gpu` | `tensorrt-cu12` 或 `tensorrt-cu13`，与 `torch.version.cuda` 一致 |
| Mac、AMD、Intel、纯 CPU | `onnxruntime`（在 CPU 上运行） | 不支持 |

- `onnxruntime` 与 `onnxruntime-gpu` 只能装一个，二者会互相覆盖。若其他节点包已安装
  `onnxruntime`，请先卸载它再安装 `onnxruntime-gpu`。
- `Export to ONNX` 与 `Fuse ONNX Attention` 需要 `onnx`（`onnxslim` 可选）。导出已在
  CUDA 与 CPU 上验证。

## 限制

- **图片尺寸**：为避免显存耗尽，PyTorch 与 ONNX 节点会拒绝超过约 2.6 MP（FP32）或
  4.7 MP（FP16/BF16）的输入，请先缩小图片；TensorRT 则受引擎的 `Max H` / `Max W` 限制。
- **尺寸取整**：宽高会向下取整到 14 的倍数（模型的 patch 大小），输出再缩放回原图尺寸。
- **Batch**：ONNX Runtime 一次处理整个 batch；PyTorch 与 TensorRT 逐张处理。
- **精度**：PyTorch 加载器的 FP16/BF16 仅在 CUDA/ROCm 上生效，其他设备按 FP32 运行。

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
