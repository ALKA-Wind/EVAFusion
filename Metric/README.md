# 融合图像指标评测

统一入口：`Metric/evaluate.py`。共享计算接口：`Metric/metrics.py`。程序读取已保存的融合图像，不加载融合网络或奖励模型。

## 使用

当前项目环境已安装所需依赖，新环境执行：

```powershell
python -m pip install -r Metric/requirements.txt
```

单个数据集（目录中的图像按不含扩展名的文件名配对，例如 `001.bmp` 与 `001.png`）：

```powershell
.\.venv\Scripts\python.exe -m Metric.evaluate --ir-dir <红外目录> --vi-dir <可见光目录> --fused-dir <融合图目录> --output-dir outputs/metrics/custom
```

批量评测项目的推理输出 `outputs/fusion/<数据集名>/`，源图像根目录默认使用项目本地资源配置：

```powershell
.\.venv\Scripts\python.exe -m Metric.evaluate --fused-root outputs/fusion --datasets M3FD TNO RoadScene --output-dir outputs/metrics
```

也可以增加 `--dataset-root <源数据根目录>` 或 `--config configs/local.json`。一个运行对应一个方法，`--method` 设置方法名；评测其他方法时选择其融合图目录和新的输出目录。

仅计算指定指标：

```powershell
.\.venv\Scripts\python.exe -m Metric.evaluate --fused-root outputs/fusion --datasets TNO --metrics CC PSNR Qabf SSIM --output-dir outputs/metrics/core
```

无参考指标可以只提供融合图：

```powershell
.\.venv\Scripts\python.exe -m Metric.evaluate --fused-dir <融合图目录> --metrics NIQE BRISQUE EN SD SF AG --output-dir outputs/metrics/no_reference
```

也支持直接运行 `python Metric/evaluate.py`。所有参数可用 `--help` 查看。

## 指标与约定

默认计算论文六项：**CC、PSNR、Qabf、SSIM、NIQE、BRISQUE**。

| 指标 | 约定 | 方向 |
|---|---|---|
| CC | 融合图与两张源图的相关系数取均值；零方差相关系数约定为 0 | ↑ |
| PSNR | 默认两项 PSNR 的 dB 均值；见下方模式说明 | ↑ |
| Qabf | Sobel 梯度和方向保真度；无源梯度时为 0 | ↑ |
| SSIM | 与两张源图分别计算后取均值，skimage 默认 7×7 窗口 | ↑ |
| NIQE | BasicSR 官方自然图统计参数，96×96 无重叠块、两尺度 | ↓ |
| BRISQUE | PIQ 0.8.0 特征与官方 SVM 参数，本地离线计算 | ↓ |
| EN / SD / SF / AG | 熵（bit）、总体标准差、空间频率、平均梯度，仅用融合图 | ↑ |
| MI | 与两张源图的 256-bin 互信息之和（bit） | ↑ |
| SCD | `corr(F-IR,VI) + corr(F-VI,IR)` | ↑ |
| MSE | 两项均方误差取均值，再除以 `255²`，范围 [0,1] | ↓ |
| VIF | PIQ VIF，融合图作失真图、源图作参考，两项求和 | ↑ |
| MS_SSIM | PIQ 五尺度 MS-SSIM，两项取均值 | ↑ |

统一把文件解码成 RGB，再用 OpenCV 转 8-bit 灰度。计算时转换为浮点数，固定峰值 255；不通过自动缩放、裁剪或配准来掩盖图像尺寸不一致。NIQE 内部按照算法要求裁去不完整的 96×96 边缘块并做两尺度特征提取。

PSNR 提供三种明确定义，运行配置会记录选择：

- `--psnr-mode mean-db`（离线评测默认）：`(PSNR(F,IR)+PSNR(F,VI))/2`。
- `--psnr-mode mean-mse`：先平均两项 MSE，再计算 `10*log10(255²/MSE_mean)`。
- `--psnr-mode legacy-normalized-255`（`test_Fusion.py` 推理默认）：原实验 `DCEvo-main/Metric/Metric_torch.py::PSNR_function`。先将三张图转 float32 并除以 255，再平均两项归一化 MSE，计算 `20*log10(255/sqrt(MSE_mean))`，保留原始 Torch 运算。离线评测需要显式选择这一模式。

历史模式比 `mean-mse` 高约 `20*log10(255)=48.1308 dB`，因此可出现论文中约 60 dB 的量级；它与常规范围一致的 PSNR 含义不同。`mean-db` 与 `mean-mse` 一般也不相等。该兼容模式没有恢复旧 `test_Fusion.py` 中的 uint8 相减/平方溢出，其他指标和图像量化方式保持不变。Qabf 的梯度强度比仍为 `min/max`，相等的非零梯度比值为 1。

## 配对与无效结果

- 默认要求红外、可见光、融合图的文件名集合一致；扩展名和大小写可以不同，同一目录中重复 stem 会报错。
- `--allow-subset` 明确选择三者交集，并在 JSON 记录排除的文件名；`--limit N` 只评测排序后的前 N 个匹配样本，适合短检查。
- 默认继续处理单项指标无法定义的情况，失败项为 JSON `null` / CSV 空白，并记录图像、指标和原因；均值只使用有效项，报告有效数和无效数。
- NIQE 至少需要两个 96×96 块，且至少两个块有完整的非退化统计；小图或平坦图可能无法定义，不会擅自放大图像或补零分。
- BRISQUE 对退化统计可能无法定义；MS-SSIM 默认五尺度要求最小边长 161，VIF 使用 PIQ 的尺寸要求。此类失败同样记录原因。
- 完全一致图像的 PSNR 为正无穷。JSON 使用字符串 `"Infinity"` 表示，并统计 `positive_infinity_count`；不会把它混成无效分数。均值涉及无穷时标准差不定义。
- `--strict` 在存在指标/图像错误时返回非零退出码，但仍保存报告；文件配对或配置错误则直接报错。

## 输出

- `per_image.csv`：每个数据集、每张图的指标。
- `summary.csv`：均值、总体标准差、有效数、有限数、无效数和无穷数。
- `results.json`：完整逐图结果、异常原因、排除列表、解析后的输入路径和评测配置。

输出目录默认带时间戳；显式指定已有非空目录会报错，避免覆盖旧结果。NIQE 在 CPU 上运行，BRISQUE/VIF/MS-SSIM 可通过 `--device cuda` 选择可用 GPU。

## 实现来源与本次验证

- [BasicSR NIQE](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/metrics/niqe.py)，采用其统计参数与特征定义，增加退化统计错误记录，缩放使用 PIQ MATLAB 兼容实现；许可证在 `assets/BasicSR_LICENSE.txt`。
- [PIQ 0.8.0](https://github.com/photosynthesis-team/piq/tree/v0.8.0)，固定特征接口和 BRISQUE SVM 参数；模型来源和 SHA256 在 `assets/sources.json`。所有参数已随项目保存，评测时不下载模型。
- 真实 TNO 融合图 `001.png`（328×595）上，NIQE 为 **4.90858924**，未改动的 BasicSR 参考为 **4.90858561**，差异约 **3.6e-6**；BRISQUE 与 PIQ 参考均为 **36.9039917**。
- 已用一张真实图像完成全部 15 项计算及 CSV/JSON 导出。只运行了 3 项指标定向用例和已有推理入口检查；未重新运行训练测试或完整训练。
