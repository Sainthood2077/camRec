# CamRec

智能摄像头场景下的多类别目标检测训练项目，基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 与 PyTorch，可用于后续边缘设备部署。

## 当前类别

| ID | 名称 |
| --- | --- |
| 0 | person |
| 1 | garbage |
| 2 | luggage |
| 3 | backpack |
| 4 | handbag |

门、车门、车窗的开/关状态已预留类别，但尚未纳入默认训练。这类状态只适合识别非常明显的开/关特征，半开或复杂视角可能误差较大。

## 仓库范围

本仓库只包含训练、数据处理脚本、配置模板和文档。

以下内容不会进入版本库：

- 数据集（`datasets/`）
- 训练输出与模型权重（`runs/`、`*.pt` 等）
- 虚拟环境、缓存和编辑器文件
- 密钥、环境变量，以及个人或本地数据

预训练权重请自行从 Ultralytics 下载，例如 `yolo11n.pt` 或 `yolo11s.pt`。

## 环境准备

需要 Python 3.10+。如需 GPU 训练，请先按本机 CUDA 版本安装对应的 PyTorch，例如：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

检查 CUDA 是否可用：

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 数据准备

请在本地按 [数据集目录约定](docs/datasets.md) 整理 YOLO 格式数据。配置模板见 `configs/dataset.yaml`，复制后按实际数据位置修改，并保持 `path` 为相对路径。

```powershell
.\.venv\Scripts\python .\scripts\normalize_datasets.py
.\.venv\Scripts\python .\scripts\prepare_coco_replacements.py
.\.venv\Scripts\python .\scripts\build_all_class_dataset.py
.\.venv\Scripts\python .\scripts\build_balanced_dataset.py
```

## 训练

```powershell
.\.venv\Scripts\python .\scripts\train_yolo11n.py --data .\datasets\camrec_all_classes_balanced\dataset.yaml --name yolo11s_camrec_5cls --device 0 --imgsz 1024 --batch -1 --epochs 120 --workers 0
```

`--batch -1` 会由 Ultralytics 按显存自动选择批次大小。训练前如需把 NVIDIA GPU 调到更高功耗模式，可以运行 `.\scripts\gpu_max_performance.ps1`（部分选项可能需要管理员权限）。

## 实时检测

```powershell
.\.venv\Scripts\python .\scripts\rtsp_detect.py --source rtsp://<host>/<stream> --model .\runs\train\<run>\weights\best.pt --device 0
```

不要把带账号或密码的拉流地址写入代码，也不要提交到仓库。

## 许可证

本项目采用 [PolyForm Noncommercial License 1.0.0](LICENSE)。允许个人学习、研究、实验以及教育、慈善等非商业用途，**禁止商业使用**。

公开数据集和预训练权重遵循其各自的许可证，使用前请自行确认是否允许训练与再分发。

