# 数据集目录约定

本仓库不包含实际图片和标注。请在本地创建 `datasets/` 目录，并按下面的结构整理数据。

```text
datasets/
  _reports/
    dataset_summary.json
  <class_name>/
    raw/
    prepared/
    meta/
  camrec_all_classes/
    images/{train,val,test}/
    labels/{train,val,test}/
    dataset.yaml
  camrec_all_classes_balanced/
    images/{train,val,test}/
    labels/{train,val,test}/
    dataset.yaml
```

## 预留与当前训练类别

当前默认训练类别：

- `person`
- `garbage`
- `luggage`
- `backpack`
- `handbag`

以下类别仅作目录预留，默认不纳入训练：

- `door_open` / `door_close`
- `car_door_open` / `car_door_close`
- `car_window_open` / `car_window_close`

门和车门的开/关识别只适合特征非常明显的情况，半开或复杂视角可能误差较大。

## 标注格式

每张图片对应一个同名 `.txt` 标签文件，每行一个目标：

```text
<class_id> <x_center> <y_center> <width> <height>
```

坐标均为归一化到 `[0, 1]` 的浮点数。默认类别映射：

- `person -> 0`
- `garbage -> 1`
- `luggage -> 2`
- `backpack -> 3`
- `handbag -> 4`

配置模板见 `configs/dataset.yaml`。`dataset.yaml` 中的 `path` 请使用相对路径，不要写入本机用户目录等绝对路径。

## 数据整理建议

- 将不同来源的标签统一映射到上述类别。
- 删除空图、损坏图与无法解析的标注。
- 对同一图像避免重复框和冲突类别。
- 对共享图像优先保留多类联合标注，不要拆成多个单类副本。
- 划分建议为 `train/val/test = 8:1:1`，并保证各分割都覆盖主要类别。

使用公开数据（例如 COCO、Open Images、TACO）前，请确认许可证允许训练用途。

## 常用脚本

```powershell
.\.venv\Scripts\python .\scripts\normalize_datasets.py
.\.venv\Scripts\python .\scripts\prepare_coco_replacements.py
.\.venv\Scripts\python .\scripts\build_all_class_dataset.py
.\.venv\Scripts\python .\scripts\build_balanced_dataset.py
```
