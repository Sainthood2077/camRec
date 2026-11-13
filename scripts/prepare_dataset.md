# 公开数据集整理指南（智能摄像头多类别）

## 1. 目标
- 统一整理为 YOLO 数据格式。
- 面向智能摄像头场景维护多类别数据集。
- 当前有效类别：
  - `0: person`
  - `1: garbage`
  - `2: luggage`
  - `3: backpack`
  - `4: handbag`

## 2. 推荐公开数据来源（示例）
- Roboflow Universe：搜索 `person`、`garbage`、`luggage`、`backpack`、`handbag` 等目标检测数据集。
- Kaggle：搜索带 bbox 标注的行人、垃圾、箱包类数据集。
- COCO / Open Images：筛选 `person`、`backpack`、`handbag`、`suitcase` 等类别后再转换。

使用前请确认每个数据集许可证允许训练用途。

## 3. 目录目标
将最终数据放入：

```text
datasets/camrec_all_classes/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

## 4. 标注格式要求
- 每张图片对应一个同名 `.txt` 标签文件。
- 每行一个目标：

```text
<class_id> <x_center> <y_center> <width> <height>
```

- 坐标均为归一化到 `[0, 1]` 的浮点数。
- 当前类别映射：
  - `person -> 0`
  - `garbage -> 1`
  - `luggage -> 2`
  - `backpack -> 3`
  - `handbag -> 4`

## 5. 合并与清洗建议
- 将不同来源标签统一映射到上述类别。
- 删除空图、损坏图与无法解析标注。
- 对同一图像避免重复框和冲突类别。
- 对共享图像优先保留多类联合标注，不要拆成多个单类副本。

## 6. 划分建议
- 按 `8:1:1` 划分 `train/val/test`。
- 确保三份数据都覆盖主要类别，避免类别偏斜。

## 7. 训练前检查清单
- 图片与标签文件一一对应。
- 标签类别与当前映射一致。
- `datasets/camrec_all_classes/dataset.yaml` 路径正确。
- 先跑 1 个 epoch 验证流程后再长训。
