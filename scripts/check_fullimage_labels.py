# -*- coding: utf-8 -*-
"""
检查垃圾数据集中的"全图标注"样本。
全图标注定义：标注框的宽高都接近整张图片（归一化 w 与 h 均接近 1.0），
即框基本覆盖全图，通常意味着标注是把整张图当成一个物体，属于异常/垃圾标注。
"""
import os
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = str(ROOT / 'datasets' / 'garbage')

# 阈值：归一化 w 和 h 都 >= 该值时视为"全图标注"
W_TH = 0.9
H_TH = 0.9

def scan_dataset(ds_path):
    """扫描一个数据集目录下所有 labels 里的 txt 标注文件"""
    results = []  # (label_file, [boxes])
    label_dirs = glob.glob(os.path.join(ds_path, "labels", "*"))
    if not label_dirs:
        return results
    for ld in label_dirs:
        for txt in glob.glob(os.path.join(ld, "*.txt")):
            boxes = []
            try:
                with open(txt, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cls = int(float(parts[0]))
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        boxes.append((cls, cx, cy, w, h))
            except Exception as e:
                print(f"  [WARN] 读取失败 {txt}: {e}")
                continue
            results.append((txt, boxes))
    return results

def main():
    # 找出 garbage 下所有数据集目录（含 labels 子目录的）
    datasets = set()
    for root, dirs, files in os.walk(BASE):
        if "labels" in dirs:
            # 该目录本身是数据集根
            datasets.add(root)
    # 去重：只要包含 labels 的目录
    datasets = sorted(datasets)

    print("=" * 80)
    print("发现包含 labels 的数据集目录：")
    for d in datasets:
        print("  ", d)
    print("=" * 80)

    report = {}
    for ds in datasets:
        ds_name = os.path.relpath(ds, BASE)
        entries = scan_dataset(ds)
        full = []       # 全图标注样本
        total_boxes = 0
        total_files = len(entries)
        for txt, boxes in entries:
            total_boxes += len(boxes)
            for (cls, cx, cy, w, h) in boxes:
                if w >= W_TH and h >= H_TH:
                    full.append((os.path.relpath(txt, ds), cls, cx, cy, w, h))
        report[ds_name] = {
            "total_files": total_files,
            "total_boxes": total_boxes,
            "fullimage_count": len(full),
            "fullimage_samples": full,
        }

    # 输出报告
    for ds_name, info in report.items():
        print("\n" + "#" * 80)
        print(f"数据集: {ds_name}")
        print(f"  标注文件数: {info['total_files']}")
        print(f"  标注框总数: {info['total_boxes']}")
        print(f"  全图标注样本数: {info['fullimage_count']}")
        if info['fullimage_samples']:
            print("  全图标注样本明细（文件 | cls | cx | cy | w | h）：")
            for txt, cls, cx, cy, w, h in info['fullimage_samples']:
                print(f"    {txt}  cls={cls} cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")

    # 写 json
    out = str(ROOT / 'datasets' / '_reports' / 'fullimage_labels_report.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out}")

if __name__ == "__main__":
    main()
