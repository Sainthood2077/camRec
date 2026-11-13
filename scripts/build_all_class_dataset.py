from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from normalize_datasets import CANONICAL_CLASSES, DATASETS_DIR, ROOT, ensure_dir, reset_dir


OUTPUT_ROOT = DATASETS_DIR / "camrec_all_classes"
SPLITS = ("train", "val", "test")


def load_summary() -> dict:
    summary_path = DATASETS_DIR / "_reports" / "dataset_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a merged YOLO dataset from prepared per-class datasets.")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep the full canonical label space even when some classes have no ready data.",
    )
    return parser.parse_args()


def write_dataset_yaml(dataset_root: Path, class_names: list[str]) -> None:
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    (dataset_root / "dataset.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=False, sort_keys=False),
        encoding="utf-8",
    )


def remap_labels(src_label: Path, dst_label: Path, target_class_id: int) -> int:
    written = 0
    lines_out: list[str] = []
    for raw_line in src_label.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        parts[0] = str(target_class_id)
        lines_out.append(" ".join(parts))
        written += 1
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return written


def materialize_image(src_image: Path, dst_image: Path) -> None:
    try:
        os.link(src_image, dst_image)
    except OSError:
        shutil.copy2(src_image, dst_image)


def initialize_output() -> None:
    reset_dir(OUTPUT_ROOT)
    for split in SPLITS:
        ensure_dir(OUTPUT_ROOT / "images" / split)
        ensure_dir(OUTPUT_ROOT / "labels" / split)


def get_ready_class_names(summary: dict, include_empty: bool) -> list[str]:
    ready_classes = {
        report["class_name"]
        for report in summary.get("class_reports", [])
        if report.get("status") == "ready"
    }
    return CANONICAL_CLASSES if include_empty else [name for name in CANONICAL_CLASSES if name in ready_classes]


def load_dataset_names(prepared_root: Path) -> list[str]:
    dataset_yaml = prepared_root / "dataset.yaml"
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    names = payload.get("names", [])
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def process_single_class_source(
    prepared_root: Path,
    class_name: str,
    class_names: list[str],
    source_prefix: str,
    merged_counts: defaultdict[str, dict[str, int]],
) -> None:
    target_class_id = class_names.index(class_name)
    for split in SPLITS:
        src_images_dir = prepared_root / "images" / split
        src_labels_dir = prepared_root / "labels" / split
        if not src_images_dir.exists() or not src_labels_dir.exists():
            continue

        for src_image in src_images_dir.iterdir():
            if not src_image.is_file():
                continue
            src_label = src_labels_dir / f"{src_image.stem}.txt"
            if not src_label.exists():
                continue

            dst_name = f"{source_prefix}_{src_image.name}"
            dst_image = OUTPUT_ROOT / "images" / split / dst_name
            dst_label = OUTPUT_ROOT / "labels" / split / f"{Path(dst_name).stem}.txt"
            materialize_image(src_image, dst_image)
            boxes_written = remap_labels(src_label, dst_label, target_class_id)
            merged_counts[class_name]["images"] += 1
            merged_counts[class_name]["boxes"] += boxes_written


def process_multiclass_source(
    prepared_root: Path,
    class_names: list[str],
    source_prefix: str,
    merged_counts: defaultdict[str, dict[str, int]],
) -> None:
    source_names = load_dataset_names(prepared_root)
    source_name_to_index = {name: index for index, name in enumerate(source_names)}
    source_index_to_target_index = {
        index: class_names.index(name)
        for name, index in source_name_to_index.items()
        if name in class_names
    }

    for split in SPLITS:
        src_images_dir = prepared_root / "images" / split
        src_labels_dir = prepared_root / "labels" / split
        if not src_images_dir.exists() or not src_labels_dir.exists():
            continue

        for src_image in src_images_dir.iterdir():
            if not src_image.is_file():
                continue
            src_label = src_labels_dir / f"{src_image.stem}.txt"
            if not src_label.exists():
                continue

            remapped_lines: list[str] = []
            classes_in_image: set[str] = set()
            for raw_line in src_label.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                source_class_id = int(parts[0])
                target_class_id = source_index_to_target_index.get(source_class_id)
                if target_class_id is None:
                    continue
                class_name = class_names[target_class_id]
                parts[0] = str(target_class_id)
                remapped_lines.append(" ".join(parts))
                classes_in_image.add(class_name)
                merged_counts[class_name]["boxes"] += 1

            if not remapped_lines:
                continue

            dst_name = f"{source_prefix}_{src_image.name}"
            dst_image = OUTPUT_ROOT / "images" / split / dst_name
            dst_label = OUTPUT_ROOT / "labels" / split / f"{Path(dst_name).stem}.txt"
            materialize_image(src_image, dst_image)
            dst_label.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
            for class_name in classes_in_image:
                merged_counts[class_name]["images"] += 1


def build_from_explicit_sources(summary: dict, include_empty: bool) -> dict:
    initialize_output()
    class_names = get_ready_class_names(summary, include_empty)
    if not class_names:
        raise RuntimeError("No ready datasets were found. Run the dataset preparation step first.")

    merged_counts = defaultdict(lambda: {"images": 0, "boxes": 0})
    for source in summary.get("build_sources", []):
        prepared_root = ROOT / Path(source["prepared_dataset"])
        source_prefix = str(source.get("source_name") or prepared_root.name)
        if source["kind"] == "multiclass":
            process_multiclass_source(prepared_root, class_names, source_prefix, merged_counts)
        elif source["kind"] == "single_class":
            process_single_class_source(prepared_root, str(source["class_name"]), class_names, source_prefix, merged_counts)
        else:
            raise RuntimeError(f"Unsupported build source kind: {source['kind']}")

    skipped_classes = [
        {"class_name": report["class_name"], "status": report["status"]}
        for report in summary.get("class_reports", [])
        if report["class_name"] not in class_names
    ]
    write_dataset_yaml(OUTPUT_ROOT, class_names)
    report = {
        "dataset_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "classes": class_names,
        "merged_counts": dict(merged_counts),
        "skipped_classes": skipped_classes,
    }
    (OUTPUT_ROOT / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_from_class_reports(summary: dict, include_empty: bool) -> dict:
    initialize_output()
    class_names = get_ready_class_names(summary, include_empty)
    if not class_names:
        raise RuntimeError("No ready datasets were found. Run the dataset preparation step first.")

    merged_counts = defaultdict(lambda: {"images": 0, "boxes": 0})
    skipped_classes: list[dict[str, str]] = []

    for report in summary.get("class_reports", []):
        class_name = report["class_name"]
        class_status = report["status"]
        if class_name not in class_names:
            skipped_classes.append({"class_name": class_name, "status": class_status})
            continue
        if class_status != "ready" or not report.get("prepared_dataset"):
            skipped_classes.append({"class_name": class_name, "status": class_status})
            continue

        prepared_root = ROOT / Path(report["prepared_dataset"])
        process_single_class_source(prepared_root, class_name, class_names, class_name, merged_counts)

    write_dataset_yaml(OUTPUT_ROOT, class_names)
    report = {
        "dataset_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "classes": class_names,
        "merged_counts": dict(merged_counts),
        "skipped_classes": skipped_classes,
    }
    (OUTPUT_ROOT / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    summary = load_summary()
    if summary.get("build_sources"):
        report = build_from_explicit_sources(summary, include_empty=args.include_empty)
    else:
        report = build_from_class_reports(summary, include_empty=args.include_empty)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
