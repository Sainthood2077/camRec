from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from normalize_datasets import (
    CANONICAL_CLASSES,
    DATASETS_DIR,
    ROOT,
    SEED,
    SPLITS,
    ensure_dir,
    reset_dir,
    write_dataset_yaml,
    write_overall_report,
)


OPENIMAGES_ROOT = DATASETS_DIR / "_sources" / "openimages_v6_bags"
OPENIMAGES_PREPARED_ROOT = OPENIMAGES_ROOT / "prepared" / "yolo_people_bags_openimages"
OPENIMAGES_SHARED_CLASSES = ["person", "luggage", "backpack", "handbag"]
OPENIMAGES_BAG_CLASSES = ["luggage", "backpack", "handbag"]
OPENIMAGES_LABEL_NAME_MAP = {
    "Person": "person",
    "Suitcase": "luggage",
    "Backpack": "backpack",
    "Handbag": "handbag",
}
OPENIMAGES_CLASS_DESCRIPTION_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
OPENIMAGES_BBOX_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "val": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
}
OPENIMAGES_IMAGE_URL_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

MENDELEY_TRASH_ROOT = DATASETS_DIR / "_sources" / "mendeley_trash_detection"
MENDELEY_TRASH_ARCHIVE = MENDELEY_TRASH_ROOT / "downloads" / "trash_detection_dataset.zip"
MENDELEY_TRASH_EXTRACTED = MENDELEY_TRASH_ROOT / "extracted"
MENDELEY_TRASH_PREPARED = MENDELEY_TRASH_ROOT / "prepared" / "yolo_garbage_mendeley"
MENDELEY_TRASH_URL = "https://data.mendeley.com/public-api/zip/z732f9pwxt/download/1"
MENDELEY_TRASH_REFERER = "https://data.mendeley.com/datasets/z732f9pwxt/1"
HTTP_TIMEOUT_SECONDS = 30
HTTP_RETRY_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand CamRec public datasets with validated Open Images bag data and Mendeley garbage data."
    )
    parser.add_argument(
        "--oi-train-limit-per-class",
        type=int,
        default=600,
        help="Maximum Open Images train images to keep for each bag class.",
    )
    parser.add_argument(
        "--oi-val-limit-per-class",
        type=int,
        default=120,
        help="Maximum Open Images validation images to keep for each bag class.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Reserved for future use. Downloads currently run sequentially for stability.",
    )
    return parser.parse_args()


def load_summary() -> dict:
    summary_path = DATASETS_DIR / "_reports" / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Dataset summary not found: {summary_path}. Run scripts/prepare_coco_replacements.py first."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def open_url_with_retries(request: str | urllib.request.Request):
    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRY_COUNT + 1):
        try:
            return urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == HTTP_RETRY_COUNT:
                break
            wait_seconds = attempt * 2
            print(f"[network] retry {attempt}/{HTTP_RETRY_COUNT - 1} after {wait_seconds}s: {error}")
            time.sleep(wait_seconds)
    if last_error is not None:
        raise RuntimeError(f"Failed to open remote URL after {HTTP_RETRY_COUNT} attempts: {last_error}") from last_error
    raise RuntimeError("Failed to open remote URL for an unknown reason.")


def read_remote_text(url: str) -> io.TextIOWrapper:
    response = open_url_with_retries(url)
    return io.TextIOWrapper(response, encoding="utf-8", newline="")


def download_with_headers(url: str, destination: Path, headers: dict[str, str] | None = None) -> None:
    if destination.exists():
        return
    ensure_dir(destination.parent)
    request = urllib.request.Request(url, headers=headers or {})
    with open_url_with_retries(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_zip_once(archive_path: Path, target_dir: Path) -> None:
    marker = target_dir / ".extract_complete"
    if marker.exists():
        return
    reset_dir(target_dir)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)
    marker.write_text("ok\n", encoding="utf-8")


def load_openimages_label_ids() -> dict[str, str]:
    name_to_label_id: dict[str, str] = {}
    with read_remote_text(OPENIMAGES_CLASS_DESCRIPTION_URL) as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 2:
                continue
            label_id, class_name = row
            mapped_name = OPENIMAGES_LABEL_NAME_MAP.get(class_name)
            if mapped_name:
                name_to_label_id[mapped_name] = label_id
    missing = sorted(set(OPENIMAGES_SHARED_CLASSES) - set(name_to_label_id))
    if missing:
        raise RuntimeError(f"Open Images label IDs not found for: {missing}")
    return name_to_label_id


def should_skip_openimages_box(row: dict[str, str]) -> bool:
    return row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1"


def yolo_line_from_normalized_box(class_id: int, row: dict[str, str]) -> str | None:
    x_min = float(row["XMin"])
    x_max = float(row["XMax"])
    y_min = float(row["YMin"])
    y_max = float(row["YMax"])
    width = max(0.0, min(1.0, x_max) - max(0.0, x_min))
    height = max(0.0, min(1.0, y_max) - max(0.0, y_min))
    if width <= 1e-6 or height <= 1e-6:
        return None
    x_center = max(0.0, min(1.0, (x_min + x_max) / 2))
    y_center = max(0.0, min(1.0, (y_min + y_max) / 2))
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def finalize_openimages_record(
    records: list[dict[str, object]],
    record: dict[str, object] | None,
    bag_image_counts: dict[str, int],
    per_class_limit: int,
) -> bool:
    if not record:
        return False
    bag_classes = sorted(set(record["bag_classes"]))  # type: ignore[arg-type]
    if not bag_classes:
        return False
    if per_class_limit > 0 and all(bag_image_counts[class_name] >= per_class_limit for class_name in bag_classes):
        return False
    records.append(record)
    for class_name in bag_classes:
        bag_image_counts[class_name] += 1
    return per_class_limit > 0 and all(bag_image_counts[class_name] >= per_class_limit for class_name in OPENIMAGES_BAG_CLASSES)


def collect_openimages_records(
    split: str,
    bbox_url: str,
    label_ids: dict[str, str],
    per_class_limit: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    target_label_to_name = {label_id: class_name for class_name, label_id in label_ids.items()}
    class_to_id = {name: index for index, name in enumerate(OPENIMAGES_SHARED_CLASSES)}
    bag_image_counts = {class_name: 0 for class_name in OPENIMAGES_BAG_CLASSES}
    records: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    randomizer = random.Random(SEED)

    with read_remote_text(bbox_url) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_name = target_label_to_name.get(row.get("LabelName", ""))
            if label_name is None or should_skip_openimages_box(row):
                continue
            image_id = row["ImageID"]
            if current is None or current["image_id"] != image_id:
                should_stop = finalize_openimages_record(records, current, bag_image_counts, per_class_limit)
                if should_stop:
                    break
                current = {
                    "image_id": image_id,
                    "split": split,
                    "lines": [],
                    "classes": set(),
                    "bag_classes": set(),
                }
            line = yolo_line_from_normalized_box(class_to_id[label_name], row)
            if line is None:
                continue
            current["lines"].append(line)  # type: ignore[index]
            current["classes"].add(label_name)  # type: ignore[index]
            if label_name in OPENIMAGES_BAG_CLASSES:
                current["bag_classes"].add(label_name)  # type: ignore[index]

    finalize_openimages_record(records, current, bag_image_counts, per_class_limit)
    randomizer.shuffle(records)
    records.sort(key=lambda item: str(item["image_id"]))
    return records, bag_image_counts


def download_openimages_image(split: str, image_id: str, destination: Path) -> None:
    if destination.exists():
        return
    ensure_dir(destination.parent)
    image_url = OPENIMAGES_IMAGE_URL_TEMPLATE.format(split="validation" if split == "val" else split, image_id=image_id)
    with open_url_with_retries(image_url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def prepare_openimages_bags(train_limit: int, val_limit: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    reset_dir(OPENIMAGES_PREPARED_ROOT)
    for split in SPLITS:
        ensure_dir(OPENIMAGES_PREPARED_ROOT / "images" / split)
        ensure_dir(OPENIMAGES_PREPARED_ROOT / "labels" / split)

    print("[openimages] loading class IDs")
    label_ids = load_openimages_label_ids()
    split_limits = {"train": train_limit, "val": val_limit}
    split_records: dict[str, list[dict[str, object]]] = {}
    split_bag_image_counts: dict[str, dict[str, int]] = {}
    class_metrics = {
        class_name: {"images": 0, "boxes": 0, "split_image_counts": {split: 0 for split in SPLITS}}
        for class_name in OPENIMAGES_SHARED_CLASSES
    }

    for split, bbox_url in OPENIMAGES_BBOX_URLS.items():
        print(f"[openimages] collecting {split} records (limit per bag class: {split_limits[split]})")
        records, bag_counts = collect_openimages_records(split, bbox_url, label_ids, split_limits[split])
        split_records[split] = records
        split_bag_image_counts[split] = bag_counts
        print(f"[openimages] selected {len(records)} {split} images: {bag_counts}")

    converted_images = 0
    boxes_written = 0
    for split, records in split_records.items():
        print(f"[openimages] downloading {len(records)} {split} images")
        for index, record in enumerate(records, start=1):
            image_id = str(record["image_id"])
            source_split = "validation" if split == "val" else split
            output_image = OPENIMAGES_PREPARED_ROOT / "images" / split / f"openimages_{source_split}_{image_id}.jpg"
            output_label = OPENIMAGES_PREPARED_ROOT / "labels" / split / f"{output_image.stem}.txt"
            download_openimages_image(source_split, image_id, output_image)
            lines = list(record["lines"])  # type: ignore[arg-type]
            output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            converted_images += 1
            boxes_written += len(lines)
            classes_in_image = set(record["classes"])  # type: ignore[arg-type]
            for class_name in classes_in_image:
                class_metrics[class_name]["images"] += 1
                class_metrics[class_name]["split_image_counts"][split] += 1
            for raw_line in lines:
                class_id = int(raw_line.split()[0])
                class_name = OPENIMAGES_SHARED_CLASSES[class_id]
                class_metrics[class_name]["boxes"] += 1
            if index % 50 == 0 or index == len(records):
                print(f"[openimages] {split} progress: {index}/{len(records)}")

    write_dataset_yaml(OPENIMAGES_PREPARED_ROOT / "dataset.yaml", OPENIMAGES_PREPARED_ROOT, OPENIMAGES_SHARED_CLASSES)

    class_reports: list[dict[str, object]] = []
    for class_name in OPENIMAGES_SHARED_CLASSES:
        metrics = class_metrics[class_name]
        source_category = "Suitcase" if class_name == "luggage" else class_name.title()
        class_reports.append(
            {
                "class_name": class_name,
                "source": str(OPENIMAGES_ROOT.relative_to(ROOT)),
                "source_dataset": "Open Images V6",
                "source_category": source_category,
                "status": "ready" if metrics["boxes"] else "missing_images",
                "local_images_found": metrics["images"],
                "converted_images": metrics["images"],
                "missing_images": 0,
                "boxes_written": metrics["boxes"],
                "split_image_counts": metrics["split_image_counts"],
                "prepared_dataset": str(OPENIMAGES_PREPARED_ROOT.relative_to(ROOT)),
                "prepared_dataset_mode": "multiclass",
                "notes": [
                    "Selected only images containing luggage/backpack/handbag to avoid person-only expansion.",
                    "Kept co-occurring person boxes on bag-positive images to reduce false negatives.",
                ],
            }
        )

    stats = {
        "prepared_dataset": str(OPENIMAGES_PREPARED_ROOT.relative_to(ROOT)),
        "source_dataset": "Open Images V6",
        "class_names": OPENIMAGES_SHARED_CLASSES,
        "train_bag_image_counts": split_bag_image_counts.get("train", {}),
        "val_bag_image_counts": split_bag_image_counts.get("val", {}),
        "converted_images": converted_images,
        "boxes_written": boxes_written,
    }
    return class_reports, stats


def prepare_mendeley_garbage() -> dict[str, object]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": MENDELEY_TRASH_REFERER,
    }
    print("[mendeley] downloading archive if needed")
    download_with_headers(MENDELEY_TRASH_URL, MENDELEY_TRASH_ARCHIVE, headers=headers)
    extract_zip_once(MENDELEY_TRASH_ARCHIVE, MENDELEY_TRASH_EXTRACTED)

    source_root = MENDELEY_TRASH_EXTRACTED / "Trash Detection Dataset" / "CUSTOM_DATASET"
    if not source_root.exists():
        raise FileNotFoundError(f"Mendeley trash dataset layout not found: {source_root}")

    reset_dir(MENDELEY_TRASH_PREPARED)
    for split in SPLITS:
        ensure_dir(MENDELEY_TRASH_PREPARED / "images" / split)
        ensure_dir(MENDELEY_TRASH_PREPARED / "labels" / split)

    split_dir_names = {"train": "train", "val": "valid", "test": "test"}
    converted_images = 0
    boxes_written = 0
    split_image_counts = {split: 0 for split in SPLITS}
    for split, source_split in split_dir_names.items():
        print(f"[mendeley] converting {split} split")
        image_dir = source_root / source_split / "images"
        label_dir = source_root / source_split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            source_label = label_dir / f"{image_path.stem}.txt"
            if not source_label.exists():
                continue
            output_image = MENDELEY_TRASH_PREPARED / "images" / split / f"mendeley_{image_path.name}"
            output_label = MENDELEY_TRASH_PREPARED / "labels" / split / f"{output_image.stem}.txt"
            shutil.copy2(image_path, output_image)
            remapped_lines: list[str] = []
            for raw_line in source_label.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                parts[0] = "0"
                remapped_lines.append(" ".join(parts))
            output_label.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")
            converted_images += 1
            split_image_counts[split] += 1
            boxes_written += len(remapped_lines)

    write_dataset_yaml(MENDELEY_TRASH_PREPARED / "dataset.yaml", MENDELEY_TRASH_PREPARED, ["garbage"])
    return {
        "class_name": "garbage",
        "source": str(MENDELEY_TRASH_ROOT.relative_to(ROOT)),
        "source_dataset": "Trash Detection Dataset (Mendeley Data)",
        "status": "ready",
        "local_images_found": converted_images,
        "converted_images": converted_images,
        "missing_images": 0,
        "boxes_written": boxes_written,
        "split_image_counts": split_image_counts,
        "prepared_dataset": str(MENDELEY_TRASH_PREPARED.relative_to(ROOT)),
        "notes": [
            "Mapped BIODEGRADABLE/CARDBOARD/GLASS/METAL/PAPER/PLASTIC into a single garbage class.",
            "Source is a public CC BY 4.0 YOLO detection dataset from Mendeley Data.",
        ],
    }


def merge_reports(base_report: dict[str, object], extra_report: dict[str, object]) -> dict[str, object]:
    merged = dict(base_report)
    merged["source"] = "merged"
    source_datasets: list[str] = []
    for report in (base_report, extra_report):
        if report.get("prepared_dataset"):
            source_datasets.append(str(report["prepared_dataset"]))
        for path in report.get("source_datasets", []):
            source_datasets.append(str(path))
    merged["source_datasets"] = sorted(set(source_datasets))
    merged["local_images_found"] = int(base_report.get("local_images_found", 0)) + int(extra_report.get("local_images_found", 0))
    merged["converted_images"] = int(base_report.get("converted_images", 0)) + int(extra_report.get("converted_images", 0))
    merged["missing_images"] = int(base_report.get("missing_images", 0)) + int(extra_report.get("missing_images", 0))
    merged["boxes_written"] = int(base_report.get("boxes_written", 0)) + int(extra_report.get("boxes_written", 0))
    merged["split_image_counts"] = {
        split: int(base_report.get("split_image_counts", {}).get(split, 0))
        + int(extra_report.get("split_image_counts", {}).get(split, 0))
        for split in SPLITS
    }
    merged["prepared_dataset"] = None
    merged.pop("prepared_dataset_mode", None)
    merged["notes"] = list(dict.fromkeys([*base_report.get("notes", []), *extra_report.get("notes", [])]))
    return merged


def build_class_report_index(class_reports: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(report["class_name"]): report for report in class_reports}


def update_summary_with_expansions(summary: dict, garbage_report: dict[str, object], openimages_reports: list[dict[str, object]]) -> dict:
    current_reports = build_class_report_index(summary.get("class_reports", []))
    openimages_index = build_class_report_index(openimages_reports)

    for class_name in ("person", "luggage", "backpack", "handbag"):
        if class_name not in current_reports:
            current_reports[class_name] = openimages_index[class_name]
        else:
            current_reports[class_name] = merge_reports(current_reports[class_name], openimages_index[class_name])

    if "garbage" not in current_reports:
        current_reports["garbage"] = garbage_report
    else:
        current_reports["garbage"] = merge_reports(current_reports["garbage"], garbage_report)

    updated_reports: list[dict[str, object]] = []
    seen = set()
    for class_name in CANONICAL_CLASSES:
        report = current_reports.get(class_name)
        if report:
            updated_reports.append(report)
            seen.add(class_name)
    for class_name, report in current_reports.items():
        if class_name not in seen:
            updated_reports.append(report)

    existing_sources = list(summary.get("build_sources", []))
    extra_sources = [
        {
            "kind": "multiclass",
            "prepared_dataset": str(OPENIMAGES_PREPARED_ROOT.relative_to(ROOT)),
            "source_name": "openimages_people_bags",
        },
        {
            "kind": "single_class",
            "class_name": "garbage",
            "prepared_dataset": str(MENDELEY_TRASH_PREPARED.relative_to(ROOT)),
            "source_name": "garbage_mendeley",
        },
    ]
    existing_keys = {
        (
            str(item.get("kind")),
            str(item.get("class_name")),
            str(item.get("prepared_dataset")),
            str(item.get("source_name")),
        )
        for item in existing_sources
    }
    for source in extra_sources:
        key = (
            str(source.get("kind")),
            str(source.get("class_name")),
            str(source.get("prepared_dataset")),
            str(source.get("source_name")),
        )
        if key not in existing_keys:
            existing_sources.append(source)
            existing_keys.add(key)

    payload = {
        "seed": summary.get("seed", SEED),
        "migrations": summary.get("migrations", []),
        "class_reports": updated_reports,
        "build_sources": existing_sources,
    }
    return payload


def main() -> None:
    args = parse_args()
    summary = load_summary()
    openimages_reports, openimages_stats = prepare_openimages_bags(
        train_limit=args.oi_train_limit_per_class,
        val_limit=args.oi_val_limit_per_class,
    )
    garbage_report = prepare_mendeley_garbage()
    updated_summary = update_summary_with_expansions(summary, garbage_report, openimages_reports)
    write_overall_report(
        list(updated_summary.get("migrations", [])),
        list(updated_summary["class_reports"]),
        build_sources=list(updated_summary["build_sources"]),
    )
    print(
        json.dumps(
            {
                "openimages_stats": openimages_stats,
                "garbage_report": garbage_report,
                "build_sources": updated_summary["build_sources"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
