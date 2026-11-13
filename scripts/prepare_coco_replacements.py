from __future__ import annotations

import argparse
import json
import random
import shutil
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
    convert_luggage,
    convert_taco,
    ensure_dir,
    reset_dir,
    write_dataset_yaml,
    write_overall_report,
    yolo_line,
)


COCO_ROOT = DATASETS_DIR / "_sources" / "coco2017"
COCO_ARCHIVES = {
    "annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "archive": COCO_ROOT / "downloads" / "annotations_trainval2017.zip",
        "marker": COCO_ROOT / "annotations" / "instances_train2017.json",
    },
    "train": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "archive": COCO_ROOT / "downloads" / "train2017.zip",
        "marker": COCO_ROOT / "train2017",
    },
    "val": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "archive": COCO_ROOT / "downloads" / "val2017.zip",
        "marker": COCO_ROOT / "val2017",
    },
}
COCO_TARGETS = {
    "person": "person",
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "luggage",
}
COCO_SHARED_CLASSES = ["person", "luggage", "backpack", "handbag"]
COCO_NON_PERSON_CLASSES = {"luggage", "backpack", "handbag"}
COCO_SPLIT_MAP = {
    "train": ("train2017", "instances_train2017.json"),
    "val": ("val2017", "instances_val2017.json"),
    "test": None,
}
EMPTY_CLASS_EXCLUDES = {"person", "garbage", "luggage", "backpack", "handbag"}
DEFAULT_PERSON_ONLY_RATIO = 0.2
COCO_PREPARED_ROOT = COCO_ROOT / "prepared" / "yolo_people_bags_balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare balanced COCO replacements for person and bag classes.")
    parser.add_argument(
        "--person-only-ratio",
        type=float,
        default=DEFAULT_PERSON_ONLY_RATIO,
        help="How many person-only images to keep relative to bag-positive images.",
    )
    return parser.parse_args()


def ensure_coco_dirs() -> None:
    ensure_dir(COCO_ROOT)
    ensure_dir(COCO_ROOT / "downloads")


def download_file(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    print(f"[download] {url}")
    existing_size = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url)
    if existing_size > 0:
        request.add_header("Range", f"bytes={existing_size}-")

    with urllib.request.urlopen(request) as response:
        total = response.headers.get("Content-Length")
        content_range = response.headers.get("Content-Range", "")
        supports_resume = response.status == 206 and existing_size > 0
        if supports_resume:
            total_size = existing_size + (int(total) if total and total.isdigit() else 0)
            downloaded = existing_size
            mode = "ab"
            print(f"[resume] {destination.name}: {existing_size / 1024 / 1024 / 1024:.2f} GB already present")
        else:
            if existing_size > 0:
                print(f"[restart] server did not honor range request: {content_range or 'no content-range'}")
            total_size = int(total) if total and total.isdigit() else 0
            downloaded = 0
            mode = "wb"

        next_report = 256 * 1024 * 1024
        while next_report <= downloaded:
            next_report += 256 * 1024 * 1024

        with destination.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total_size:
                        print(
                            f"[download] {destination.name}: "
                            f"{downloaded / 1024 / 1024 / 1024:.2f} / {total_size / 1024 / 1024 / 1024:.2f} GB"
                        )
                    else:
                        print(f"[download] {destination.name}: {downloaded / 1024 / 1024 / 1024:.2f} GB")
                    next_report += 256 * 1024 * 1024
    print(f"[done] downloaded {destination}")


def extract_archive(archive_path: Path, marker: Path, delete_archive: bool) -> None:
    if marker.exists():
        print(f"[skip] extracted marker exists: {marker}")
        if delete_archive and archive_path.exists():
            archive_path.unlink()
        return

    print(f"[extract] {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(COCO_ROOT)
    print(f"[done] extracted {archive_path.name}")
    if delete_archive and archive_path.exists():
        archive_path.unlink()
        print(f"[cleanup] removed archive {archive_path.name}")


def ensure_coco_source(delete_archives: bool) -> None:
    ensure_coco_dirs()
    for payload in COCO_ARCHIVES.values():
        marker = payload["marker"]
        archive_path = payload["archive"]
        if marker.exists():
            if delete_archives and archive_path.exists():
                archive_path.unlink()
            continue
        download_file(payload["url"], archive_path)
        extract_archive(archive_path, marker, delete_archives)


def load_instances_payload(annotation_name: str) -> dict:
    annotation_path = COCO_ROOT / "annotations" / annotation_name
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def verify_categories() -> dict[str, int]:
    payload = load_instances_payload("instances_train2017.json")
    categories = {item["name"]: int(item["id"]) for item in payload.get("categories", [])}
    missing = sorted(set(COCO_TARGETS) - set(categories))
    if missing:
        raise RuntimeError(f"COCO categories not found: {missing}")
    return {name: categories[name] for name in COCO_TARGETS}


def prepare_output_dirs(prepared_root: Path) -> None:
    reset_dir(prepared_root)
    for split in SPLITS:
        ensure_dir(prepared_root / "images" / split)
        ensure_dir(prepared_root / "labels" / split)


def build_split_records(
    images_dir_name: str,
    annotation_name: str,
    category_id_to_name: dict[int, str],
) -> tuple[list[dict[str, object]], int]:
    payload = load_instances_payload(annotation_name)
    images_by_id = {int(item["id"]): item for item in payload.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)

    for annotation in payload.get("annotations", []):
        target_name = category_id_to_name.get(int(annotation["category_id"]))
        if target_name is None:
            continue
        if int(annotation.get("iscrowd", 0)) == 1:
            continue
        annotations_by_image[int(annotation["image_id"])].append(
            {
                "class_name": target_name,
                "bbox": list(annotation["bbox"]),
            }
        )

    records: list[dict[str, object]] = []
    missing_images = 0
    for image_id, annotations in annotations_by_image.items():
        image_info = images_by_id.get(image_id)
        if not image_info:
            continue
        image_path = COCO_ROOT / images_dir_name / image_info["file_name"]
        if not image_path.exists():
            missing_images += 1
            continue
        records.append(
            {
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "images_dir_name": images_dir_name,
                "image_path": image_path,
                "width": float(image_info["width"]),
                "height": float(image_info["height"]),
                "labels": annotations,
            }
        )
    return records, missing_images


def select_balanced_records(records: list[dict[str, object]], person_only_ratio: float) -> tuple[list[dict[str, object]], dict[str, int]]:
    bag_positive = [record for record in records if any(label["class_name"] in COCO_NON_PERSON_CLASSES for label in record["labels"])]
    bag_positive_ids = {int(record["image_id"]) for record in bag_positive}
    person_only = [
        record
        for record in records
        if int(record["image_id"]) not in bag_positive_ids and any(label["class_name"] == "person" for label in record["labels"])
    ]

    keep_person_only = min(len(person_only), int(len(bag_positive) * person_only_ratio))
    person_only_shuffled = list(person_only)
    random.Random(SEED).shuffle(person_only_shuffled)
    selected_person_only = person_only_shuffled[:keep_person_only]

    selected = list(bag_positive)
    selected.extend(selected_person_only)
    selected.sort(key=lambda record: record["file_name"])
    stats = {
        "bag_positive_images": len(bag_positive),
        "person_only_images_seen": len(person_only),
        "person_only_images_kept": keep_person_only,
        "selected_total_images": len(selected),
    }
    return selected, stats


def convert_coco_balanced_multiclass(person_only_ratio: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    prepare_output_dirs(COCO_PREPARED_ROOT)
    class_to_id = {name: index for index, name in enumerate(COCO_SHARED_CLASSES)}
    category_ids = verify_categories()
    category_id_to_name = {category_ids[coco_name]: class_name for coco_name, class_name in COCO_TARGETS.items()}

    shared_stats = {
        "prepared_dataset": str(COCO_PREPARED_ROOT.relative_to(ROOT)),
        "prepared_dataset_mode": "multiclass",
        "source": str(COCO_ROOT.relative_to(ROOT)),
        "source_dataset": "COCO 2017",
        "class_names": COCO_SHARED_CLASSES,
        "person_only_ratio": person_only_ratio,
        "selection": {},
        "class_metrics": {
            class_name: {
                "images": 0,
                "boxes": 0,
                "split_image_counts": {split: 0 for split in SPLITS},
            }
            for class_name in COCO_SHARED_CLASSES
        },
        "converted_images": 0,
        "boxes_written": 0,
        "missing_images": 0,
    }

    for split, split_config in COCO_SPLIT_MAP.items():
        if split_config is None:
            continue
        images_dir_name, annotation_name = split_config
        split_records, missing_images = build_split_records(images_dir_name, annotation_name, category_id_to_name)
        selected_records, selection_stats = select_balanced_records(split_records, person_only_ratio)
        shared_stats["selection"][split] = selection_stats
        shared_stats["missing_images"] += missing_images

        for record in selected_records:
            output_image = COCO_PREPARED_ROOT / "images" / split / f"coco_{images_dir_name}_{record['file_name']}"
            output_label = COCO_PREPARED_ROOT / "labels" / split / f"{output_image.stem}.txt"
            shutil.copy2(record["image_path"], output_image)

            label_lines: list[str] = []
            classes_in_image: set[str] = set()
            for label in record["labels"]:
                class_name = str(label["class_name"])
                line = yolo_line(
                    class_to_id[class_name],
                    list(label["bbox"]),
                    float(record["width"]),
                    float(record["height"]),
                )
                if line:
                    label_lines.append(line)
                    classes_in_image.add(class_name)
                    shared_stats["class_metrics"][class_name]["boxes"] += 1

            if not label_lines:
                output_image.unlink(missing_ok=True)
                continue

            output_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            shared_stats["converted_images"] += 1
            shared_stats["boxes_written"] += len(label_lines)
            for class_name in classes_in_image:
                shared_stats["class_metrics"][class_name]["images"] += 1
                shared_stats["class_metrics"][class_name]["split_image_counts"][split] += 1

    write_dataset_yaml(COCO_PREPARED_ROOT / "dataset.yaml", COCO_PREPARED_ROOT, COCO_SHARED_CLASSES)

    class_reports: list[dict[str, object]] = []
    for class_name in COCO_SHARED_CLASSES:
        metrics = shared_stats["class_metrics"][class_name]
        class_reports.append(
            {
                "class_name": class_name,
                "source": str(COCO_ROOT.relative_to(ROOT)),
                "source_dataset": "COCO 2017",
                "source_category": class_name if class_name != "luggage" else "suitcase",
                "status": "ready" if metrics["boxes"] else "missing_images",
                "local_images_found": metrics["images"],
                "converted_images": metrics["images"],
                "missing_images": shared_stats["missing_images"],
                "boxes_written": metrics["boxes"],
                "split_image_counts": metrics["split_image_counts"],
                "prepared_dataset": str(COCO_PREPARED_ROOT.relative_to(ROOT)),
                "prepared_dataset_mode": "multiclass",
                "notes": [
                    "Shared COCO dataset keeps co-occurring labels in the same image.",
                    "Person-only images are downsampled to avoid overwhelming bag classes.",
                ],
            }
        )
    return class_reports, shared_stats


def build_class_report_index(class_reports: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {report["class_name"]: report for report in class_reports}


def merge_class_reports(base_report: dict[str, object], extra_report: dict[str, object]) -> dict[str, object]:
    merged = dict(base_report)
    merged["source"] = "merged"
    merged["source_datasets"] = [str(base_report.get("prepared_dataset")), str(extra_report.get("prepared_dataset"))]
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
    merged["notes"] = [
        "Merged from balanced COCO luggage and local luggage dataset.",
    ]
    return merged


def inspect_remaining_empty_classes() -> list[dict[str, object]]:
    entries = []
    for class_name in CANONICAL_CLASSES:
        if class_name in EMPTY_CLASS_EXCLUDES:
            continue
        class_dir = DATASETS_DIR / class_name
        raw_files = [path for path in class_dir.glob("raw/**/*") if path.is_file()]
        status = "empty" if not raw_files else "has_raw_files"
        payload = {
            "class_name": class_name,
            "status": status,
            "raw_file_count": len(raw_files),
        }
        (class_dir / "meta" / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        entries.append(payload)
    return entries


def write_class_meta(class_reports: list[dict[str, object]]) -> None:
    for report in class_reports:
        class_dir = DATASETS_DIR / report["class_name"]
        ensure_dir(class_dir / "meta")
        (class_dir / "meta" / "status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_coco_source(delete_archives=True)

    garbage_report = convert_taco()
    local_luggage_report = convert_luggage()
    coco_class_reports, coco_shared_stats = convert_coco_balanced_multiclass(args.person_only_ratio)
    coco_index = build_class_report_index(coco_class_reports)

    luggage_report = merge_class_reports(coco_index["luggage"], local_luggage_report)
    class_reports = [
        coco_index["person"],
        garbage_report,
        luggage_report,
        coco_index["backpack"],
        coco_index["handbag"],
    ]
    class_reports.extend(inspect_remaining_empty_classes())
    write_class_meta(class_reports)

    build_sources = [
        {
            "kind": "multiclass",
            "prepared_dataset": str(COCO_PREPARED_ROOT.relative_to(ROOT)),
            "source_name": "coco_people_bags_balanced",
        },
        {
            "kind": "single_class",
            "class_name": "garbage",
            "prepared_dataset": str(garbage_report["prepared_dataset"]),
            "source_name": "garbage_local",
        },
        {
            "kind": "single_class",
            "class_name": "luggage",
            "prepared_dataset": str(local_luggage_report["prepared_dataset"]),
            "source_name": "luggage_local",
        },
    ]
    write_overall_report([], class_reports, build_sources=build_sources)
    print(
        json.dumps(
            {
                "class_reports": class_reports,
                "build_sources": build_sources,
                "coco_shared_stats": coco_shared_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
