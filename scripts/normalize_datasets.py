from __future__ import annotations

import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
REPORTS_DIR = DATASETS_DIR / "_reports"
SEED = 20260622
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)

CANONICAL_CLASSES = [
    "person",
    "garbage",
    "luggage",
    "backpack",
    "handbag",
    "door_open",
    "door_close",
    "car_door_open",
    "car_door_close",
    "car_window_open",
    "car_window_close",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def safe_move(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    if dst.exists():
        return False
    shutil.move(str(src), str(dst))
    return True


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    ensure_dir(path)


def split_items(items: list[str]) -> dict[str, list[str]]:
    if not items:
        return {split: [] for split in SPLITS}

    shuffled = list(items)
    random.Random(SEED).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * SPLIT_RATIOS[0])
    val_count = int(total * SPLIT_RATIOS[1])
    if total >= 3:
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
    if train_count + val_count >= total:
        val_count = max(0, total - train_count - 1)
    test_count = total - train_count - val_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }


def yolo_line(class_id: int, bbox: list[float], width: float, height: float) -> str | None:
    if width <= 0 or height <= 0:
        return None

    x_min, y_min, box_w, box_h = bbox
    x_max = x_min + box_w
    y_max = y_min + box_h
    x_min = max(0.0, min(x_min, width))
    y_min = max(0.0, min(y_min, height))
    x_max = max(0.0, min(x_max, width))
    y_max = max(0.0, min(y_max, height))
    clipped_w = x_max - x_min
    clipped_h = y_max - y_min
    if clipped_w <= 1 or clipped_h <= 1:
        return None

    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    norm_w = clipped_w / width
    norm_h = clipped_h / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def _yaml_dataset_path(yaml_path: Path, dataset_root: Path) -> str:
    yaml_dir = yaml_path.parent.resolve()
    root = dataset_root.resolve()
    if yaml_dir == root:
        return "."
    try:
        return root.relative_to(yaml_dir).as_posix()
    except ValueError:
        return "."


def write_dataset_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {_yaml_dataset_path(path, dataset_root)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        lines.append(f"  - {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_class_layout() -> None:
    for class_name in CANONICAL_CLASSES:
        class_dir = DATASETS_DIR / class_name
        ensure_dir(class_dir / "raw")
        ensure_dir(class_dir / "prepared")
        ensure_dir(class_dir / "meta")


def migrate_legacy_layout() -> list[str]:
    changes: list[str] = []

    people_dir = DATASETS_DIR / "people"
    person_raw_dir = DATASETS_DIR / "person" / "raw"
    if people_dir.exists():
        for child in people_dir.iterdir():
            target = person_raw_dir / child.name
            if safe_move(child, target):
                changes.append(f"moved {child.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
        if people_dir.exists() and not any(people_dir.iterdir()):
            people_dir.rmdir()
            changes.append(f"removed empty {people_dir.relative_to(ROOT)}")

    garbage_legacy = DATASETS_DIR / "garbage" / "TACO"
    garbage_target = DATASETS_DIR / "garbage" / "raw" / "TACO"
    if safe_move(garbage_legacy, garbage_target):
        changes.append(f"moved {garbage_legacy.relative_to(ROOT)} -> {garbage_target.relative_to(ROOT)}")

    luggage_archive = DATASETS_DIR / "luggage" / "archive"
    luggage_target = DATASETS_DIR / "luggage" / "raw" / "archive"
    if safe_move(luggage_archive, luggage_target):
        changes.append(f"moved {luggage_archive.relative_to(ROOT)} -> {luggage_target.relative_to(ROOT)}")

    luggage_zip = DATASETS_DIR / "luggage" / "archive.zip"
    luggage_zip_target = DATASETS_DIR / "luggage" / "raw" / "archive.zip"
    if safe_move(luggage_zip, luggage_zip_target):
        changes.append(f"moved {luggage_zip.relative_to(ROOT)} -> {luggage_zip_target.relative_to(ROOT)}")

    return changes


def convert_taco() -> dict[str, object]:
    source_root = DATASETS_DIR / "garbage" / "raw" / "TACO"
    annotations_path = source_root / "data" / "annotations.json"
    prepared_root = DATASETS_DIR / "garbage" / "prepared" / "yolo_taco_local"
    meta_path = DATASETS_DIR / "garbage" / "meta" / "status.json"
    result: dict[str, object] = {
        "class_name": "garbage",
        "source": str(source_root.relative_to(ROOT)),
        "status": "missing_source",
        "local_images_found": 0,
        "converted_images": 0,
        "missing_images": 0,
        "boxes_written": 0,
    }

    if not annotations_path.exists():
        meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images_by_id = {item["id"]: item for item in payload.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    local_records: list[dict[str, object]] = []
    missing_images = 0
    for image_id, image_info in images_by_id.items():
        image_path = source_root / "data" / image_info["file_name"]
        if image_path.exists():
            local_records.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "annotations": annotations_by_image.get(image_id, []),
                }
            )
        else:
            missing_images += 1

    reset_dir(prepared_root)
    for split in SPLITS:
        ensure_dir(prepared_root / "images" / split)
        ensure_dir(prepared_root / "labels" / split)

    split_plan = split_items([str(item["image_id"]) for item in local_records])
    split_lookup = {item_id: split for split, ids in split_plan.items() for item_id in ids}

    boxes_written = 0
    split_image_counts = {split: 0 for split in SPLITS}
    for record in local_records:
        image_id = str(record["image_id"])
        split = split_lookup[image_id]
        image_path = Path(record["image_path"])
        relative_name = image_path.relative_to(source_root / "data").with_suffix("")
        file_stub = f"taco_{slugify(relative_name.as_posix())}"
        output_image = prepared_root / "images" / split / f"{file_stub}{image_path.suffix.lower()}"
        output_label = prepared_root / "labels" / split / f"{file_stub}.txt"

        shutil.copy2(image_path, output_image)
        label_lines: list[str] = []
        for annotation in record["annotations"]:
            line = yolo_line(
                0,
                list(annotation["bbox"]),
                float(record["width"]),
                float(record["height"]),
            )
            if line:
                label_lines.append(line)
        boxes_written += len(label_lines)
        output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        split_image_counts[split] += 1

    write_dataset_yaml(prepared_root / "dataset.yaml", prepared_root, ["garbage"])
    result.update(
        {
            "status": "ready",
            "local_images_found": len(local_records),
            "converted_images": len(local_records),
            "missing_images": missing_images,
            "boxes_written": boxes_written,
            "split_image_counts": split_image_counts,
            "prepared_dataset": str(prepared_root.relative_to(ROOT)),
        }
    )
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_float(element: ET.Element | None, default: float = 0.0) -> float:
    if element is None or element.text is None or not element.text.strip():
        return default
    return float(element.text.strip())


def convert_luggage() -> dict[str, object]:
    source_root = DATASETS_DIR / "luggage" / "raw" / "archive"
    images_root = source_root / "suitcase" / "suitcase"
    annotations_root = source_root / "annotation" / "annotation"
    prepared_root = DATASETS_DIR / "luggage" / "prepared" / "yolo_luggage_local"
    meta_path = DATASETS_DIR / "luggage" / "meta" / "status.json"
    result: dict[str, object] = {
        "class_name": "luggage",
        "source": str(source_root.relative_to(ROOT)),
        "status": "missing_source",
        "local_images_found": 0,
        "converted_images": 0,
        "boxes_written": 0,
    }

    if not images_root.exists() or not annotations_root.exists():
        meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    xml_files = sorted(annotations_root.glob("*.xml"))
    reset_dir(prepared_root)
    for split in SPLITS:
        ensure_dir(prepared_root / "images" / split)
        ensure_dir(prepared_root / "labels" / split)

    split_plan = split_items([xml_file.stem for xml_file in xml_files])
    split_lookup = {item_id: split for split, ids in split_plan.items() for item_id in ids}

    converted = 0
    boxes_written = 0
    missing_images = 0
    split_image_counts = {split: 0 for split in SPLITS}
    for xml_file in xml_files:
        split = split_lookup[xml_file.stem]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_name = root.findtext("filename", default="").strip()
        if not image_name:
            continue
        image_path = images_root / image_name
        if not image_path.exists():
            missing_images += 1
            continue

        width = parse_float(root.find("./size/width"))
        height = parse_float(root.find("./size/height"))
        if width <= 0 or height <= 0:
            continue

        file_stub = f"luggage_{slugify(Path(image_name).stem)}"
        output_image = prepared_root / "images" / split / f"{file_stub}{image_path.suffix.lower()}"
        output_label = prepared_root / "labels" / split / f"{file_stub}.txt"
        shutil.copy2(image_path, output_image)

        label_lines: list[str] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name", default="") or "").strip().lower()
            if name != "suitcase":
                continue
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = parse_float(bbox.find("xmin"))
            ymin = parse_float(bbox.find("ymin"))
            xmax = parse_float(bbox.find("xmax"))
            ymax = parse_float(bbox.find("ymax"))
            line = yolo_line(0, [xmin, ymin, xmax - xmin, ymax - ymin], width, height)
            if line:
                label_lines.append(line)
        boxes_written += len(label_lines)
        output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        converted += 1
        split_image_counts[split] += 1

    write_dataset_yaml(prepared_root / "dataset.yaml", prepared_root, ["luggage"])
    result.update(
        {
            "status": "ready",
            "local_images_found": len(list(images_root.glob("*.jpg"))),
            "converted_images": converted,
            "missing_images": missing_images,
            "boxes_written": boxes_written,
            "split_image_counts": split_image_counts,
            "prepared_dataset": str(prepared_root.relative_to(ROOT)),
        }
    )
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def inspect_person() -> dict[str, object]:
    person_root = DATASETS_DIR / "person"
    raw_root = person_root / "raw" / "CrowdHuman"
    meta_path = person_root / "meta" / "status.json"
    image_files = list(raw_root.glob("**/*.jpg")) + list(raw_root.glob("**/*.png"))
    sample_files = list(raw_root.glob("**/train_samples.json")) + list(raw_root.glob("**/val_samples.json")) + list(raw_root.glob("**/test_samples.json"))
    result: dict[str, object] = {
        "class_name": "person",
        "source": str(raw_root.relative_to(ROOT)) if raw_root.exists() else "datasets/person/raw/CrowdHuman",
        "status": "missing_images",
        "local_images_found": len(image_files),
        "sample_index_files": len(sample_files),
        "prepared_dataset": None,
        "notes": [
            "CrowdHuman raw metadata exists, but no local image files were found.",
            "Add CrowdHuman Images directories before converting person data to YOLO.",
            "When converting, keep only full body boxes as person.",
        ],
    }
    if image_files:
        result["status"] = "images_present_needs_conversion"
    meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def inspect_empty_classes() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for class_name in CANONICAL_CLASSES:
        if class_name in {"person", "garbage", "luggage"}:
            continue
        class_dir = DATASETS_DIR / class_name
        raw_files = [path for path in class_dir.glob("raw/**/*") if path.is_file()]
        status = "empty"
        if raw_files:
            status = "has_raw_files"
        payload = {
            "class_name": class_name,
            "status": status,
            "raw_file_count": len(raw_files),
        }
        (class_dir / "meta" / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        entries.append(payload)
    return entries


def write_overall_report(
    migrations: list[str],
    class_reports: list[dict[str, object]],
    build_sources: list[dict[str, object]] | None = None,
) -> None:
    ensure_dir(REPORTS_DIR)
    summary_path = REPORTS_DIR / "dataset_summary.json"
    payload = {
        "seed": SEED,
        "migrations": migrations,
        "class_reports": class_reports,
    }
    if build_sources is not None:
        payload["build_sources"] = build_sources
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    prepare_class_layout()
    migrations = migrate_legacy_layout()
    class_reports = [
        convert_taco(),
        convert_luggage(),
        inspect_person(),
    ]
    class_reports.extend(inspect_empty_classes())
    write_overall_report(migrations, class_reports)
    print(json.dumps({"migrations": migrations, "class_reports": class_reports}, indent=2))


if __name__ == "__main__":
    main()
