from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

from normalize_datasets import DATASETS_DIR, ROOT, SPLITS, ensure_dir, reset_dir, write_dataset_yaml


DEFAULT_SOURCE_ROOT = DATASETS_DIR / "camrec_all_classes"
DEFAULT_OUTPUT_ROOT = DATASETS_DIR / "camrec_all_classes_balanced"
DEFAULT_BALANCE_CLASSES = ["garbage", "luggage", "backpack", "handbag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a person-reduced balanced dataset from the merged CamRec dataset.")
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Source merged YOLO dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for the balanced YOLO dataset.",
    )
    parser.add_argument(
        "--balance-classes",
        default=",".join(DEFAULT_BALANCE_CLASSES),
        help="Comma-separated classes to balance. Person is reduced indirectly and is not balanced by default.",
    )
    parser.add_argument(
        "--target-images-per-class",
        type=int,
        default=0,
        help="Target image count per balance class. Use 0 to auto-select the minimum available class count.",
    )
    return parser.parse_args()


def load_class_names(dataset_root: Path) -> list[str]:
    payload = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8"))
    names = payload.get("names", [])
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def parse_label_file(label_path: Path, class_names: list[str]) -> tuple[list[str], set[str], Counter]:
    kept_lines: list[str] = []
    image_classes: set[str] = set()
    box_counts: Counter = Counter()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_name = class_names[int(parts[0])]
        kept_lines.append(line)
        image_classes.add(class_name)
        box_counts[class_name] += 1
    return kept_lines, image_classes, box_counts


def gather_records(dataset_root: Path, class_names: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split in SPLITS:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file():
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            label_lines, image_classes, box_counts = parse_label_file(label_path, class_names)
            if not image_classes:
                continue
            records.append(
                {
                    "id": f"{split}/{image_path.name}",
                    "split": split,
                    "image_path": image_path,
                    "label_path": label_path,
                    "label_lines": label_lines,
                    "classes": image_classes,
                    "box_counts": box_counts,
                }
            )
    return records


def compute_image_counts(records: list[dict[str, object]], class_names: list[str]) -> Counter:
    counts: Counter = Counter()
    for record in records:
        for class_name in record["classes"]:  # type: ignore[index]
            counts[class_name] += 1
    for class_name in class_names:
        counts.setdefault(class_name, 0)
    return counts


def score_record(
    record: dict[str, object],
    deficits: dict[str, int],
    rarity_weights: dict[str, float],
    balance_classes: list[str],
) -> tuple[float, int, int, int, int]:
    classes = record["classes"]  # type: ignore[index]
    helped_classes = [class_name for class_name in balance_classes if class_name in classes and deficits[class_name] > 0]
    weighted_gain = sum(rarity_weights[class_name] * deficits[class_name] for class_name in helped_classes)
    person_penalty = 1 if "person" in classes else 0
    overshoot_penalty = sum(1 for class_name in balance_classes if class_name in classes and deficits[class_name] == 0)
    return (
        weighted_gain,
        len(helped_classes),
        -person_penalty,
        -overshoot_penalty,
        -len(classes),
    )


def select_balanced_records(
    records: list[dict[str, object]],
    balance_classes: list[str],
    target_images_per_class: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_image_counts = compute_image_counts(records, balance_classes + ["person"])
    if target_images_per_class <= 0:
        target_images_per_class = min(source_image_counts[class_name] for class_name in balance_classes)

    rarity_weights = {
        class_name: 1.0 / max(1, source_image_counts[class_name])
        for class_name in balance_classes
    }
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selected_counts: Counter = Counter()

    while True:
        deficits = {
            class_name: max(0, target_images_per_class - selected_counts[class_name])
            for class_name in balance_classes
        }
        if all(value == 0 for value in deficits.values()):
            break

        best_record: dict[str, object] | None = None
        best_score: tuple[float, int, int, int, int] | None = None
        for record in records:
            if record["id"] in selected_ids:
                continue
            classes = record["classes"]  # type: ignore[index]
            if not any(class_name in classes and deficits[class_name] > 0 for class_name in balance_classes):
                continue
            score = score_record(record, deficits, rarity_weights, balance_classes)
            if best_score is None or score > best_score:
                best_record = record
                best_score = score

        if best_record is None:
            break

        selected.append(best_record)
        selected_ids.add(str(best_record["id"]))
        for class_name in best_record["classes"]:  # type: ignore[index]
            selected_counts[class_name] += 1

    # Remove overshoot records when they are no longer needed for the balanced classes.
    changed = True
    while changed:
        changed = False
        for record in sorted(
            selected,
            key=lambda item: (
                "person" in item["classes"],  # type: ignore[index]
                -len(item["classes"]),  # type: ignore[arg-type]
                str(item["id"]),
            ),
            reverse=True,
        ):
            classes = record["classes"]  # type: ignore[index]
            if any(class_name in balance_classes and selected_counts[class_name] - 1 < target_images_per_class for class_name in classes):
                continue
            selected.remove(record)
            selected_ids.remove(str(record["id"]))
            for class_name in classes:
                selected_counts[class_name] -= 1
            changed = True
            break

    stats = {
        "target_images_per_class": target_images_per_class,
        "balance_classes": balance_classes,
        "selected_train_image_counts": dict(compute_image_counts(selected, balance_classes + ["person"])),
        "selected_total_images": len(selected),
    }
    return selected, stats


def materialize_image(src_image: Path, dst_image: Path) -> None:
    try:
        os.link(src_image, dst_image)
    except OSError:
        shutil.copy2(src_image, dst_image)


def write_balanced_dataset(
    output_root: Path,
    class_names: list[str],
    selected_records: list[dict[str, object]],
    stats: dict[str, object],
) -> dict[str, object]:
    reset_dir(output_root)
    for split in SPLITS:
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    merged_counts = {class_name: {"images": 0, "boxes": 0} for class_name in class_names}
    train_merged_counts = {class_name: {"images": 0, "boxes": 0} for class_name in class_names}
    split_image_counts = {split: 0 for split in SPLITS}
    for record in selected_records:
        split = str(record["split"])
        src_image = Path(record["image_path"])
        src_label = Path(record["label_path"])
        dst_image = output_root / "images" / split / src_image.name
        dst_label = output_root / "labels" / split / src_label.name
        materialize_image(src_image, dst_image)
        shutil.copy2(src_label, dst_label)
        split_image_counts[split] += 1
        for class_name in record["classes"]:  # type: ignore[index]
            merged_counts[class_name]["images"] += 1
            if split == "train":
                train_merged_counts[class_name]["images"] += 1
        for class_name, box_count in record["box_counts"].items():  # type: ignore[index]
            merged_counts[class_name]["boxes"] += int(box_count)
            if split == "train":
                train_merged_counts[class_name]["boxes"] += int(box_count)

    write_dataset_yaml(output_root / "dataset.yaml", output_root, class_names)
    report = {
        "dataset_root": str(output_root.relative_to(ROOT)),
        "classes": class_names,
        "balance_target_images": stats["target_images_per_class"],
        "balance_classes": stats["balance_classes"],
        "split_image_counts": split_image_counts,
        "merged_counts": merged_counts,
        "train_merged_counts": train_merged_counts,
        "source_train_image_counts": stats["source_train_image_counts"],
        "selected_train_image_counts": stats["selected_train_image_counts"],
        "selected_total_images": stats["selected_total_images"],
    }
    (output_root / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    balance_classes = [item.strip() for item in str(args.balance_classes).split(",") if item.strip()]
    class_names = load_class_names(source_root)
    records = gather_records(source_root, class_names)
    train_records = [record for record in records if record["split"] == "train"]
    holdout_records = [record for record in records if record["split"] != "train"]
    selected_train_records, stats = select_balanced_records(train_records, balance_classes, args.target_images_per_class)
    stats["source_train_image_counts"] = dict(compute_image_counts(train_records, balance_classes + ["person"]))
    final_records = selected_train_records + holdout_records
    report = write_balanced_dataset(output_root, class_names, final_records, stats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
