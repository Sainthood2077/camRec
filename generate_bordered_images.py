import math
from pathlib import Path

import cv2


BORDER_COLOR = (0, 0, 255)  # BGR: red
BORDER_THICKNESS = 1
LABEL_PIXEL_HEIGHT = 16
LABEL_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_SIZES = [224, 336]


def letterbox_yolo(
    image,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scale_fill=False,
    scaleup=True,
    stride=32,
):
    """
    YOLO letterbox:
    - keep aspect ratio
    - resize with unchanged ratio
    - pad to target shape
    """
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw = dw % stride
        dh = dh % stride
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def export_letterbox_640(input_path: Path) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    output, _, _ = letterbox_yolo(image, new_shape=(640, 640))
    output_path = input_path.with_name(f"{input_path.stem}_letterbox_640{input_path.suffix}")
    cv2.imwrite(str(output_path), output)
    print(f"Exported: {output_path.name}")


def get_overlap_stride(width: int, height: int, size: int) -> int:
    if size > width or size > height:
        raise ValueError(f"Window size {size} exceeds image dimensions {width}x{height}")

    remaining_w = width - size
    remaining_h = height - size
    common = math.gcd(remaining_w, remaining_h)
    if common == 0:
        return 1

    best_stride = 1
    divisor = 1
    while divisor * divisor <= common:
        if common % divisor == 0:
            paired = common // divisor
            if divisor < size:
                best_stride = max(best_stride, divisor)
            if paired < size:
                best_stride = max(best_stride, paired)
        divisor += 1

    return best_stride


def get_exact_fit_positions(length: int, size: int, stride: int) -> list[int]:
    if size > length:
        raise ValueError(f"Window size {size} exceeds dimension length {length}")

    last_start = length - size
    positions = list(range(0, last_start + 1, stride))
    if not positions or positions[-1] != last_start:
        positions.append(last_start)
    return positions


def draw_window_grid(image, size: int, stride: int, exact_fit: bool) -> tuple:
    output = image.copy()
    height, width = output.shape[:2]
    label = f"{size}*{size}"
    _, base_height = cv2.getTextSize("Hg", LABEL_FONT, 1.0, LABEL_THICKNESS)[0]
    font_scale = LABEL_PIXEL_HEIGHT / max(base_height, 1)

    if exact_fit:
        y_positions = get_exact_fit_positions(height, size, stride)
        x_positions = get_exact_fit_positions(width, size, stride)
    else:
        y_positions = list(range(0, height, stride))
        x_positions = list(range(0, width, stride))

    for top in y_positions:
        for left in x_positions:
            right = min(left + size - 1, width - 1)
            bottom = min(top + size - 1, height - 1)
            cv2.rectangle(output, (left, top), (right, bottom), BORDER_COLOR, BORDER_THICKNESS)

            box_w = right - left + 1
            box_h = bottom - top + 1
            is_full_square = box_w == size and box_h == size
            if not is_full_square:
                continue

            (text_w, text_h), baseline = cv2.getTextSize(
                label, LABEL_FONT, font_scale, LABEL_THICKNESS
            )
            text_x = left + (box_w - text_w) // 2
            text_y = top + (box_h + text_h) // 2 - baseline // 2
            cv2.putText(
                output,
                label,
                (text_x, text_y),
                LABEL_FONT,
                font_scale,
                BORDER_COLOR,
                LABEL_THICKNESS,
                cv2.LINE_AA,
            )

    return output, x_positions, y_positions


def export_with_border(input_path: Path) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    height, width = image.shape[:2]

    for size in WINDOW_SIZES:
        non_overlap_output, _, _ = draw_window_grid(image, size, size, exact_fit=False)
        non_overlap_path = input_path.with_name(
            f"{input_path.stem}_{size}px_no_overlap{input_path.suffix}"
        )
        cv2.imwrite(str(non_overlap_path), non_overlap_output)
        print(f"Exported: {non_overlap_path.name}")

        overlap_stride = get_overlap_stride(width, height, size)
        overlap_output, _, _ = draw_window_grid(image, size, overlap_stride, exact_fit=True)
        overlap_path = input_path.with_name(
            f"{input_path.stem}_{size}px_overlap_stride{overlap_stride}{input_path.suffix}"
        )
        cv2.imwrite(str(overlap_path), overlap_output)
        print(f"Exported: {overlap_path.name}")


if __name__ == "__main__":
    export_with_border(Path(__file__).parent / "test.png")
