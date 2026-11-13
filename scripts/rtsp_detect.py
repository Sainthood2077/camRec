"""RTSP real-time YOLO detection — boxes only (no labels / conf text)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2


def redact_source(source: str) -> str:
    """Hide username/password in stream URLs before logging."""
    try:
        parts = urlsplit(source)
    except ValueError:
        return source
    if not parts.scheme or not (parts.username or parts.password):
        return source
    host = parts.hostname or ''
    if parts.port:
        host = f'{host}:{parts.port}'
    netloc = f'***:***@{host}' if host else '***:***'
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

from ultralytics import YOLO

# Class colors (BGR) for person / garbage / luggage / backpack / handbag
DEFAULT_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
]

DEFAULT_MODEL = (
    Path("runs")
    / "detect"
    / "runs"
    / "train"
    / "yolo11s_camrec_5cls_balanced_1024-3"
    / "weights"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTSP real-time YOLO detect (boxes only).")
    p.add_argument(
        "--source",
        required=True,
        help="RTSP URL or video path / camera index.",
    )
    p.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to trained weights (default: latest yolo11s balanced best.pt).",
    )
    p.add_argument("--imgsz", type=int, default=1024, help="Inference size.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--device", default="0", help="Device, e.g. 0 or cpu.")
    p.add_argument("--line-width", type=int, default=2, help="Box line thickness.")
    p.add_argument(
        "--max-fps",
        type=float,
        default=0,
        help="Cap display FPS (0 = unlimited / as fast as inference).",
    )
    p.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before RTSP reconnect.",
    )
    return p.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    # Prefer TCP for RTSP stability on LAN cameras
    if source.lower().startswith("rtsp://"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        # camera index
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
    return cap


def draw_boxes_only(frame, result, line_width: int = 2) -> None:
    """In-place draw detection boxes only (no class / conf text)."""
    if result.boxes is None or len(result.boxes) == 0:
        return
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), cls_id in zip(boxes, classes):
        color = DEFAULT_COLORS[int(cls_id) % len(DEFAULT_COLORS)]
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            line_width,
        )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path.resolve()}")

    print(f"[rtsp_detect] model  = {model_path.resolve()}")
    print(f"[rtsp_detect] source = {redact_source(args.source)}")
    print(f"[rtsp_detect] imgsz  = {args.imgsz}  conf={args.conf}  device={args.device}")
    print("[rtsp_detect] keys: q = quit, r = reconnect")

    model = YOLO(str(model_path))
    window = "CamRec RTSP Detect (boxes only)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_interval = (1.0 / args.max_fps) if args.max_fps and args.max_fps > 0 else 0.0
    last_show = 0.0
    cap: cv2.VideoCapture | None = None

    try:
        while True:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                print("[rtsp_detect] opening stream ...")
                cap = open_capture(args.source)
                if not cap.isOpened():
                    print(
                        f"[rtsp_detect] open failed, retry in {args.reconnect_delay}s"
                    )
                    time.sleep(args.reconnect_delay)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print(
                    f"[rtsp_detect] frame read failed, reconnect in {args.reconnect_delay}s"
                )
                cap.release()
                cap = None
                time.sleep(args.reconnect_delay)
                continue

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            draw_boxes_only(frame, results[0], line_width=args.line_width)

            now = time.perf_counter()
            if frame_interval <= 0 or (now - last_show) >= frame_interval:
                cv2.imshow(window, frame)
                last_show = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                print("[rtsp_detect] manual reconnect")
                if cap is not None:
                    cap.release()
                cap = None
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[rtsp_detect] stopped")


if __name__ == "__main__":
    main()
