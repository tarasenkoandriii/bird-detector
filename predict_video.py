#!/usr/bin/env python3
"""Детекція птахів на кожному кадрі відео зі збереженням FPS і розміру кадрів."""

import hashlib, json, math, os, shutil, subprocess, tempfile, time
from pathlib import Path
from common import UkrainianParser, ROOT, resolve_device


def parser():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Шлях до вхідного відео")
    p.add_argument("--output", required=True, type=Path, help="Шлях до вихідного MP4")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--iou", type=float, default=0.5, help="Поріг CPU NMS для повторних рамок тієї самої птиці"
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto", help="auto, cpu, mps або індекс пристрою CUDA")
    p.add_argument("--overwrite", action="store_true", help="Дозволити заміну наявного результату")
    return p


def validate(a):
    if not a.input.is_file():
        raise ValueError(f"Вхідне відео не існує: {a.input}")
    if not a.weights.is_file():
        raise ValueError(f"Файл ваг не існує: {a.weights}")
    if a.input.resolve() == a.output.resolve():
        raise ValueError("Вхідний і вихідний файли мають бути різними.")
    if a.output.suffix.lower() != ".mp4":
        raise ValueError("Вихідний файл має мати розширення .mp4.")
    if a.output.exists() and not a.overwrite:
        raise ValueError("Вихідний файл уже існує; явно додайте --overwrite.")
    if not 0 < a.conf < 1:
        raise ValueError("--conf має бути між 0 та 1.")
    if not 0 < a.iou < 1:
        raise ValueError("--iou має бути між 0 та 1.")
    if a.imgsz < 32 or a.imgsz % 32:
        raise ValueError("--imgsz має бути додатним числом, кратним 32.")


def suppress_duplicates(boxes, iou):
    import cv2

    if not boxes:
        return []
    xywh = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1, _ in boxes]
    indices = cv2.dnn.NMSBoxes(xywh, [b[4] for b in boxes], 0.0, iou)
    return [boxes[int(i)] for i in indices.flatten()] if len(indices) else []


def draw(frame, boxes):
    import cv2

    h, w = frame.shape[:2]
    for x0, y0, x1, y1, confidence in boxes:
        x0, x1 = sorted((max(0, min(w - 1, round(x0))), max(0, min(w - 1, round(x1)))))
        y0, y1 = sorted((max(0, min(h - 1, round(y0))), max(0, min(h - 1, round(y1)))))
        cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 220, 90), 2)
        label = f"bird {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(th + baseline + 4, y0)
        cv2.rectangle(
            frame,
            (x0, max(0, top - th - baseline - 4)),
            (min(w - 1, x0 + tw + 6), top),
            (30, 220, 90),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x0 + 3, top - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def video_info(cap):
    import cv2

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or min(w, h) <= 0:
        raise ValueError("Некоректний FPS або розмір кадру відео.")
    return fps, w, h


def run(a, model_factory=None):
    validate(a)
    import cv2

    cap = cv2.VideoCapture(str(a.input.resolve()))
    writer = None
    temporary = []
    try:
        if not cap.isOpened():
            raise ValueError("Не вдалося декодувати вхідне відео.")
        fps, w, h = video_info(cap)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        device = resolve_device(a.device)
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        model = model_factory(str(a.weights.resolve()))
        from inference import SafeDetectionPredictor

        if dict(model.names) != {0: "bird"}:
            raise ValueError("Потрібні ваги одного класу з names {0: bird}.")
        a.output.parent.mkdir(parents=True, exist_ok=True)

        def temp():
            f = tempfile.NamedTemporaryFile(
                suffix=".mp4", prefix=".bird-", dir=a.output.parent, delete=False
            )
            f.close()
            p = Path(f.name)
            temporary.append(p)
            return p

        raw = temp()
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise ValueError("Не вдалося відкрити запис MP4.")
        count = 0
        detected = 0
        total_boxes = 0
        samples = []
        start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(
                source=frame,
                conf=a.conf,
                imgsz=a.imgsz,
                device=device,
                verbose=False,
                predictor=SafeDetectionPredictor,
            )
            data = results[0].boxes.data.cpu().tolist() if results[0].boxes is not None else []
            boxes = [[float(v) for v in row[:5]] for row in data if int(row[5]) == 0]
            boxes = suppress_duplicates(boxes, a.iou)
            detected += bool(boxes)
            total_boxes += len(boxes)
            writer.write(draw(frame, boxes))
            count += 1
            if boxes and len(samples) < 5:
                samples.append({"frame": count - 1, "detections": boxes})
            if count % 25 == 0:
                print(
                    f"{count} кадрів | {count/(time.perf_counter()-start):.2f} кадрів/с", flush=True
                )
        elapsed = time.perf_counter() - start
        writer.release()
        writer = None
        cap.release()
        if not count:
            raise ValueError("У відео немає кадрів, які вдалося декодувати.")
        if expected > 0 and abs(count - expected) > 1:
            raise ValueError(
                f"Неповне декодування: прочитано {count}, заголовок указує {expected} кадрів."
            )
        encoded = raw
        codec = "mp4v"
        if shutil.which("ffmpeg") and w % 2 == 0 and h % 2 == 0:
            encoded = temp()
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(raw),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(encoded),
                ],
                check=True,
                capture_output=True,
            )
            codec = "h264"
        check = cv2.VideoCapture(str(encoded))
        written = 0
        try:
            if not check.isOpened():
                raise ValueError("Не вдалося повторно відкрити вихідне відео.")
            out_fps, out_w, out_h = video_info(check)
            while True:
                ok, _ = check.read()
                if not ok:
                    break
                written += 1
        finally:
            check.release()
        if written != count or (out_w, out_h) != (w, h) or abs(out_fps - fps) > 0.01:
            raise ValueError("Перевірка результату не пройшла: змінилися кадри, FPS або розміри.")
        if a.output.exists() and not a.overwrite:
            raise ValueError("Вихідний файл з’явився під час обробки; перезапис скасовано.")
        os.replace(encoded, a.output)
        report = {
            "input": str(a.input),
            "output": str(a.output),
            "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
            "device": device,
            "imgsz": a.imgsz,
            "conf": a.conf,
            "iou": a.iou,
            "frames": count,
            "output_frames": written,
            "skipped_frames": 0,
            "width": w,
            "height": h,
            "fps": fps,
            "duration_seconds": count / fps,
            "processing_seconds": elapsed,
            "processing_fps": count / elapsed,
            "frames_with_detections": detected,
            "total_detections": total_boxes,
            "codec": codec,
            "audio": "не копіюється",
            "sample_detections": samples,
        }
        a.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return report
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for p in temporary:
            p.unlink(missing_ok=True)


def main():
    p = parser()
    a = p.parse_args()
    try:
        run(a)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as e:
        p.exit(1, f"Помилка: {e}\n")
    except KeyboardInterrupt:
        p.exit(130, "Обробку перервано; незавершений файл видалено.\n")


if __name__ == "__main__":
    main()
