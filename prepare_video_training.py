#!/usr/bin/env python3
"""Перевірені відеокадри: польоти, дрібні птахи та негативні приклади."""

import hashlib, io, json, zipfile, subprocess, collections
from pathlib import Path, PurePosixPath
from PIL import Image
import yaml
from common import ROOT, UkrainianParser
from prepare_video_test import REV, SHA, file_sha256, SEQUENCES, fingerprint
from tools.dataset import audit, prepare

ALLOWED = {
    "bird-1": "train",
    "bird-6": "train",
    "bird-18": "train",
    "bird-7": "val",
    "bird-8": "val",
}


def validate_selection(selection):
    if selection.get("reviewed") is not True:
        raise ValueError("Потрібен візуальний перегляд вибірки.")
    for r in selection["records"]:
        if r["sequence"] in SEQUENCES or ALLOWED.get(r["sequence"]) != r["split"]:
            raise ValueError("Заборонений ролик або неправильний split.")
        if r["kind"] not in {"positive", "negative_full", "negative_crop"}:
            raise ValueError("Невідомий тип кадру.")


def crop_clear(crop, box, pad=32):
    a, b, c, d = crop
    x, y, w, h = box
    return c <= x - pad or a >= x + w + pad or d <= y - pad or b >= y + h + pad


def small_target(box, size):
    scale = 640 / max(size)
    return box[2] * scale < 32 and box[3] * scale < 32


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "datasets/bird-video")
    p.add_argument("--selection", type=Path, default=ROOT / "tools/video_training_selection.json")
    p.add_argument("--archive", type=Path, default=ROOT / "datasets/cache-video/bird.zip")
    a = p.parse_args()
    raw = a.out.with_name(a.out.name + "-raw")
    if a.out.exists() or raw.exists():
        p.error("Результат уже існує; задайте новий --out.")
    selection = json.loads(a.selection.read_text())
    validate_selection(selection)
    if not a.archive.exists():
        a.archive.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.archive.with_suffix(".zip.part")
        if not tmp.exists() or file_sha256(tmp) != SHA:
            subprocess.run(
                [
                    "curl",
                    "-fL",
                    "--retry",
                    "3",
                    "--max-time",
                    "1800",
                    "-C",
                    "-",
                    "-o",
                    str(tmp),
                    f"https://huggingface.co/datasets/l-lt/LaSOT/resolve/{REV}/bird.zip",
                ],
                check=True,
            )
        if file_sha256(tmp) != SHA:
            raise ValueError("Неправильний SHA256 джерела.")
        tmp.rename(a.archive)
    if file_sha256(a.archive) != SHA:
        raise ValueError("Пошкоджений архів.")
    protected = []
    for path in [ROOT / "datasets/bird/audit.json", ROOT / "datasets/bird-expanded/audit.json"]:
        if path.exists():
            protected += json.loads(path.read_text())["records"]
    hold = ROOT / "datasets/video-holdout/manifest.json"
    if hold.exists():
        protected += json.loads(hold.read_text())["records"]
    raw.mkdir(parents=True)
    provenance = {}
    excluded = []
    with zipfile.ZipFile(a.archive) as z:
        index = {}
        for name in z.namelist():
            path = PurePosixPath(name)
            for seq in ALLOWED:
                if seq in path.parts:
                    index[(seq, path.name)] = name
        for r in selection["records"]:
            seq = r["sequence"]
            blob = z.read(index[(seq, f'{r["frame"]:08d}.jpg')])
            if hashlib.sha256(blob).hexdigest() != r["source_sha256"]:
                raise ValueError("Змінився вихідний кадр.")
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            w, h = im.size
            labels = []
            if r["kind"] == "positive":
                x, y, bw, bh = r["bbox_xywh"]
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(w, x + bw), min(h, y + bh)
                if not (x0 < x1 and y0 < y1):
                    raise ValueError("Некоректний bbox.")
                labels = [[0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]]
            elif r["kind"] == "negative_crop":
                if not crop_clear(r["crop_xyxy"], r["bbox_xywh"]):
                    raise ValueError("Негативна вирізка перетинає птаха.")
                im = im.crop(r["crop_xyxy"])
            pixel, dh, usable = fingerprint(im)
            if any(
                pixel == c["pixel"]
                or (usable and c["hash_usable"] and (dh ^ c["dhash"]).bit_count() <= 4)
                for c in protected
            ):
                excluded.append(
                    {
                        "file": r["file"],
                        "kind": r["kind"],
                        "reason": "Схожість із захищеною вибіркою",
                    }
                )
                continue
            folder = raw / r["split"]
            (folder / "images").mkdir(parents=True, exist_ok=True)
            (folder / "labels").mkdir(exist_ok=True)
            stem = f'{seq}-{r["frame"]:08d}-{r["kind"]}'
            ip = folder / "images" / f"{stem}.jpg"
            if r["kind"] == "negative_crop":
                im.save(ip, quality=95)
            else:
                ip.write_bytes(blob)
            (folder / "labels" / f"{stem}.txt").write_text(
                "".join(" ".join(map(str, b)) + "\n" for b in labels)
            )
            small = bool(labels and small_target([x0, y0, x1 - x0, y1 - y0], [w, h]))
            provenance[str(ip.resolve())] = {**r, "small_at_640": small}
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": {0: "bird"}}))
    before = audit(raw, distance=4)
    counts = prepare(before, a.out, 26, None)
    after = audit(a.out, distance=4)
    if after["errors"] or any(p["leakage"] for p in after["pairs"]):
        raise ValueError("Аудит не пройшов.")
    chosen = json.loads((a.out / "manifest.json").read_text())["records"]
    rows = [
        {
            **provenance[r["image"]],
            "output_image": str(Path(r["output_image"]).relative_to(a.out.resolve())),
        }
        for r in chosen
    ]
    summary = {
        "counts": counts,
        "positives": dict(collections.Counter(r["split"] for r in rows if r["kind"] == "positive")),
        "negatives": dict(collections.Counter(r["split"] for r in rows if r["kind"] != "positive")),
        "negative_full_frames": sum(r["kind"] == "negative_full" for r in rows),
        "negative_crops": sum(r["kind"] == "negative_crop" for r in rows),
        "training_flight_frames": sum(r["flight"] and r["split"] == "train" for r in rows),
        "training_small_targets_at_640": sum(
            r["small_at_640"] and r["split"] == "train" for r in rows
        ),
        "audit_errors": len(after["errors"]),
        "cross_split_pairs": sum(p["leakage"] for p in after["pairs"]),
        "pairs_after": len(after["pairs"]),
        "similarity_exclusions": excluded,
        "reserved_video_sequences_excluded": list(SEQUENCES),
        "source_revision": REV,
        "archive_sha256": SHA,
        "selection_sha256": file_sha256(a.selection),
        "review_method": selection["review_method"],
        "note": "Малий навчальний відеонабір; негативні повні кадри й фонові вирізки обліковані окремо. Нові валідаційні ролики не використовуються для навчання.",
    }
    (a.out / "source.json").write_text(
        json.dumps({"summary": summary, "records": rows}, indent=2, ensure_ascii=False)
    )
    (a.out / "audit.json").write_text(json.dumps(after, indent=2, ensure_ascii=False))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
