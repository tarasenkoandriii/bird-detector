#!/usr/bin/env python3
"""Фіксований відеонабір LaSOT: різні ролики для валідації та тесту."""

import hashlib, json, subprocess, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from common import ROOT, UkrainianParser

REV = "a97464600e8c1ab91eb91a47332a20732830585e"
SHA = "33e05e600732608fe7c35413b94558c959ffbe145eb03a918fa3d7d07f949e39"
SEQUENCES = {"bird-2": "val", "bird-3": "test", "bird-15": "test", "bird-17": "test"}


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(image):
    rgb = image.convert("RGB")
    w, h = rgb.size
    pixel = hashlib.sha256(f"{w},{h}:".encode() + rgb.tobytes()).hexdigest()
    gray = rgb.resize((9, 8)).convert("L")
    px = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
    dh = sum((px[y * 9 + x] > px[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
    return pixel, dh, max(px) - min(px) > 15


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "datasets/video-holdout")
    p.add_argument("--archive", type=Path, default=ROOT / "datasets/cache-video/bird.zip")
    p.add_argument("--cub-audit", type=Path, default=ROOT / "datasets/bird-expanded/audit.json")
    a = p.parse_args()
    if a.out.exists():
        p.error("Набір уже існує; задайте новий --out.")
    if not a.cub_audit.is_file():
        p.error("Потрібен аудит CUB для перевірки перетинів.")
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.archive.with_suffix(".zip.part")
    if not a.archive.exists() and tmp.exists() and file_sha256(tmp) == SHA:
        tmp.rename(a.archive)
    if not a.archive.exists():
        tmp = a.archive.with_suffix(".zip.part")
        subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
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
            raise ValueError("Невірний SHA256 архіву.")
        tmp.rename(a.archive)
    if file_sha256(a.archive) != SHA:
        raise ValueError("Пошкоджений архів.")
    cub = json.loads(a.cub_audit.read_text())["records"]
    previous_audit = ROOT / "datasets/bird/audit.json"
    if previous_audit.is_file() and previous_audit.resolve() != a.cub_audit.resolve():
        cub += json.loads(previous_audit.read_text())["records"]
    a.out.mkdir(parents=True)
    rows = []
    excluded = []
    with zipfile.ZipFile(a.archive) as z:
        names = z.namelist()
        for seq, split in SEQUENCES.items():

            def member(basename):
                found = [
                    n
                    for n in names
                    if PurePosixPath(n).name == basename and seq in PurePosixPath(n).parts
                ]
                if len(found) != 1:
                    raise ValueError(f"Неоднозначний файл {seq}/{basename}")
                return found[0]

            gt = np.loadtxt(z.open(member("groundtruth.txt")), delimiter=",").reshape(-1, 4)
            occ = np.loadtxt(z.open(member("full_occlusion.txt")), delimiter=",").reshape(-1)
            out = np.loadtxt(z.open(member("out_of_view.txt")), delimiter=",").reshape(-1)
            if not len(gt) == len(occ) == len(out):
                raise ValueError("Довжини розмітки не збігаються.")
            desc = z.read(member("nlp.txt")).decode().strip()
            for index in np.unique(np.linspace(0, len(gt) - 1, 100, dtype=int)):
                name = member(f"{index+1:08d}.jpg")
                blob = z.read(name)
                import io

                with Image.open(io.BytesIO(blob)) as im:
                    w, h = im.size
                    pixel, dh, usable = fingerprint(im)
                if any(
                    pixel == c["pixel"]
                    or (usable and c["hash_usable"] and (dh ^ c["dhash"]).bit_count() <= 4)
                    for c in cub
                ):
                    excluded.append(
                        {"sequence": seq, "frame": int(index + 1), "reason": "Схожість із CUB"}
                    )
                    continue
                target = gt[index].tolist()
                if not np.isfinite(target).all() or target[2] <= 0 or target[3] <= 0:
                    raise ValueError("Некоректний bbox.")
                folder = a.out / split / seq
                folder.mkdir(parents=True, exist_ok=True)
                file = folder / f"{index+1:08d}.jpg"
                file.write_bytes(blob)
                rows.append(
                    {
                        "split": split,
                        "sequence": seq,
                        "frame": int(index + 1),
                        "image": str(file.relative_to(a.out)),
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "pixel": pixel,
                        "dhash": dh,
                        "hash_usable": usable,
                        "size": [w, h],
                        "bbox_xywh": target,
                        "visible": bool(occ[index] == 0 and out[index] == 0),
                        "full_occlusion": bool(occ[index]),
                        "out_of_view": bool(out[index]),
                        "description": desc,
                    }
                )
    pairs = []
    for i, x in enumerate(rows):
        for y in rows[:i]:
            if x["split"] == y["split"]:
                continue
            if x["pixel"] == y["pixel"] or (
                x["hash_usable"] and y["hash_usable"] and (x["dhash"] ^ y["dhash"]).bit_count() <= 4
            ):
                pairs.append([x["sequence"], x["frame"], y["sequence"], y["frame"]])
    if pairs:
        raise ValueError("Знайдено схожі кадри між відеовибірками; набір потребує перегляду.")
    manifest = {
        "source": "LaSOT, офіційні тестові послідовності bird",
        "revision": REV,
        "archive_sha256": SHA,
        "sequences": SEQUENCES,
        "sampling": "100 рівномірних індексів у кожному ролику, до запуску моделі",
        "annotation_scope": "Один цільовий птах на кадр. Інші птахи можуть бути нерозмічені.",
        "cub_similarity_exclusions": excluded,
        "cross_split_pairs": pairs,
        "records": rows,
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "frames": len(rows),
                "excluded": len(excluded),
                "visible": sum(x["visible"] for x in rows),
                "sequences": SEQUENCES,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
