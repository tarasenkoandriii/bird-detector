#!/usr/bin/env python3
"""Дослідницька підвибірка CUB без ключа: перші шарди train/test, фіксований seed."""

import argparse, hashlib, io, json, random, shutil, subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import UkrainianParser
import pyarrow.parquet as pq
from PIL import Image
import yaml

REPO = "bentrevett/caltech-ucsd-birds-200-2011"
REV = "1ef09e021b0b65b40337f6f285909656f407f6e0"
SHA = {
    "train": "fb4590d7fbd018c1cd0df7674dd4055ca91312df72790a2fddcab74e15fcefa4",
    "test": "fb0c8ac3d2357353689f07444f0d29e27a36f76faaaad1a13451494816a5c608",
}


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Каталог результату вже існує")
    a.cache.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ["train", "test"]:
        target = a.cache / f"cub-{split}.parquet"
        if not target.exists():
            tmp = target.with_suffix(".part")
            print("Завантаження", split, flush=True)
            # curl використовує системне сховище сертифікатів macOS.
            subprocess.run(
                [
                    "curl",
                    "-fL",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "1200",
                    "-o",
                    str(tmp),
                    f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/data/{split}-00000-of-00002.parquet",
                ],
                check=True,
            )
            tmp.rename(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != SHA[split]:
            raise ValueError(
                f"Контрольна сума не збігається: {target}; видаліть пошкоджений кеш і повторіть"
            )
        files[split] = target
    rows = {s: pq.read_table(f).to_pylist() for s, f in files.items()}
    rng = random.Random(26)
    rng.shuffle(rows["train"])
    rng.shuffle(rows["test"])
    subsets = {
        "train": rows["train"][:1000],
        "val": rows["train"][1000:1200],
        "test": rows["test"][:100],
    }
    a.out.mkdir(parents=True)
    provenance = []
    rejected = []
    for split, records in subsets.items():
        images = a.out / split / "images"
        labels = a.out / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        for n, r in enumerate(records):
            raw = r["image"]["bytes"]
            im = Image.open(io.BytesIO(raw))
            w, h = im.size
            x0, y0, x1, y1 = r["bbox"]
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                rejected.append({"split": split, "index": n, "bbox": r["bbox"], "size": [w, h]})
                continue
            stem = f"cub_{split}_{n:05d}"
            im.convert("RGB").save(images / f"{stem}.jpg", quality=95)
            (labels / f"{stem}.txt").write_text(
                f"0 {(x0+x1)/2/w:.9f} {(y0+y1)/2/h:.9f} {(x1-x0)/w:.9f} {(y1-y0)/h:.9f}\n"
            )
            provenance.append(
                {
                    "output": stem,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "species_id": r["label"],
                    "official_split": "test" if split == "test" else "train",
                }
            )
    (a.out / "data.yaml").write_text(yaml.safe_dump({"names": {0: "bird"}}))
    (a.out / "source.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "revision": REV,
                "scope": "Лише шард 0 із 2 офіційних train/test; це не повний CUB і не Roboflow Birds",
                "seed": 26,
                "license": "Лише некомерційні дослідження та навчання; права на зображення належать власникам",
                "official_source": "https://www.vision.caltech.edu/datasets/cub_200_2011/",
                "sha256": {s: hashlib.sha256(f.read_bytes()).hexdigest() for s, f in files.items()},
                "rejected_invalid_bbox": rejected,
                "records": provenance,
            },
            indent=2,
        )
    )
    print("Записано", len(provenance), "виключено некоректних bbox:", len(rejected))


if __name__ == "__main__":
    main()
