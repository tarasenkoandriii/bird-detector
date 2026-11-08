#!/usr/bin/env python3
"""Повний CUB train із захистом попередніх split і новим резервним test."""

import hashlib
import io
import json
import random
import subprocess
from pathlib import Path
from common import ROOT, UkrainianParser
from tools.download_public_cub import REPO, REV

SHARDS = {
    "train-00000-of-00002.parquet": "fb4590d7fbd018c1cd0df7674dd4055ca91312df72790a2fddcab74e15fcefa4",
    "train-00001-of-00002.parquet": "7d56af56f59182c2544cbba974c92a9c69e7e72f984b035d2d9ea7af23cd763c",
    "test-00000-of-00002.parquet": "fb0c8ac3d2357353689f07444f0d29e27a36f76faaaad1a13451494816a5c608",
    "test-00001-of-00002.parquet": "180fe23f22750c6f9e96f4ce986be8b1268cbb3c6dba6222ac902704ae24a286",
}


def choose_splits(records, old_train, old_val, old_test, new_val_count=400, test_count=500):
    rng = random.Random(26)
    available = [x for x in records["train"] if x[0] not in old_train | old_val | old_test]
    rng.shuffle(available)
    new_val = {h for h, _ in available[:new_val_count]}
    val_ids = old_val | new_val
    test = [x for x in records["test"] if x[0] not in old_train | old_val | old_test]
    rng.shuffle(test)
    subsets = {
        "train": [x for x in records["train"] if x[0] not in val_ids | old_test],
        "val": [x for x in records["train"] if x[0] in val_ids],
        "test": test[:test_count],
    }
    return subsets, new_val


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=ROOT / "datasets/bird/source.json")
    p.add_argument("--out", type=Path, default=ROOT / "datasets/bird-expanded")
    p.add_argument("--cache", type=Path, default=ROOT / "datasets/cache-expanded")
    a = p.parse_args()
    raw = a.out.with_name(a.out.name + "-raw")
    if a.out.exists() or raw.exists():
        p.error("Результат уже існує; задайте інший --out.")
    if not a.previous.is_file() or not a.previous.with_name("audit.json").is_file():
        p.error("Потрібен source.json попередньої вибірки для захисту split.")
    import pyarrow.parquet as pq
    from PIL import Image
    import yaml
    from tools.dataset import audit, prepare

    old = json.loads(a.previous.read_text())
    old_train = {x["source_sha256"] for x in old["records"] if x["output"].startswith("cub_train_")}
    old_val = {x["source_sha256"] for x in old["records"] if x["output"].startswith("cub_val_")}
    old_test = {x["source_sha256"] for x in old["records"] if x["output"].startswith("cub_test_")}
    if not old_train or not old_val or not old_test:
        p.error("Нерозпізнаний попередній manifest.")
    a.cache.mkdir(parents=True, exist_ok=True)
    records = {"train": [], "test": []}
    rejected = []
    for name, sha in SHARDS.items():
        f = a.cache / name
        if not f.exists():
            print("Завантаження", name, flush=True)
            tmp = f.with_suffix(".part")
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
                    "-C",
                    "-",
                    "-o",
                    str(tmp),
                    f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/data/{name}",
                ],
                check=True,
            )
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha:
                raise ValueError("SHA256 не збігається: " + name)
            tmp.rename(f)
        if hashlib.sha256(f.read_bytes()).hexdigest() != sha:
            raise ValueError("Пошкоджений кеш: " + name)
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                blob = row["image"]["bytes"]
                h = hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as im:
                    w, height = im.size
                x0, y0, x1, y1 = row["bbox"]
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= height):
                    rejected.append({"source_sha256": h, "bbox": row["bbox"], "size": [w, height]})
                    continue
                records[name.split("-")[0]].append((h, row))
    subsets, new_val = choose_splits(records, old_train, old_val, old_test)
    raw.mkdir(parents=True)
    mapping = {}
    provenance = []
    for split, items in subsets.items():
        (raw / split / "images").mkdir(parents=True)
        (raw / split / "labels").mkdir()
        for h, row in items:
            ip = raw / split / "images" / f"{h[:24]}.jpg"
            with Image.open(io.BytesIO(row["image"]["bytes"])) as im:
                w, height = im.size
                im.convert("RGB").save(ip, quality=95)
            x0, y0, x1, y1 = row["bbox"]
            (raw / split / "labels" / f"{h[:24]}.txt").write_text(
                f"0 {(x0+x1)/2/w:.9f} {(y0+y1)/2/height:.9f} {(x1-x0)/w:.9f} {(y1-y0)/height:.9f}\n"
            )
            mapping[str(ip.resolve())] = h
            provenance.append({"source_sha256": h, "split": split, "species_id": row["label"]})
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": {0: "bird"}}))
    print("Аудит", {k: len(v) for k, v in subsets.items()}, flush=True)
    report = audit(raw, distance=4)
    previous_audit = json.loads(a.previous.with_name("audit.json").read_text())
    old_test_records = [r for r in previous_audit["records"] if r["split"] == "test"]
    protected_test_matches = []
    for r in report["records"]:
        if any(
            r["pixel"] == old["pixel"]
            or r["raw"] == old["raw"]
            or (
                r["hash_usable"]
                and old["hash_usable"]
                and (r["dhash"] ^ old["dhash"]).bit_count() <= 4
            )
            for old in old_test_records
        ):
            protected_test_matches.append(r)
    for r in protected_test_matches:
        Path(r["image"]).unlink()
        Path(r["label"]).unlink()
    if protected_test_matches:
        report = audit(raw, distance=4)
    # Нові val-кандидати, схожі на раніше навчальні кадри, не є новою validation.
    prior_groups = {r["group"] for r in report["records"] if mapping[r["image"]] in old_train}
    excluded = [
        r
        for r in report["records"]
        if r["split"] == "val" and mapping[r["image"]] in new_val and r["group"] in prior_groups
    ]
    for r in excluded:
        Path(r["image"]).unlink()
        Path(r["label"]).unlink()
    if excluded:
        report = audit(raw, distance=4)
    counts = prepare(report, a.out, 26, None)
    post = audit(a.out, distance=4)
    if post["errors"] or any(x["leakage"] for x in post["pairs"]):
        raise RuntimeError("Після підготовки аудит не пройшов.")
    (a.out / "source.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "revision": REV,
                "shards_sha256": SHARDS,
                "seed": 26,
                "license": old["license"],
                "previous_source_sha256": hashlib.sha256(a.previous.read_bytes()).hexdigest(),
                "records_before_deduplication": provenance,
                "rejected_invalid_bbox": rejected,
                "new_val_excluded_for_previous_train_similarity": len(excluded),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    (a.out / "audit.json").write_text(json.dumps(post, indent=2, ensure_ascii=False))
    summary = {
        "repo": REPO,
        "revision": REV,
        "audit_errors": len(post["errors"]),
        "counts": counts,
        "species": len({x["species_id"] for x in provenance}),
        "invalid_bbox_rejected": len(rejected),
        "pairs_before": len(report["pairs"]),
        "pairs_after": len(post["pairs"]),
        "leakage_after": sum(x["leakage"] for x in post["pairs"]),
        "previous_train": len(old_train),
        "previous_val": len(old_val),
        "previous_test_excluded": len(old_test),
        "additional_previous_test_similarity_excluded": len(protected_test_matches),
        "new_val_excluded_for_previous_train_similarity": len(excluded),
        "note": "Test не оцінювався моделлю. DHash — евристика, не гарантія відсутності всіх семантичних дублікатів. CUB не є відеодатасетом.",
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
