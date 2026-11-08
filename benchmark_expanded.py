#!/usr/bin/env python3
"""Короткий CPU bbox-benchmark; без оцінювання test і без заміни робочих ваг."""

import hashlib, json, random, subprocess, sys, time
from pathlib import Path
from common import ROOT, UkrainianParser
import yaml


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/bird-expanded/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", default="expanded-benchmark")
    a = p.parse_args()
    root = a.data.resolve().parent
    if not a.data.is_file():
        p.error("Спочатку prepare_expanded.py.")
    run = ROOT / "runs/experiments" / a.name
    if run.exists():
        p.error("Запуск уже існує; задайте нове --name.")
    work = ROOT / "work" / a.name
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(26)
    counts = {}
    sample_counts = {}
    for split, size in [("train", 128), ("val", 64)]:
        images = sorted((root / split / "images").glob("*.jpg"))
        counts[split] = len(images)
        rng.shuffle(images)
        images = images[:size]
        sample_counts[split] = len(images)
        if not images:
            p.error("Порожня вибірка " + split)
        (work / f"{split}.txt").write_text("".join(str(x.resolve()) + "\n" for x in images))
    cfg = work / "data.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": str(work / "train.txt"),
                "val": str(work / "val.txt"),
                "names": {0: "bird"},
            }
        )
    )
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_boxes.py"),
            "--data",
            str(cfg),
            "--epochs",
            "1",
            "--imgsz",
            str(a.imgsz),
            "--model",
            str(ROOT / "weights/best.pt"),
            "--name",
            a.name,
        ],
        check=True,
    )
    wall = time.perf_counter() - start
    timings = json.loads((run / "phase_timing.json").read_text())
    train = timings["train_seconds"][0]
    val = timings["validation_seconds"][0]
    projected = (
        train * counts["train"] / sample_counts["train"]
        + val * counts["val"] / sample_counts["val"]
    )
    report = {
        "weights_sha256": hashlib.sha256((ROOT / "weights/best.pt").read_bytes()).hexdigest(),
        "device": "cpu",
        "imgsz": a.imgsz,
        "batch": 8,
        "strategy": "bbox_only",
        "sample_counts": sample_counts,
        "full_counts": counts,
        "wall_seconds": wall,
        "phase_timing": timings,
        "projected_epoch_seconds": projected,
        "projected_3_epochs_hours": 3 * projected / 3600,
        "estimated_range_3_epochs_hours": [3 * projected * 0.6 / 3600, 3 * projected * 1.4 / 3600],
        "note": "Екстраполяція малого холодного запуску, не гарантія часу. Test не використовувався; робочі ваги не замінено. Benchmark-checkpoint не є кандидатом на реліз.",
    }
    out = ROOT / "reports/expanded_benchmark.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
