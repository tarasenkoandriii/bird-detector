#!/usr/bin/env python3
"""Download and prepare the pinned non-commercial CUB research subset."""

import argparse, json, shutil, subprocess, sys
from pathlib import Path
from common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=ROOT / "datasets/cache")
    p.add_argument(
        "--source",
        type=Path,
        help="Use an existing Roboflow-style YOLO export instead of downloading CUB",
    )
    p.add_argument("--out", type=Path, default=ROOT / "datasets/bird")
    a = p.parse_args()
    if a.out.exists():
        p.error("Output exists. Use a new --out; existing data is never overwritten.")
    source = a.source.resolve() if a.source else a.out.resolve().with_name(a.out.name + "-raw")
    if not a.source:
        if source.exists():
            p.error("Raw directory exists. Pass --source or choose a new --out.")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/download_public_cub.py"),
                "--out",
                str(source),
                "--cache",
                str(a.cache.resolve()),
            ],
            check=True,
        )
    from tools.dataset import audit, prepare

    report = audit(source, distance=4)
    target = a.out.resolve()
    counts = prepare(report, target, 26, None)
    # Keep audit evidence and data lineage with the generated dataset.
    (target / "audit.json").write_text(json.dumps(report, indent=2))
    if (source / "source.json").is_file():
        shutil.copy2(source / "source.json", target / "source.json")
    print("Prepared:", counts, "\nData YAML:", target / "data.yaml")


if __name__ == "__main__":
    main()
