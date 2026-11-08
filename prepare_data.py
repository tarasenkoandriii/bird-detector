#!/usr/bin/env python3
"""Завантаження та підготовка зафіксованої дослідницької підвибірки CUB."""

import json, shutil, subprocess, sys
from pathlib import Path
from common import UkrainianParser, ROOT


def main():
    p = UkrainianParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=ROOT / "datasets/cache")
    p.add_argument(
        "--source",
        type=Path,
        help="Використати наявний YOLO export замість завантаження CUB",
    )
    p.add_argument("--out", type=Path, default=ROOT / "datasets/bird")
    a = p.parse_args()
    if a.out.exists():
        p.error(
            "Каталог результату вже існує. Задайте новий --out; наявні дані не перезаписуються."
        )
    source = a.source.resolve() if a.source else a.out.resolve().with_name(a.out.name + "-raw")
    if not a.source:
        if source.exists():
            p.error("Каталог вихідних даних уже існує. Задайте --source або новий --out.")
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
    # Зберігаємо звіт аудиту та походження даних разом із датасетом.
    (target / "audit.json").write_text(json.dumps(report, indent=2))
    if (source / "source.json").is_file():
        shutil.copy2(source / "source.json", target / "source.json")
    print("Підготовлено:", counts, "\nYAML датасету:", target / "data.yaml")


if __name__ == "__main__":
    main()
